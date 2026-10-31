"""Docker Model Runner's host-native inference API (Metal, CUDA, or CPU)."""
from __future__ import annotations

import html
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from threading import Event, RLock, Thread
from urllib.parse import quote

import httpx

from . import downloads, hf

STATE = Path(os.getenv("HUB_STATE_DIR", "/state")) / "model-runner.json"
_lock = RLock()
_selection: dict | None = None
_cancel: Event | None = None
_discovery: dict = {"at": 0.0, "connection": None, "error": None}
_discovery_lock = RLock()
RECOMMENDED = ["ai/qwen3:0.6B-Q4_K_M", "ai/qwen3:8B-Q4_K_M"]
logger = logging.getLogger(__name__)


def _http(connection: dict, timeout: float = 10) -> httpx.Client:
    transport = httpx.HTTPTransport(uds="/var/run/docker.sock") if connection.get("socket") else None
    return httpx.Client(base_url=connection["url"], transport=transport, timeout=timeout, trust_env=False)


def request(connection: dict, method: str, path: str, **kwargs) -> httpx.Response:
    timeout = kwargs.pop("timeout", 10)
    with _http(connection, timeout) as client:
        response = client.request(method, path, **kwargs)
        if response.is_error:
            raise RuntimeError(f"Model Runner ({response.status_code}): {response.text[:1000]}")
        return response


def discover() -> dict | None:
    with _discovery_lock:
        if time.monotonic() - _discovery["at"] < 10:
            return _discovery["connection"]
        configured = os.getenv("HUB_MODEL_RUNNER_URL")
        candidates = [{"url": configured}] if configured else [
            {"url": "http://localhost/exp/vDD4.40/", "socket": True},
            {"url": "http://model-runner.docker.internal:12434/"},
            {"url": "http://host.docker.internal:12434/"},
        ]
        errors = []
        connection = None
        for candidate in candidates:
            try:
                response = request(candidate, "GET", "models", timeout=1)
                if not isinstance(response.json(), list):
                    raise ValueError("Model Runner returned an invalid model list")
                engines = request(candidate, "GET", "engines/status", timeout=1).json()
                if not isinstance(engines, dict) or not all(isinstance(value, str) for value in engines.values()):
                    raise ValueError("Model Runner returned invalid engine status")
                connection = {**candidate, "engines": engines}
                break
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                errors.append(str(exc))
        _discovery.update(at=time.monotonic(), connection=connection, error="; ".join(errors) or None)
        return connection


def connection() -> dict:
    found = discover()
    if found is None:
        raise RuntimeError("Docker Model Runner is unavailable. Enable it in Docker Desktop Settings > AI.")
    return found


def _size_gib(size: str) -> float | None:
    match = re.fullmatch(r"([\d.]+)\s*(GiB|MiB|KiB|GB|MB|KB|B)", size)
    if not match:
        return None
    value, unit = match.groups()
    factor = {"GiB": 2**30, "MiB": 2**20, "KiB": 2**10, "GB": 10**9, "MB": 10**6, "KB": 10**3, "B": 1}
    return round(float(value) * factor[unit] / 2**30, 2)


def _reference(value: str) -> str:
    value = value.removeprefix("docker.io/")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:-]*", value) or any(
        part in ("", ".", "..") for part in value.split("/")
    ):
        raise ValueError("Invalid Model Runner model reference")
    return value


def _model_path(reference: str) -> str:
    return "models/" + quote(_reference(reference), safe="/:")


def _model(model: dict, ref: str | None = None) -> dict:
    config = model.get("config") or {}
    tags = model.get("tags") or []
    model_id = _reference(ref or (tags[0] if tags else model["id"]))
    context = config.get("context_size")
    if not context:
        gguf = config.get("gguf") or {}
        context = gguf.get(f"{config.get('architecture')}.context_length")
    maximum = int(context) if context else None
    context = min(maximum or 8192, 8192)
    return {
        "id": model_id, "size_gib": _size_gib(config.get("size") or ""), "last_used": 0,
        "context_hint": context, "max_context": maximum, "context_learned": False,
        "quant": config.get("quantization") or config.get("format"),
        "verdict": "unknown", "reason": "Model Runner selects the native engine and manages host memory.",
        "headroom_gib": None, "backend": "model-runner",
    }


def models() -> list[dict]:
    return [_model(m) for m in request(connection(), "GET", "models").json()]


def recommended() -> list[dict]:
    conn = connection()
    return [_model(request(conn, "GET", _model_path(ref), params={"remote": "true"}).json(), ref)
            for ref in RECOMMENDED]


def search(query: str, limit: int = 25) -> list[dict]:
    # GGUF is portable across Model Runner's Metal/CUDA/CPU engines.
    results = hf.api.list_models(search=query, filter="gguf", pipeline_tag="text-generation",
                                 sort="downloads", limit=min(limit, 50))
    return [{
        "id": "hf.co/" + m.id + ":Q4_K_M", "quant": "GGUF Q4_K_M", "verdict": "unknown",
        "reason": "Model Runner imports this GGUF model. The repository must provide Q4_K_M weights.",
        "downloads": m.downloads, "likes": m.likes, "gated": bool(m.gated),
    } for m in results]


def _pull(job: downloads.Job) -> None:
    layers: dict[str, tuple[int, int]] = {}
    try:
        conn = connection()
        with _http(conn, 30) as client, client.stream(
            "POST", "models/create", json={"from": job.repo_id, "bearer-token": os.getenv("HF_TOKEN", "")},
        ) as response:
            if response.is_error:
                response.read()
                raise RuntimeError(f"Model Runner ({response.status_code}): {response.text[:1000]}")
            job.status = "downloading"
            for line in response.iter_lines():
                if job.cancel.is_set():
                    raise downloads.Cancelled()
                if not line:
                    continue
                try:
                    progress = json.loads(line)
                except json.JSONDecodeError:
                    # Desktop versions emit HTML-escaped JSON in their text progress stream.
                    progress = json.loads(html.unescape(line))
                if not isinstance(progress, dict):
                    raise ValueError("Model Runner returned invalid download progress")
                error_detail = progress.get("errorDetail")
                if progress.get("error") or error_detail or progress.get("type") == "error":
                    message = error_detail.get("message") if isinstance(error_detail, dict) else error_detail
                    raise RuntimeError(
                        progress.get("error") or message or progress.get("message") or "Native pull failed"
                    )
                detail = progress.get("layer", progress.get("progressDetail", {}))
                if detail is None:
                    detail = {}
                if not isinstance(detail, dict):
                    raise ValueError("Model Runner returned invalid byte counts")
                total = detail.get("size", detail.get("total", 0))
                current = detail.get("current", 0)
                overall = progress.get("total", 0)
                if any(type(value) is not int or value < 0 for value in (total, current, overall)) or current > total:
                    raise ValueError("Model Runner returned invalid byte counts")
                if total:
                    layer_id = detail.get("id", progress.get("id", ""))
                    if not isinstance(layer_id, str):
                        raise ValueError("Model Runner returned an invalid layer ID")
                    layers[layer_id] = (current, total)
                    with job._lock:
                        job.total = overall or sum(size for _, size in layers.values())
                    job.add(sum(done for done, _ in layers.values()) - job.done)
        if job.cancel.is_set():
            raise downloads.Cancelled()
        if not any(m["id"].removesuffix(":latest") == job.repo_id.removesuffix(":latest") for m in models()):
            raise RuntimeError("Model Runner finished the pull without registering the requested model")
        with job._lock:
            job.done, job.status = job.total, "done"
    except downloads.Cancelled:
        job.status = "cancelled"
    except (httpx.HTTPError, RuntimeError, ValueError, KeyError) as exc:
        job.status, job.error = ("cancelled", None) if job.cancel.is_set() else ("error", str(exc))
    finally:
        job.finished = time.time()


def pull(repo_id: str) -> downloads.Job:
    repo_id = _reference(repo_id)
    if ":" not in repo_id.rsplit("/", 1)[-1]:
        repo_id += ":latest"
    return downloads.start(repo_id, worker=_pull)


def delete(repo_id: str) -> bool:
    repo_id = _reference(repo_id)
    if selected_model() == repo_id:
        stop()
    request(connection(), "DELETE", _model_path(repo_id))
    downloads.forget_completed(repo_id)
    return True


def _load() -> dict:
    global _selection
    if _selection is None:
        try:
            _selection = json.loads(STATE.read_text())
        except FileNotFoundError:
            _selection = {}
    return _selection


def _save(selection: dict) -> None:
    global _selection
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(selection))
    temp.replace(STATE)
    _selection = selection


def selected_model() -> str | None:
    with _lock:
        return _load().get("model")


def _record_error(selection: dict, message: str) -> None:
    global _selection
    failed = {**selection, "state": "exited", "error": message}
    try:
        _save(failed)
    except OSError as exc:
        _selection = {**failed, "error": f"{message}; could not save native state: {exc}"}
        logger.exception("Could not persist Model Runner failure state")


def _unload(conn: dict, model: str) -> None:
    request(conn, "POST", "engines/unload", json={"models": [model]})


def stop() -> None:
    global _cancel
    with _lock:
        selected = _load()
        if _cancel:
            _cancel.set()
        if selected.get("model"):
            _unload(connection(), selected["model"])
        _save({})


def _preload(conn: dict, selection: dict, cancel: Event) -> None:
    try:
        request(conn, "POST", "engines/v1/chat/completions",
                json={"model": selection["model"], "messages": []},
                headers={"X-Preload-Only": "true"}, timeout=300)
        with _lock:
            current = _load()
            if cancel.is_set() or current.get("attempt") != selection["attempt"]:
                if current.get("model") != selection["model"]:
                    _unload(conn, selection["model"])
                return
            _save({**selection, "state": "running"})
    except (httpx.HTTPError, RuntimeError, OSError) as exc:
        with _lock:
            if not cancel.is_set() and _load().get("attempt") == selection["attempt"]:
                _record_error(selection, str(exc))


def start(repo_id: str, context: int | None = None, util=None, graphs=False, extra=None, weights_gib=None) -> dict:
    global _cancel
    repo_id = _reference(repo_id)
    if graphs or util is not None or extra:
        raise ValueError("CUDA graph, GPU utilization and extra CUDA flags do not apply to Model Runner.")
    conn = connection()
    metadata = request(conn, "GET", _model_path(repo_id)).json()
    model = _model(metadata, repo_id)
    context = context or model["context_hint"]
    maximum = model["max_context"]
    if context <= 0 or (maximum is not None and context > maximum):
        raise ValueError(f"Context must be between 1 and {maximum} tokens." if maximum else
                         "Context must be a positive number of tokens.")
    with _lock:
        stop()
        selection = {"model": repo_id, "model_id": metadata["id"], "context": context, "state": "created",
                     "attempt": uuid.uuid4().hex, "started_at": time.time()}
        _save(selection)
        try:
            request(conn, "POST", "engines/_configure",
                    json={"model": repo_id, "context-size": context, "keep_alive": "-1"})
        except (httpx.HTTPError, RuntimeError) as exc:
            _record_error(selection, str(exc))
            raise
        cancel = _cancel = Event()
        Thread(target=_preload, args=(conn, selection, cancel), daemon=True).start()
    return {"model": repo_id, "context": context, "util": None, "args": [], "backend": "model-runner"}


def status() -> dict:
    conn = connection()
    with _lock:
        selected = dict(_load())
    model = selected.get("model")
    running = request(conn, "GET", "engines/ps").json() if model else []
    aliases = [model, "docker.io/" + model, selected.get("model_id")] if model else []
    if model and not selected.get("model_id"):
        aliases.append(request(conn, "GET", _model_path(model)).json()["id"])
    loaded = next((r for r in running if r["model_name"] in aliases), None)
    state = selected.get("state", "stopped")
    ready = loaded is not None and not loaded.get("loading")
    if ready:
        state = "running"
    if state == "running" and not ready:
        state = "created" if loaded and loaded.get("loading") else "stopped"
    context = selected.get("context")
    return {
        "docker_ok": True, "state": state, "ready": ready, "model": model,
        "context": context, "container_id": selected.get("attempt"), "port": 12434,
        "url": os.getenv("HUB_MODEL_RUNNER_PUBLIC_URL", "http://localhost:12434/engines/v1"),
        "served": {"id": model, "max_model_len": context} if ready else None,
        "backend": "model-runner", "engine": loaded.get("backend_name") if loaded else None,
        "error": selected.get("error"), "kv_tokens": None,
        "startup": {
            "container_id": selected["attempt"], "started_at": selected["started_at"],
            "elapsed_seconds": max(0, time.time() - selected["started_at"]),
            "stage": "Loading model with Docker Model Runner", "progress": None, "last_log_at": None,
            "message": "Model Runner is starting the native engine. Detailed logs are available in Docker Desktop.",
        } if state == "created" else None,
    }


def resume() -> None:
    with _lock:
        selected = dict(_load())
    if selected.get("model") and selected.get("state") in ("created", "running"):
        try:
            if not status()["ready"]:
                start(selected["model"], selected["context"])
        except (httpx.HTTPError, RuntimeError, ValueError, OSError) as exc:
            with _lock:
                _record_error(selected, f"Could not restore native model: {exc}")


def chat(messages: list[dict], thinking: bool = False, max_tokens: int = 512) -> dict:
    selected = status()
    if not selected["ready"]:
        raise RuntimeError("Model Runner is not ready yet.")
    response = request(connection(), "POST", "engines/v1/chat/completions", timeout=300, json={
        "model": selected["model"], "messages": messages, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }).json()
    message = response["choices"][0]["message"]
    return {"content": message.get("content") or message.get("reasoning_content") or "",
            "usage": response.get("usage", {})}
