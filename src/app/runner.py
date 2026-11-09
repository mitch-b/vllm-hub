"""Starts/stops the vLLM container as a sibling of this app through the Docker socket."""
from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path
from threading import Event, RLock

import docker
import httpx
from docker.errors import DockerException, ImageNotFound, NotFound
from docker.types import DeviceRequest
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HOME

from . import capacity, profiles
from .gpu import gpu_info, reset_probe

CONTAINER = os.getenv("VLLM_CONTAINER", "vllm")
IMAGE = os.getenv("VLLM_IMAGE", "vllm/vllm-openai:latest")
NETWORK = os.getenv("VLLM_NETWORK", "vllm-hub-net")
HOST_PORT = int(os.getenv("VLLM_PORT", "8000"))
# Where the *Docker host* finds the HF cache (named volume or host path); shared with vLLM.
CACHE_SOURCE = os.getenv("HF_CACHE_SOURCE", "vllm-hub-hf-cache")
BASE_URL = os.getenv("VLLM_BASE_URL", f"http://{CONTAINER}:8000")

_client: docker.DockerClient | None = None
_lifecycle_lock = RLock()
_start_cancel: Event | None = None


class StartCancelled(RuntimeError):
    pass


class GPUUnavailable(RuntimeError):
    pass


def client() -> docker.DockerClient:
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


def _container():
    try:
        return client().containers.get(CONTAINER)
    except NotFound:
        return None


def _ensure_network() -> None:
    try:
        client().networks.get(NETWORK)
    except NotFound:
        client().networks.create(NETWORK)


def _timestamp(value: str) -> float | None:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.timestamp() if dt.year > 1970 and dt.tzinfo else None
    except (ValueError, TypeError):
        return None


_ERROR_LINE = re.compile(r"(?:Error|Exception)\b.*", re.IGNORECASE)


def _crash_reason(c) -> str:
    """Last error from a crash-looping container, with a hint for unsupported-architecture failures."""
    try:
        logs = c.logs(tail=60).decode(errors="replace")
    except DockerException:
        return "vLLM keeps crashing; open Console for details."
    logs = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", logs)
    lines = [m.group(0).strip() for line in logs.splitlines() if (m := _ERROR_LINE.search(line))]
    reason = lines[-1][:300] if lines else "vLLM keeps crashing; open Console for details."
    if re.search(r"does not recognize this architecture|not supported|unsupported", logs, re.IGNORECASE):
        reason += (" The vLLM image may be too old for this model: set VLLM_IMAGE to a newer image,"
                   " then Stop and Run again.")
    return reason


def _startup(c) -> dict:
    """Report only milestones and counts actually emitted by this startup attempt."""
    started = _timestamp(c.attrs.get("State", {}).get("StartedAt", ""))
    active = c.status == "running"
    result = {
        "container_id": c.id, "started_at": started if active else None,
        "elapsed_seconds": max(0, time.time() - started) if active and started else None,
        "stage": "Starting engine" if active else "Waiting for container",
        "progress": None, "last_log_at": None, "error": None,
    }
    if c.status == "restarting":
        result["stage"] = "Container restarting"
        result["error"] = _crash_reason(c)
    if not active:
        return result
    if not started:
        result["error"] = "Container startup timestamp unavailable"
        return result
    try:
        logs = c.logs(tail=300, since=int(started), timestamps=True).decode(errors="replace")
    except DockerException as exc:
        result["error"] = f"Startup logs unavailable: {exc}"
        return result
    stamp = None
    # Bound parsing work, including unusually large individual log records.
    for line in logs[-262144:].splitlines():
        timestamp, _, message = line.partition(" ")
        parsed = _timestamp(timestamp)
        if parsed is not None:
            stamp = parsed
            line = message
        if stamp is None or stamp < started:
            continue
        line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
        if not line.strip():
            continue
        result["last_log_at"] = max(result["last_log_at"] or stamp, stamp)
        lower = line.lower()
        stage = None
        if re.search(r"loading (?:safetensors|pt|np_?cache) checkpoint shards"
                     r"|multi-thread loading (?:pt )?shards", lower):
            stage = "Loading weights"
        elif "capturing cuda graph" in lower or "capturing cudagraph" in lower:
            stage = "Capturing CUDA graphs"
        elif "graph capturing finished" in lower or "cuda graph capturing finished" in lower:
            stage = "Waiting for API readiness"
        elif "init engine" in lower or "application startup complete" in lower:
            stage = "Waiting for API readiness"
        elif "torch.compile" in lower or "compiling" in lower:
            stage = "Compiling model"
        elif ("loading model weights took" in lower or "model loading took" in lower
              or "gpu kv cache size" in lower or "available kv cache memory" in lower
              or "warming up" in lower or "warmup" in lower or "profile run" in lower):
            stage = "Warming up"
        elif "starting to load model" in lower:
            stage = "Loading weights"
        if stage:
            result["stage"], result["progress"] = stage, None
            if stage in ("Loading weights", "Capturing CUDA graphs"):
                count = re.search(r"\b(\d+)\s*/\s*(\d+)\b", line)
                if count:
                    done, total = map(int, count.groups())
                    if 0 <= done <= total and total > 0:
                        result["progress"] = {"completed": done, "total": total}
    return result


def status() -> dict:
    try:
        c = _container()
    except Exception as exc:
        return {"docker_ok": False, "error": str(exc), "state": "unknown"}
    if c is None:
        return {"docker_ok": True, "state": "stopped", "model": None, "ready": False}
    state = c.status
    model = c.labels.get("vllm-hub.model")
    ready, served = False, None
    if state == "running":
        try:
            r = httpx.get(f"{BASE_URL}/v1/models", timeout=2)
            if r.status_code == 200:
                ready, served = True, r.json()["data"][0]
        except Exception:
            pass
    kv = None
    if ready and model and c.labels.get("vllm-hub.util"):
        try:
            kv = capacity.learn(c, model, float(c.labels["vllm-hub.util"]))
        except Exception:
            pass
    return {
        "docker_ok": True, "state": state, "model": model, "ready": ready, "served": served,
        "error": _crash_reason(c) if state in ("exited", "dead")
        and (state == "dead" or c.attrs.get("State", {}).get("ExitCode", 0) != 0) else None,
        "util": c.labels.get("vllm-hub.util"),
        "image": c.attrs.get("Config", {}).get("Image"),
        "container_id": c.id,
        "port": HOST_PORT, "context": c.labels.get("vllm-hub.context"), "kv_tokens": kv,
        "url": f"http://localhost:{HOST_PORT}/v1",
        "startup": _startup(c) if not ready and state in ("running", "created", "restarting") else None,
    }


def stop() -> None:
    with _lifecycle_lock:
        if _start_cancel is not None:
            _start_cancel.set()
        c = _container()
        if c is not None:
            c.remove(force=True)
        reset_probe()


def _wait_vram_free(cancel: Event, timeout: float = 20) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cancel.is_set():
            raise StartCancelled("Model start cancelled")
        g = gpu_info()
        if not g or g["used_mib"] < 0.25 * g["total_mib"]:
            return
        if cancel.wait(1):
            raise StartCancelled("Model start cancelled")


def _spark_parser() -> str:
    # Keep the official plugin in the shared HF cache; no additional host bind mount is needed.
    path = hf_hub_download(profiles.NEMOTRON_SUPER, "super_v3_reasoning_parser.py",
                           revision="ff433f5493e25d631c9f12b5d55c674229923d02")
    relative = Path(path).relative_to(Path(HF_HOME))
    return str(Path("/root/.cache/huggingface") / relative)


def _ensure_image(image: str) -> None:
    try:
        client().images.get(image)
    except ImageNotFound:
        client().images.pull(image)


def start(model_id: str, context: int | None = None, util: float | None = None,
          graphs: bool | None = None, extra: list[str] | None = None, weights_gib: float | None = None) -> dict:
    global _start_cancel
    initial_gpu = gpu_info()
    if initial_gpu is None:
        raise GPUUnavailable("CUDA vLLM requires an NVIDIA GPU. Enable Docker Model Runner for Metal or CPU inference.")
    spark = profiles.spark_recipe(model_id, initial_gpu)
    if spark and context is not None and not 2048 <= context <= profiles.SPARK_CONTEXT:
        raise ValueError("NVIDIA DGX Spark recipe context must be between 2048 and 1000000 tokens.")
    with _lifecycle_lock:
        stop()
        cancel = _start_cancel = Event()
    _wait_vram_free(cancel)
    flags = profiles.spark_flags(_spark_parser()) if spark else None
    if cancel.is_set():
        raise StartCancelled("Model start cancelled")
    image = profiles.SPARK_IMAGE if spark else IMAGE
    if spark:
        _ensure_image(image)
    g = gpu_info()
    if util is None:
        util = 0.90 if spark else profiles.choose_util(g["total_mib"], g["used_mib"]) if g else 0.85
    if not context:
        total_gib = (g["total_mib"] / 1024) if g else 24
        context = (capacity.auto_context(model_id, util, profiles.SPARK_CONTEXT) or profiles.SPARK_CONTEXT
                   ) if spark else (capacity.auto_context(model_id, util)
                                    or profiles.choose_context(weights_gib, total_gib, util))
    args = profiles.build_vllm_args(model_id, context, util, graphs=spark if graphs is None else graphs,
                                   max_seqs=4 if spark else 2, extra=extra, flags=flags)

    env = dict(profiles.SPARK_ENV) if spark else {}
    if os.getenv("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    with _lifecycle_lock:
        if cancel.is_set():
            raise StartCancelled("Model start cancelled")
        _ensure_network()
        client().containers.run(
            image, _command(args, image), name=CONTAINER, detach=True, ipc_mode="host",
            restart_policy={"Name": "unless-stopped"},  # comes back with Docker after a reboot
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
            ports={"8000/tcp": HOST_PORT}, network=NETWORK, environment=env,
            volumes={CACHE_SOURCE: {"bind": "/root/.cache/huggingface", "mode": "rw"}},
            labels={"vllm-hub.model": model_id, "vllm-hub.context": str(context), "vllm-hub.util": f"{util:.2f}",
                    "vllm-hub.managed": "true"},
        )
    return {"model": model_id, "context": context, "util": util, "args": args, "image": image}


def _command(args: list[str], image: str | None = None) -> list[str]:
    # NVIDIA's NGC vLLM image (DGX Spark) has a generic entrypoint, so it needs the explicit `vllm serve`.
    if "nvidia/vllm" in (image or IMAGE) and args[:1] == ["--model"]:
        return ["vllm", "serve", args[1], *args[2:]]
    return args


def log_stream(tail: int = 100):
    c = _container()
    if c is None:
        yield b"vLLM is not running.\n"
        return
    yield from c.logs(stream=True, follow=True, tail=tail)


def chat(messages: list[dict], thinking: bool = False, max_tokens: int = 512) -> dict:
    s = status()
    if not s.get("ready"):
        raise RuntimeError("vLLM is not ready yet.")
    body = {
        "model": s["served"]["id"], "messages": messages, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=body, timeout=300)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    usage = r.json().get("usage", {})
    return {"content": msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or "",
            "usage": usage}
