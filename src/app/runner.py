"""Starts/stops the vLLM container as a sibling of this app through the Docker socket."""
from __future__ import annotations

import os
import re
import time
from datetime import datetime

import docker
import httpx
from docker.errors import DockerException, NotFound
from docker.types import DeviceRequest

from . import capacity, profiles
from .gpu import gpu_info

CONTAINER = os.getenv("VLLM_CONTAINER", "vllm")
IMAGE = os.getenv("VLLM_IMAGE", "vllm/vllm-openai:latest")
NETWORK = os.getenv("VLLM_NETWORK", "vllm-hub-net")
HOST_PORT = int(os.getenv("VLLM_PORT", "8000"))
# Where the *Docker host* finds the HF cache (named volume or host path); shared with vLLM.
CACHE_SOURCE = os.getenv("HF_CACHE_SOURCE", "vllm-hub-hf-cache")
BASE_URL = os.getenv("VLLM_BASE_URL", f"http://{CONTAINER}:8000")

_client: docker.DockerClient | None = None


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
        "port": HOST_PORT, "context": c.labels.get("vllm-hub.context"), "kv_tokens": kv,
        "url": f"http://localhost:{HOST_PORT}/v1",
        "startup": _startup(c) if not ready and state in ("running", "created", "restarting") else None,
    }


def stop() -> None:
    c = _container()
    if c is not None:
        c.remove(force=True)


def _wait_vram_free(timeout: float = 20) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        g = gpu_info()
        if not g or g["used_mib"] < 0.25 * g["total_mib"]:
            return
        time.sleep(1)


def start(model_id: str, context: int | None = None, util: float | None = None,
          graphs: bool = False, extra: list[str] | None = None, weights_gib: float | None = None) -> dict:
    stop()
    _wait_vram_free()
    g = gpu_info()
    if util is None:
        util = profiles.choose_util(g["total_mib"], g["used_mib"]) if g else 0.85
    if not context:
        total_gib = (g["total_mib"] / 1024) if g else 24
        context = capacity.auto_context(model_id, util) or profiles.choose_context(weights_gib, total_gib, util)
    args = profiles.build_vllm_args(model_id, context, util, graphs=graphs, extra=extra)

    _ensure_network()
    env = {}
    if os.getenv("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    client().containers.run(
        IMAGE, args, name=CONTAINER, detach=True, ipc_mode="host",
        restart_policy={"Name": "unless-stopped"},  # comes back with Docker after a reboot
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
        ports={"8000/tcp": HOST_PORT}, network=NETWORK, environment=env,
        volumes={CACHE_SOURCE: {"bind": "/root/.cache/huggingface", "mode": "rw"}},
        labels={"vllm-hub.model": model_id, "vllm-hub.context": str(context), "vllm-hub.util": f"{util:.2f}",
                "vllm-hub.managed": "true"},
    )
    return {"model": model_id, "context": context, "util": util, "args": args}


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
