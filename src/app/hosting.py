"""Capability-based backend selection; never infer the server OS from its browser."""
from __future__ import annotations

from . import model_runner, runner
from .gpu import gpu_info


def detect() -> dict:
    native = model_runner.discover()
    native_host = _native(native) if native else None
    gpu = None if native_host and native_host["accelerator"] == "Metal" else gpu_info()
    if native and model_runner.selected_model():
        return native_host
    # Existing NVIDIA installations keep their model/cache/flags and endpoint.
    if gpu:
        return {"backend": "cuda", "gpu": gpu, "local_inference": {
            "available": True, "reason": None, "message": None,
        }}
    if native:
        return native_host
    runtime = runner.status()
    message = (
        "Enable Docker Model Runner in Docker Desktop Settings > AI, then enable host-side TCP "
        "access to use the model from Copilot. On Linux, install docker-model-plugin and run "
        "docker model install-runner. The hub reconnects automatically. "
        "NVIDIA hosts can also use the NVIDIA container runtime."
    )
    if not runtime["docker_ok"]:
        message = "Docker is unreachable. Check Docker and the /var/run/docker.sock mount. " + message
    return {"backend": None, "gpu": gpu, "local_inference": {
        "available": False, "reason": "runtime_unavailable", "message": message,
    }}


def _native(connection: dict) -> dict:
    engines = " ".join(connection.get("engines", {}).values()).lower()
    accelerator = "Metal" if "metal" in engines else "CUDA" if "cuda" in engines else "Auto"
    return {"backend": "model-runner", "gpu": None, "accelerator": accelerator,
            "local_inference": {"available": True, "reason": None, "message": None}}


def engine():
    host = detect()
    backend = host["backend"]
    if backend == "model-runner":
        return model_runner
    if backend == "cuda":
        return runner
    raise runner.GPUUnavailable(host["local_inference"]["message"])
