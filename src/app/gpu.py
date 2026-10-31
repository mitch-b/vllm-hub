"""GPU detection via nvidia-smi (injected into the container by the NVIDIA runtime)."""
from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import time
from threading import Lock

from docker.errors import DockerException
from docker.types import DeviceRequest
from requests.exceptions import RequestException

_probe_lock = Lock()
_probe = {"at": 0.0, "gpu": None, "error": None}
logger = logging.getLogger(__name__)
QUERY = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
         "--format=csv,noheader,nounits"]


def _parse(output: str) -> dict:
    rows = output.strip().splitlines()
    name, total, used, driver = [part.strip() for part in rows[0].split(",")]
    return {"name": name, "total_mib": int(total), "used_mib": int(used), "driver": driver,
            "gpu_count": len(rows)}


def _docker_gpu() -> dict | None:
    # Only the short-lived probe requests NVIDIA devices; the hub itself boots on any host.
    from . import runner

    with _probe_lock:
        lifetime = 5 if _probe["gpu"] else 60
        if _probe["at"] and time.monotonic() - _probe["at"] < lifetime:
            return _probe["gpu"]
        probe = None
        try:
            client = runner.client()
            own = client.containers.get(socket.gethostname())
            probe = client.containers.create(
                own.attrs["Image"], entrypoint=QUERY, command=[],
                network_mode="none", environment={"NVIDIA_DRIVER_CAPABILITIES": "utility"},
                device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
                labels={"vllm-hub.probe": "nvidia"},
            )
            probe.start()
            result = probe.wait(timeout=10)
            output = probe.logs()
            if result["StatusCode"] != 0:
                raise ValueError(output.decode())
            _probe.update(gpu=_parse(output.decode()), error=None)
        except (DockerException, RequestException, ValueError, IndexError, KeyError) as exc:
            _probe.update(gpu=None, error=f"NVIDIA probe unavailable: {exc}")
        finally:
            if probe is not None:
                try:
                    probe.remove(force=True)
                except (DockerException, RequestException):
                    logger.exception("Could not remove NVIDIA probe container")
        _probe["at"] = time.monotonic()
        return _probe["gpu"]


def reset_probe() -> None:
    with _probe_lock:
        _probe["at"] = 0.0


def gpu_info() -> dict | None:
    override = os.getenv("VRAM_TOTAL_MIB")
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                QUERY,
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout
            return _parse(out)
        except (subprocess.SubprocessError, ValueError, IndexError):
            pass
    if override:
        return {"name": "GPU (VRAM_TOTAL_MIB override)", "total_mib": int(override), "used_mib": 0,
                "driver": "?", "gpu_count": 1}
    return _docker_gpu()
