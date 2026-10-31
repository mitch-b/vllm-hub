"""GPU detection via nvidia-smi (injected into the container by the NVIDIA runtime)."""
from __future__ import annotations

import os
import shutil
import subprocess


def gpu_info() -> dict | None:
    override = os.getenv("VRAM_TOTAL_MIB")
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip().splitlines()
            name, total, used, driver = [p.strip() for p in out[0].split(",")]
            return {"name": name, "total_mib": int(total), "used_mib": int(used), "driver": driver,
                    "gpu_count": len(out)}
        except (subprocess.SubprocessError, ValueError, IndexError):
            pass
    if override:
        return {"name": "GPU (VRAM_TOTAL_MIB override)", "total_mib": int(override), "used_mib": 0,
                "driver": "?", "gpu_count": 1}
    return None
