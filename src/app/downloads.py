"""Background model downloads with byte-level progress, speed and ETA."""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from pathlib import Path

from huggingface_hub import scan_cache_dir, snapshot_download, try_to_load_from_cache
from huggingface_hub.utils import tqdm as hf_tqdm

from . import hf, profiles


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, repo_id: str):
        self.id = uuid.uuid4().hex[:8]
        self.repo_id = repo_id
        self.status = "queued"  # queued | downloading | done | error | cancelled
        self.total = 0
        self.done = 0
        self.error: str | None = None
        self.started = time.time()
        self.finished: float | None = None
        self.cancel = threading.Event()
        self._lock = threading.Lock()
        self._samples: deque[tuple[float, int]] = deque(maxlen=20)

    def add(self, n: int) -> None:
        if self.cancel.is_set():
            raise Cancelled()
        with self._lock:
            self.done = max(0, min(self.done + n, self.total or self.done + n))
            self._samples.append((time.time(), self.done))

    def view(self) -> dict:
        speed = 0.0
        with self._lock:
            if len(self._samples) >= 2:
                (t0, d0), (t1, d1) = self._samples[0], self._samples[-1]
                if t1 > t0 and time.time() - t1 < 10:
                    speed = max(0, (d1 - d0) / (t1 - t0))
            remaining = max(self.total - self.done, 0)
            finalizing = self.status == "downloading" and self.total > 0 and remaining == 0
            percent = round(100 * self.done / self.total, 1) if self.total else 0.0
            return {
                "id": self.id, "repo_id": self.repo_id, "status": self.status, "total": self.total,
                "done": self.done, "speed_bps": speed,
                "eta_s": remaining / speed if speed > 0 and self.status == "downloading" and remaining else None,
                "error": self.error, "finalizing": finalizing,
                "percent": min(percent, 99.9) if self.status != "done" else 100.0,
            }


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def _cached_bytes(repo_id: str, files: list[tuple[str, int]]) -> int:
    done = 0
    for name, size in files:
        path = try_to_load_from_cache(repo_id, name)
        if isinstance(path, str):
            cached = Path(path)
            if cached.is_file() and cached.stat().st_size == size:
                done += size
    return done


def _run(job: Job) -> None:
    try:
        files = hf.repo_files(job.repo_id)
        wanted = set(profiles.select_files([f for f, _ in files]))
        files = [(f, s) for f, s in files if f in wanted]
        if not files:
            raise RuntimeError("No downloadable model files found")
        job.total = sum(s for _, s in files)
        job.done = _cached_bytes(job.repo_id, files)
        job.status = "downloading"

        class Bar(hf_tqdm):
            def __init__(self, *args, **kwargs):
                # Xet reports both network transfer and reconstructed file bytes.
                # Only the latter shares the selected files' size denominator.
                self._bytes = kwargs.get("unit") == "B" and not (kwargs.get("name") or "").endswith(".transfer")
                super().__init__(*args, **kwargs)
                if self._bytes and kwargs.get("initial"):
                    job.add(int(kwargs["initial"]))

            def update(self, n=1):
                if job.cancel.is_set():
                    raise Cancelled()
                r = super().update(n)
                if self._bytes and n:
                    job.add(int(n))
                return r

        snapshot = Path(snapshot_download(job.repo_id, allow_patterns=sorted(wanted), tqdm_class=Bar))
        if job.cancel.is_set():
            raise Cancelled()
        for name, size in files:
            path = snapshot / name
            if not path.is_file() or (size and path.stat().st_size != size):
                raise RuntimeError(f"Snapshot file is missing or incomplete: {name}")
        with job._lock:
            job.done = job.total
            job.status = "done"
    except Cancelled:
        job.status = "cancelled"
    except Exception as exc:  # surfaced in the UI
        job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        job.finished = time.time()


def start(repo_id: str, worker=None) -> Job:
    with _lock:
        for j in _jobs.values():
            if j.repo_id == repo_id and j.status in ("queued", "downloading"):
                return j
        job = Job(repo_id)
        _jobs[job.id] = job
    threading.Thread(target=worker or _run, args=(job,), daemon=True).start()
    return job


def cancel(job_id: str) -> bool:
    job = _jobs.get(job_id)
    if job and job.status in ("queued", "downloading"):
        job.cancel.set()
        return True
    return False


def list_jobs() -> list[dict]:
    return [j.view() for j in sorted(_jobs.values(), key=lambda j: -j.started)]


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def cached_models() -> list[dict]:
    try:
        info = scan_cache_dir()
    except Exception:
        return []
    models = [
        {"id": r.repo_id, "size_gib": round(r.size_on_disk / 2**30, 1), "last_used": r.last_accessed}
        for r in info.repos if r.repo_type == "model"
    ]
    return sorted(models, key=lambda m: m["id"].lower())


def delete_cached(repo_id: str) -> bool:
    info = scan_cache_dir()
    hashes = [rev.commit_hash for r in info.repos if r.repo_id == repo_id and r.repo_type == "model"
              for rev in r.revisions]
    if not hashes:
        return False
    info.delete_revisions(*hashes).execute()
    forget_completed(repo_id)
    return True


def forget_completed(repo_id: str) -> None:
    # Completed jobs must not reintroduce a deleted model during UI reconciliation.
    with _lock:
        for job_id in [j.id for j in _jobs.values() if j.repo_id == repo_id and j.status == "done"]:
            del _jobs[job_id]
