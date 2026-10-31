"""Self-update: compare the running image with the published one and swap the container in place."""
from __future__ import annotations

import json
import os
import socket
import threading
import time

import httpx

from . import downloads, runner

IMAGE = os.getenv("HUB_IMAGE", "ghcr.io/mitch-b/vllm-hub:latest")
CHECK = os.getenv("HUB_UPDATE_CHECK", "true").lower() not in ("0", "false", "no")
CACHE_SECS = 1800
UPDATER_NAME = "vllm-hub-updater"
INDEX_TYPES = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])

_cache: dict = {"at": 0.0, "remote": None, "error": None}
_state: dict = {"status": "idle", "error": None}
_lock = threading.Lock()


def split_ref(ref: str) -> tuple[str, str, str]:
    """'ghcr.io/org/name:tag' -> ('ghcr.io', 'org/name', 'tag')."""
    name, _, tag = ref.rpartition(":") if ":" in ref.split("/")[-1] else (ref, "", "latest")
    registry, _, repo = name.partition("/")
    return registry, repo, tag or "latest"


def _self():
    try:
        return runner.client().containers.get(socket.gethostname())
    except Exception:
        return None


_snap: dict = {}


def image_snapshot(c) -> dict:
    """Metadata of the image this container runs, remembered by image id.

    Once the tag moves to a newer image (our own pull, or a manual `docker pull`), the old image can
    become impossible to inspect (containerd image store), so capture it at startup and keep it.
    """
    image_id = c.attrs.get("Image")
    if _snap.get("id") != image_id:
        try:
            a = runner.client().api.inspect_image(image_id)
        except Exception:
            return {"id": image_id, "labels": {}, "repo_digests": [], "arch": "amd64", "config": None}
        _snap.clear()
        _snap.update(id=image_id, labels=(a.get("Config") or {}).get("Labels") or {}, config=a.get("Config"),
                     repo_digests=a.get("RepoDigests") or [], arch=a.get("Architecture") or "amd64")
    return _snap


def warm() -> None:
    c = _self()
    if c is not None:
        image_snapshot(c)


def local_info() -> dict:
    """What we're running: image ref, digests and OCI labels. Empty when not in a container."""
    c = _self()
    if c is None:
        return {"container": None}
    img = image_snapshot(c)
    labels = img["labels"]
    repo = split_ref(IMAGE)[0] + "/" + split_ref(IMAGE)[1]
    digests = [d.split("@", 1)[1] for d in img["repo_digests"] if d.startswith(repo + "@")]
    return {
        "container": c.name,
        "image": (c.attrs.get("Config") or {}).get("Image"),
        "digests": digests,
        "arch": img["arch"],
        "version": labels.get("org.opencontainers.image.version"),
        "revision": labels.get("org.opencontainers.image.revision"),
        "created": labels.get("org.opencontainers.image.created"),
    }


def remote_info(arch: str = "amd64") -> dict:
    """Digest and OCI labels of the published tag, via the anonymous registry API."""
    registry, repo, tag = split_ref(IMAGE)
    base = f"https://{registry}/v2/{repo}"
    with httpx.Client(timeout=10, follow_redirects=True) as h:
        tok = h.get(f"https://{registry}/token", params={"scope": f"repository:{repo}:pull"}).json().get("token")
        auth = {"Authorization": f"Bearer {tok}"} if tok else {}
        r = h.get(f"{base}/manifests/{tag}", headers={**auth, "Accept": INDEX_TYPES})
        r.raise_for_status()
        digest, doc = r.headers.get("Docker-Content-Digest"), r.json()
        if "manifests" in doc:
            plat = [m for m in doc["manifests"] if (m.get("platform") or {}).get("architecture") == arch]
            if not plat:
                return {"digest": digest}
            doc = h.get(f"{base}/manifests/{plat[0]['digest']}", headers={**auth, "Accept": INDEX_TYPES}).json()
        cfg = h.get(f"{base}/blobs/{doc['config']['digest']}", headers=auth).json()
    labels = (cfg.get("config") or {}).get("Labels") or {}
    return {
        "digest": digest,
        "version": labels.get("org.opencontainers.image.version"),
        "revision": labels.get("org.opencontainers.image.revision"),
        "created": labels.get("org.opencontainers.image.created"),
    }


def check(force: bool = False) -> dict:
    local = local_info()
    managed = bool(local.get("container")) and split_ref(local.get("image") or "")[:2] == split_ref(IMAGE)[:2]
    out = {"image": IMAGE, "check_enabled": CHECK, "local": local, "managed": managed, "remote": None,
           "error": None, "update_available": False, "state": dict(_state)}
    if not CHECK:
        return out
    if force or time.time() - _cache["at"] > CACHE_SECS:
        try:
            _cache.update(remote=remote_info(local.get("arch") or "amd64"), error=None)
        except Exception as exc:
            _cache.update(error=f"Could not reach {split_ref(IMAGE)[0]}: {exc}")
        _cache["at"] = time.time()
    remote = _cache["remote"]
    out.update(remote=remote, error=_cache["error"], checked_at=_cache["at"])
    if remote and remote.get("digest"):
        out["update_available"] = remote["digest"] not in local.get("digests", [])
    return out


def _run_update(container_id: str) -> None:
    try:
        _state.update(status="pulling", error=None)
        registry_repo, tag = IMAGE.rsplit(":", 1) if ":" in IMAGE.split("/")[-1] else (IMAGE, "latest")
        dc = runner.client()
        snap = image_snapshot(dc.containers.get(container_id))
        old_cfg = json.dumps(snap["config"]) if snap["config"] else ""
        dc.images.pull(registry_repo, tag=tag)
        _state.update(status="restarting")
        try:
            dc.containers.get(UPDATER_NAME).remove(force=True)
        except Exception:
            pass
        dc.containers.run(
            IMAGE, ["python", "-m", "app.updater", container_id, IMAGE, old_cfg], name=UPDATER_NAME, detach=True,
            auto_remove=True, volumes={"/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"}},
            labels={"vllm-hub.role": "updater"},
        )
    except Exception as exc:
        _state.update(status="error", error=str(exc))


def start_update() -> dict:
    c = _self()
    if c is None:
        raise RuntimeError("The hub is not running in a container, so it can't update itself.")
    if split_ref((c.attrs.get("Config") or {}).get("Image") or "")[:2] != split_ref(IMAGE)[:2]:
        raise RuntimeError(f"This container doesn't run {IMAGE}; update it with docker compose instead.")
    busy = [j["repo_id"] for j in downloads.list_jobs() if j["status"] in ("queued", "downloading")]
    if busy:
        raise RuntimeError(f"Wait for downloads to finish first ({', '.join(busy)}).")
    with _lock:
        if _state["status"] in ("pulling", "restarting"):
            return dict(_state)
        _state.update(status="pulling", error=None)
    threading.Thread(target=_run_update, args=(c.id,), daemon=True).start()
    return dict(_state)
