"""Recreate the hub container on a new image. Runs in a short-lived helper container.

Usage: python -m app.updater <container-id> <image>

Keeps the container's name, ports, volumes, env, GPU access, networks and compose labels, but drops
settings that came from the old image so the new image's defaults apply. Rolls back if the new
container isn't healthy within two minutes.
"""
from __future__ import annotations

import json
import sys
import time

import docker

HEALTH_TIMEOUT = 120


def inherited(container_value, image_value):
    """Container config values that just mirror the old image should be left to the new image."""
    return container_value is None or container_value == image_value


def new_config(attrs: dict, old_image_cfg: dict, image: str) -> dict:
    cfg = dict(attrs["Config"])
    cfg["Image"] = image
    cfg.pop("Hostname", None)
    old_env = set(old_image_cfg.get("Env") or [])
    cfg["Env"] = [e for e in cfg.get("Env") or [] if e not in old_env]
    old_labels = old_image_cfg.get("Labels") or {}
    cfg["Labels"] = {k: v for k, v in (cfg.get("Labels") or {}).items() if old_labels.get(k) != v}
    for key in ("Cmd", "Entrypoint", "Healthcheck", "WorkingDir", "ExposedPorts", "User", "StopSignal"):
        if inherited(cfg.get(key), old_image_cfg.get(key)):
            cfg.pop(key, None)
    cfg["HostConfig"] = attrs["HostConfig"]
    endpoints = {}
    for net, ep in (attrs.get("NetworkSettings", {}).get("Networks") or {}).items():
        aliases = [a for a in (ep.get("Aliases") or []) if not attrs["Id"].startswith(a)]
        endpoints[net] = {"Aliases": aliases or None, "Links": ep.get("Links")}
    cfg["NetworkingConfig"] = {"EndpointsConfig": endpoints}
    return cfg


def wait_healthy(c) -> bool:
    deadline = time.time() + HEALTH_TIMEOUT
    while time.time() < deadline:
        c.reload()
        health = (c.attrs.get("State", {}).get("Health") or {}).get("Status")
        if c.status == "exited" or health == "unhealthy":
            return False
        if health == "healthy" or (health is None and c.status == "running"):
            return True
        time.sleep(2)
    return False


def main(container_id: str, image: str, old_image_cfg: str = "") -> int:
    """Blue/green-style swap: everything slow happens while the old hub still serves.

    The image is already pulled and the new container is created (ports are only bound on start),
    so the only gap is stop-old -> start-new, typically around a second.
    """
    dc = docker.from_env()
    old = dc.containers.get(container_id)
    name = old.name
    # The hub captures the old image's config before pulling: once the tag moves, the old image may
    # no longer be inspectable (e.g. with the containerd image store).
    image_cfg = json.loads(old_image_cfg) if old_image_cfg else dc.api.inspect_image(old.attrs["Image"])["Config"]
    cfg = new_config(old.attrs, image_cfg, image)
    print(f"updating {name} to {image}", flush=True)
    for stale in (f"{name}-next", f"{name}-old"):
        try:
            dc.containers.get(stale).remove(force=True)
        except docker.errors.NotFound:
            pass
    resp = dc.api.post(dc.api._url("/containers/create"), params={"name": f"{name}-next"}, json=cfg)
    resp.raise_for_status()
    new = dc.containers.get(resp.json()["Id"])
    t0 = time.monotonic()
    try:
        old.stop(timeout=3)
        new.start()
        healthy = wait_healthy(new)
    except Exception as exc:
        print(f"start failed: {exc}", flush=True)
        healthy = False
    if healthy:
        print(f"new hub healthy after {time.monotonic() - t0:.1f}s", flush=True)
        old.rename(f"{name}-old")
        new.rename(name)
        old.remove()
        print("update complete", flush=True)
        return 0
    print("new container unhealthy, rolling back", flush=True)
    try:
        print(new.logs(tail=40).decode(errors="replace"), flush=True)
    except Exception:
        pass
    new.remove(force=True)
    old.start()
    return 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
