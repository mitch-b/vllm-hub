from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import downloads, main, runner, updater, updates

IMG = "ghcr.io/mitch-b/vllm-hub:latest"


def fake_self(image_ref=IMG, digests=("sha256:old",)):
    snap = {"id": "sha256:img", "labels": {"org.opencontainers.image.revision": "a" * 40}, "arch": "amd64",
            "repo_digests": [f"ghcr.io/mitch-b/vllm-hub@{d}" for d in digests], "config": {"Env": ["A=1"]}}
    updates._snap.clear()
    updates._snap.update(snap)
    cfg = {"Image": image_ref}
    return SimpleNamespace(id="c0ffee", name="vllm-hub-hub-1", attrs={"Image": "sha256:img", "Config": cfg})


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(updates, "IMAGE", IMG)
    monkeypatch.setattr(updates, "CHECK", True)
    updates._cache.update(at=0.0, remote=None, error=None)
    updates._state.update(status="idle", error=None)
    monkeypatch.setattr(downloads, "list_jobs", lambda: [])


def test_split_ref():
    assert updates.split_ref(IMG) == ("ghcr.io", "mitch-b/vllm-hub", "latest")
    assert updates.split_ref("ghcr.io/mitch-b/vllm-hub") == ("ghcr.io", "mitch-b/vllm-hub", "latest")
    assert updates.split_ref("localhost:5000/x/y:dev") == ("localhost:5000", "x/y", "dev")


def test_version_reports_update(monkeypatch):
    monkeypatch.setattr(updates, "_self", lambda: fake_self())
    monkeypatch.setattr(updates, "remote_info", lambda arch: {"digest": "sha256:new", "revision": "b" * 40})
    body = TestClient(main.app).get("/api/version").json()
    assert body["managed"] and body["update_available"]
    assert body["local"]["digests"] == ["sha256:old"]


def test_version_up_to_date_and_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(updates, "_self", lambda: fake_self())
    monkeypatch.setattr(updates, "remote_info", lambda arch: calls.append(1) or {"digest": "sha256:old"})
    api = TestClient(main.app)
    assert api.get("/api/version").json()["update_available"] is False
    api.get("/api/version")
    assert len(calls) == 1
    api.get("/api/version?refresh=1")
    assert len(calls) == 2


def test_version_offline_and_outside_container(monkeypatch):
    def offline(arch):
        raise OSError("no network")

    monkeypatch.setattr(updates, "_self", lambda: None)
    monkeypatch.setattr(updates, "remote_info", offline)
    body = TestClient(main.app).get("/api/version").json()
    assert body["managed"] is False and body["update_available"] is False and "no network" in body["error"]


def test_update_refused_outside_container_or_foreign_image(monkeypatch):
    api = TestClient(main.app)
    monkeypatch.setattr(updates, "_self", lambda: None)
    assert api.post("/api/update").status_code == 409
    monkeypatch.setattr(updates, "_self", lambda: fake_self("mcr.microsoft.com/devcontainers/python:3.12"))
    assert api.post("/api/update").status_code == 409


def test_update_refused_while_downloading(monkeypatch):
    monkeypatch.setattr(updates, "_self", lambda: fake_self())
    monkeypatch.setattr(downloads, "list_jobs", lambda: [{"repo_id": "org/m", "status": "downloading"}])
    r = TestClient(main.app).post("/api/update")
    assert r.status_code == 409 and "org/m" in r.json()["detail"]


def test_update_starts_background_swap(monkeypatch):
    ran = []
    monkeypatch.setattr(updates, "_self", lambda: fake_self())
    monkeypatch.setattr(updates, "_run_update", lambda cid: ran.append(cid))
    r = TestClient(main.app).post("/api/update")
    assert r.status_code == 200 and r.json()["status"] == "pulling"
    assert ran == ["c0ffee"]


def test_run_update_pulls_then_launches_helper(monkeypatch):
    log = []
    old = fake_self()

    def get(name):
        if name == "c0ffee":
            return old
        raise LookupError(name)

    def gone(ref):
        raise LookupError("tag moved; old image no longer inspectable")

    containers = SimpleNamespace(get=get, run=lambda image, cmd, **kw: log.append(("run", image, cmd, kw["name"])))
    images = SimpleNamespace(pull=lambda repo, tag: log.append(("pull", repo, tag)))
    monkeypatch.setattr(runner, "client", lambda: SimpleNamespace(containers=containers, images=images,
                                                                   api=SimpleNamespace(inspect_image=gone)))
    updates._run_update("c0ffee")
    assert log[0] == ("pull", "ghcr.io/mitch-b/vllm-hub", "latest")
    cmd = ["python", "-m", "app.updater", "c0ffee", IMG, '{"Env": ["A=1"]}']
    assert log[1] == ("run", IMG, cmd, "vllm-hub-updater")
    assert updates._state["status"] == "restarting"


def test_new_config_keeps_user_settings_and_drops_image_defaults():
    image_cfg = {"Env": ["PATH=/usr/bin", "HF_HOME=/cache"], "Labels": {"org.opencontainers.image.revision": "old"},
                 "Cmd": ["uvicorn", "x"], "WorkingDir": "/srv", "Healthcheck": {"Test": ["CMD", "old"]}}
    attrs = {
        "Id": "abc123def456",
        "Config": {"Hostname": "abc123def456", "Image": "ghcr.io/mitch-b/vllm-hub:latest",
                   "Env": ["PATH=/usr/bin", "HF_HOME=/cache", "VLLM_PORT=8000"],
                   "Labels": {"org.opencontainers.image.revision": "old", "com.docker.compose.service": "hub"},
                   "Cmd": ["uvicorn", "x"], "WorkingDir": "/srv", "Healthcheck": {"Test": ["CMD", "old"]}},
        "HostConfig": {"PortBindings": {"7860/tcp": [{"HostIp": "127.0.0.1", "HostPort": "7860"}]}},
        "NetworkSettings": {"Networks": {"vllm-hub-net": {"Aliases": ["hub", "abc123def456"], "Links": None}}},
    }
    cfg = updater.new_config(attrs, image_cfg, "ghcr.io/mitch-b/vllm-hub:latest")
    assert cfg["Env"] == ["VLLM_PORT=8000"]
    assert cfg["Labels"] == {"com.docker.compose.service": "hub"}
    assert not {"Hostname", "Cmd", "WorkingDir", "Healthcheck"} & cfg.keys()
    assert cfg["HostConfig"] == attrs["HostConfig"]
    assert cfg["NetworkingConfig"]["EndpointsConfig"]["vllm-hub-net"]["Aliases"] == ["hub"]


def test_auto_tick(monkeypatch):
    started = []
    monkeypatch.setattr(updates, "_self", lambda: fake_self())
    monkeypatch.setattr(updates, "remote_info", lambda arch: {"digest": "sha256:new"})
    monkeypatch.setattr(updates, "_run_update", lambda cid: started.append(cid))
    monkeypatch.setattr(updates, "AUTO", False)
    assert updates.auto_tick() == "disabled"
    monkeypatch.setattr(updates, "AUTO", True)
    monkeypatch.setattr(downloads, "list_jobs", lambda: [{"repo_id": "org/m", "status": "downloading"}])
    assert updates.auto_tick().startswith("deferred") and not started
    monkeypatch.setattr(downloads, "list_jobs", lambda: [])
    assert updates.auto_tick() == "updating" and started == ["c0ffee"]
    monkeypatch.setattr(updates, "remote_info", lambda arch: {"digest": "sha256:old"})
    updates._state.update(status="idle")
    assert updates.auto_tick() == "current"


def test_pinned_tag_is_not_managed(monkeypatch):
    monkeypatch.setattr(updates, "_self", lambda: fake_self("ghcr.io/mitch-b/vllm-hub:20261031-1042"))
    monkeypatch.setattr(updates, "remote_info", lambda arch: {"digest": "sha256:new"})
    assert TestClient(main.app).get("/api/version").json()["managed"] is False
    assert TestClient(main.app).post("/api/update").status_code == 409
