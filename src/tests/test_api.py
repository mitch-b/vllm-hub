import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from docker.errors import NotFound
from fastapi.testclient import TestClient

from app import downloads, main, runner

GPU = {"name": "Test GPU", "total_mib": 24564, "used_mib": 1000, "driver": "1", "gpu_count": 1}
INDEX = Path(main.STATIC / "index.html").read_text(encoding="utf-8")


class FakeContainers:
    def get(self, name):
        raise NotFound("no container")


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(main, "gpu_info", lambda: GPU)
    monkeypatch.setattr(runner, "client", lambda: SimpleNamespace(containers=FakeContainers()))
    return TestClient(main.app)


def test_healthz(api):
    assert api.get("/healthz").json() == {"ok": True}


def test_index_served(api):
    r = api.get("/")
    assert r.status_code == 200
    assert "<title>vLLM Hub</title>" in r.text


def test_system_shape(api):
    body = api.get("/api/system").json()
    assert body["gpu"] == GPU
    assert 0.3 <= body["recommended_util"] <= 0.95
    assert body["vllm"] == {"docker_ok": True, "state": "stopped", "model": None, "ready": False}
    assert body["image"]


def test_system_without_docker_or_gpu(monkeypatch):
    def boom():
        raise RuntimeError("docker socket missing")

    monkeypatch.setattr(main, "gpu_info", lambda: None)
    monkeypatch.setattr(runner, "client", boom)
    body = TestClient(main.app).get("/api/system").json()
    assert body["gpu"] is None and body["recommended_util"] is None
    assert body["vllm"]["docker_ok"] is False


def test_models_lists_cache_with_context_hint(api, monkeypatch):
    monkeypatch.setattr(downloads, "cached_models", lambda: [{"id": "org/m", "size_gib": 4.0, "last_used": 0}])
    [m] = api.get("/api/models").json()
    assert m["id"] == "org/m" and m["context_hint"] >= 2048


def test_delete_unknown_model_404(api, monkeypatch):
    monkeypatch.setattr(downloads, "scan_cache_dir", lambda: SimpleNamespace(repos=[]))
    assert api.delete("/api/models/org/missing").status_code == 404


@pytest.mark.parametrize("repo_id", ["", "   ", "no-slash"])
def test_pull_rejects_bad_repo_id(api, repo_id):
    assert api.post("/api/pull", json={"repo_id": repo_id}).status_code == 400


def test_jobs_and_cancel_unknown(api):
    assert isinstance(api.get("/api/jobs").json(), list)
    assert api.post("/api/jobs/nope/cancel").status_code == 404


@pytest.mark.parametrize("util", [0.05, 0.99])
def test_run_rejects_out_of_range_util(api, util):
    assert api.post("/api/run", json={"repo_id": "org/m", "util": util}).status_code == 422


def test_run_reports_runner_errors(api, monkeypatch):
    def fail(*args):
        raise RuntimeError("no GPU")

    monkeypatch.setattr(runner, "start", fail)
    r = api.post("/api/run", json={"repo_id": "org/m"})
    assert r.status_code == 500 and "no GPU" in r.json()["detail"]


def test_chat_reports_upstream_errors(api, monkeypatch):
    def fail(*args):
        raise RuntimeError("vLLM not running")

    monkeypatch.setattr(runner, "chat", fail)
    r = api.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502


def test_every_ui_action_is_handled():
    acts = set(re.findall(r'data-act="([\w-]+)"', INDEX))
    handled = set(re.findall(r"case '([\w-]+)':", INDEX)) | {"conn-flag"}
    assert acts, "no data-act attributes found"
    assert acts <= handled, f"unhandled UI actions: {sorted(acts - handled)}"


def test_ui_only_calls_existing_api_routes():
    routes = {r.path for r in main.app.routes}
    calls = {re.sub(r"\$\{[^}]+\}", "{x}", c).split("?")[0] for c in re.findall(r"[`'\"](/api/[^`'\"]+)", INDEX)}
    pattern = [re.compile("^" + re.sub(r"\{[^}:]+\}", "[^/]+", re.sub(r"\{[^}]+:path\}", ".+", p)) + "$")
               for p in routes]
    missing = [c for c in calls if not any(p.match(c + "x" if c.endswith("/") else c.replace("{x}", "x"))
                                           for p in pattern)]
    assert not missing, f"UI calls unknown API routes: {missing}"
