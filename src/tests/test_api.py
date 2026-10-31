import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from docker.errors import APIError, NotFound
from fastapi.testclient import TestClient

from app import downloads, hf, hosting, main, model_runner, runner

GPU = {"name": "Test GPU", "total_mib": 24564, "used_mib": 1000, "driver": "1", "gpu_count": 1}
INDEX = Path(main.STATIC / "index.html").read_text(encoding="utf-8")


class FakeContainers:
    def get(self, name):
        raise NotFound("no container")


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "gpu_info", lambda: GPU)
    monkeypatch.setattr(hosting, "gpu_info", lambda: main.gpu_info())
    monkeypatch.setattr(model_runner, "discover", lambda: None)
    monkeypatch.setattr(model_runner, "STATE", tmp_path / "model-runner.json")
    monkeypatch.setattr(model_runner, "_selection", {})
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
    assert body["hosting"]["local_inference"] == {"available": True, "reason": None, "message": None}


def test_system_without_docker_or_gpu(api, monkeypatch):
    def boom():
        raise RuntimeError("docker socket missing")

    monkeypatch.setattr(main, "gpu_info", lambda: None)
    monkeypatch.setattr(runner, "client", boom)
    body = TestClient(main.app).get("/api/system").json()
    assert body["gpu"] is None and body["recommended_util"] is None
    assert body["vllm"]["docker_ok"] is False


def test_system_without_gpu_with_docker(api, monkeypatch):
    monkeypatch.setattr(main, "gpu_info", lambda: None)
    body = api.get("/api/system").json()
    assert body["gpu"] is None and body["recommended_util"] is None
    assert body["vllm"]["docker_ok"] is True
    assert body["hosting"]["local_inference"]["reason"] == "runtime_unavailable"
    assert api.get("/healthz").json() == {"ok": True}
    assert api.get("/").status_code == 200


def test_run_without_gpu_preserves_existing_container(api, monkeypatch):
    monkeypatch.setattr(runner, "gpu_info", lambda: None)

    def unexpected_client():
        pytest.fail("An unsupported start must not touch Docker or stop a running model")

    monkeypatch.setattr(runner, "client", unexpected_client)
    response = api.post("/api/run", json={"repo_id": "org/model"})
    assert response.status_code == 409
    assert "NVIDIA GPU" in response.json()["detail"]
    assert "Model Runner" in response.json()["detail"]


@pytest.mark.parametrize("gpu,docker_ok,reason", [
    (None, True, "runtime_unavailable"),
    (GPU, False, "docker_unavailable"),
    (None, False, "docker_unavailable"),
    (GPU, True, None),
])
def test_hosting_capabilities(api, monkeypatch, gpu, docker_ok, reason):
    monkeypatch.setattr(main, "gpu_info", lambda: gpu)
    monkeypatch.setattr(runner, "status", lambda: {"docker_ok": docker_ok})
    support = api.get("/api/system").json()["hosting"]["local_inference"]
    assert support["available"] is (reason is None)
    assert support["reason"] == reason
    assert bool(support["message"]) is (reason is not None)


def test_run_with_no_available_backend_is_actionable(api, monkeypatch):
    monkeypatch.setattr(main, "gpu_info", lambda: None)
    response = api.post("/api/run", json={"repo_id": "org/model"})
    assert response.status_code == 409
    assert "Settings > AI" in response.json()["detail"]


@pytest.fixture
def native_api(api, monkeypatch):
    monkeypatch.setattr(hosting, "detect", lambda: {
        "backend": "model-runner", "gpu": None, "accelerator": "Metal",
        "local_inference": {"available": True, "reason": None, "message": None},
    })
    monkeypatch.setattr(model_runner, "selected_model", lambda: "fixture/model")
    return api


@pytest.mark.parametrize("method,path,handler,body,result", [
    ("GET", "/api/models", "models", None, [{"id": "fixture/model"}]),
    ("GET", "/api/recommended", "recommended", None, [{"id": "fixture/model"}]),
    ("GET", "/api/search?q=coder", "search", None, [{"id": "fixture/model"}]),
    ("DELETE", "/api/models/fixture/model", "delete", None, {"ok": True}),
    ("POST", "/api/pull", "pull", {"repo_id": "fixture/model"}, {"id": "fixture-job"}),
    ("POST", "/api/run", "start", {"repo_id": "fixture/model"}, {"backend": "model-runner"}),
    ("POST", "/api/stop", "stop", None, {"ok": True}),
    ("POST", "/api/chat", "chat", {"messages": [{"role": "user", "content": "hi"}]}, {"content": "answer"}),
])
def test_native_api_dispatch(native_api, monkeypatch, method, path, handler, body, result):
    calls = []

    def operation(*args):
        calls.append(args)
        return SimpleNamespace(view=lambda: result) if handler == "pull" else result

    monkeypatch.setattr(model_runner, handler, operation)
    response = native_api.request(method, path, json=body)
    assert response.status_code == 200 and response.json() == result
    assert len(calls) == 1


@pytest.mark.parametrize("sort", ["popular", "recent", "updated", "trending"])
@pytest.mark.parametrize("native", [False, True])
def test_search_sort_dispatch(api, monkeypatch, sort, native):
    calls = []
    if native:
        monkeypatch.setattr(hosting, "detect", lambda: {"backend": "model-runner"})
    def search(*args):
        calls.append(args)
        return [{"id": "fixture/model", "uploaded_at": "2026-10-01T00:00:00+00:00",
                 "updated_at": None, "max_context": 32768, "context_hint": 8192}]
    monkeypatch.setattr(model_runner if native else hf, "search", search)
    response = api.get(f"/api/search?sort={sort}&limit=10")
    assert response.status_code == 200
    assert calls == [("", 10, sort)]
    assert response.json()[0]["max_context"] == 32768


@pytest.mark.parametrize("query", ["sort=intelligence", "limit=0"])
def test_search_validates_sort_and_limit(api, query):
    assert api.get("/api/search?" + query).status_code == 422


def test_search_keeps_existing_large_limit_clamp(api, monkeypatch):
    calls = []
    monkeypatch.setattr(hf, "search", lambda *args: calls.append(args) or [])
    assert api.get("/api/search?q=coder&limit=100").status_code == 200
    assert calls == [("coder", 50, "popular")]


def test_search_failure_is_explicit(api, monkeypatch):
    def fail(*args):
        raise RuntimeError("Catalog unavailable")
    monkeypatch.setattr(hf, "search", fail)
    response = api.get("/api/search?sort=trending")
    assert response.status_code == 502
    assert "Catalog unavailable" in response.json()["detail"]


def test_native_api_system_and_diagnostics(native_api, monkeypatch):
    runtime = {"docker_ok": True, "state": "stopped", "backend": "model-runner"}
    monkeypatch.setattr(model_runner, "status", lambda: runtime)
    body = native_api.get("/api/system").json()
    assert body["vllm"] == runtime
    assert body["hosting"]["accelerator"] == "Metal"
    assert not any(body["hosting"]["features"].values())
    assert native_api.get("/api/activity").json()["supported"] is False
    assert "Docker Desktop" in native_api.get("/api/logs").text


def test_native_status_connection_failure_is_explicit(native_api, monkeypatch):
    def disconnected():
        raise RuntimeError("Native management connection lost")

    monkeypatch.setattr(model_runner, "status", disconnected)
    response = native_api.get("/api/system")
    assert response.status_code == 502
    assert "Native management connection lost" in response.json()["detail"]


@pytest.fixture
def starting(api, monkeypatch):
    started = "2026-10-31T16:00:00.123456789Z"
    epoch = runner._timestamp(started)
    c = SimpleNamespace(
        id="first-container", status="running", labels={"vllm-hub.model": "org/model"},
        attrs={"State": {"StartedAt": started}},
    )
    calls = []

    def logs(**kwargs):
        calls.append(kwargs)
        return c.output

    c.output = b""
    c.logs = logs
    monkeypatch.setattr(runner, "_container", lambda: c)
    monkeypatch.setattr(runner.time, "time", lambda: epoch + 125)
    monkeypatch.setattr(runner.httpx, "get", lambda *a, **kw: SimpleNamespace(status_code=503))
    return c, calls


def test_system_startup_timer_survives_refresh_and_resets_for_new_attempt(api, starting, monkeypatch):
    c, calls = starting
    first = api.get("/api/system").json()["vllm"]["startup"]
    assert first["container_id"] == "first-container"
    assert first["started_at"] == pytest.approx(1793462400.123456, rel=0, abs=0.00001)
    assert first["elapsed_seconds"] == 125
    monkeypatch.setattr(runner.time, "time", lambda: first["started_at"] + 130)
    refreshed = api.get("/api/system").json()["vllm"]["startup"]
    assert refreshed["started_at"] == first["started_at"]
    assert refreshed["elapsed_seconds"] == 130
    # Docker can restart the same ID; StartedAt, not its creation time, identifies the attempt.
    c.attrs["State"]["StartedAt"] = "2026-10-31T16:02:00.123456789Z"
    restarted = api.get("/api/system").json()["vllm"]["startup"]
    assert restarted["container_id"] == first["container_id"]
    assert restarted["elapsed_seconds"] == 10
    c.id = "second-container"
    c.labels["vllm-hub.model"] = "org/other"
    assert api.get("/api/system").json()["vllm"]["startup"]["container_id"] == "second-container"
    assert calls[0] == {"tail": 300, "since": int(first["started_at"]), "timestamps": True}


@pytest.mark.parametrize("message,stage,progress", [
    ("Starting to load model org/model", "Loading weights", None),
    ("Loading safetensors checkpoint shards: 25%|xx| 2/8 [00:03]", "Loading weights",
     {"completed": 2, "total": 8}),
    ("Loading pt checkpoint shards: 0%| | 0/4 [00:00]", "Loading weights",
     {"completed": 0, "total": 4}),
    ("Loading np_cache checkpoint shards: 2/4", "Loading weights",
     {"completed": 2, "total": 4}),
    ("Multi-thread loading shards: 3/4", "Loading weights",
     {"completed": 3, "total": 4}),
    ("Loading model weights took 40.0 seconds", "Warming up", None),
    ("Model loading took 14 GiB and 40 seconds", "Warming up", None),
    ("GPU KV cache size: 66,901 tokens", "Warming up", None),
    ("Using torch.compile for the model", "Compiling model", None),
    ("Capturing CUDA graphs (mixed prefill-decode): 50%|xx| 4/8 [00:10]",
     "Capturing CUDA graphs", {"completed": 4, "total": 8}),
    ("Graph capturing finished in 20 secs", "Waiting for API readiness", None),
    ("Application startup complete.", "Waiting for API readiness", None),
    ("init engine (profile, create kv cache, warmup model) took 123 s",
     "Waiting for API readiness", None),
    ("Unrecognized future version output", "Starting engine", None),
    ("Loading safetensors checkpoint shards: 2/0", "Loading weights", None),
    ("Loading safetensors checkpoint shards: 9/8", "Loading weights", None),
])
def test_system_startup_measured_stages(api, starting, message, stage, progress):
    c, _ = starting
    c.output = f"2026-10-31T16:00:10.000000000Z {message}\n".encode()
    s = api.get("/api/system").json()["vllm"]["startup"]
    assert s["stage"] == stage
    assert s["progress"] == progress
    assert s["last_log_at"] == runner._timestamp("2026-10-31T16:00:10Z")


def test_startup_ignores_previous_attempt_and_clears_completed_stage_counts(api, starting):
    c, _ = starting
    c.output = (
        b"2026-10-31T16:00:00.000000000Z Capturing CUDA graphs: 8/8\n"
        b"2026-10-31T16:00:10Z \x1b[32mLoading safetensors checkpoint shards: 1/8\x1b[0m"
        b"\rLoading safetensors checkpoint shards: 8/8\n"
    )
    s = api.get("/api/system").json()["vllm"]["startup"]
    assert s["stage"] == "Loading weights"
    assert s["progress"] == {"completed": 8, "total": 8}
    c.output += b"2026-10-31T16:00:15Z Loading model weights took 15 seconds\n"
    s = api.get("/api/system").json()["vllm"]["startup"]
    assert s["stage"] == "Warming up" and s["progress"] is None
    c.output = b"2026-10-31T15:59:59Z Loading model weights took 15 seconds\n"
    s = api.get("/api/system").json()["vllm"]["startup"]
    assert s["stage"] == "Starting engine" and s["last_log_at"] is None


def test_startup_reports_log_errors_without_breaking_system(api, starting):
    c, _ = starting

    def fail(**kwargs):
        raise APIError("logging driver does not support reading")

    c.logs = fail
    s = api.get("/api/system").json()["vllm"]["startup"]
    assert s["elapsed_seconds"] == 125
    assert "Startup logs unavailable" in s["error"]
    assert "logging driver" in s["error"]


@pytest.mark.parametrize("state", ["created", "restarting", "exited", "dead"])
def test_system_startup_nonrunning_states_do_not_read_old_logs(api, starting, state):
    c, calls = starting
    c.status = state
    s = api.get("/api/system").json()["vllm"]["startup"]
    if state in ("exited", "dead"):
        assert s is None
    else:
        assert s["elapsed_seconds"] is None and s["started_at"] is None
        assert s["stage"] == ("Container restarting" if state == "restarting" else "Waiting for container")
    assert not calls


def test_system_ready_clears_startup_without_reading_logs(api, starting, monkeypatch):
    _, calls = starting
    monkeypatch.setattr(runner.httpx, "get", lambda *a, **kw: SimpleNamespace(
        status_code=200, json=lambda: {"data": [{"id": "model"}]},
    ))
    v = api.get("/api/system").json()["vllm"]
    assert v["ready"] is True and v["startup"] is None
    assert not calls


@pytest.mark.parametrize("timestamp", ["", "not a timestamp", "0001-01-01T00:00:00Z"])
def test_system_invalid_start_timestamp_is_explicit(api, starting, timestamp):
    c, calls = starting
    c.attrs["State"]["StartedAt"] = timestamp
    s = api.get("/api/system").json()["vllm"]["startup"]
    assert s["started_at"] is None and s["elapsed_seconds"] is None
    assert s["error"] == "Container startup timestamp unavailable"
    assert not calls


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


def test_jobs_distinguishes_finalizing_from_complete(api, monkeypatch):
    job = downloads.Job("org/model")
    job.total = job.done = 100
    job.status = "downloading"
    monkeypatch.setattr(downloads, "_jobs", {job.id: job})
    [pending] = api.get("/api/jobs").json()
    assert pending["finalizing"] and pending["percent"] < 100
    job.status = "done"
    [complete] = api.get("/api/jobs").json()
    assert not complete["finalizing"] and complete["percent"] == 100


def test_delete_model_does_not_leave_a_completed_job_to_reappear(api, monkeypatch):
    job = downloads.Job("org/model")
    job.status = "done"
    monkeypatch.setattr(downloads, "_jobs", {job.id: job})
    monkeypatch.setattr(downloads, "scan_cache_dir", lambda: SimpleNamespace(
        repos=[SimpleNamespace(repo_id=job.repo_id, repo_type="model",
                               revisions=[SimpleNamespace(commit_hash="revision")])],
        delete_revisions=lambda *args: SimpleNamespace(execute=lambda: None),
    ))
    assert api.delete("/api/models/org/model").status_code == 200
    assert api.get("/api/jobs").json() == []


@pytest.mark.parametrize("util", [0.05, 0.99])
def test_run_rejects_out_of_range_util(api, util):
    assert api.post("/api/run", json={"repo_id": "org/m", "util": util}).status_code == 422


def test_run_reports_runner_errors(api, monkeypatch):
    def fail(*args):
        raise RuntimeError("no GPU")

    monkeypatch.setattr(runner, "start", fail)
    r = api.post("/api/run", json={"repo_id": "org/m"})
    assert r.status_code == 500 and "no GPU" in r.json()["detail"]


@pytest.fixture
def runtime(api, monkeypatch):
    containers = SimpleNamespace(current=None, runs=[])

    def get(name):
        if containers.current is None:
            raise NotFound("no container")
        return containers.current

    def remove(force):
        assert force
        containers.current = None

    def run(image, args, **kwargs):
        containers.runs.append(kwargs)
        containers.current = SimpleNamespace(
            id=f"container-{len(containers.runs)}", status="running", labels=kwargs["labels"],
            attrs={"State": {"StartedAt": "2026-10-31T16:00:00Z"}},
            remove=remove, logs=lambda **kw: b"Starting to load model\n",
        )
        return containers.current

    containers.get, containers.run = get, run
    monkeypatch.setattr(runner, "client", lambda: SimpleNamespace(containers=containers))
    monkeypatch.setattr(runner, "gpu_info", lambda: GPU)
    monkeypatch.setattr(runner, "_ensure_network", lambda: None)
    monkeypatch.setattr(runner, "_start_cancel", None)
    monkeypatch.setattr(runner.httpx, "get", lambda *a, **kw: SimpleNamespace(status_code=503))
    return containers


def test_stop_removes_loading_container_and_explicit_run_still_works(api, runtime):
    request = {"repo_id": "org/model", "context": 8192, "util": 0.9}
    assert api.post("/api/run", json=request).status_code == 200
    assert api.get("/api/system").json()["vllm"]["container_id"] == "container-1"
    assert runtime.runs[0]["restart_policy"] == {"Name": "unless-stopped"}
    assert api.post("/api/stop").json() == {"ok": True}
    assert api.post("/api/stop").json() == {"ok": True}
    assert api.get("/api/system").json()["vllm"]["state"] == "stopped"
    assert api.post("/api/run", json=request).status_code == 200
    assert runtime.current.id == "container-2"


def test_cancel_interrupts_start_while_waiting_for_vram(api, runtime, monkeypatch):
    waiting = Event()

    def busy_gpu():
        waiting.set()
        return dict(GPU, used_mib=GPU["total_mib"])

    monkeypatch.setattr(runner, "gpu_info", busy_gpu)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(api.post, "/api/run", json={"repo_id": "org/model", "context": 8192})
        try:
            assert waiting.wait(2)
            assert api.post("/api/stop").status_code == 200
            response = pending.result(timeout=2)
            assert response.status_code == 409
            assert response.json()["detail"] == "Model start cancelled"
            assert not runtime.runs
        finally:
            runner.stop()


@pytest.mark.parametrize("replacement", [False, True])
def test_cancelled_preparation_cannot_create_a_container(api, runtime, monkeypatch, replacement):
    preparing, release = Event(), Event()
    build = runner.profiles.build_vllm_args

    def prepare(model, *args, **kwargs):
        if model == "org/first":
            preparing.set()
            assert release.wait(5)
        return build(model, *args, **kwargs)

    monkeypatch.setattr(runner.profiles, "build_vllm_args", prepare)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(api.post, "/api/run", json={
            "repo_id": "org/first", "context": 8192, "util": 0.9,
        })
        try:
            assert preparing.wait(2)
            if replacement:
                assert api.post("/api/run", json={
                    "repo_id": "org/second", "context": 8192, "util": 0.9,
                }).status_code == 200
            else:
                assert api.post("/api/stop").status_code == 200
            release.set()
            assert pending.result(timeout=2).status_code == 409
            assert len(runtime.runs) == int(replacement)
            if replacement:
                assert runtime.current.labels["vllm-hub.model"] == "org/second"
            else:
                assert runtime.current is None
        finally:
            release.set()


def test_stop_racing_container_creation_removes_the_new_container(api, runtime, monkeypatch):
    creating, release, stopping = Event(), Event(), Event()
    run = runtime.run

    def create(*args, **kwargs):
        creating.set()
        assert release.wait(5)
        return run(*args, **kwargs)

    def stop():
        stopping.set()
        return api.post("/api/stop")

    monkeypatch.setattr(runtime, "run", create)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(api.post, "/api/run", json={
            "repo_id": "org/model", "context": 8192, "util": 0.9,
        })
        try:
            assert creating.wait(2)
            cancelled = pool.submit(stop)
            assert stopping.wait(2)
            release.set()
            assert pending.result(timeout=2).status_code == 200
            assert cancelled.result(timeout=2).status_code == 200
            assert runtime.current is None
        finally:
            release.set()


def test_logs_returns_container_output_and_stopped_message(api, runtime):
    assert api.get("/api/logs").text == "vLLM is not running.\n"
    assert api.post("/api/run", json={
        "repo_id": "org/model", "context": 8192, "util": 0.9,
    }).status_code == 200
    calls = []

    def logs(**kwargs):
        calls.append(kwargs)
        return iter([b"Loading weights\n", b"Shard 1/4\n"])

    runtime.current.logs = logs
    assert api.get("/api/logs").text == "Loading weights\nShard 1/4\n"
    assert calls == [{"stream": True, "follow": True, "tail": 100}]


def test_chat_reports_upstream_errors(api, monkeypatch):
    def fail(*args):
        raise RuntimeError("vLLM not running")

    monkeypatch.setattr(runner, "chat", fail)
    r = api.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502


METRICS = """# HELP vllm:num_requests_running Number of requests in model execution batches.
vllm:num_requests_running{{engine="0",model_name="m"}} {running}
vllm:num_requests_waiting{{engine="0",model_name="m"}} 1.0
vllm:kv_cache_usage_perc{{engine="0",model_name="m"}} 0.25
vllm:prompt_tokens_total{{engine="0",model_name="m"}} {prompt}
vllm:generation_tokens_total{{engine="0",model_name="m"}} {gen}
vllm:request_success_total{{engine="0",finished_reason="stop",model_name="m"}} 3.0
vllm:request_success_total{{engine="0",finished_reason="length",model_name="m"}} 1.0
vllm:time_to_first_token_seconds_sum{{engine="0",model_name="m"}} 2.0
vllm:time_to_first_token_seconds_count{{engine="0",model_name="m"}} 4.0
vllm:prefix_cache_queries_total{{engine="0",model_name="m"}} 200.0
vllm:prefix_cache_hits_total{{engine="0",model_name="m"}} 150.0
"""
ACCESS = b"""2026-10-31T16:00:01.000000000Z INFO:     172.27.0.2:1 - "GET /v1/models HTTP/1.1" 200 OK
2026-10-31T16:00:02.000000000Z INFO:     172.27.0.1:2 - "POST /v1/chat/completions HTTP/1.1" 200 OK
2026-10-31T16:00:03.000000000Z INFO:     172.27.0.1:2 - "GET /metrics HTTP/1.1" 200 OK
2026-10-31T16:00:04.000000000Z INFO:     172.27.0.2:3 - "POST /v1/chat/completions HTTP/1.1" 200 OK
2026-10-31T16:00:05.000000000Z INFO:     10.0.0.9:4 - "POST /v1/completions?x=1 HTTP/1.1" 400 Bad Request
2026-10-31T16:00:05.500000000Z (APIServer pid=1) INFO Engine 000: Avg prompt throughput: 3.1 tokens/s
"""


@pytest.fixture
def serving(api, monkeypatch):
    from app import activity

    activity.reset()
    started = "2026-10-31T16:00:00.000000000Z"
    c = SimpleNamespace(id="served", status="running", labels={"vllm-hub.model": "org/model"},
                        attrs={"State": {"StartedAt": started}})
    log_calls = []

    def logs(**kw):
        log_calls.append(kw)
        return c.output

    c.output, c.logs = ACCESS, logs
    counters = {"running": 2.0, "prompt": 1000.0, "gen": 100.0}
    monkeypatch.setattr(runner, "_container", lambda: c)
    monkeypatch.setattr(activity.httpx, "get", lambda *a, **kw: SimpleNamespace(
        raise_for_status=lambda: None, text=METRICS.format(**counters)))
    monkeypatch.setattr(activity, "_gateways", lambda: {"172.27.0.1"})
    monkeypatch.setattr(activity, "_own_ips", lambda: {"172.27.0.2"})
    monkeypatch.setattr(activity.time, "time", lambda: runner._timestamp(started) + 20)
    yield activity, c, counters, log_calls
    activity.reset()


def test_activity_is_empty_without_a_served_model(api):
    from app import activity

    activity.reset()
    activity.sample()
    assert api.get("/api/activity").json() == {"serving": False, "interval": activity.INTERVAL}


def test_activity_reports_rates_callers_and_recent_requests(api, serving):
    activity, c, counters, log_calls = serving
    t0 = runner._timestamp(c.attrs["State"]["StartedAt"])
    activity.sample(now=t0 + 10)
    counters.update(prompt=1500.0, gen=150.0)
    activity.sample(now=t0 + 15)
    body = api.get("/api/activity").json()
    assert body["serving"] is True
    assert body["current"] == {"running": 2, "waiting": 1, "kv_cache": 0.25, "gen_tps": 10.0, "prompt_tps": 100.0}
    assert body["totals"]["requests"] == 4 and body["totals"]["avg_ttft_s"] == 0.5
    assert body["totals"]["prefix_cache_hit"] == 0.75
    assert len(body["history"]) == 1
    # Hub status polling and scrapes are ignored; each caller is labelled and counted once.
    callers = {x["client"]: x for x in body["callers"]}
    assert set(callers) == {"172.27.0.1", "172.27.0.2", "10.0.0.9"}
    assert callers["172.27.0.1"]["label"] == "This computer" and callers["172.27.0.1"]["requests"] == 1
    assert callers["172.27.0.2"]["label"] == "vLLM Hub"
    assert callers["10.0.0.9"]["errors"] == 1 and callers["10.0.0.9"]["label"] is None
    assert [r["path"] for r in body["recent"]] == ["/v1/completions", "/v1/chat/completions", "/v1/chat/completions"]
    # The second read starts after the last log line, so the same requests are not counted twice.
    assert log_calls[1]["since"] == pytest.approx(t0 + 5.5)


def test_activity_resets_for_a_new_container_and_when_stopped(api, serving, monkeypatch):
    activity, c, counters, _ = serving
    t0 = runner._timestamp(c.attrs["State"]["StartedAt"])
    activity.sample(now=t0 + 10)
    activity.sample(now=t0 + 15)
    assert api.get("/api/activity").json()["callers"]
    c.id, c.output = "replacement", b""
    activity.sample(now=t0 + 20)
    body = api.get("/api/activity").json()
    assert body["callers"] == [] and body["history"] == []
    monkeypatch.setattr(runner, "_container", lambda: None)
    activity.sample(now=t0 + 25)
    assert api.get("/api/activity").json()["serving"] is False


def test_activity_hides_a_loading_model(api, serving, monkeypatch):
    activity, *_ = serving

    def down(*a, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr(activity.httpx, "get", down)
    activity.sample()
    assert api.get("/api/activity").json()["serving"] is False


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
