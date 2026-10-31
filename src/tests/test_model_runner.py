import json
from threading import Event, Thread
from types import SimpleNamespace

import httpx
import pytest
from docker.errors import APIError, NotFound
from fastapi.testclient import TestClient

from app import downloads, gpu, hosting, main, model_runner, runner


class ImmediateThread:
    def __init__(self, target, args=(), **kwargs):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


@pytest.fixture
def native(monkeypatch, tmp_path):
    state = SimpleNamespace(calls=[], loaded=[], block=None, release=None, fail=None,
                            progress=b'{"id":"weights","progressDetail":{"current":100,"total":100}}\n')
    conn = {"url": "http://native.test/", "engines": {"llama.cpp": "Running: llama.cpp metal"}}
    model = {
        "id": "sha256:fixture", "tags": ["ai/qwen3:0.6B-Q4_K_M"], "created": 0,
        "config": {"format": "gguf", "size": "456.11 MiB", "context_size": 8192},
    }

    def handle(request):
        body = json.loads(request.content) if request.content else None
        state.calls.append((request.method, request.url.path, body))
        if state.fail and request.url.path == state.fail:
            return httpx.Response(503, text="Native engine unavailable")
        if request.url.path == "/models":
            return httpx.Response(200, json=[model])
        if request.url.path.startswith("/models/") and request.url.path != "/models/create":
            return httpx.Response(200, json=model if request.method == "GET" else {})
        if request.url.path == "/models/create":
            return httpx.Response(200, content=state.progress)
        if request.url.path == "/engines/status":
            return httpx.Response(200, json=conn["engines"])
        if request.url.path == "/engines/ps":
            return httpx.Response(200, json=state.loaded)
        if request.url.path == "/engines/_configure":
            return httpx.Response(202)
        if request.url.path == "/engines/unload":
            state.loaded = [row for row in state.loaded if row["model_name"] not in body["models"]]
            return httpx.Response(200, json={"unloaded_runners": 1})
        if request.url.path == "/engines/v1/chat/completions":
            if request.headers.get("X-Preload-Only") == "true":
                if state.block:
                    state.block.set()
                    assert state.release.wait(5)
                state.loaded = [r for r in state.loaded if r["model_name"] != body["model"]]
                state.loaded.append({"model_name": body["model"], "backend_name": "llama.cpp", "loading": False})
                return httpx.Response(200)
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "Native answer"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            })
        pytest.fail(f"Unexpected native request: {request.method} {request.url}")

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(model_runner, "_http", lambda c, timeout=10: httpx.Client(
        base_url=c["url"], transport=transport, timeout=timeout))
    monkeypatch.setattr(model_runner, "discover", lambda: conn)
    monkeypatch.setattr(model_runner, "STATE", tmp_path / "model-runner.json")
    monkeypatch.setattr(model_runner, "_selection", {})
    monkeypatch.setattr(model_runner, "_cancel", None)
    monkeypatch.setattr(model_runner, "Thread", ImmediateThread)
    monkeypatch.setattr(hosting, "gpu_info", lambda: None)
    monkeypatch.setattr(main, "gpu_info", lambda: None)
    monkeypatch.setattr(downloads, "_jobs", {})
    monkeypatch.setattr(runner, "client", lambda: pytest.fail("Native inference must not manage CUDA containers"))
    monkeypatch.setattr(model_runner.hf.api, "model_info", lambda repo, **kwargs: SimpleNamespace(
        siblings=[SimpleNamespace(rfilename="model-Q4_K_M.gguf", size=2**30)], gated=False,
        card_data={"base_model": "Qwen/original"},
    ))
    state.api, state.model, state.conn = TestClient(main.app), model, conn
    return state


def test_native_system_auto_selects_metal_without_nvidia(native, monkeypatch):
    monkeypatch.setattr(hosting, "gpu_info", lambda: pytest.fail("Metal hosts must not request NVIDIA devices"))
    response = native.api.get("/api/system")
    assert response.status_code == 200
    body = response.json()
    assert body["gpu"] is None
    assert body["hosting"]["backend"] == "model-runner"
    assert body["hosting"]["accelerator"] == "Metal"
    assert body["hosting"]["local_inference"]["available"]
    assert body["hosting"]["features"] == {
        "cuda_options": False, "logs": False, "activity": False, "gpu_fit": False,
    }
    assert body["vllm"]["state"] == "stopped"


def test_native_models_recommendations_and_delete(native):
    model = native.api.get("/api/models").json()[0]
    assert model["size_gib"] == 0.45
    assert model["context_hint"] == 8192
    assert model["verdict"] == "unknown"
    assert [m["id"] for m in native.api.get("/api/recommended").json()] == [
        f"hf.co/{repo.lower()}:Q4_K_M" for repo in model_runner.RECOMMENDED
    ]
    job = downloads.Job(model["id"])
    job.status = "done"
    downloads._jobs[job.id] = job
    assert native.api.delete("/api/models/" + model["id"]).json() == {"ok": True}
    assert downloads.list_jobs() == []
    assert ("DELETE", "/models/" + model["id"], None) in native.calls


def test_native_search_returns_portable_gguf_references(native, monkeypatch):
    calls = []

    def search(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(id="fixture/coder-GGUF", downloads=10, likes=1, gated=False,
                                siblings=[SimpleNamespace(rfilename="coder-Q4_K_M.gguf")])]

    monkeypatch.setattr(model_runner.hf.api, "list_models", search)
    rows = native.api.get("/api/search?q=coder").json()
    assert rows[0]["id"] == "hf.co/fixture/coder-gguf:Q4_K_M"
    assert calls[0]["filter"] == "gguf"


def test_native_run_chat_stop_and_restart_use_only_owned_model(native):
    model = native.model["tags"][0]
    native.loaded = [{"model_name": "ai/unrelated-model", "backend_name": "llama.cpp"}]
    response = native.api.post("/api/run", json={"repo_id": model})
    assert response.status_code == 200
    assert response.json()["context"] == 8192
    assert response.json()["args"] == [] and response.json()["util"] is None
    configured = next(body for method, path, body in native.calls if path == "/engines/_configure")
    assert configured == {"model": model, "context-size": 8192, "keep_alive": "-1"}
    runtime = native.api.get("/api/system").json()["vllm"]
    assert runtime["ready"] is True
    assert runtime["served"] == {"id": model, "max_model_len": 8192}
    assert runtime["url"] == "http://localhost:12434/engines/v1"
    chat = native.api.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}).json()
    assert chat["content"] == "Native answer"
    assert native.api.get("/api/activity").json()["supported"] is False
    assert "Docker Desktop" in native.api.get("/api/logs").text
    # Reloading the hub uses persisted selection, not its in-memory startup timer.
    saved = model_runner.STATE.read_text()
    model_runner._selection = None
    assert native.api.get("/api/system").json()["vllm"]["ready"] is True
    assert model_runner.STATE.read_text() == saved
    assert native.api.post("/api/stop").status_code == 200
    assert native.loaded == [{"model_name": "ai/unrelated-model", "backend_name": "llama.cpp"}]
    unloads = [body for method, path, body in native.calls if path == "/engines/unload"]
    assert unloads == [{"models": [model]}]
    assert not any(body and body.get("all") for _, _, body in native.calls)
    assert native.api.get("/api/system").json()["vllm"]["state"] == "stopped"
    assert native.api.post("/api/run", json={"repo_id": model, "context": 4096}).status_code == 200


@pytest.mark.parametrize("options", [{"graphs": True}, {"util": 0.9}, {"extra": ["--kv-cache-dtype", "fp8"]},
                                     {"context": 16384}, {"context": -1}])
def test_native_rejects_cuda_options_and_invalid_context_before_stopping(native, options):
    response = native.api.post("/api/run", json={"repo_id": native.model["tags"][0], **options})
    assert response.status_code == 400
    assert not any(path in ("/engines/_configure", "/engines/unload") for _, path, _ in native.calls)


def test_native_loading_cancel_unloads_a_late_preload(native, monkeypatch):
    native.block, native.release = Event(), Event()
    threads = []

    def background(**kwargs):
        thread = Thread(**kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(model_runner, "Thread", background)
    model = native.model["tags"][0]
    try:
        assert native.api.post("/api/run", json={"repo_id": model}).status_code == 200
        assert native.block.wait(2)
        runtime = native.api.get("/api/system").json()["vllm"]
        assert runtime["state"] == "created" and not runtime["ready"]
        assert runtime["startup"]["elapsed_seconds"] >= 0
        assert native.api.post("/api/stop").status_code == 200
    finally:
        native.release.set()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
    assert not native.loaded
    assert model_runner.selected_model() is None


def test_native_preload_failure_surfaces_and_never_reports_ready(native):
    native.fail = "/engines/v1/chat/completions"
    assert native.api.post("/api/run", json={"repo_id": native.model["tags"][0]}).status_code == 200
    runtime = native.api.get("/api/system").json()["vllm"]
    assert runtime["state"] == "exited" and not runtime["ready"]
    assert "Native engine unavailable" in runtime["error"]


def test_native_pull_progress_completion_cancel_and_error(native, monkeypatch):
    job = downloads.Job(native.model["tags"][0])
    model_runner._pull(job)
    assert job.status == "done" and job.done == job.total == 100
    job = downloads.Job(native.model["tags"][0])
    job.cancel.set()
    model_runner._pull(job)
    assert job.status == "cancelled" and job.finished is not None
    native.fail = "/models/create"
    job = downloads.Job(native.model["tags"][0])
    model_runner._pull(job)
    assert job.status == "error" and "Native engine unavailable" in job.error
    monkeypatch.setattr(model_runner, "pull", lambda ref: job)
    assert native.api.post("/api/pull", json={"repo_id": job.repo_id}).json()["status"] == "error"


def test_native_catalog_errors_are_explicit(native, monkeypatch):
    native.fail = "/models"
    assert native.api.get("/api/models").status_code == 502
    def invalid(repo, **kwargs):
        raise ValueError("Publisher unavailable")

    monkeypatch.setattr(model_runner.hf.api, "model_info", invalid)
    rows = native.api.get("/api/recommended").json()
    assert all(row["verdict"] == "unsupported" and "Publisher unavailable" in row["reason"] for row in rows)


def test_native_curated_sources_sizes_and_partial_catalog_failure(native, monkeypatch):
    def info(repo, **kwargs):
        if repo == "Qwen/Qwen3-14B-GGUF":
            raise ValueError("Model metadata unavailable")
        return SimpleNamespace(gated=False, card_data={"base_model": "Qwen/original"}, siblings=[
            SimpleNamespace(rfilename="model-Q4_K_M.gguf", size=2**30),
            SimpleNamespace(rfilename="model-Q4_K_M-00001-of-00002.gguf", size=2**29),
            SimpleNamespace(rfilename="model-Q4_K_M-00002-of-00002.gguf", size=2**29),
            SimpleNamespace(rfilename="model-Q8_0.gguf", size=2**31),
        ])

    monkeypatch.setattr(model_runner.hf.api, "model_info", info)
    rows = native.api.get("/api/recommended").json()
    assert len(rows) == 6
    assert rows[0]["source_label"] == "Official publisher"
    assert rows[1]["source_label"] == "Established GGUF converter"
    assert rows[0]["source_model"] == "Qwen/original"
    assert rows[0]["size_gib"] == 1
    assert len([row for row in rows if row["verdict"] == "unsupported"]) == 1


def test_native_search_prioritizes_publishers_without_hiding_labeled_community_results(native, monkeypatch):
    def model(repo, file="model-Q4_K_M.gguf"):
        return SimpleNamespace(id=repo, downloads=1, likes=0, gated=False,
                               siblings=[SimpleNamespace(rfilename=file)])

    monkeypatch.setattr(model_runner.hf.api, "list_models", lambda **kwargs: [
        model("fixture/community"), model("unsloth/conversion"), model("Qwen/official"),
        model("fixture/no-q4", "model-Q8_0.gguf"),
    ])
    rows = native.api.get("/api/search?q=qwen").json()
    assert [row["publisher"] for row in rows] == ["Qwen", "unsloth", "fixture"]
    assert rows[-1]["source_label"] == "Community upload"


def test_native_hugging_face_canonical_registration_and_reference_reconciliation(native):
    native.model["tags"] = ["huggingface.co/qwen/qwen3-0.6b-gguf:Q4_K_M"]
    job = downloads.Job(model_runner._reference("hf.co/Qwen/Qwen3-0.6B-GGUF:Q4_K_M"))
    model_runner._pull(job)
    assert job.status == "done"
    assert model_runner.models()[0]["id"] == job.repo_id
    assert model_runner.start(native.model["tags"][0])["model"] == job.repo_id
    assert model_runner.selected_model() == job.repo_id
    assert model_runner.delete("hf.co/Qwen/Qwen3-0.6B-GGUF:Q4_K_M")
    assert model_runner.selected_model() is None


@pytest.mark.parametrize("escaped", [False, True])
def test_native_current_progress_protocol_and_canonical_tags(native, escaped):
    native.model["tags"] = ["docker.io/ai/qwen3:0.6B-Q4_K_M"]
    messages = [
        {"type": "progress", "total": 100, "layer": {"id": "weights", "size": 80, "current": 40}},
        {"type": "progress", "total": 100, "layer": {"id": "weights", "size": 80, "current": 80}},
        {"type": "progress", "total": 100, "layer": {"id": "config", "size": 20, "current": 20}},
        {"type": "success", "total": 0, "layer": {"size": 0, "current": 0}},
    ]
    stream = "\n".join(json.dumps(message) for message in messages)
    native.progress = (stream.replace('"', "&#34;") if escaped else stream).encode()
    job = downloads.Job("ai/qwen3:0.6B-Q4_K_M")
    model_runner._pull(job)
    assert job.status == "done" and job.total == job.done == 100
    assert [done for _, done in job._samples] == [40, 80, 100]
    assert native.api.get("/api/models").json()[0]["id"] == job.repo_id
    assert model_runner.start(native.model["tags"][0])["model"] == job.repo_id
    assert model_runner.delete(native.model["tags"][0])
    assert model_runner.selected_model() is None


@pytest.mark.parametrize("progress", [
    b'{"type":"error","message":"Registry denied the pull"}\n',
    b'{"errorDetail":"Registry denied the pull"}\n',
    b'{"layer":{"size":100,"current":"invalid"}}\n',
    b'{"progressDetail":[]}\n',
    b'[]\n',
    b'{malformed}\n',
])
def test_native_malformed_or_failed_progress_never_marks_a_cached_model_done(native, progress):
    native.progress = progress
    job = downloads.Job(native.model["tags"][0])
    model_runner._pull(job)
    assert job.status == "error" and job.error
    assert job.finished is not None


@pytest.mark.parametrize("reference", ["ai/../models", "ai//model", "ai/model?remote=true", "ai/model#fragment"])
def test_native_rejects_invalid_references_before_any_runtime_change(native, reference):
    assert native.api.post("/api/pull", json={"repo_id": reference}).status_code == 400
    assert native.api.post("/api/run", json={"repo_id": reference}).status_code == 400
    with pytest.raises(ValueError, match="reference"):
        model_runner.delete(reference)
    assert not native.calls


@pytest.mark.parametrize("replacement", ["ai/qwen3:0.6B-Q4_K_M", "ai/qwen3:8B-Q4_K_M"])
def test_native_replacement_is_not_unloaded_by_an_older_preload(native, monkeypatch, replacement):
    native.block, native.release = Event(), Event()
    threads = []

    def background(**kwargs):
        thread = Thread(**kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(model_runner, "Thread", background)
    try:
        model_runner.start(native.model["tags"][0])
        assert native.block.wait(2)
        native.block = None
        model_runner.start(replacement)
        threads[-1].join(5)
        assert not threads[-1].is_alive()
    finally:
        native.release.set()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
    assert model_runner.selected_model() == replacement
    assert model_runner.status()["ready"]
    assert native.loaded == [{"model_name": replacement, "backend_name": "llama.cpp", "loading": False}]


def test_native_restart_recovers_model_runner_loss_using_persisted_context(native):
    model_runner.start(native.model["tags"][0], context=4096)
    old_attempt = model_runner.status()["container_id"]
    native.loaded = []
    model_runner._selection = None
    model_runner.resume()
    runtime = model_runner.status()
    assert runtime["ready"] and runtime["context"] == 4096
    assert runtime["container_id"] != old_attempt


def test_native_readiness_uses_persisted_digest_without_depending_on_cached_metadata(native):
    model_runner.start(native.model["tags"][0])
    native.loaded = [{"model_name": native.model["id"], "backend_name": "llama.cpp", "loading": False}]
    native.fail = "/models/" + native.model["tags"][0]
    assert model_runner.status()["ready"]
    assert json.loads(model_runner.STATE.read_text())["model_id"] == native.model["id"]


def test_native_unknown_context_maximum_does_not_become_a_fictitious_8k_limit(native):
    native.model["config"].pop("context_size")
    model = model_runner.models()[0]
    assert model["max_context"] is None and model["context_hint"] == 8192
    assert model_runner.start(model["id"], context=16384)["context"] == 16384


def test_native_configuration_failure_remains_owned_and_explicit(native):
    native.fail = "/engines/_configure"
    response = native.api.post("/api/run", json={"repo_id": native.model["tags"][0]})
    assert response.status_code == 500
    runtime = model_runner.status()
    assert runtime["state"] == "exited" and not runtime["ready"]
    assert "Native engine unavailable" in runtime["error"]
    assert native.api.post("/api/stop").status_code == 200


def test_native_state_failure_is_reported_even_when_error_cannot_be_persisted(native, monkeypatch, caplog):
    selection = {"model": native.model["tags"][0], "state": "created", "attempt": "fixture", "started_at": 0}
    model_runner._selection = selection
    native.fail = "/engines/v1/chat/completions"

    def fail(selection):
        raise OSError("State disk unavailable")

    monkeypatch.setattr(model_runner, "_save", fail)
    model_runner._preload(native.conn, selection, Event())
    assert model_runner.status()["state"] == "exited"
    assert "State disk unavailable" in model_runner.status()["error"]
    assert "Could not persist Model Runner failure state" in caplog.text


def test_cuda_remains_default_for_nvidia_and_native_selection_is_sticky(native, monkeypatch):
    native.conn["engines"] = {"llama.cpp": "Running: CUDA"}
    nvidia = {"total_mib": 24576, "used_mib": 0, "name": "Test NVIDIA"}
    monkeypatch.setattr(hosting, "gpu_info", lambda: nvidia)
    assert hosting.detect()["backend"] == "cuda"
    model_runner._selection = {"model": native.model["tags"][0]}
    assert hosting.detect()["backend"] == "model-runner"


def test_gpu_probe_uses_own_image_and_caches_unavailable_devices(monkeypatch):
    monkeypatch.setattr(gpu.time, "monotonic", lambda: 1.0)
    calls = []
    removed = []

    def create(image, **kwargs):
        calls.append((image, kwargs))
        return SimpleNamespace(
            start=lambda: None, wait=lambda timeout: {"StatusCode": 0},
            logs=lambda: b"Test NVIDIA, 24576, 1000, 1\n",
            remove=lambda **kwargs: removed.append(kwargs),
        )

    monkeypatch.setattr(gpu, "_probe", {"at": 0.0, "gpu": None, "error": None})
    monkeypatch.setattr(runner, "client", lambda: SimpleNamespace(containers=SimpleNamespace(
        get=lambda name: SimpleNamespace(attrs={"Image": "sha256:own-hub"}), create=create,
    )))
    assert gpu._docker_gpu()["total_mib"] == 24576
    assert gpu._docker_gpu()["used_mib"] == 1000
    assert len(calls) == 1
    assert calls[0][0] == "sha256:own-hub"
    assert calls[0][1]["network_mode"] == "none"
    assert calls[0][1]["device_requests"][0]["Capabilities"] == [["gpu"]]
    assert calls[0][1]["labels"] == {"vllm-hub.probe": "nvidia"}
    assert removed == [{"force": True}]
    gpu.reset_probe()
    monkeypatch.setattr(runner, "client", lambda: SimpleNamespace(containers=SimpleNamespace(
        get=lambda name: (_ for _ in ()).throw(NotFound("no NVIDIA runtime")),
    )))
    assert gpu._docker_gpu() is None
    assert "no NVIDIA runtime" in gpu._probe["error"]


def test_gpu_probe_cleans_up_after_cdi_start_failure_and_caches_failure(monkeypatch):
    monkeypatch.setattr(gpu.time, "monotonic", lambda: 1.0)
    removed, created = [], []

    def fail():
        raise APIError("no known GPU vendor found")

    def create(*args, **kwargs):
        created.append(kwargs)
        return SimpleNamespace(start=fail, remove=lambda **kwargs: removed.append(kwargs))

    monkeypatch.setattr(gpu, "_probe", {"at": 0.0, "gpu": None, "error": None})
    monkeypatch.setattr(runner, "client", lambda: SimpleNamespace(containers=SimpleNamespace(
        get=lambda name: SimpleNamespace(attrs={"Image": "sha256:own-hub"}), create=create,
    )))
    assert gpu._docker_gpu() is None
    assert gpu._docker_gpu() is None
    assert len(created) == 1 and removed == [{"force": True}]
    assert "no known GPU vendor" in gpu._probe["error"]


def test_model_runner_auto_discovery_socket_fallback_and_cache(monkeypatch):
    monkeypatch.setattr(model_runner.time, "monotonic", lambda: 1.0)
    original = model_runner.discover
    calls = []
    monkeypatch.setattr(model_runner, "_discovery", {"at": 0.0, "connection": None, "error": None})
    monkeypatch.delenv("HUB_MODEL_RUNNER_URL", raising=False)

    def request(conn, method, path, **kwargs):
        calls.append((conn, path))
        if conn.get("socket"):
            raise RuntimeError("Desktop proxy is unavailable")
        return httpx.Response(200, json=[] if path == "models" else {"llama.cpp": "Running: metal"})

    monkeypatch.setattr(model_runner, "request", request)
    connection = original()
    assert connection["url"] == "http://model-runner.docker.internal:12434/"
    assert original() == connection
    assert len(calls) == 3
    assert calls[0][0]["socket"] is True
    assert model_runner._discovery["error"] == "Desktop proxy is unavailable"
