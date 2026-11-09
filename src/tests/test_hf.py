import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from huggingface_hub import ModelInfo

from app import capacity, hf, profiles


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    monkeypatch.setattr(hf, "_context_cache", {})
    monkeypatch.setattr(hf, "_cache", {})
    monkeypatch.setattr(hf, "_weight_cache", {})
    monkeypatch.setattr(hf, "gpu_info", lambda: None)
    monkeypatch.setattr(hf, "_total_gib", lambda: 24)
    monkeypatch.setattr(capacity, "auto_context", lambda *args: None)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"max_position_embeddings": 32768}))
    monkeypatch.setattr(hf, "hf_hub_download", lambda *args: str(config))
    info = ModelInfo(id="fixture/model", downloads=100, likes=5, tags=["awq"],
                     safetensors={"parameters": {"F16": 3 * 2**30}, "total": 3 * 2**30},
                     createdAt="2026-10-01T00:00:00Z", lastModified="2026-10-30T00:00:00Z")
    return info, config


def test_packed_checkpoint_uses_stored_tensor_bytes(catalog, monkeypatch):
    info, config = catalog
    info.id = profiles.NEMOTRON_SUPER
    info.tags = ["nvfp4"]
    info.safetensors = SimpleNamespace(parameters={
        "U8": 112764911616, "BF16": 6020553728, "F8_E4M3": 4825546752, "F32": 20992,
    })
    index = config.with_name("model.safetensors.index.json")
    index.write_text(json.dumps({"metadata": {"total_size": 80297329824}}))
    calls = []

    def download(model_id, filename):
        calls.append(filename)
        return str(index if filename.endswith("index.json") else config)

    monkeypatch.setattr(hf, "hf_hub_download", download)
    for _ in range(2):
        row = hf._summarize(info, 121)
        assert row["weights_gib"] == 74.8
        assert row["verdict"] == "fits"
        assert row["size_error"] is None
    assert calls.count("model.safetensors.index.json") == 1
    assert profiles.weights_gib(info.safetensors.parameters) > 120


@pytest.mark.parametrize("metadata", [{}, {"total_size": 0}, {"total_size": True},
                                      {"total_size": "unknown"}])
def test_packed_size_failure_is_unknown_not_inflated(catalog, metadata, caplog):
    info, config = catalog
    info.tags = ["nvfp4"]
    config.write_text(json.dumps({"metadata": metadata}))
    row = hf._summarize(info, 121)
    assert row["weights_gib"] is None
    assert row["verdict"] == "unknown"
    assert row["size_error"] == "Could not read packed checkpoint size."
    assert "Could not read packed checkpoint size" in caplog.text


def test_spark_context_preview_uses_recipe_limit(catalog, monkeypatch):
    monkeypatch.setattr(hf, "gpu_info", lambda: {
        "name": "NVIDIA GB10", "unified_memory": True, "gpu_count": 1, "total_mib": 124000,
    })
    row = hf.context_preview(profiles.NEMOTRON_SUPER, 74.8, 121, "fits")
    assert row["max_context"] == row["context_hint"] == 1_000_000
    assert row["launch_profile"]["graphs"] is True
    assert not row["context_learned"]
    calls = []
    monkeypatch.setattr(capacity, "auto_context", lambda *args: calls.append(args) or 524288)
    row = hf.context_preview(profiles.NEMOTRON_SUPER, 74.8, 121, "fits")
    assert row["context_hint"] == 524288 and row["context_learned"]
    assert calls == [(profiles.NEMOTRON_SUPER, 0.9, 1_000_000)]


def test_spark_recommended_only_on_supported_hardware(catalog, monkeypatch):
    calls = []
    monkeypatch.setattr(hf, "_one", lambda model, total: calls.append(model))
    hf.recommended()
    assert profiles.NEMOTRON_SUPER not in calls
    monkeypatch.setattr(hf, "gpu_info", lambda: {
        "name": "NVIDIA GB10", "unified_memory": True, "gpu_count": 1, "total_mib": 124000,
    })
    hf.recommended()
    assert profiles.NEMOTRON_SUPER in calls


@pytest.mark.parametrize("sort,upstream", [
    ("popular", "downloads"), ("recent", "createdAt"),
    ("updated", "lastModified"), ("trending", "trendingScore"),
])
def test_search_sort_and_metadata(catalog, monkeypatch, sort, upstream):
    info, _ = catalog
    calls = []
    def models(**kwargs):
        calls.append(kwargs)
        return [info]
    monkeypatch.setattr(hf.api, "list_models", models)
    row = hf.search("", sort=sort)[0]
    assert calls[0]["sort"] == upstream
    assert {"createdAt", "lastModified"}.issubset(calls[0]["expand"])
    assert row["uploaded_at"] == "2026-10-01T00:00:00+00:00"
    assert row["updated_at"] == "2026-10-30T00:00:00+00:00"
    assert row["max_context"] == 32768
    assert row["context_hint"] == min(profiles.choose_context(6, 24, 0.9), 32768)
    assert row["context_learned"] is False


def test_measured_context_capped_by_model_metadata(catalog, monkeypatch):
    info, _ = catalog
    monkeypatch.setattr(capacity, "auto_context", lambda *args: 65536)
    row = hf._summarize(info, 24)
    assert row["context_hint"] == 32768 and row["context_learned"]


@pytest.mark.parametrize("total,params,tags", [
    (None, {"F16": 3 * 2**30}, []), (24, None, []),
    (24, {"F16": 30 * 2**30}, []), (24, {"F16": 3 * 2**30}, ["gguf"]),
])
def test_no_context_fit_claim_without_usable_weights_and_gpu(catalog, total, params, tags):
    info, _ = catalog
    info.safetensors = SimpleNamespace(parameters=params)
    info.tags = tags
    row = hf._summarize(info, total)
    assert row["context_hint"] is None
    assert row["max_context"] == 32768


@pytest.mark.parametrize("config,expected", [
    ({}, None), ({"max_position_embeddings": -1}, None),
    ({"max_position_embeddings": True}, None),
    ({"max_position_embeddings": "unknown"}, None),
    ({"max_position_embeddings": 4096, "text_config": {"max_position_embeddings": 131072}}, 131072),
])
def test_model_maximum_is_reported_not_invented(config, expected):
    assert hf.config_max_context(config) == expected


def test_context_metadata_failure_and_cache(catalog, monkeypatch, caplog):
    calls = []
    def missing(*args):
        calls.append(args)
        raise OSError("Metadata unavailable")
    monkeypatch.setattr(hf, "hf_hub_download", missing)
    for _ in range(2):
        row = hf._summarize(catalog[0], 24)
        assert row["max_context"] is None
        assert row["context_error"] == "Could not read model context metadata."
    assert len(calls) == 1
    assert "Metadata unavailable" in caplog.text


def test_cached_recommendation_recomputes_gpu_fit(catalog, monkeypatch):
    monkeypatch.setattr(hf.api, "model_info", lambda *a, **kw: catalog[0])
    assert hf._one("fixture/model", 24)["verdict"] == "fits"
    row = hf._one("fixture/model", None)
    assert row["verdict"] == "unknown" and row["context_hint"] is None


@pytest.mark.parametrize("maximum,expected", [(131072, 8192), (4096, 4096), (None, 8192), (-1, 8192)])
def test_native_context_default_not_a_fit_estimate(maximum, expected):
    info = SimpleNamespace(gguf={"context_length": maximum},
                           created_at=datetime(2026, 10, 1, tzinfo=timezone.utc))
    row = hf.native_metadata(info)
    assert row["context_hint"] == expected
    assert row["max_context"] == (maximum if maximum and maximum > 0 else None)
    assert row["context_learned"] is False
    assert row["updated_at"] is None


@pytest.mark.parametrize("sort,upstream", list(hf.SORTS.items()))
def test_native_sort_preserved(monkeypatch, sort, upstream):
    from app import model_runner
    calls = []
    def models(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(id="fixture/gguf", downloads=1, likes=1, gated=False,
                                siblings=[SimpleNamespace(rfilename="model-Q4_K_M.gguf")],
                                gguf={"context_length": 65536})]
    monkeypatch.setattr(hf.api, "list_models", models)
    row = model_runner.search("", sort=sort)[0]
    assert calls[0]["sort"] == upstream
    assert row["max_context"] == 65536 and row["context_hint"] == 8192
