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
    monkeypatch.setattr(hf, "_total_gib", lambda: 24)
    monkeypatch.setattr(capacity, "auto_context", lambda *args: None)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"max_position_embeddings": 32768}))
    monkeypatch.setattr(hf, "hf_hub_download", lambda *args: str(config))
    info = ModelInfo(id="fixture/model", downloads=100, likes=5, tags=["awq"],
                     safetensors={"parameters": {"F16": 3 * 2**30}, "total": 3 * 2**30},
                     createdAt="2026-10-01T00:00:00Z", lastModified="2026-10-30T00:00:00Z")
    return info, config


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
