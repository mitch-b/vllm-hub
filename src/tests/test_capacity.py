from types import SimpleNamespace

from app import capacity

LOG = b"INFO [kv_cache_utils.py:2464] GPU KV cache size: 58,254 tokens, Maximum concurrency for 32,768 tokens: 1.78x\n"


def test_learn_records_capacity_once(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity, "STATE", tmp_path / "s" / "capacity.json")
    calls = []
    c = SimpleNamespace(id="ctr1", logs=lambda tail: calls.append(1) or LOG)
    assert capacity.learn(c, "org/m", 0.9) == 58254
    assert capacity.learn(c, "org/m", 0.9) == 58254
    assert len(calls) == 1
    assert capacity.kv_tokens("org/m", 0.90) == 58254
    assert capacity.kv_tokens("org/m", 0.85) is None


def test_learn_without_log_line(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity, "STATE", tmp_path / "capacity.json")
    assert capacity.learn(SimpleNamespace(id="ctr2", logs=lambda tail: b"loading"), "org/m", 0.9) is None


def test_fit_context():
    assert capacity.fit_context(58254, None) == 56320
    assert capacity.fit_context(58254, 40960) == 40960
    assert capacity.fit_context(500, None) == 2048


def test_auto_context_uses_model_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity, "STATE", tmp_path / "capacity.json")
    capacity.record("org/m", 0.9, 300000)
    monkeypatch.setattr(capacity, "model_max_len", lambda m: 262144)
    assert capacity.auto_context("org/m", 0.9) == 262144
    assert capacity.auto_context("org/other", 0.9) is None
