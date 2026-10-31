"""Learns each model's real context capacity from vLLM's startup log, so "auto" context uses all the VRAM."""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

STATE = Path(os.getenv("HUB_STATE_DIR", "/state")) / "capacity.json"
KV_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
MARGIN = 0.97  # activation peaks vary a little between runs
_lock = threading.Lock()
_seen: set[str] = set()


def _load() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def kv_tokens(model_id: str, util: float) -> int | None:
    return _load().get(model_id, {}).get(f"{util:.2f}")


def record(model_id: str, util: float, tokens: int) -> None:
    with _lock:
        data = _load()
        data.setdefault(model_id, {})[f"{util:.2f}"] = tokens
        try:
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps(data, indent=1))
        except OSError:
            pass


def learn(container, model_id: str, util: float) -> int | None:
    """Parse the KV capacity from a ready vLLM container's log (once per container)."""
    if container.id in _seen:
        return kv_tokens(model_id, util)
    m = KV_RE.search(container.logs(tail=4000).decode(errors="replace"))
    if not m:
        return None
    _seen.add(container.id)
    tokens = int(m.group(1).replace(",", ""))
    record(model_id, util, tokens)
    return tokens


def model_max_len(model_id: str) -> int | None:
    """The model's own maximum positions, from config.json in the local HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(model_id, "config.json")
        cfg = json.loads(Path(path).read_text()) if isinstance(path, str) else {}
    except Exception:
        return None
    for c in (cfg, cfg.get("text_config") or {}):
        if c.get("max_position_embeddings"):
            return int(c["max_position_embeddings"])
    return None


def fit_context(tokens: int, model_max: int | None) -> int:
    """Largest context (multiple of 1024) that fits the learned KV capacity and the model's limit."""
    ctx = int(tokens * MARGIN) // 1024 * 1024
    if model_max:
        ctx = min(ctx, model_max)
    return max(ctx, 2048)


def auto_context(model_id: str, util: float) -> int | None:
    tokens = kv_tokens(model_id, util)
    return fit_context(tokens, model_max_len(model_id)) if tokens else None
