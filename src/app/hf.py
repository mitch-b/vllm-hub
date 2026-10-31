"""Hugging Face search, curated recommendations and size/fit enrichment."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi

from . import profiles
from .gpu import gpu_info

api = HfApi()

# Coding-capable picks that are worth checking against the current GPU.
RECOMMENDED = [
    "cyankiwi/Qwen3.6-27B-AWQ-INT4",
    "QuantTrio/Qwen3.6-27B-AWQ",
    "Qwen/Qwen3.6-35B-A3B-FP8",
    "QuantTrio/Qwen3.6-35B-A3B-AWQ",
    "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
    "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "mistralai/Devstral-Small-2505",
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-8B",
]

_EXPAND = ["downloads", "likes", "tags", "safetensors", "gated", "pipeline_tag"]
_cache: dict[str, tuple[float, dict]] = {}
CACHE_TTL = 600


def _total_gib() -> float | None:
    gpu = gpu_info()
    return gpu["total_mib"] / 1024 if gpu else None


def _summarize(info, total_gib: float | None) -> dict:
    tags = list(info.tags or [])
    st = getattr(info, "safetensors", None)
    params = getattr(st, "parameters", None) if st else None
    w = profiles.weights_gib(params)
    fit = profiles.fit_verdict(w, total_gib, tags)
    return {
        "id": info.id,
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "gated": bool(getattr(info, "gated", False)),
        "quant": profiles.quant_label(tags, info.id),
        "weights_gib": round(w, 1) if w else None,
        **fit,
    }


def search(query: str, limit: int = 25) -> list[dict]:
    total = _total_gib()
    results = api.list_models(
        search=query, pipeline_tag="text-generation", sort="downloads",
        limit=limit, expand=_EXPAND,
    )
    return [_summarize(m, total) for m in results]


def _one(model_id: str, total: float | None) -> dict | None:
    hit = _cache.get(model_id)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    try:
        info = api.model_info(model_id, expand=_EXPAND)
    except Exception:
        return None
    out = _summarize(info, total)
    _cache[model_id] = (time.time(), out)
    return out


_RANK = {"fits": 0, "tight": 1, "unknown": 2, "too_big": 3, "unsupported": 4}


def recommended() -> list[dict]:
    total = _total_gib()
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for r in pool.map(lambda m: _one(m, total), RECOMMENDED) if r]
    # Best usable first; among usable models prefer the biggest (most capable) that fits.
    rows.sort(key=lambda r: (_RANK[r["verdict"]], -(r["weights_gib"] or 0)))
    return rows


def repo_files(model_id: str) -> list[tuple[str, int]]:
    info = api.model_info(model_id, files_metadata=True)
    return [(s.rfilename, s.size or 0) for s in (info.siblings or [])]
