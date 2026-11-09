"""Hugging Face search, curated recommendations and size/fit enrichment."""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from huggingface_hub import HfApi, ModelInfo, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from . import capacity, profiles
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

SORTS = {"popular": "downloads", "recent": "createdAt", "updated": "lastModified", "trending": "trendingScore"}
_EXPAND = ["downloads", "likes", "tags", "safetensors", "gated", "pipeline_tag", "createdAt", "lastModified"]
_cache: dict[str, tuple[float, ModelInfo]] = {}
_context_cache: dict[str, tuple[float, dict]] = {}
_weight_cache: dict[str, tuple[float, float | None]] = {}
CACHE_TTL = 600
logger = logging.getLogger(__name__)


def dates(info) -> dict:
    def iso(value):
        return value.isoformat() if hasattr(value, "isoformat") else value
    return {"uploaded_at": iso(getattr(info, "created_at", None)),
            "updated_at": iso(getattr(info, "last_modified", None))}


def config_max_context(config: dict) -> int | None:
    for part in (config.get("text_config") or {}, config):
        if not isinstance(part, dict):
            raise ValueError("Model text_config is not an object")
        value = part.get("max_position_embeddings")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _context_metadata(model_id: str) -> dict:
    hit = _context_cache.get(model_id)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    try:
        path = hf_hub_download(model_id, "config.json")
        config = json.loads(Path(path).read_text())
        if not isinstance(config, dict):
            raise ValueError("config.json is not an object")
        out = {"max_context": config_max_context(config), "context_error": None}
    except (HfHubHTTPError, httpx.HTTPError, OSError, ValueError) as exc:
        logger.warning("Could not read context metadata for %s: %s", model_id, exc)
        out = {"max_context": None, "context_error": "Could not read model context metadata."}
    _context_cache[model_id] = (time.time(), out)
    return out


def context_preview(model_id: str, w: float | None, total: float | None, verdict: str) -> dict:
    metadata = _context_metadata(model_id)
    profile = profiles.launch_profile(model_id, gpu_info()) if model_id == profiles.NEMOTRON_SUPER else None
    if profile:
        metadata = {**metadata, "max_context": profile["max_context"], "context_error": None,
                    "launch_profile": profile}
    maximum = metadata["max_context"]
    usable = total and verdict in ("fits", "tight")
    learned = (capacity.auto_context(model_id, 0.90, maximum) if profile
               else capacity.auto_context(model_id, 0.90)) if usable else None
    estimate = (profile["default_context"] if profile else profiles.choose_context(w, total, 0.90)
                ) if usable and w else None
    hint = learned or estimate
    if hint and maximum:
        hint = min(hint, maximum)
    return {**metadata, "context_hint": hint, "context_learned": bool(learned)}


def native_metadata(info) -> dict:
    gguf = getattr(info, "gguf", None) or {}
    maximum = gguf.get("context_length")
    maximum = maximum if isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0 else None
    return {**dates(info), "max_context": maximum, "context_hint": min(maximum or 8192, 8192),
            "context_learned": False}


def _total_gib() -> float | None:
    gpu = gpu_info()
    return gpu["total_mib"] / 1024 if gpu else None


def _packed_weights_gib(model_id: str) -> float | None:
    """Packed dtype parameter counts can be logical values, not stored bytes."""
    hit = _weight_cache.get(model_id)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    try:
        path = hf_hub_download(model_id, "model.safetensors.index.json")
        size = json.loads(Path(path).read_text())["metadata"]["total_size"]
        if not isinstance(size, (int, float)) or isinstance(size, bool) or size <= 0:
            raise ValueError("Invalid checkpoint tensor size")
        weight = size / profiles.GIB
    except (HfHubHTTPError, httpx.HTTPError, OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Could not read packed checkpoint size for %s: %s", model_id, exc)
        weight = None
    _weight_cache[model_id] = (time.time(), weight)
    return weight


def _summarize(info, total_gib: float | None) -> dict:
    tags = list(info.tags or [])
    st = getattr(info, "safetensors", None)
    params = getattr(st, "parameters", None) if st else None
    quant = profiles.quant_label(tags, info.id)
    packed = quant in ("NVFP4", "MXFP4") or "gpt-oss" in info.id.lower()
    w = _packed_weights_gib(info.id) if packed else profiles.weights_gib(params)
    fit = profiles.fit_verdict(w, total_gib, tags)
    return {
        "id": info.id,
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "gated": bool(getattr(info, "gated", False)),
        "quant": quant,
        "size_error": "Could not read packed checkpoint size." if packed and w is None else None,
        "weights_gib": round(w, 1) if w else None,
        **fit,
        **dates(info),
        **context_preview(info.id, w, total_gib, fit["verdict"]),
    }


def search(query: str, limit: int = 25, sort: str = "popular") -> list[dict]:
    total = _total_gib()
    results = api.list_models(
        search=query, pipeline_tag="text-generation", sort=SORTS[sort],
        limit=min(max(limit, 1), 50), expand=_EXPAND,
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda m: _summarize(m, total), results))


def _one(model_id: str, total: float | None) -> dict | None:
    hit = _cache.get(model_id)
    if hit and time.time() - hit[0] < CACHE_TTL:
        info = hit[1]
    else:
        try:
            info = api.model_info(model_id, expand=_EXPAND)
        except Exception:
            return None
        _cache[model_id] = (time.time(), info)
    return _summarize(info, total)


_RANK = {"fits": 0, "tight": 1, "unknown": 2, "too_big": 3, "unsupported": 4}


def recommended() -> list[dict]:
    total = _total_gib()
    picks = list(RECOMMENDED)
    if profiles.spark_recipe(profiles.NEMOTRON_SUPER, gpu_info()):
        picks.insert(0, profiles.NEMOTRON_SUPER)
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for r in pool.map(lambda m: _one(m, total), picks) if r]
    # Best usable first; weight size is a fit heuristic, not an intelligence score.
    rows.sort(key=lambda r: (_RANK[r["verdict"]], -(r["weights_gib"] or 0)))
    return rows


def repo_files(model_id: str) -> list[tuple[str, int]]:
    info = api.model_info(model_id, files_metadata=True)
    return [(s.rfilename, s.size or 0) for s in (info.siblings or [])]
