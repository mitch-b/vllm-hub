"""Pure logic: model sizing, GPU fit verdicts, context choice and vLLM flag profiles."""
from __future__ import annotations

import re

GIB = 2**30

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 0.5, "I16": 2,
    "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1,
}

CONTEXT_STEPS = [4096, 8192, 16384, 32768, 65536, 131072]

# Headroom model (GiB): memory the OS/display keeps, plus vLLM activations/overhead.
OS_RESERVE_GIB = 1.5
VLLM_OVERHEAD_GIB = 1.4
MIN_KV_GIB = 0.5
FIT_COMFORT_GIB = 3.0
# Conservative KV cache bytes per token, per GiB of weights (fp8 KV cache).
KV_BYTES_PER_TOKEN_PER_GIB = 4096

UNSUPPORTED_TAGS = {"gguf", "mlx", "onnx", "coreml", "tflite"}


def weights_gib(safetensors_params: dict | None) -> float | None:
    """Weight size from the Hub's per-dtype parameter counts (packed INT4 shows up as I32)."""
    if not safetensors_params:
        return None
    total = sum(DTYPE_BYTES.get(dt, 2) * int(n) for dt, n in safetensors_params.items())
    return total / GIB if total else None


def quant_label(tags: list[str], model_id: str) -> str:
    haystack = " ".join(tags).lower() + " " + model_id.lower()
    for key, label in (
        ("nvfp4", "NVFP4"), ("mxfp4", "MXFP4"), ("awq", "AWQ"), ("gptq", "GPTQ"),
        ("autoround", "AutoRound"), ("fp8", "FP8"), ("int4", "INT4"), ("int8", "INT8"),
        ("bnb", "bitsandbytes"), ("gguf", "GGUF"), ("mlx", "MLX"),
    ):
        if key in haystack:
            return label
    return "BF16/FP16"


def fit_verdict(w_gib: float | None, total_gib: float | None, tags: list[str] | None = None) -> dict:
    tags = [t.lower() for t in (tags or [])]
    if UNSUPPORTED_TAGS & set(tags):
        return {"verdict": "unsupported", "headroom_gib": None,
                "reason": "Format not supported by vLLM (GGUF/MLX/ONNX)."}
    if w_gib is None:
        return {"verdict": "unknown", "headroom_gib": None, "reason": "Model size unknown."}
    if total_gib is None:
        return {"verdict": "unknown", "headroom_gib": None, "reason": "GPU not detected."}
    headroom = (total_gib - OS_RESERVE_GIB) - (w_gib + VLLM_OVERHEAD_GIB + MIN_KV_GIB)
    if headroom < 0:
        verdict, reason = "too_big", "Weights will not fit in VRAM."
    elif headroom < FIT_COMFORT_GIB:
        verdict, reason = "tight", "Fits, but only with a short context window."
    else:
        verdict, reason = "fits", "Fits with room for a long context."
    return {"verdict": verdict, "headroom_gib": round(headroom, 1), "reason": reason}


def choose_util(total_mib: int, used_mib: int, reserve_mib: int = 1536) -> float:
    """GPU memory fraction for vLLM, based on VRAM free *after* any old vLLM was stopped."""
    util = (total_mib - used_mib - reserve_mib) / total_mib
    if util >= 0.50:
        # WDDM reports idle VRAM as "used"; 0.90 starts reliably (0.93 hung)
        return 0.90
    util = min(0.90, int(util * 100) / 100)
    return util if util >= 0.3 else 0.85


def choose_context(w_gib: float | None, total_gib: float, util: float) -> int:
    """Largest standard context that a conservative KV-cache estimate supports."""
    if w_gib is None:
        return 8192
    kv_gib = util * total_gib - w_gib - VLLM_OVERHEAD_GIB
    if kv_gib <= 0:
        return CONTEXT_STEPS[0]
    per_token = KV_BYTES_PER_TOKEN_PER_GIB * max(w_gib, 1.0)
    tokens = kv_gib * GIB / per_token
    best = CONTEXT_STEPS[0]
    for step in CONTEXT_STEPS:
        if step <= tokens:
            best = step
    return best


def model_flags(model_id: str) -> list[str]:
    """Reasoning/tool-call parser flags for known model families (Hermes as the fallback)."""
    n = model_id.lower()
    if re.search(r"qwen3[._-]?[5-9]|qwen3\.", n):
        return ["--language-model-only", "--reasoning-parser", "qwen3",
                "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"]
    if "qwen3-coder" in n:
        return ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"]
    if "qwen3" in n:
        return ["--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    if re.search(r"devstral|mistral", n):
        return ["--tokenizer-mode", "mistral", "--config-format", "mistral", "--load-format", "mistral",
                "--enable-auto-tool-choice", "--tool-call-parser", "mistral"]
    if re.search(r"llama-?3", n):
        return ["--enable-auto-tool-choice", "--tool-call-parser", "llama3_json"]
    if "gpt-oss" in n:
        return ["--enable-auto-tool-choice", "--tool-call-parser", "openai"]
    if "deepseek-r1" in n:
        return ["--reasoning-parser", "deepseek_r1"]
    return ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]


def served_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def build_vllm_args(model_id: str, context: int, util: float, graphs: bool = False,
                    max_seqs: int = 2, extra: list[str] | None = None) -> list[str]:
    args = [
        "--model", model_id, "--served-model-name", served_name(model_id),
        "--max-model-len", str(context), "--kv-cache-dtype", "fp8",
        "--gpu-memory-utilization", f"{util:.2f}", "--max-num-seqs", str(max_seqs),
        *model_flags(model_id),
    ]
    if not graphs:
        args.append("--enforce-eager")
    return args + list(extra or [])


# Files vLLM does not need; skipped to avoid downloading duplicate weights.
_SKIP_SUFFIXES = (".gguf", ".onnx", ".msgpack", ".h5", ".ot", ".tflite", ".pth", ".pt")


def select_files(filenames: list[str]) -> list[str]:
    has_sharded = any(f.endswith(".safetensors") and not f.startswith("consolidated") and "/" not in f
                      for f in filenames)
    has_safetensors = any(f.endswith(".safetensors") for f in filenames)
    keep = []
    for f in filenames:
        low = f.lower()
        if low.startswith(("original/", "onnx/", "openvino/", ".git")) or low.endswith(_SKIP_SUFFIXES):
            continue
        if low.endswith(".bin") and has_safetensors:
            continue
        if low.startswith("consolidated") and low.endswith(".safetensors") and has_sharded:
            continue
        keep.append(f)
    return keep
