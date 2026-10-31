---
name: model-sizing
description: How vLLM Hub picks GPU memory, context length and Copilot token limits (profiles.py, capacity.py, tokenLimits in the UI). Use before changing recommendations, run flags, context defaults or Connect snippets.
---

# Model sizing

The fit ratings, utilization flags and learned KV-cache rules below apply to
the CUDA vLLM backend. `src/app/model_runner.py` handles native engines
separately: Auto context defaults to 8K, capped by reported model metadata.
Missing maximum-context metadata remains unknown, not an invented 8K limit.
Do not apply CUDA flags or VRAM-fit claims to native models. Both backends
must preserve the Connect prompt + output = context invariant.

- `src/app/profiles.py` holds the curated models and is covered by `src/tests/test_profiles.py`. It contains:
  - per-family flags (tool-call and reasoning parsers, KV dtype),
  - the fit ratings,
  - `choose_context`, a conservative context estimate.
- `src/app/capacity.py` parses vLLM's "GPU KV cache size: N tokens" log line and stores the value per model and GPU utilization in `/state/capacity.json`. It is covered by `src/tests/test_capacity.py`.
  - `fit_context` is N × 0.97, floored to a multiple of 1024, capped at the model's max, with a minimum of 2048.
  - **Auto** context prefers this learned value over the estimate.
- GPU memory utilization defaults to 0.90. On Windows/WDDM, 0.93 is known to hang a 24 GB card. `nvidia-smi` used-VRAM is unreliable there, so don't size from it.
- vLLM rejects requests where prompt + max_tokens > max_model_len. The UI's `tokenLimits(ctx)` sets:
  - output = ctx/4 floored to a multiple of 1024, clamped to 2048..32768,
  - prompt = ctx − output.
  - Keep prompt + output = context.
- Connect warns below 48K context because Copilot agent prompts are large.
- An unlearned model can get a small (8K) Auto context on its first run. That run learns capacity, and the next Auto run uses it.
