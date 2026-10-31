# vLLM Hub — agent instructions

FastAPI app (`src/app`) with a single-file vanilla JS UI (`src/app/static/index.html`). It manages a sibling `vllm` container through the Docker socket. `compose.yml` at the repo root runs it.

## Before opening a PR

Run from `src/` (the dev container has everything; otherwise use `python:3.12-slim`):

```bash
ruff check .
pytest -q
node tests/check_ui.mjs
```

The PR check **CI passed** must be green to merge. It runs the above, actionlint, hadolint, `docker compose config`, an image build and a smoke test of the API. Merges to `main` re-run CI, then publish `ghcr.io/mitch-b/vllm-hub` (`latest`, a US Central build-time tag like `20261031-1042`, `sha-*`, and semver for `v*.*.*`).

## Rules

- Keep the root clean: code, tests and the Dockerfile live in `src/`.
- Every API route change needs a test in `src/tests/test_api.py`. Tests must not need Docker, a GPU or the network; fake `runner.client`, `main.gpu_info` and `downloads.*` with `monkeypatch`.
- Every `data-act="..."` in the UI needs a matching `case` in the click handler; every `/api/...` the UI calls must exist. Tests enforce both.
- Sizing logic lives in `src/app/profiles.py` and is covered by `src/tests/test_profiles.py`. On Windows/WDDM, `nvidia-smi` used-VRAM is unreliable; GPU memory utilization defaults to 0.90 and 0.93 is known to hang a 24 GB card.
- `src/app/capacity.py` learns each model's KV-cache capacity from vLLM's "GPU KV cache size" log line (keyed by model and util) into `/state/capacity.json`; Auto context prefers it over the `profiles.choose_context` estimate. Copilot token limits in the UI must satisfy prompt + output = context.
- The hub must only bind to `127.0.0.1` (it holds the Docker socket).
- Self-update (`src/app/updates.py`, `src/app/updater.py`) runs the **new** image's `app.updater` in a helper container, so keep `python -m app.updater <container-id> <image> [old-image-config-json]` backward compatible. Keep the `/healthz` healthcheck fast; the swap and rollback depend on it. Test with fakes in `src/tests/test_updates.py`.
- Update `README.md` when behavior, configuration or the API changes.
