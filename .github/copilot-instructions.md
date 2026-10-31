# vLLM Hub — agent instructions

FastAPI app (`src/app`) with a single-file vanilla JS UI (`src/app/static/index.html`). It manages a sibling `vllm` container through the Docker socket. `compose.yml` at the repo root runs it.

## Before opening a PR

Run from `src/` (the dev container has everything; otherwise use `python:3.12-slim`):

```bash
ruff check .
pytest -q
node tests/check_ui.mjs
```

The PR check **CI passed** must be green to merge. It runs the above, actionlint, hadolint, `docker compose config`, an image build and a smoke test of the API. Merges to `main` re-run CI, then publish `ghcr.io/mitch-b/vllm-hub` (`latest`, `sha-*`, and semver tags for `v*.*.*`).

## Rules

- Keep the root clean: code, tests and the Dockerfile live in `src/`.
- Every API route change needs a test in `src/tests/test_api.py`. Tests must not need Docker, a GPU or the network; fake `runner.client`, `main.gpu_info` and `downloads.*` with `monkeypatch`.
- Every `data-act="..."` in the UI needs a matching `case` in the click handler; every `/api/...` the UI calls must exist. Tests enforce both.
- Sizing logic lives in `src/app/profiles.py` and is covered by `src/tests/test_profiles.py`. On Windows/WDDM, `nvidia-smi` used-VRAM is unreliable; GPU memory utilization defaults to 0.90 and 0.93 is known to hang a 24 GB card.
- The hub must only bind to `127.0.0.1` (it holds the Docker socket).
- Update `README.md` when behavior, configuration or the API changes.
