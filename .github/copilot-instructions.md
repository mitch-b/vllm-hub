# vLLM Hub — agent instructions

Task-specific playbooks live in `.agents/skills/`: `contributing` (always), `ui-changes`, `model-sizing`, `self-update-and-release`. Read the ones that match your change before starting.

FastAPI app (`src/app`) with a single-file vanilla JS UI (`src/app/static/index.html`). It automatically selects a sibling CUDA `vllm` container or Docker Model Runner's host-native engine. `compose.yml` runs the hub without GPU requests; only the CUDA server and short-lived NVIDIA probes request GPUs.

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
- Every API route change needs a test in `src/tests/test_api.py`. Tests must not need Docker, a GPU or the network; fake `runner.client`, `main.gpu_info`, `hosting`, `model_runner` and `downloads.*` with `monkeypatch`. Native lifecycle/protocol tests use HTTP fakes in `src/tests/test_model_runner.py`.
- Every `data-act="..."` in the UI needs a matching `case` in the click handler; every `/api/...` the UI calls must exist. Tests enforce both.
- CUDA sizing logic lives in `src/app/profiles.py` and is covered by `src/tests/test_profiles.py`. On Windows/WDDM, `nvidia-smi` used-VRAM is unreliable; GPU memory utilization defaults to 0.90 and 0.93 is known to hang a 24 GB card. Native engines use metadata-based context defaults and no CUDA flags or GPU-fit claims.
- `src/app/capacity.py` learns each model's KV-cache capacity from vLLM's "GPU KV cache size" log line (keyed by model and util) into `/state/capacity.json`; Auto context prefers it over the `profiles.choose_context` estimate. Copilot token limits in the UI must satisfy prompt + output = context.
- `src/app/hosting.py` selects backends from server capabilities, not the browser OS. Honor `hosting.features` in the UI. Native selection/context/digest persist in `/state/model-runner.json`; Stop unloads only the selected model, never all of Model Runner.
- The hub binds to `127.0.0.1` by default (it holds the Docker socket and has no auth); LAN access is opt-in through `HUB_BIND`.
- UI changes (`src/app/static/`) get screenshots on the PR automatically. If you add or change a visible surface, add or update a scene in `src/tests/ui/screenshots.py` and its data in `src/tests/ui/fixtures/` (fake data only: no secrets, personal paths or hostnames). Preview locally with the command in the README's "UI screenshots" section. Never commit image files; CI rejects them. Images live on the `ui-screenshots` branch, which the workflows manage.
- Self-update (`src/app/updates.py`, `src/app/updater.py`) runs the **new** image's `app.updater` in a helper container, so keep `python -m app.updater <container-id> <image> [old-image-config-json]` backward compatible. Keep the `/healthz` healthcheck fast; the swap and rollback depend on it. Test with fakes in `src/tests/test_updates.py`.
- Update `README.md` when behavior, configuration or the API changes.
