---
name: contributing
description: How to make and ship any change to vLLM Hub - layout, local validation without host Python, rules CI enforces, and the PR flow. Use before starting any code, workflow or docs change in this repo.
---

# Contributing to vLLM Hub

## Layout

- `src/app/`: FastAPI app.
  - `main.py`: routes.
  - `runner.py`: the sibling `vllm` container.
  - `downloads.py`: Hugging Face pulls.
  - `profiles.py`, `capacity.py`: sizing.
  - `updates.py`, `updater.py`: self-update.
- `src/app/static/index.html`: the entire UI (vanilla JS, no build step).
- `src/tests/`: the pytest suite, `check_ui.mjs`, and `ui/` (Playwright screenshots).
- The root holds only `.github/`, `.devcontainer/`, `.agents/`, `compose.yml`, the README and dotfiles. Put code in `src/`.

## Validate locally (from `src/`)

Use the dev container, or Docker if Python or Node aren't installed:

```bash
ruff check .
pytest -q
node tests/check_ui.mjs
docker run --rm -v "$PWD/..:/repo" -w /repo rhysd/actionlint:1.7.7   # when workflows change
```

PowerShell without the dev container:

```powershell
docker run --rm -v "${PWD}:/w" -w /w python:3.12-slim sh -c "pip install -q -r requirements-dev.txt && pytest -q && ruff check ."
docker run --rm -v "${PWD}:/w" -w /w node:22-slim node tests/check_ui.mjs
```

## Rules CI enforces

- The **CI passed** check must be green to merge to `main`. The branch is protected; no direct pushes.
- Every API route change needs a test in `src/tests/test_api.py`.
  - Tests must not need Docker, a GPU or the network.
  - Fake `runner.client`, `main.gpu_info` and `downloads.*` with `monkeypatch`.
- Every `data-act` in the UI needs a `case` in the click handler, and every `/api/...` the UI calls must exist.
- No image files anywhere in the tree (see the `ui-changes` skill).
- Workflows must pass actionlint, which also runs shellcheck on `run:` blocks. Shell scripts must use LF line endings (enforced by `.gitattributes`).

## Shipping

1. Branch from `main` (`feat/...`, `fix/...`).
2. Commit. Update `README.md` when behavior, configuration or the API changes, and `.github/copilot-instructions.md` or these skills when the rules change.
3. Run `gh pr create --fill --base main`, then `gh pr merge <n> --squash --delete-branch --auto`.
4. Merging publishes `ghcr.io/mitch-b/vllm-hub:latest` plus a US Central timestamp tag (`YYYYMMDD-HHMM`).
   - Hubs with `HUB_AUTO_UPDATE=true` install it automatically, so a broken `main` reaches users quickly.
   - Keep `/healthz` working.

## PR descriptions

Fill `.github/pull_request_template.md` (`gh pr create --fill` uses the commit message; edit with `gh pr edit --body-file`). Keep it short; a reviewer should read it in under a minute.

- **Why**: the problem or goal in 1-2 sentences, plus `Fixes #N` if applicable.
- **What changed**: 1-3 bullets of intent or design that the diff doesn't make obvious. No file-by-file lists.
- **Verified**: the exact commands you ran and what you saw (for example `pytest -q` shows 51 passed, or `curl /api/version` returns `auto_update: true`). Unit tests alone don't prove a runtime change works; exercise it.
- **Review focus** (optional): the risky part, a tradeoff, or anything you didn't test. Be honest about gaps.

Don't add checklists for things CI already enforces, emoji headers, or generated summaries of every file.

## Safety

- The hub holds the Docker socket and has no auth. It binds to `127.0.0.1` unless the user sets `HUB_BIND`. Don't widen defaults.
- Never commit `.env`, tokens, or personal paths and hostnames, including in test fixtures.
