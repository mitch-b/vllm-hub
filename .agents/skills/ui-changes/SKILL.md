---
name: ui-changes
description: Rules for changing the vLLM Hub web UI (src/app/static/index.html) - required PR screenshots via Playwright, adding scenes and fixtures, and keeping images off main. Use whenever a change touches the UI or src/tests/ui.
---

# UI changes and screenshots

Any PR touching `src/app/static/**` or `src/tests/ui/**` runs the **UI screenshots** CI job. It must pass for **CI passed** to go green. A follow-up workflow then posts the images as one sticky PR comment and updates it on every push. You don't upload anything yourself.

## How it works

- `src/tests/ui/screenshots.py` serves `index.html` at `http://hub.test/` and answers every `/api/*` call from `src/tests/ui/fixtures/`.
  - No GPU, Docker or network is needed, and the output is deterministic.
  - Each scene is a function `scene(browser) -> page` listed in `SCENES`.
  - The output file name is the scene name; `run_dialog` saves as `run.png`.
- Images live on the orphan `ui-screenshots` branch. Never commit to it by hand.
  - `main/` holds the README images; `pr-N/<sha7>/` holds the PR comment images.
  - The branch is always a single force-pushed commit.
  - A weekly job deletes screenshots of PRs closed more than 7 days ago.
- `main` must not contain `*.png`, `*.jpg`, `*.jpeg`, `*.gif` or `*.webp` files. CI fails if it does.

## When you change the UI

1. Make the change in `index.html`. Keep `data-act` attributes and click-handler cases in sync, and check with `node tests/check_ui.mjs`.
2. If you add or significantly change a visible surface (panel, dialog, tab or state), add or extend a scene:
   - Start from `open_page(b)` or an existing scene such as `overview(b)`.
   - Interact through stable selectors (`#id`, `[data-act=...]:visible`).
   - Call `wait_for_selector` for the final state instead of using fixed sleeps.
   - Add the scene to `SCENES`.
   - For per-scene API data, pass `open_page(b, {"<api-key>": data})`. The key is the first path segment after `/api/`.
3. If the UI calls a new endpoint, add `fixtures/<key>.json` and add `<key>` to the names `open_page` loads.
   - Fixture data must be fake or sanitized: no tokens, usernames, personal paths or hostnames.
   - Timestamps are shifted relative to `CAPTURED`, so store absolute epochs from that time.
4. Preview locally from `src/`, then look at the PNGs in `.shots/` (gitignored). Add `--scene <name>` to render one scene.

   ```powershell
   docker run --rm -v "${PWD}:/w" -w /w mcr.microsoft.com/playwright/python:v1.49.0-noble sh -c "pip install -q -r tests/ui/requirements.txt && python tests/ui/screenshots.py --out .shots"
   ```

5. If a new scene belongs in the README, reference `https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/<scene>.png`. The image appears after the PR merges and the `main` publish runs.

## Gotchas

- `playwright==` in `tests/ui/requirements.txt` must match the container tag (`v1.49.0-noble`) in `ci.yml` and `ui-screenshots.yml`. Bump all three together.
- Don't use `page.clock`. It freezes `performance.now`, and the playground then shows "Infinity tok/s".
- Hidden duplicate buttons exist (Discover and Library), so add `:visible` to `data-act` selectors.
- Fork PRs get screenshots too. The privileged workflow only copies validated PNGs from the artifact and never runs PR code.
