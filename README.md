# vLLM Hub

A local web UI for running Hugging Face models on [vLLM](https://docs.vllm.ai) with your NVIDIA GPU. Pick a model, watch it download, and the hub starts vLLM with settings chosen for your GPU. It exposes an OpenAI-compatible endpoint at `http://localhost:8000/v1`.

## Features

- **GPU-aware recommendations**: a curated list of coding models, each rated `fits`, `tight` or `too big` against your VRAM.
- **Search Hugging Face** or paste any `org/model`.
- **Pull progress**: reconstructed file bytes, speed and ETA, with cancel. Network transfer bytes are not counted again as file progress. **Finalizing snapshot** means all counted bytes are present but Hugging Face is still finishing the snapshot; the model becomes available only after the download returns and its selected files are checked. The Library refreshes on completion even after a reload or missed poll. Each model appears once in the Library, including while downloading. Run another cached model while a download continues in the background; **Download & run** still switches to the new model when it finishes. Already-cached files are skipped.
- **Run / Stop** with automatic flags: GPU memory use, context length, KV-cache dtype, tool-call and reasoning parsers per model family. You can override the context and enable CUDA graphs.
- **Local models**: list and delete what's cached.
- **Live vLLM logs** and a quick test chat.

## Screenshots

| Overview | Connect Copilot |
| --- | --- |
| [![Overview: running model, GPU and library](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/overview.png)](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/overview.png) | [![Copy-paste commands for Copilot CLI, app and SDK](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/connect.png)](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/connect.png) |
| **Discover models** | **Run a model** |
| [![Search Hugging Face and recommendations rated for your GPU](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/discover.png)](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/discover.png) | [![Run dialog with context and GPU memory settings](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/run.png)](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/run.png) |
| **Playground** | **Console** |
| [![Quick test chat with tokens per second](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/playground.png)](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/playground.png) | [![Live vLLM logs](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/console.png)](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/console.png) |

These images are rendered from mocked data by CI and refresh automatically when UI changes merge.

## Quick start

Requirements: Docker with the NVIDIA container runtime (Docker Desktop + WSL2 on Windows).

No clone needed. Grab the compose file and start it; it pulls the published image `ghcr.io/mitch-b/vllm-hub`:

```powershell
mkdir vllm-hub; cd vllm-hub
curl.exe -fsSLO https://raw.githubusercontent.com/mitch-b/vllm-hub/main/compose.yml
docker compose up -d
```

From a clone, `docker compose up -d` does the same; add `--build` to run your local source instead.

Open <http://localhost:7860>. To reuse an existing Hugging Face cache, create a `.env` next to `compose.yml` (see [.env.example](.env.example)):

```
HF_CACHE_SOURCE=C:/Users/you/.cache/huggingface
```

Gated models need `HF_TOKEN` in `.env`.

## Updates

The hub checks GitHub's registry every 30 minutes. When a newer image is published, the version pill in the top bar turns into **Update available**. Click it to see what changed, then **Update now**:

1. The new image downloads in the background while the hub keeps working.
2. A short-lived `vllm-hub-updater` container creates the new hub ahead of time, then swaps it in. The UI is unreachable for about a second and reloads itself.
3. If the new version isn't healthy within two minutes, the previous one is restarted.

The model server is a separate container and keeps serving throughout, so Copilot sessions aren't interrupted. Updates wait until downloads finish.

Prefer the command line? `docker compose pull && docker compose up -d`. Set `HUB_UPDATE_CHECK=false` to turn off the check, or `HUB_AUTO_UPDATE=true` in `.env` to install new versions automatically (checked every 30 minutes; it waits for downloads to finish).

Images are tagged with their build time in US Central (`20261031-1042` = Oct 31 2026, 10:42 AM), plus `latest`, `sha-<commit>`, and semver for `v*.*.*` git tags. Pin a build with `image: ghcr.io/mitch-b/vllm-hub:20261031-1042` in a compose override; pinned hubs don't self-update.

## Use it from GitHub Copilot

Open **Connect** in the UI while a model is serving. It generates ready-to-paste commands with the right model ID and token limits.

### Copilot CLI

```powershell
$env:COPILOT_PROVIDER_BASE_URL = "http://localhost:8000/v1"
$env:COPILOT_MODEL = "qwen3.6-27b-awq-int4"
$env:COPILOT_PROVIDER_MAX_PROMPT_TOKENS = "24576"   # context minus output
$env:COPILOT_PROVIDER_MAX_OUTPUT_TOKENS = "8192"
copilot --autopilot --yolo -i "your task"   # drop --autopilot/--yolo for a normal session
```

To keep it permanently, add a provider to `~/.copilot/providers.json` (the UI's "Save as provider" tab generates it) and run `copilot --model vllm-hub/<model>`.

### Copilot app

Settings → **Model providers** → **Add provider** → **OpenAI-compatible**. Base URL `http://localhost:8000/v1`, model = the served name, no API key. Then pick it from the model picker. (BYOK in the app is in public preview.)

### Copilot SDK (BYOK)

```ts
provider: { type: "openai", baseUrl: "http://localhost:8000/v1", apiKey: "none" }
// model: the served name shown in the UI, e.g. "qwen3.6-27b-awq-int4"
```

Agentic use needs tool calling (enabled automatically) and a large context, because Copilot's instructions and tool definitions use a big share of the prompt. The Connect tab sets output to a quarter of the context (2K–32K) and the prompt to the rest, so requests never overflow it, and warns below 48K.

### Context sizing

The first time a model runs, the hub reads vLLM's measured KV-cache size from its logs and saves it in the `vllm-hub-state` volume. From then on, **Auto** context uses about 97% of that measured capacity (capped at the model's own limit) instead of a conservative estimate. Run a model, then stop and run it again with Auto to use it. For example, a 27B AWQ model on a 24 GB card goes from 32K to about 56K. If you need more context, choose a smaller model: less VRAM for weights means more for context.

A running model restarts with Docker (`unless-stopped`); **Stop** or **Cancel** removes it
and cancels any pending startup so it stays off until you choose **Run** again.
Stopping also clears queued **Download & run** actions without stopping the downloads.
The Console reconnects after a dropped stream or replacement container and reports
connection errors instead of leaving an empty panel.

### Startup feedback

While a model starts, the timer uses Docker's actual start time, so refreshing the page
or restarting the hub does not reset it. A new model container or Docker restart starts
a new timer. Before Docker starts the container, the UI says **Waiting for container**.

Startup stages (weight loading, warm-up, compilation and CUDA graph capture) and
shard/graph counts are taken from recent vLLM logs when available, not an estimated
overall percentage. Last-log activity shows when the server last emitted output;
a quiet period is not proof of a hang. Log formats vary by vLLM version, so unrecognized
output keeps the generic stage and the **Console** remains available for details.
`/api/system` includes a `vllm.startup` object while starting, with the container ID,
start time, elapsed seconds, stage, optional counts, last-log timestamp and any log-read
error; it is `null` once ready or exited.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | empty | Hugging Face token for gated models |
| `HF_CACHE_SOURCE` | `vllm-hub-hf-cache` | Named volume or host path for the model cache |
| `VLLM_IMAGE` | `vllm/vllm-openai:latest` | vLLM image to run |
| `VLLM_PORT` | `8000` | Host port for the vLLM API |
| `HUB_UPDATE_CHECK` | `true` | Check the registry for a newer hub image |
| `HUB_AUTO_UPDATE` | `false` | Install newer hub images automatically |
| `HUB_BIND` | `127.0.0.1` | UI listen address; `0.0.0.0` for LAN access (no login, so trusted networks only) |

## API

`GET /api/system` · `GET /api/search?q=` · `GET /api/recommended` · `GET|DELETE /api/models` · `POST /api/pull` · `GET /api/jobs` · `POST /api/jobs/{id}/cancel` · `POST /api/run` · `POST /api/stop` · `GET /api/logs` · `POST /api/chat` · `GET /api/version?refresh=1` · `POST /api/update` · `GET /healthz`

## Notes

- **Security**: the hub mounts the Docker socket, which is root-equivalent on the host. It binds to `127.0.0.1` only; don't expose it.
- **Windows VRAM**: WDDM evicts idle VRAM, so `nvidia-smi` can show little used while a model is loaded. The hub uses 0.90 GPU memory utilization by default; 0.93 was observed to hang on a 4090.
- **Eager mode**: vLLM runs with `--enforce-eager` unless you tick CUDA graphs, trading speed for lower VRAM and faster startup.
- The vLLM container is named `vllm` and joins the `vllm-hub-net` network. Only one model runs at a time.

## Development

```
.agents/skills/ playbooks for coding agents
.devcontainer/  Python 3.12 dev container
.github/        CI, CD, UI screenshots, Dependabot
compose.yml     runs the hub
src/            app code, tests, Dockerfile
```

### Dev container (no local Python needed)

Open the folder in VS Code and choose **Reopen in Container**. It installs the dependencies, enables pytest in the Testing view, and talks to the host's Docker. It also gets the GPU when one is available and shares the `vllm-hub-hf-cache` volume and `vllm-hub-net` network with the compose stack.

```bash
cd src
pytest -q
uvicorn app.main:app --reload --port 7860   # stop the compose hub first, or use another port
```

### Without the dev container

```powershell
docker run --rm -v "${PWD}/src:/w" -w /w python:3.12-slim sh -c "pip install -q -r requirements-dev.txt && pytest -q"
```

### CI/CD

- **CI** (every PR, about 1 minute): two parallel jobs. One runs ruff, pytest (unit and API tests with Docker and the GPU faked), a UI script syntax check and actionlint. The other runs hadolint and `docker compose config`, then builds the image with layer caching and smoke-tests its API. The **CI passed** check gates merges to `main`.
- **CD** (merge to `main`, or a `v*.*.*` tag): re-runs CI, then publishes a multi-arch `ghcr.io/mitch-b/vllm-hub` image (`latest`, `sha-*`, semver) with a build provenance attestation.

### UI screenshots

PRs that touch `src/app/static/` or `src/tests/ui/` must pass a **UI screenshots** CI job. It renders every scene in `src/tests/ui/screenshots.py` with Playwright against fixed API fixtures (`src/tests/ui/fixtures/`), so screenshots are identical across machines and need no GPU. A follow-up workflow posts them as a single comment on the PR that's updated on every push.

Images never land on `main`; CI fails if an image file is committed. They live on the `ui-screenshots` branch, which always holds exactly one force-pushed commit: `main/` for this README and `pr-N/<sha>/` for PRs. A weekly job deletes screenshots of PRs closed for more than 7 days, so the branch stays small and its history never grows.

To preview locally (from `src/`):

```powershell
docker run --rm -v "${PWD}:/w" -w /w mcr.microsoft.com/playwright/python:v1.49.0-noble sh -c "pip install -q -r tests/ui/requirements.txt && python tests/ui/screenshots.py --out .shots"
```

When you add a UI surface, add a scene (and fixture data if needed) so reviewers can see it.

Agent guidance lives in `.github/copilot-instructions.md`.
