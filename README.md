# vLLM Hub

A local web UI for running Hugging Face models on [vLLM](https://docs.vllm.ai) with your NVIDIA GPU. Pick a model, watch it download, and the hub starts vLLM with settings chosen for your GPU. It exposes an OpenAI-compatible endpoint at `http://localhost:8000/v1`.

## Features

- **GPU-aware recommendations**: a curated list of coding models, each rated `fits`, `tight` or `too big` against your VRAM.
- **Search Hugging Face** or paste any `org/model`.
- **Pull progress**: bytes, speed and ETA, with cancel. Already-cached files are skipped.
- **Run / Stop** with automatic flags: GPU memory use, context length, KV-cache dtype, tool-call and reasoning parsers per model family. You can override the context and enable CUDA graphs.
- **Local models**: list and delete what's cached.
- **Live vLLM logs** and a quick test chat.

## Quick start

Requirements: Docker with the NVIDIA container runtime (Docker Desktop + WSL2 on Windows).

```powershell
cp .env.example .env     # optional
docker compose up -d --build
```

Open <http://localhost:7860>. To reuse an existing Hugging Face cache, set in `.env`:

```
HF_CACHE_SOURCE=C:/Users/you/.cache/huggingface
```

Gated models need `HF_TOKEN` in `.env`.

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

Agentic use needs tool calling (enabled automatically) and a context of at least 16K; use 32K on a 24 GB card for large prompts.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | empty | Hugging Face token for gated models |
| `HF_CACHE_SOURCE` | `vllm-hub-hf-cache` | Named volume or host path for the model cache |
| `VLLM_IMAGE` | `vllm/vllm-openai:latest` | vLLM image to run |
| `VLLM_PORT` | `8000` | Host port for the vLLM API |

## API

`GET /api/system` · `GET /api/search?q=` · `GET /api/recommended` · `GET|DELETE /api/models` · `POST /api/pull` · `GET /api/jobs` · `POST /api/jobs/{id}/cancel` · `POST /api/run` · `POST /api/stop` · `GET /api/logs` · `POST /api/chat` · `GET /healthz`

## Notes

- **Security**: the hub mounts the Docker socket, which is root-equivalent on the host. It binds to `127.0.0.1` only; don't expose it.
- **Windows VRAM**: WDDM evicts idle VRAM, so `nvidia-smi` can show little used while a model is loaded. The hub uses 0.90 GPU memory utilization by default; 0.93 was observed to hang on a 4090.
- **Eager mode**: vLLM runs with `--enforce-eager` unless you tick CUDA graphs, trading speed for lower VRAM and faster startup.
- The vLLM container is named `vllm` and joins the `vllm-hub-net` network. Only one model runs at a time.

## Development

```
.devcontainer/  Python 3.12 dev container
.github/        CI, CD, Dependabot
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

Agent guidance lives in `.github/copilot-instructions.md`.
