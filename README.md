# vLLM Hub

A local web UI for running models on Windows, Linux, or macOS. Pick a model and
the hub downloads and serves it with an OpenAI-compatible API. NVIDIA hosts use
[vLLM](https://docs.vllm.ai); other hosts automatically use
[Docker Model Runner](https://docs.docker.com/ai/model-runner/), including native
Metal acceleration on Apple Silicon and supported CPU/GPU backends on Linux.
There is one Compose file and no platform-specific setup flag.

## Features

- **Automatic backend and catalog**: GPU-rated CUDA coding models on NVIDIA hosts; portable GGUF recommendations on Model Runner hosts, without misleading VRAM-fit claims.
- **Search Hugging Face** or paste a model reference.
- **Pull progress**: file bytes, speed and ETA, with cancel. The Library refreshes on completion even after a reload or missed poll. Each model appears once in the Library, including while downloading. Run another cached model while a download continues in the background; **Download & run** still switches to the new model when it finishes. Already-cached files are skipped. Hugging Face snapshots verify selected files before becoming available; Model Runner downloads verify that the model was registered.
- **Run / Stop** with automatic context defaults and backend-specific configuration. CUDA vLLM also sets GPU memory use, KV-cache dtype, tool-call and reasoning parsers per model family; you can enable CUDA graphs.
- **Local models**: list and delete what's cached.
- **Quick test chat** on either backend; live vLLM logs on CUDA.
- **Activity** on CUDA: live tokens/s, active requests, KV-cache use, callers and recent requests while a model is serving.

For Hugging Face snapshots, progress counts reconstructed file bytes rather
than counting network transfers again. **Finalizing snapshot** means the bytes
are present but Hugging Face is still finishing and verifying the snapshot;
the model is available only after that completes.

## Screenshots

| Overview | Connect Copilot |
| --- | --- |
| [![Overview: running model, GPU and library](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/overview.png)](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/overview.png) | [![Copy-paste commands for Copilot CLI, app and SDK](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/connect.png)](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/connect.png) |
| **Discover models** | **Run a model** |
| [![Search Hugging Face and recommendations rated for your GPU](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/discover.png)](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/discover.png) | [![Run dialog with context and GPU memory settings](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/run.png)](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/run.png) |
| **Playground** | **Console** |
| [![Quick test chat with tokens per second](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/playground.png)](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/playground.png) | [![Live vLLM logs](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/console.png)](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/console.png) |
| **Activity** | |
| [![Live throughput, callers and recent requests](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/activity.png)](https://raw.githubusercontent.com/mitch-b/vllm-hub/ui-screenshots/main/activity.png) | |

These images are rendered from mocked data by CI and refresh automatically when UI changes merge.

## Quick start

Requirements: Docker. NVIDIA inference needs the NVIDIA container runtime
(Docker Desktop + WSL2 on Windows). On Apple Silicon, enable Docker Model Runner
once in Docker Desktop **Settings > AI**, including **host-side TCP support** for
Copilot access. The equivalent command on Docker Desktop is:

```bash
docker desktop enable model-runner --tcp=12434 --cors=none
```

On Linux without an NVIDIA runtime, install Docker's `docker-model-plugin`
package and run `docker model install-runner` once.

No clone needed. Grab the compose file and start it; it pulls the published image `ghcr.io/mitch-b/vllm-hub`:

**macOS / Linux**

```bash
mkdir -p vllm-hub && cd vllm-hub
curl -fsSLO https://raw.githubusercontent.com/mitch-b/vllm-hub/main/compose.yml
docker compose up -d
```

**Windows (PowerShell)**

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

### Automatic platform support

Run the same `docker compose up -d` on every host. The hub no longer requests
an NVIDIA device at startup, so a Mac does not fail with
`failed to discover GPU vendor from CDI: no known GPU vendor found`.

| Available capability | Automatically selected backend | Model cache | Copilot base URL |
|---|---|---|---|
| NVIDIA GPU + container runtime (Linux / Windows WSL2) | CUDA vLLM | Hub's Hugging Face cache | `http://localhost:8000/v1` |
| Docker Model Runner (Apple Silicon / supported Windows and Linux hosts) | Model Runner's native engine | Model Runner's shared model store | `http://localhost:12434/engines/v1` |
| Neither runtime ready | Setup guidance; browsing/downloads remain available | Hub cache until a runtime is ready | Not serving yet |

NVIDIA detection uses a short-lived, network-isolated probe from the already
running hub image. Only the probe and the CUDA model server request GPUs.
Metal hosts skip the NVIDIA probe entirely. Probes are time-bounded and removed
even when Docker refuses GPU access.
Existing NVIDIA installations keep their flags, learned context sizing, and
endpoint. An active native model stays on its backend until stopped.

On a Mac, Model Runner handles Metal access **on the macOS host**, not inside a
Linux container. Its GGUF engine works out of the box on Apple Silicon. Optional
MLX/vLLM Metal models need the corresponding backend installed in Model Runner
(`docker model install-runner --backend vllm`, Docker Desktop 4.62+). The hub
does not install Homebrew packages or launch arbitrary host commands.

The UI chooses the matching model catalog and cache automatically, offers
**Download & run**, **Run**, **Stop**, chat and Connect for both backends, and
hides CUDA-only options, GPU-fit claims, Console and Activity on native engines.
Native Auto context uses a conservative 8K default, capped by model metadata;
it does not reuse the CUDA KV-cache estimate or memory utilization flags.
When metadata omits the maximum context, it is reported as unknown rather than
inventing an 8K hard limit; manual context choices are validated by the native engine.
Model size is shown only when the backend reports it. Model compatibility and
tool calling still depend on the selected model and engine.

Native Discover starts with six selected Hugging Face GGUF models for general
chat and coding, from Qwen and Unsloth. Each shows its use case, weight size,
and whether the repository belongs to an official publisher or an established
converter (not the original author). Search prioritizes those publisher groups,
labels community uploads, and includes only repositories listing Q4_K_M files.
Community uploads are hidden by default; **Include community uploads** reveals
them without presenting them as official or endorsed.
Weights are not total memory requirements; leave room for macOS and context.
An unavailable recommendation is shown with an explicit verification error,
without hiding the other picks. Paste `hf.co/publisher/model:Q4_K_M` to import
a specific repository; newer architectures may require an updated native engine.

If a runtime is missing, the hub explains the one-time setup and reconnects
when it becomes available. It never changes a running model for an unsupported
start. Native Stop unloads only the model selected in this hub, not other Model
Runner users' models. The selected native model and context survive hub updates;
an interrupted native startup is recovered from Model Runner's actual status.
Model Runner is shared: changing context or stopping a selected model also
affects other clients using that same model.

## Updates

The hub checks GitHub's registry every 30 minutes. When a newer image is published, the version pill in the top bar turns into **Update available**. Click it to see what changed, then **Update now**:

1. The new image downloads in the background while the hub keeps working.
2. A short-lived `vllm-hub-updater` container creates the new hub ahead of time, then swaps it in. The UI is unreachable for about a second and reloads itself.
3. If the new version isn't healthy within two minutes, the previous one is restarted.

Inference runs independently of the hub (a CUDA vLLM container or Model Runner's native engine) and keeps serving throughout, so Copilot sessions aren't interrupted. Updates wait until downloads finish.

Prefer the command line? `docker compose pull && docker compose up -d`. Set `HUB_UPDATE_CHECK=false` to turn off the check, or `HUB_AUTO_UPDATE=true` in `.env` to install new versions automatically (checked every 30 minutes; it waits for downloads to finish).

Images are tagged with their build time in US Central (`20261031-1042` = Oct 31 2026, 10:42 AM), plus `latest`, `sha-<commit>`, and semver for `v*.*.*` git tags. Pin a build with `image: ghcr.io/mitch-b/vllm-hub:20261031-1042` in a compose override; pinned hubs don't self-update.

## Use it from GitHub Copilot

Open **Connect** in the UI while a model is serving. It generates ready-to-paste commands with the right model ID and token limits.
The CLI shell defaults to your client platform (bash/zsh on macOS/Linux,
PowerShell on Windows), with a switch for other shells. The examples below use
CUDA vLLM; native clients use the Model Runner URL and model shown in Connect.

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

Agentic use needs a tool-calling model and a large context, because Copilot's instructions and tool definitions use a big share of the prompt. CUDA parser flags are set automatically; native tool calling depends on the model and engine. The Connect tab sets output to a quarter of the context (2K–32K) and the prompt to the rest, so requests never overflow it, and warns below 48K.

### Context sizing

The measured KV-cache sizing below applies to CUDA vLLM. Native engines use
the conservative metadata-based context described under Automatic platform support.

The first time a model runs, the hub reads vLLM's measured KV-cache size from its logs and saves it in the `vllm-hub-state` volume. From then on, **Auto** context uses about 97% of that measured capacity (capped at the model's own limit) instead of a conservative estimate. Run a model, then stop and run it again with Auto to use it. For example, a 27B AWQ model on a 24 GB card goes from 32K to about 56K. If you need more context, choose a smaller model: less VRAM for weights means more for context.

A running model restarts with Docker (`unless-stopped`); **Stop** or **Cancel** removes it
and cancels any pending startup so it stays off until you choose **Run** again.
Stopping also clears queued **Download & run** actions without stopping the downloads.
The Console reconnects after a dropped stream or replacement container and reports
connection errors instead of leaving an empty panel.

### Startup feedback

The detailed stages below apply to CUDA vLLM. Native engines show a persisted
startup timer and become ready only when Model Runner confirms the model is loaded.

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

### Activity

Activity is available only on CUDA vLLM. Native engine diagnostics remain in Docker Desktop.

Open **Activity** while a model is serving to see who is using it: generation and prompt tokens/s, active and queued requests and KV-cache use (with 15-minute sparklines), totals such as average time to first token and prefix-cache hit rate, a per-caller table and the most recent requests.

The hub samples vLLM's Prometheus endpoint (`/metrics`) and its HTTP access log every 5 seconds in the background and keeps the last 15 minutes in memory. Nothing is stored or sent anywhere, and no extra containers are needed. Health checks and the hub's own status polling are left out. Callers are identified by address: everything on this computer reaches vLLM through the published port and appears as **This computer**, the playground appears as **vLLM Hub**, and other machines show their IP. `GET /api/activity` returns the same data as JSON (`serving: false` when no model is ready).

For history beyond 15 minutes or alerting, point your own Prometheus (or Grafana Alloy) at `http://localhost:8000/metrics`; vLLM's metric names start with `vllm:`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | empty | Hugging Face token for gated models |
| `HF_CACHE_SOURCE` | `vllm-hub-hf-cache` | Named volume or host path for the model cache |
| `VLLM_IMAGE` | `vllm/vllm-openai:latest` | vLLM image to run |
| `VLLM_PORT` | `8000` | Host port for the vLLM API |
| `HUB_UPDATE_CHECK` | `true` | Check the registry for a newer hub image |
| `HUB_AUTO_UPDATE` | `false` | Install newer hub images automatically |
| `HUB_ACTIVITY_INTERVAL` | `5` | Seconds between Activity samples of vLLM's metrics and access log |
| `HUB_BIND` | `127.0.0.1` | UI listen address; `0.0.0.0` for LAN access (no login, so trusted networks only) |
| `HUB_MODEL_RUNNER_URL` | auto-discovered | Optional internal Model Runner management URL (Desktop socket proxy or host gateway detected automatically) |
| `HUB_MODEL_RUNNER_PUBLIC_URL` | `http://localhost:12434/engines/v1` | Client-facing native endpoint shown in Connect; change for custom host-side TCP ports or remote clients |

## API

`GET /api/system` · `GET /api/search?q=` · `GET /api/recommended` · `GET|DELETE /api/models` · `POST /api/pull` · `GET /api/jobs` · `POST /api/jobs/{id}/cancel` · `POST /api/run` · `POST /api/stop` · `GET /api/logs` · `GET /api/activity` · `POST /api/chat` · `GET /api/version?refresh=1` · `POST /api/update` · `GET /healthz`

`GET /api/system` reports `hosting.backend` (`cuda`, `model-runner`, or null),
`hosting.local_inference` (`available`, `reason`, `message`) and
`hosting.features` (`cuda_options`, `gpu_fit`, `logs`, `activity`). Native status
includes the selected engine and its client-facing URL. This describes the
inference host, never the browser's OS. `POST /api/run` returns HTTP 409 when
neither backend is available and HTTP 400 for incompatible native/CUDA options.
`GET /api/activity` explicitly reports `supported: false` on native backends.
Status read failures return HTTP 502; the UI stops claiming a model is serving,
shows the error and reconnects automatically without replaying queued starts.

## Notes

- **Security**: the hub mounts the Docker socket, which is root-equivalent on the host. It binds to `127.0.0.1` only; don't expose it.
- **Windows VRAM**: WDDM evicts idle VRAM, so `nvidia-smi` can show little used while a model is loaded. The hub uses 0.90 GPU memory utilization by default; 0.93 was observed to hang on a 4090.
- **Eager mode**: vLLM runs with `--enforce-eager` unless you tick CUDA graphs, trading speed for lower VRAM and faster startup.
- The CUDA vLLM container is named `vllm` and joins the `vllm-hub-net` network. The hub selects one model at a time; native models run in Model Runner, not that container.

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
