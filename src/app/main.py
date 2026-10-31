from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from threading import Thread

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import capacity, downloads, hf, profiles, runner, updates
from .gpu import gpu_info


@asynccontextmanager
async def lifespan(_app):
    Thread(target=updates.warm, daemon=True).start()  # remember our image before any pull moves its tag
    if updates.AUTO:
        Thread(target=updates.auto_loop, daemon=True).start()
    yield


app = FastAPI(title="vLLM Hub", lifespan=lifespan)
STATIC = Path(__file__).parent / "static"


class RunRequest(BaseModel):
    repo_id: str
    context: int | None = None
    util: float | None = Field(default=None, gt=0.1, le=0.95)
    graphs: bool = False
    extra: list[str] = []
    weights_gib: float | None = None


class ChatRequest(BaseModel):
    messages: list[dict]
    thinking: bool = False
    max_tokens: int = 512


@app.get("/api/system")
def system():
    gpu = gpu_info()
    return {
        "gpu": gpu,
        "recommended_util": profiles.choose_util(gpu["total_mib"], gpu["used_mib"]) if gpu else None,
        "vllm": runner.status(),
        "image": runner.IMAGE,
    }


@app.get("/api/search")
def search(q: str, limit: int = 25):
    try:
        return hf.search(q, min(limit, 50))
    except Exception as exc:
        raise HTTPException(502, f"Hugging Face search failed: {exc}") from exc


@app.get("/api/recommended")
def recommended():
    return hf.recommended()


@app.get("/api/models")
def models():
    gpu = gpu_info()
    total = gpu["total_mib"] / 1024 if gpu else None
    out = []
    for m in downloads.cached_models():
        learned = capacity.auto_context(m["id"], 0.90)
        m["context_hint"] = learned or (profiles.choose_context(m["size_gib"], total, 0.85) if total else 8192)
        m["context_learned"] = bool(learned)
        out.append(m)
    return out


@app.delete("/api/models/{repo_id:path}")
def delete_model(repo_id: str):
    if not downloads.delete_cached(repo_id):
        raise HTTPException(404, "Model not in cache")
    return {"ok": True}


@app.post("/api/pull")
def pull(body: dict):
    repo_id = body.get("repo_id", "").strip()
    if "/" not in repo_id:
        raise HTTPException(400, "repo_id must look like org/model")
    return downloads.start(repo_id).view()


@app.get("/api/jobs")
def jobs():
    return downloads.list_jobs()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not downloads.cancel(job_id):
        raise HTTPException(404, "No active job with that id")
    return {"ok": True}


@app.post("/api/run")
def run(req: RunRequest):
    try:
        return runner.start(req.repo_id, req.context, req.util, req.graphs, req.extra, req.weights_gib)
    except runner.StartCancelled as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Could not start vLLM: {exc}") from exc


@app.post("/api/stop")
def stop():
    runner.stop()
    return {"ok": True}


@app.get("/api/logs")
def logs():
    return StreamingResponse(runner.log_stream(), media_type="text/plain")


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        return runner.chat(req.messages, req.thinking, req.max_tokens)
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/version")
def version(refresh: bool = False):
    return updates.check(force=refresh)


@app.post("/api/update")
def update():
    try:
        return updates.start_update()
    except Exception as exc:
        raise HTTPException(409, str(exc)) from exc


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
