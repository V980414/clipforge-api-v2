"""ClipForge API — servidor FastAPI."""
import os
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from pipeline import process_video, JobState, JOBS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("clipforge")

API_SECRET = os.environ.get("API_SECRET", "")
CLIPS_DIR = Path(os.environ.get("CLIPS_DIR", "/data/clips"))
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ClipForge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_secret(x_api_secret: Optional[str]):
    if not API_SECRET:
        raise HTTPException(500, "API_SECRET not configured on server")
    if x_api_secret != API_SECRET:
        raise HTTPException(401, "invalid api secret")


class CreateJobIn(BaseModel):
    youtube_url: HttpUrl
    callback_url: Optional[HttpUrl] = None
    callback_secret: Optional[str] = None
    job_id: Optional[str] = None  # id externo (uuid do supabase)
    max_clips: int = 8
    target_duration: int = 60  # segundos por corte


class CreateJobOut(BaseModel):
    job_id: str
    status: str


@app.get("/health")
def health():
    return {"ok": True, "service": "clipforge-api"}


@app.post("/jobs", response_model=CreateJobOut)
async def create_job(
    payload: CreateJobIn,
    background: BackgroundTasks,
    x_api_secret: Optional[str] = Header(None),
):
    _check_secret(x_api_secret)
    job_id = payload.job_id or str(uuid.uuid4())
    state = JobState(job_id=job_id, status="queued", progress=0)
    JOBS[job_id] = state
    background.add_task(
        process_video,
        job_id=job_id,
        youtube_url=str(payload.youtube_url),
        callback_url=str(payload.callback_url) if payload.callback_url else None,
        callback_secret=payload.callback_secret,
        max_clips=payload.max_clips,
        target_duration=payload.target_duration,
    )
    log.info("Job %s queued for %s", job_id, payload.youtube_url)
    return CreateJobOut(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, x_api_secret: Optional[str] = Header(None)):
    _check_secret(x_api_secret)
    state = JOBS.get(job_id)
    if not state:
        raise HTTPException(404, "job not found")
    return state.model_dump()


@app.get("/clips/{filename}")
def serve_clip(filename: str):
    # Servidos publicamente (URLs longas com uuid são suficientes).
    # Para mais segurança, gere URLs assinadas no futuro.
    path = CLIPS_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "clip not found")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/")
def root():
    return JSONResponse({"service": "ClipForge API", "docs": "/docs"})
