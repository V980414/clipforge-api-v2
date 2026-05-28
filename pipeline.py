"""Pipeline: download YouTube → transcribe → analyze with Lovable AI → cut with FFmpeg."""
import os
import json
import shutil
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

import httpx
import yt_dlp
from pydantic import BaseModel
from faster_whisper import WhisperModel

log = logging.getLogger("clipforge.pipeline")

CLIPS_DIR = Path(os.environ.get("CLIPS_DIR", "/data/clips"))
WORK_DIR = Path(os.environ.get("WORK_DIR", "/data/work"))
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
LOVABLE_API_KEY = os.environ.get("LOVABLE_API_KEY", "")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "")  # ex: https://clipforge-api.up.railway.app

# Modelo carregado lazy (1ª req demora ~30s)
_whisper: Optional[WhisperModel] = None


def get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        log.info("Loading Whisper model %s (cpu/int8)", WHISPER_MODEL_SIZE)
        _whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
class ClipOut(BaseModel):
    title: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    viral_score: int
    reason: str
    hashtags: List[str] = []
    caption: str = ""
    download_url: str = ""


class JobState(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    error: Optional[str] = None
    video_title: Optional[str] = None
    duration_seconds: Optional[float] = None
    clips: List[ClipOut] = []


JOBS: Dict[str, JobState] = {}


def _update(job_id: str, **kwargs):
    state = JOBS.get(job_id)
    if state:
        for k, v in kwargs.items():
            setattr(state, k, v)


# ──────────────────────────────────────────────────────────────────────────────
# 1) Download
# ──────────────────────────────────────────────────────────────────────────────
def download_youtube(youtube_url: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
    return {
        "path": str(out_dir / "source.mp4"),
        "title": info.get("title", "Untitled"),
        "duration": float(info.get("duration", 0)),
        "youtube_id": info.get("id"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2) Transcribe (faster-whisper)
# ──────────────────────────────────────────────────────────────────────────────
def transcribe(video_path: str) -> dict:
    model = get_whisper()
    segments, info = model.transcribe(video_path, beam_size=1, vad_filter=True)
    segs = []
    full_text_parts = []
    for s in segments:
        segs.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        full_text_parts.append(s.text.strip())
    return {
        "language": info.language,
        "segments": segs,
        "full_text": " ".join(full_text_parts),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3) AI selection (Lovable AI Gateway — Gemini Flash)
# ──────────────────────────────────────────────────────────────────────────────
async def select_clips_with_ai(
    title: str, full_text: str, segments: list, max_clips: int, target_duration: int
) -> list:
    if not LOVABLE_API_KEY:
        raise RuntimeError("LOVABLE_API_KEY not configured")

    # Prepara índice "tempo: texto" pra a IA referenciar
    indexed = "\n".join(f"[{int(s['start'])}-{int(s['end'])}] {s['text']}" for s in segments)
    if len(indexed) > 80_000:
        indexed = indexed[:80_000]

    system = (
        "Você é um editor de vídeos virais. A partir de uma transcrição com "
        "timestamps de um vídeo longo (live, podcast, pregação, entrevista), "
        "selecione os melhores momentos para virar Shorts/Reels de até "
        f"{target_duration}s cada. Priorize: histórias emocionais, frases de impacto, "
        "controvérsias respeitosas, lições práticas, humor, momentos de virada. "
        "Sempre retorne JSON válido."
    )

    user = f"""Título do vídeo: {title}

Transcrição com timestamps (segundos):
{indexed}

Selecione até {max_clips} cortes. Para cada corte:
- start_seconds e end_seconds (escolha entre os timestamps fornecidos; duração entre 20 e {target_duration} segundos)
- title: título chamativo em português (máx 60 chars)
- viral_score: 1-100
- reason: por que esse momento é forte (1 frase)
- hashtags: 3-5 hashtags relevantes em português
- caption: legenda completa pro Instagram/TikTok (2-4 frases, com emojis)

Retorne JSON puro neste formato:
{{ "clips": [ {{ "title": "...", "start_seconds": 0, "end_seconds": 0, "viral_score": 0, "reason": "...", "hashtags": [], "caption": "..." }} ] }}
"""

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            "https://ai.gateway.lovable.dev/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {LOVABLE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "google/gemini-2.5-flash",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
            },
        )
        r.raise_for_status()
        data = r.json()

    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return parsed.get("clips", [])


# ──────────────────────────────────────────────────────────────────────────────
# 4) Cut + 9:16 + subtitles (FFmpeg)
# ──────────────────────────────────────────────────────────────────────────────
def build_srt(segments: list, start: float, end: float) -> str:
    """Gera SRT relativo (0-based) cobrindo apenas o intervalo do corte."""
    def fmt(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    lines = []
    idx = 1
    for seg in segments:
        s, e, text = seg["start"], seg["end"], seg["text"]
        if e <= start or s >= end:
            continue
        rs = max(s, start) - start
        re_ = min(e, end) - start
        lines.append(f"{idx}\n{fmt(rs)} --> {fmt(re_)}\n{text.strip()}\n")
        idx += 1
    return "\n".join(lines)


def cut_clip(
    source_path: str,
    start: float,
    end: float,
    segments: list,
    out_path: Path,
    work_dir: Path,
) -> None:
    """Corta, redimensiona pra 1080x1920 (9:16) com blur de fundo, queima legenda."""
    srt_path = work_dir / f"{out_path.stem}.srt"
    srt_path.write_text(build_srt(segments, start, end), encoding="utf-8")

    duration = end - start

    # Filter complex:
    # 1) Background: scale to fill 1080x1920 + blur (vídeo borrado atrás)
    # 2) Foreground: scale to fit width 1080 mantendo aspect, centralizado
    # 3) Overlay + subtítulos
    vf = (
        "[0:v]split=2[bg][fg];"
        "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=30:5[bgblur];"
        "[fg]scale=1080:-2[fgs];"
        "[bgblur][fgs]overlay=(W-w)/2:(H-h)/2[v];"
        f"[v]subtitles='{srt_path.as_posix()}':force_style='Fontname=DejaVu Sans,"
        "Fontsize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BorderStyle=3,Outline=2,Shadow=0,MarginV=120,Alignment=2,Bold=1'[vout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-t", str(duration),
        "-i", source_path,
        "-filter_complex", vf,
        "-map", "[vout]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    log.info("ffmpeg cut %s [%.1f-%.1f]", out_path.name, start, end)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("ffmpeg failed: %s", proc.stderr[-2000:])
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-500:]}")


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────
async def process_video(
    job_id: str,
    youtube_url: str,
    callback_url: Optional[str] = None,
    callback_secret: Optional[str] = None,
    max_clips: int = 8,
    target_duration: int = 60,
):
    work = WORK_DIR / job_id
    try:
        # 1) Download
        _update(job_id, status="downloading", progress=5)
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, download_youtube, youtube_url, work)
        _update(
            job_id,
            video_title=info["title"],
            duration_seconds=info["duration"],
            status="transcribing",
            progress=25,
        )

        # 2) Transcribe
        tr = await loop.run_in_executor(None, transcribe, info["path"])
        _update(job_id, status="analyzing", progress=55)

        # 3) AI select
        ai_clips = await select_clips_with_ai(
            info["title"], tr["full_text"], tr["segments"], max_clips, target_duration
        )
        _update(job_id, status="cutting", progress=70)

        # 4) Cut + render
        out_clips: List[ClipOut] = []
        total = max(len(ai_clips), 1)
        for i, c in enumerate(ai_clips):
            start = float(c.get("start_seconds", 0))
            end = float(c.get("end_seconds", start + target_duration))
            if end <= start:
                continue
            end = min(end, start + target_duration)

            out_name = f"{job_id}_{i}.mp4"
            out_path = CLIPS_DIR / out_name

            try:
                await loop.run_in_executor(
                    None, cut_clip, info["path"], start, end, tr["segments"], out_path, work
                )
            except Exception as e:
                log.exception("clip %s failed: %s", i, e)
                continue

            url = f"{PUBLIC_BASE_URL.rstrip('/')}/clips/{out_name}" if PUBLIC_BASE_URL else f"/clips/{out_name}"
            out_clips.append(
                ClipOut(
                    title=c.get("title", f"Corte {i+1}")[:120],
                    start_seconds=start,
                    end_seconds=end,
                    duration_seconds=end - start,
                    viral_score=int(c.get("viral_score", 50)),
                    reason=c.get("reason", "")[:500],
                    hashtags=c.get("hashtags", [])[:8],
                    caption=c.get("caption", "")[:1000],
                    download_url=url,
                )
            )
            _update(job_id, progress=70 + int(25 * (i + 1) / total))

        _update(job_id, status="completed", progress=100, clips=out_clips)
        log.info("Job %s completed with %d clips", job_id, len(out_clips))

        # Webhook
        if callback_url:
            try:
                state = JOBS[job_id]
                async with httpx.AsyncClient(timeout=30.0) as client:
                    await client.post(
                        callback_url,
                        headers={"x-callback-secret": callback_secret or ""},
                        json=state.model_dump(),
                    )
            except Exception as e:
                log.warning("callback failed: %s", e)

    except Exception as e:
        log.exception("Job %s failed", job_id)
        _update(job_id, status="failed", error=str(e)[:500])
        if callback_url:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    await client.post(
                        callback_url,
                        headers={"x-callback-secret": callback_secret or ""},
                        json=JOBS[job_id].model_dump(),
                    )
            except Exception:
                pass
    finally:
        # limpa arquivo grande do download
        try:
            shutil.rmtree(work, ignore_errors=True)
        except Exception:
            pass
