"""Pipeline: download YouTube → transcribe → analyze with Lovable AI → cut with FFmpeg."""

import os
import json
import shutil
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict

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
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "")
YTDLP_COOKIES = os.environ.get("YTDLP_COOKIES_FILE", "")  # opcional: caminho pro cookies.txt

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
# 1) Download — robusto com cascata de fallbacks
# ──────────────────────────────────────────────────────────────────────────────
# Cascata (separada por "/"): yt-dlp tenta da esquerda pra direita até funcionar.
# 1. MP4 H.264 + M4A AAC até 1080p (ideal pro FFmpeg/Whisper, sem reencode pesado)
# 2. Qualquer vídeo + qualquer áudio até 1080p (yt-dlp faz merge)
# 3. Melhor MP4 progressivo (vídeo+áudio num arquivo só) até 720p
# 4. "best" — último recurso, qualquer container
_FORMAT_CASCADE = (
    "bv*[ext=mp4][vcodec^=avc1][height<=1080]+ba[ext=m4a]"
    "/bv*[height<=1080]+ba"
    "/b[ext=mp4][height<=720]"
    "/best"
)

_COMMON_YDL_OPTS = {
    "merge_output_format": "mp4",
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "retries": 5,
    "fragment_retries": 5,
    "extractor_retries": 3,
    "socket_timeout": 30,
    "concurrent_fragment_downloads": 4,
    "nocheckcertificate": True,
    "geo_bypass": True,
    # Headers pra evitar 403/formato indisponível em alguns vídeos
    "http_headers": {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8",
    },
    # Força cliente Android/web — costuma expor mais formatos que o default
    "extractor_args": {
        "youtube": {
            "player_client": ["android", "web", "ios"],
        }
    },
}


def _ydl_opts(out_dir: Path, fmt: str) -> dict:
    opts = dict(_COMMON_YDL_OPTS)
    opts["format"] = fmt
    opts["outtmpl"] = str(out_dir / "source.%(ext)s")
    if YTDLP_COOKIES and Path(YTDLP_COOKIES).exists():
        opts["cookiefile"] = YTDLP_COOKIES
    return opts


def _find_downloaded(out_dir: Path) -> Optional[Path]:
    for ext in ("mp4", "mkv", "webm", "m4a", "mov"):
        p = out_dir / f"source.{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _remux_to_mp4(src: Path, dst: Path) -> None:
    """Remux (sem reencode) pra MP4 quando o download veio em mkv/webm."""
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c", "copy", "-movflags", "+faststart",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # se copy falhar (codec incompatível com mp4), reencode rápido
        log.warning("remux copy failed, reencoding: %s", proc.stderr[-300:])
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(dst),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"remux failed: {proc.stderr[-300:]}")


def download_youtube(youtube_url: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    last_err: Optional[Exception] = None
    info = None

    # Tenta a cascata completa primeiro; se falhar inteiro, tenta cada nível isolado
    attempts = [_FORMAT_CASCADE] + _FORMAT_CASCADE.split("/")

    for fmt in attempts:
        # limpa restos da tentativa anterior
        for f in out_dir.glob("source.*"):
            try:
                f.unlink()
            except Exception:
                pass

        try:
            log.info("yt-dlp attempt with format=%s", fmt)
            with yt_dlp.YoutubeDL(_ydl_opts(out_dir, fmt)) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
            chosen_id = info.get("format_id")
            chosen_ext = info.get("ext")
            log.info(
                "yt-dlp OK: format_id=%s ext=%s height=%s",
                chosen_id, chosen_ext, info.get("height"),
            )
            break
        except yt_dlp.utils.DownloadError as e:
            last_err = e
            log.warning("yt-dlp failed with format=%s: %s", fmt, str(e)[:300])
            continue
        except Exception as e:
            last_err = e
            log.warning("yt-dlp unexpected error with format=%s: %s", fmt, str(e)[:300])
            continue

    if info is None:
        raise RuntimeError(
            f"Não foi possível baixar o vídeo após múltiplas tentativas. "
            f"Último erro: {last_err}"
        )

    downloaded = _find_downloaded(out_dir)
    if downloaded is None:
        raise RuntimeError("Download terminou mas o arquivo não foi encontrado.")

    target = out_dir / "source.mp4"
    if downloaded.suffix.lower() != ".mp4":
        log.info("Remuxing %s → source.mp4", downloaded.name)
        _remux_to_mp4(downloaded, target)
        try:
            downloaded.unlink()
        except Exception:
            pass
    elif downloaded != target:
        downloaded.rename(target)

    return {
        "path": str(target),
        "title": info.get("title", "Untitled"),
        "duration": float(info.get("duration", 0) or 0),
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
    srt_path = work_dir / f"{out_path.stem}.srt"
    srt_path.write_text(build_srt(segments, start, end), encoding="utf-8")
    duration = end - start

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

        tr = await loop.run_in_executor(None, transcribe, info["path"])
        _update(job_id, status="analyzing", progress=55)

        ai_clips = await select_clips_with_ai(
            info["title"], tr["full_text"], tr["segments"], max_clips, target_duration
        )
        _update(job_id, status="cutting", progress=70)

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
        try:
            shutil.rmtree(work, ignore_errors=True)
        except Exception:
            pass
