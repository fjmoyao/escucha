import asyncio
import logging
import shutil
from pathlib import Path

import httpx
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import PlainTextResponse

from escucha.config import Settings
from escucha.export import to_srt, to_txt
from escucha.jobs import JobRegistry, JobStatus
from escucha.models import DiarizedSegment
from escucha.pipeline import PipelineRunner

logger = logging.getLogger("escucha")

router = APIRouter()

_settings: Settings = None
_registry: JobRegistry = None
_runner: PipelineRunner = None


def init_routes(settings: Settings, registry: JobRegistry, runner: PipelineRunner) -> None:
    """Inject dependencies into the routes module. Called once at startup."""
    global _settings, _registry, _runner
    _settings = settings
    _registry = registry
    _runner = runner


MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB
UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024     # 8 MB


def _on_pipeline_done(task: asyncio.Task) -> None:
    """Surface unhandled exceptions from background pipeline tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Background pipeline task raised: %s", exc, exc_info=exc)


@router.get("/api/health")
async def health_check() -> dict:
    """Return system status. Used by the frontend to display warnings."""
    ffmpeg_ok = bool(
        _settings.ffmpeg_path.exists() or shutil.which("ffmpeg") is not None
    )
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{_settings.ollama_base_url}/api/version")
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "ffmpeg": ffmpeg_ok,
        "ollama": ollama_ok,
        "cuda": _settings.resolved_device == "cuda",
        "device": _settings.resolved_device,
        "whisper_model": _settings.whisper_model,
        "whisper_ready": _runner.whisper_ready if _runner else False,
        "diarization_ready": _runner.diarization_ready if _runner else False,
        "hf_token_set": _settings.hf_token is not None,
        "anthropic_key_set": _settings.anthropic_api_key is not None,
    }


@router.post("/api/jobs", status_code=201)
async def create_job(
    file: UploadFile = File(...),
    num_speakers: int | None = Form(default=None),
    language: str = Form(default="es"),
    summarize: bool = Form(default=True),
    use_claude: bool = Form(default=False),
) -> dict:
    """Accept an MP4 upload and start processing in the background."""
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(400, "Only MP4 files are accepted.")

    try:
        job = _registry.create()
    except RuntimeError:
        raise HTTPException(503, "A job is already running. Please wait.")

    upload_path = _settings.upload_dir / f"{job.job_id}.mp4"
    loop = asyncio.get_running_loop()

    try:
        # Open and write off the event loop to keep the server responsive
        # during multi-MB uploads.
        with open(upload_path, "wb") as f:
            written = 0
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                await loop.run_in_executor(None, f.write, chunk)
                written += len(chunk)
                if written > MAX_FILE_SIZE:
                    raise HTTPException(400, "File exceeds 2 GB limit.")
    except HTTPException:
        upload_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        upload_path.unlink(missing_ok=True)
        logger.exception("Upload failed for job %s", job.job_id)
        raise HTTPException(500, f"Failed to save upload: {e}")

    task = asyncio.create_task(
        _runner.run(
            job=job,
            input_path=upload_path,
            language=None if language == "auto" else language,
            num_speakers=num_speakers,
            summarize=summarize,
            use_claude=use_claude,
        )
    )
    task.add_done_callback(_on_pipeline_done)

    return {
        "job_id": job.job_id,
        "status": "queued",
        "message": f"Job created. Connect to /api/jobs/{job.job_id}/ws for progress.",
    }


@router.get("/api/jobs")
async def list_jobs() -> list[dict]:
    """Return a snapshot of every known job — useful for debugging and UIs."""
    return [
        {
            "job_id": j.job_id,
            "status": j.status.value,
            "progress": j.progress,
            "current_step": j.current_step.value,
            "step_detail": j.step_detail,
            "error": j.error,
        }
        for j in _registry.all_jobs()
    ]


@router.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict:
    """Poll job status (fallback for when WebSocket is unavailable)."""
    job = _registry.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "current_step": job.current_step.value,
        "step_detail": job.step_detail,
        "error": job.error,
    }


@router.websocket("/api/jobs/{job_id}/ws")
async def job_websocket(websocket: WebSocket, job_id: str) -> None:
    """Stream pipeline progress via WebSocket."""
    job = _registry.get(job_id)
    if job is None:
        await websocket.close(code=4004, reason="Job not found")
        return

    await websocket.accept()
    _registry.add_listener(job, websocket)

    # Catch up if the job already finished before the WS connected.
    if job.status == JobStatus.COMPLETED:
        await websocket.send_json({"type": "completed", "progress": 100.0})
        await websocket.close()
        return
    if job.status == JobStatus.FAILED:
        await websocket.send_json({"type": "error", "message": job.error})
        await websocket.close()
        return

    try:
        # Block until the client disconnects (or the server closes us).
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _registry.remove_listener(job, websocket)


@router.get("/api/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> dict:
    """Fetch the full result payload of a completed job."""
    job = _registry.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, "Job is not yet completed.")
    return {"job_id": job.job_id, **job.result}


@router.get("/api/jobs/{job_id}/export/{fmt}")
async def export_transcript(job_id: str, fmt: str) -> PlainTextResponse:
    """Download transcript as TXT or SRT."""
    if fmt not in ("txt", "srt"):
        raise HTTPException(400, "Format must be 'txt' or 'srt'.")

    job = _registry.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, "Job is not yet completed.")

    segments = [DiarizedSegment(**s) for s in job.result["segments"]]
    content = to_txt(segments) if fmt == "txt" else to_srt(segments)

    return PlainTextResponse(
        content=content,
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="transcription.{fmt}"'
        },
    )
