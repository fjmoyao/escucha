import asyncio
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import PlainTextResponse
import httpx
import torch

from escucha.config import Settings
from escucha.jobs import JobRegistry, JobStatus
from escucha.pipeline import PipelineRunner
from escucha.export import to_txt, to_srt

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


@router.get("/api/health")
async def health_check() -> dict:
    """Return system status for frontend warnings."""
    ffmpeg_ok = _settings.ffmpeg_path.exists() or shutil.which("ffmpeg") is not None
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
        "cuda": torch.cuda.is_available(),
        "device": _settings.resolved_device,
        "whisper_model": _settings.whisper_model,
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
    try:
        with open(upload_path, "wb") as f:
            while chunk := await file.read(8 * 1024 * 1024):
                f.write(chunk)
                if f.tell() > MAX_FILE_SIZE:
                    upload_path.unlink()
                    raise HTTPException(400, "File exceeds 2 GB limit.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to save upload: {e}")

    asyncio.create_task(
        _runner.run(
            job=job,
            input_path=upload_path,
            language=None if language == "auto" else language,
            num_speakers=num_speakers,
            summarize=summarize,
            use_claude=use_claude,
        )
    )

    return {
        "job_id": job.job_id,
        "status": "queued",
        "message": f"Job created. Connect to /api/jobs/{job.job_id}/ws for progress.",
    }


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
    """WebSocket endpoint for streaming progress updates."""
    job = _registry.get(job_id)
    if job is None:
        await websocket.close(code=4004, reason="Job not found")
        return

    await websocket.accept()
    _registry.add_listener(job, websocket)

    if job.status == JobStatus.COMPLETED:
        await websocket.send_json({"type": "completed", "progress": 100.0})
        await websocket.close()
        return
    if job.status == JobStatus.FAILED:
        await websocket.send_json({"type": "error", "message": job.error})
        await websocket.close()
        return

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _registry.remove_listener(job, websocket)


@router.get("/api/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> dict:
    """Fetch the full result of a completed job."""
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

    from escucha.models import DiarizedSegment
    segments = [DiarizedSegment(**s) for s in job.result["segments"]]

    if fmt == "txt":
        content = to_txt(segments)
    else:
        content = to_srt(segments)

    return PlainTextResponse(
        content=content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="transcription.{fmt}"'},
    )
