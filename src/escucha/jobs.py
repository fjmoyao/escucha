import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from fastapi import WebSocket


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStep(str, Enum):
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    MERGING = "merging"
    SUMMARIZING = "summarizing"
    DONE = "done"


@dataclass
class Job:
    """Represents a single transcription job."""
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    current_step: PipelineStep = PipelineStep.EXTRACTING_AUDIO
    step_detail: str = ""
    error: str | None = None
    result: dict | None = None
    listeners: list[WebSocket] = field(default_factory=list, repr=False)


class JobRegistry:
    """In-memory registry of jobs. Single-job concurrency enforced.

    This class is NOT thread-safe by design — all access happens
    on the asyncio event loop.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._active_job_id: str | None = None

    def create(self) -> Job:
        """Create a new job. Raises RuntimeError if a job is already active."""
        if self._active_job_id is not None:
            active = self._jobs[self._active_job_id]
            if active.status == JobStatus.PROCESSING:
                raise RuntimeError("A job is already running")
        job_id = uuid.uuid4().hex[:8]
        job = Job(job_id=job_id)
        self._jobs[job_id] = job
        self._active_job_id = job_id
        return job

    def get(self, job_id: str) -> Job | None:
        """Return a job by ID, or None."""
        return self._jobs.get(job_id)

    async def update_progress(
        self,
        job: Job,
        progress: float,
        step: PipelineStep,
        detail: str = "",
    ) -> None:
        """Update job progress and broadcast to all WebSocket listeners."""
        job.status = JobStatus.PROCESSING
        job.progress = round(progress, 1)
        job.current_step = step
        job.step_detail = detail
        await self._broadcast(job, {
            "type": "progress",
            "progress": job.progress,
            "current_step": step.value,
            "step_detail": detail,
        })

    async def mark_completed(self, job: Job, result: dict) -> None:
        """Mark a job as completed and broadcast."""
        job.status = JobStatus.COMPLETED
        job.progress = 100.0
        job.current_step = PipelineStep.DONE
        job.result = result
        await self._broadcast(job, {"type": "completed", "progress": 100.0})
        await self._close_listeners(job)

    async def mark_failed(self, job: Job, error: str) -> None:
        """Mark a job as failed and broadcast the error."""
        job.status = JobStatus.FAILED
        job.error = error
        await self._broadcast(job, {"type": "error", "message": error})
        await self._close_listeners(job)

    def add_listener(self, job: Job, ws: WebSocket) -> None:
        """Register a WebSocket as a progress listener for a job."""
        job.listeners.append(ws)

    def remove_listener(self, job: Job, ws: WebSocket) -> None:
        """Unregister a WebSocket listener."""
        if ws in job.listeners:
            job.listeners.remove(ws)

    async def _broadcast(self, job: Job, message: dict) -> None:
        """Send a JSON message to all listeners. Remove dead sockets."""
        dead: list[WebSocket] = []
        for ws in job.listeners:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            job.listeners.remove(ws)

    async def _close_listeners(self, job: Job) -> None:
        """Close all WebSocket connections for a job."""
        for ws in job.listeners:
            try:
                await ws.close()
            except Exception:
                pass
        job.listeners.clear()
