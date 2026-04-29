"""Data contracts for the Escucha pipeline.

All structures are dataclasses for simplicity and zero-dependency use
in internal pipeline code. Pydantic is used only at the API boundary
(FastAPI handles serialization automatically from dicts).
"""

from dataclasses import dataclass, field


@dataclass
class RawSegment:
    """A single transcript segment from faster-whisper, before speaker assignment.

    Attributes:
        start: Start time in seconds from beginning of audio.
        end: End time in seconds.
        text: Transcribed text for this segment.
        words: Optional word-level timestamps. Each entry is a dict with
               keys 'start' (float), 'end' (float), 'word' (str).
               Empty list if word timestamps were not generated.
    """
    start: float
    end: float
    text: str
    words: list[dict] = field(default_factory=list)


@dataclass
class DiarizationSegment:
    """A speaker turn identified by pyannote.audio.

    Attributes:
        speaker: Speaker label (e.g., 'SPEAKER_00', 'SPEAKER_01').
        start: Start time in seconds.
        end: End time in seconds.
    """
    speaker: str
    start: float
    end: float


@dataclass
class DiarizedSegment:
    """A transcript segment with an assigned speaker label.

    This is the primary output structure: it combines the text from
    RawSegment with the speaker from DiarizationSegment.

    Attributes:
        speaker: Assigned speaker label.
        start: Start time in seconds.
        end: End time in seconds.
        text: Transcribed text.
    """
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class JobState:
    """Serializable snapshot of a job's current state.

    Used by the GET /api/jobs/{job_id} polling endpoint.

    Attributes:
        job_id: Unique 8-character hex identifier.
        status: One of: 'queued', 'processing', 'completed', 'failed'.
        progress: Percentage from 0.0 to 100.0.
        current_step: Pipeline step name.
        step_detail: Human-readable detail string (e.g., 'Segment 5/40').
        error: Error message if status is 'failed', else None.
    """
    job_id: str
    status: str
    progress: float
    current_step: str
    step_detail: str = ""
    error: str | None = None
