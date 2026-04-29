from collections.abc import Generator
from pathlib import Path
from faster_whisper import WhisperModel
from escucha.models import RawSegment


class TranscriptionError(Exception):
    """Raised when transcription fails."""


def load_whisper_model(
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "int8",
) -> WhisperModel:
    """Load and return a faster-whisper model.

    Args:
        model_size: Whisper model name (tiny, base, small, medium, large-v3).
        device: 'cuda' or 'cpu'.
        compute_type: 'int8', 'float16', or 'float32'.
            NOTE: 'float16' is not supported on CPU; caller must
            pass 'int8' or 'float32' for CPU. Use Settings.effective_compute_type.

    Returns:
        A loaded WhisperModel ready for inference.

    Raises:
        TranscriptionError: If model loading fails (e.g., out of memory).
    """
    try:
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        raise TranscriptionError(f"Failed to load Whisper model '{model_size}': {e}") from e


def transcribe(
    model: WhisperModel,
    audio_path: Path,
    language: str | None = "es",
) -> Generator[RawSegment, None, None]:
    """Transcribe audio and yield segments as they are decoded.

    Each yielded segment represents one Whisper-decoded chunk (typically
    up to 30 seconds). Segments are yielded as they are produced, enabling
    the caller to compute progress incrementally.

    Args:
        model: A loaded WhisperModel instance.
        audio_path: Path to the 16kHz mono WAV file.
        language: ISO 639-1 language code. Default 'es' (Spanish).

    Yields:
        RawSegment with start, end (seconds), and text fields.

    Raises:
        TranscriptionError: If transcription fails.
    """
    try:
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
            word_timestamps=True,
        )
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}") from e

    for seg in segments_iter:
        yield RawSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            words=[
                {"start": w.start, "end": w.end, "word": w.word}
                for w in (seg.words or [])
            ],
        )
