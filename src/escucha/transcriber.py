from collections.abc import Generator
from pathlib import Path

from faster_whisper import WhisperModel

from escucha.models import RawSegment


class TranscriptionError(Exception):
    """Raised when transcription fails."""


# Bias for Spanish content: hints punctuation style and common openings.
# Whisper uses this as a soft prior, not a hard constraint.
_DEFAULT_SPANISH_PROMPT = (
    "Reunion de trabajo en espanol. Conversacion entre varios hablantes. "
    "Incluye signos de puntuacion como puntos, comas, signos de interrogacion."
)


def load_whisper_model(
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "int8",
) -> WhisperModel:
    """Load and return a faster-whisper model.

    Args:
        model_size: Model name (tiny, base, small, medium, large-v3).
        device: 'cuda' or 'cpu'.
        compute_type: 'int8', 'float16', or 'float32'.
            Note: 'float16' is not supported on CPU. Use Settings.effective_compute_type.

    Returns:
        Loaded WhisperModel ready for inference.

    Raises:
        TranscriptionError: If model loading fails (network, OOM, bad name).
    """
    try:
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        raise TranscriptionError(
            f"Failed to load Whisper model '{model_size}' on {device}: {e}"
        ) from e


def transcribe(
    model: WhisperModel,
    audio_path: Path,
    language: str | None = "es",
    *,
    beam_size: int = 5,
    initial_prompt: str | None = None,
    vad_min_silence_ms: int = 500,
) -> Generator[RawSegment, None, None]:
    """Transcribe audio and yield segments lazily.

    Each yielded segment represents one Whisper-decoded chunk (typically up
    to 30s of audio). Yielding lazily lets the caller report progress.

    Args:
        model: A loaded WhisperModel instance.
        audio_path: Path to a WAV file (any sample rate; Whisper resamples).
        language: ISO 639-1 code, or None for auto-detection.
            Do not pass the literal string "auto" — faster-whisper rejects it.
        beam_size: Beam search width. Higher = better quality, slower.
        initial_prompt: Soft prior that biases punctuation/style. If language
            is 'es' and no prompt is given, a default Spanish prompt is used.
        vad_min_silence_ms: Silero VAD minimum silence between speech chunks.

    Yields:
        RawSegment with start, end (seconds), text, and word-level timestamps.

    Raises:
        TranscriptionError: If transcription fails.
    """
    prompt = initial_prompt
    if prompt is None and language == "es":
        prompt = _DEFAULT_SPANISH_PROMPT

    try:
        segments_iter, _info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": vad_min_silence_ms},
            word_timestamps=True,
            initial_prompt=prompt,
            condition_on_previous_text=False,  # reduces hallucinated repetition
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
