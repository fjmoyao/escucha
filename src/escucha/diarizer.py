from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline as PyannotePipeline
from scipy import signal as scipy_signal

from escucha.models import DiarizationSegment


class DiarizationError(Exception):
    """Raised when diarization fails."""


_TARGET_SAMPLE_RATE = 16000


def load_diarization_pipeline(
    hf_token: str,
    device: str = "cuda",
) -> PyannotePipeline:
    """Load the pyannote speaker-diarization pipeline.

    Args:
        hf_token: HuggingFace access token.
            User must accept model terms at:
            - https://huggingface.co/pyannote/speaker-diarization-3.1
            - https://huggingface.co/pyannote/speaker-diarization-community-1
        device: 'cuda' or 'cpu'.

    Returns:
        A loaded pyannote Pipeline ready for inference.

    Raises:
        DiarizationError: If loading fails (invalid token, terms not accepted, network).
    """
    try:
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except Exception as e:
        raise DiarizationError(
            "Failed to load pyannote pipeline. Verify HF_TOKEN is valid and that "
            "you accepted the model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1 and "
            "https://huggingface.co/pyannote/speaker-diarization-community-1. "
            f"Underlying error: {e}"
        ) from e

    try:
        pipeline.to(torch.device(device))
    except Exception as e:
        raise DiarizationError(f"Failed to move pipeline to {device}: {e}") from e

    return pipeline


def diarize(
    pipeline: PyannotePipeline,
    audio_path: Path,
    num_speakers: int | None = None,
) -> list[DiarizationSegment]:
    """Run speaker diarization on an audio file.

    Loads audio with ``soundfile`` (which bundles its own libsndfile DLL) so
    we never depend on ``torchcodec`` / FFmpeg shared libraries being present
    in the process PATH. The waveform is converted to mono and resampled to
    16 kHz before being passed to pyannote as an in-memory tensor dict.

    Args:
        pipeline: A loaded pyannote Pipeline.
        audio_path: Path to a WAV file.
        num_speakers: If provided, constrains the expected speaker count.
            None means auto-detect.

    Returns:
        List of DiarizationSegment in chronological order.

    Raises:
        DiarizationError: If audio loading or inference fails.
    """
    waveform = _load_waveform(audio_path)

    audio_input = {"waveform": waveform, "sample_rate": _TARGET_SAMPLE_RATE}

    try:
        if num_speakers is not None:
            output = pipeline(audio_input, num_speakers=num_speakers)
        else:
            output = pipeline(audio_input)
    except Exception as e:
        raise DiarizationError(f"Diarization inference failed: {e}") from e

    annotation = _annotation_from_output(output)

    results: list[DiarizationSegment] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        results.append(
            DiarizationSegment(speaker=speaker, start=turn.start, end=turn.end)
        )
    return results


def _load_waveform(audio_path: Path) -> torch.Tensor:
    """Load a WAV file as a (1, samples) float32 torch.Tensor at 16 kHz."""
    try:
        # always_2d -> (samples, channels). dtype float32 matches torch model input.
        data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    except Exception as e:
        raise DiarizationError(
            f"Failed to read audio with soundfile (libsndfile): {e}"
        ) from e

    # Mono: average channels.
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    # (samples, 1) -> (1, samples)
    data = data.T

    # Resample if needed. Use scipy's polyphase resampler — high quality
    # and stays in numpy land (no torchaudio dependency).
    if sample_rate != _TARGET_SAMPLE_RATE:
        data = _resample(data, sample_rate, _TARGET_SAMPLE_RATE)

    return torch.from_numpy(np.ascontiguousarray(data))


def _resample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Polyphase resample a (channels, samples) array."""
    from math import gcd
    g = gcd(src_rate, dst_rate)
    up = dst_rate // g
    down = src_rate // g
    # resample_poly works channel-by-channel along axis=1
    return scipy_signal.resample_poly(data, up, down, axis=1).astype(np.float32)


def _annotation_from_output(output) -> object:
    """Extract the pyannote ``Annotation`` from a diarization pipeline result.

    pyannote-audio 4.0 returns a ``DiarizeOutput`` dataclass whose
    ``exclusive_speaker_diarization`` field is the Annotation we want
    (no overlapping turns, ideal for transcript merging).

    pyannote-audio 3.x returns the Annotation directly.
    """
    if hasattr(output, "exclusive_speaker_diarization"):
        return output.exclusive_speaker_diarization
    if hasattr(output, "speaker_diarization"):
        return output.speaker_diarization
    return output
