from pathlib import Path
import torch
import torchaudio
from pyannote.audio import Pipeline as PyannotePipeline
from escucha.models import DiarizationSegment


class DiarizationError(Exception):
    """Raised when diarization fails."""


def load_diarization_pipeline(
    hf_token: str,
    device: str = "cuda",
) -> PyannotePipeline:
    """Load the pyannote speaker-diarization pipeline.

    Args:
        hf_token: HuggingFace access token.
            User must accept model terms at:
            https://huggingface.co/pyannote/speaker-diarization-3.1
        device: 'cuda' or 'cpu'.

    Returns:
        A loaded pyannote Pipeline ready for inference.

    Raises:
        DiarizationError: If loading fails (e.g., invalid token, terms not accepted).
    """
    try:
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except Exception as e:
        raise DiarizationError(
            f"Failed to load pyannote pipeline. "
            f"Ensure HF_TOKEN is valid and model terms are accepted: {e}"
        ) from e

    pipeline.to(torch.device(device))
    return pipeline


def diarize(
    pipeline: PyannotePipeline,
    audio_path: Path,
    num_speakers: int | None = None,
) -> list[DiarizationSegment]:
    """Run speaker diarization on an audio file.

    Args:
        pipeline: A loaded pyannote Pipeline.
        audio_path: Path to the 16kHz mono WAV file.
        num_speakers: If provided, constrains the expected speaker count.
            None = auto-detect.

    Returns:
        List of DiarizationSegment sorted by start time.

    Raises:
        DiarizationError: If diarization fails.
    """
    try:
        # Load audio via torchaudio to avoid torchcodec/FFmpeg-DLL dependency.
        # pyannote accepts a pre-loaded waveform dict as well as a file path.
        waveform, sample_rate = torchaudio.load(str(audio_path))
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}

        if num_speakers is not None:
            diarization = pipeline(audio_input, num_speakers=num_speakers)
        else:
            diarization = pipeline(audio_input)
    except Exception as e:
        raise DiarizationError(f"Diarization failed: {e}") from e

    annotation = diarization.exclusive_speaker_diarization
    results: list[DiarizationSegment] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        results.append(DiarizationSegment(
            speaker=speaker,
            start=turn.start,
            end=turn.end,
        ))

    return results
