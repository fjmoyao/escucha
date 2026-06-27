import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from escucha.diarizer import diarize, load_diarization_pipeline, DiarizationError
from escucha.models import DiarizationSegment


# A real-shaped fake WAV: 1 second of silence at 16 kHz, mono, float32.
_FAKE_AUDIO_MONO = (np.zeros((16000, 1), dtype=np.float32), 16000)


class _FakeAnnotation:
    """Minimal stand-in for pyannote's Annotation. We avoid MagicMock here
    because MagicMock auto-creates every attribute — that breaks the v3
    test where ``hasattr(annotation, 'exclusive_speaker_diarization')``
    must return False."""

    def __init__(self, turns):
        self._turns = turns

    def itertracks(self, yield_label=True):
        return [
            (SimpleNamespace(start=s, end=e), None, label)
            for s, e, label in self._turns
        ]


def _make_mock_pipeline(api: str = "v4"):
    """Build a mock pyannote pipeline.

    api='v4': pipeline returns ``DiarizeOutput`` with ``exclusive_speaker_diarization``.
    api='v3': pipeline returns the Annotation directly.
    """
    pipeline = MagicMock()
    annotation = _FakeAnnotation([
        (0.0, 4.5, "SPEAKER_00"),
        (4.5, 9.0, "SPEAKER_01"),
        (9.0, 13.0, "SPEAKER_00"),
    ])
    if api == "v4":
        pipeline.return_value = SimpleNamespace(
            exclusive_speaker_diarization=annotation
        )
    else:
        pipeline.return_value = annotation
    return pipeline


def test_diarize_returns_segments(tmp_path):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 32)
    with patch("soundfile.read", return_value=_FAKE_AUDIO_MONO):
        results = diarize(_make_mock_pipeline("v4"), wav)
    assert len(results) == 3
    assert all(isinstance(r, DiarizationSegment) for r in results)
    assert results[0].speaker == "SPEAKER_00"
    assert results[1].speaker == "SPEAKER_01"
    assert results[2].speaker == "SPEAKER_00"


def test_diarize_handles_pyannote_3x_annotation(tmp_path):
    """Backwards-compat: pyannote 3.x returns Annotation directly."""
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 32)
    with patch("soundfile.read", return_value=_FAKE_AUDIO_MONO):
        results = diarize(_make_mock_pipeline("v3"), wav)
    assert len(results) == 3
    assert results[0].speaker == "SPEAKER_00"


def test_diarize_with_num_speakers(tmp_path):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 32)
    pipeline = _make_mock_pipeline("v4")
    with patch("soundfile.read", return_value=_FAKE_AUDIO_MONO):
        diarize(pipeline, wav, num_speakers=2)
    pipeline.assert_called_once()
    assert pipeline.call_args.kwargs.get("num_speakers") == 2


def test_diarize_converts_stereo_to_mono(tmp_path):
    """If the WAV is stereo, the diarizer must average to mono before passing
    it to pyannote (which only accepts mono)."""
    wav = tmp_path / "stereo.wav"
    wav.write_bytes(b"\x00" * 32)
    stereo = (np.zeros((16000, 2), dtype=np.float32), 16000)  # (samples, channels)
    pipeline = _make_mock_pipeline("v4")
    with patch("soundfile.read", return_value=stereo):
        diarize(pipeline, wav)
    audio_arg = pipeline.call_args.args[0]
    # After mono conversion the tensor must be (1, samples).
    assert audio_arg["waveform"].shape[0] == 1


def test_diarize_resamples_to_16khz(tmp_path):
    """Audio at the wrong sample rate must be resampled to 16 kHz."""
    wav = tmp_path / "wrong_sr.wav"
    wav.write_bytes(b"\x00" * 32)
    audio_44k = (np.zeros((44100, 1), dtype=np.float32), 44100)
    pipeline = _make_mock_pipeline("v4")
    with patch("soundfile.read", return_value=audio_44k):
        diarize(pipeline, wav)
    audio_arg = pipeline.call_args.args[0]
    assert audio_arg["sample_rate"] == 16000


def test_load_pipeline_invalid_token():
    with patch(
        "pyannote.audio.Pipeline.from_pretrained",
        side_effect=Exception("401 Unauthorized"),
    ):
        with pytest.raises(DiarizationError, match="Failed to load"):
            load_diarization_pipeline(hf_token="invalid_token_xyz", device="cpu")


def test_diarize_wraps_audio_load_errors(tmp_path):
    """If soundfile can't read the file, raise DiarizationError, not RuntimeError."""
    wav = tmp_path / "bad.wav"
    wav.write_bytes(b"\x00")
    pipeline = _make_mock_pipeline("v4")
    with patch("soundfile.read", side_effect=RuntimeError("bad header")):
        with pytest.raises(DiarizationError, match="Failed to read audio"):
            diarize(pipeline, wav)
