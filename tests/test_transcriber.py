import subprocess
from pathlib import Path
import pytest
from escucha.transcriber import load_whisper_model, transcribe, TranscriptionError
from escucha.models import RawSegment


@pytest.fixture(scope="module")
def tiny_model():
    """Load the tiny Whisper model once for all tests in this module."""
    return load_whisper_model("tiny", device="cpu", compute_type="int8")


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    """5-second silent WAV at 16 kHz mono."""
    out = tmp_path / "silent.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
         "-t", "5", str(out)],
        check=True, capture_output=True,
    )
    return out


def test_load_model_returns_model(tiny_model):
    assert tiny_model is not None


def test_load_model_invalid_name_raises():
    with pytest.raises(TranscriptionError, match="Failed to load"):
        load_whisper_model("nonexistent_model_xyz", device="cpu", compute_type="int8")


def test_transcribe_returns_list(tiny_model, silent_wav):
    segments = list(transcribe(tiny_model, silent_wav, language="es"))
    assert isinstance(segments, list)
    for seg in segments:
        assert isinstance(seg, RawSegment)
        assert seg.end >= seg.start


def test_transcribe_invalid_audio_raises(tiny_model, tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_text("not audio")
    with pytest.raises(TranscriptionError):
        list(transcribe(tiny_model, bad, language="es"))
