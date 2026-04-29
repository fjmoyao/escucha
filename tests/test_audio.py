import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from escucha.audio import extract_audio, AudioExtractionError


@pytest.fixture
def tiny_mp4(tmp_path: Path) -> Path:
    """Generate a 1-second MP4 with a sine tone audio track using FFmpeg."""
    out = tmp_path / "test.mp4"
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-f", "lavfi", "-i", "color=c=black:size=64x64:rate=1:duration=1",
         "-shortest", str(out)],
        check=True, capture_output=True,
    )
    return out


def test_extract_audio_produces_wav(tiny_mp4, tmp_path):
    out = tmp_path / "out.wav"
    result = extract_audio(tiny_mp4, out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_extract_audio_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_audio(tmp_path / "missing.mp4", tmp_path / "out.wav")


def test_extract_audio_invalid_file(tmp_path):
    bad = tmp_path / "fake.mp4"
    bad.write_text("this is not a video file")
    with pytest.raises(AudioExtractionError):
        extract_audio(bad, tmp_path / "out.wav")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_extract_audio_creates_no_window(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"\x00")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        try:
            extract_audio(bad, tmp_path / "out.wav")
        except Exception:
            pass
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("creationflags") == subprocess.CREATE_NO_WINDOW
