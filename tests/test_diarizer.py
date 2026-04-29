from unittest.mock import MagicMock
import pytest
from escucha.diarizer import diarize, load_diarization_pipeline, DiarizationError
from escucha.models import DiarizationSegment


def _make_mock_pipeline():
    """Build a mock pyannote pipeline that returns three fixed speaker turns."""
    mock_pipeline = MagicMock()

    t0 = MagicMock(); t0.start = 0.0;  t0.end = 4.5
    t1 = MagicMock(); t1.start = 4.5;  t1.end = 9.0
    t2 = MagicMock(); t2.start = 9.0;  t2.end = 13.0

    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = [
        (t0, None, "SPEAKER_00"),
        (t1, None, "SPEAKER_01"),
        (t2, None, "SPEAKER_00"),
    ]
    mock_pipeline.return_value = mock_diarization
    return mock_pipeline


def test_diarize_returns_segments(tmp_path):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 32)
    results = diarize(_make_mock_pipeline(), wav)
    assert len(results) == 3
    assert all(isinstance(r, DiarizationSegment) for r in results)
    assert results[0].speaker == "SPEAKER_00"
    assert results[1].speaker == "SPEAKER_01"
    assert results[2].speaker == "SPEAKER_00"


def test_diarize_with_num_speakers(tmp_path):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"\x00" * 32)
    mock_pipeline = _make_mock_pipeline()
    diarize(mock_pipeline, wav, num_speakers=2)
    mock_pipeline.assert_called_once()
    call_kwargs = mock_pipeline.call_args.kwargs
    assert call_kwargs.get("num_speakers") == 2


def test_load_pipeline_invalid_token():
    with pytest.raises(DiarizationError, match="Failed to load"):
        load_diarization_pipeline(hf_token="invalid_token_xyz", device="cpu")
