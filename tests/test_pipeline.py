"""Unit tests for helpers in escucha.pipeline.

The full pipeline is covered end-to-end via tests/test_api.py — these tests
cover the small pure helpers in isolation.
"""

from pathlib import Path
import pytest

from escucha.models import DiarizedSegment
from escucha.pipeline import _format_timestamp, _save_outputs


def test_format_timestamp_basic():
    assert _format_timestamp(0) == "00:00:00"
    assert _format_timestamp(65) == "00:01:05"
    assert _format_timestamp(3661) == "01:01:01"


def test_save_outputs_writes_transcript_only(tmp_path: Path):
    diarized = [
        DiarizedSegment(speaker="SPEAKER_00", start=0.0, end=2.0, text="Hola."),
        DiarizedSegment(speaker="SPEAKER_01", start=2.0, end=5.0, text="Buenos dias."),
    ]
    tx_path, sm_path = _save_outputs("abc12345", diarized, summary="", base_dir=tmp_path)
    assert tx_path.exists()
    assert sm_path is None
    body = tx_path.read_text(encoding="utf-8")
    assert "[00:00:00] SPEAKER_00: Hola." in body
    assert "[00:00:02] SPEAKER_01: Buenos dias." in body
    assert tx_path.parent == tmp_path / "output"
    assert "abc12345" in tx_path.name


def test_save_outputs_writes_summary_when_present(tmp_path: Path):
    diarized = [DiarizedSegment(speaker="SPEAKER_00", start=0.0, end=1.0, text="Hi.")]
    tx_path, sm_path = _save_outputs(
        "deadbeef", diarized, summary="Hablaron de cosas.", base_dir=tmp_path
    )
    assert sm_path is not None
    assert sm_path.exists()
    assert sm_path.read_text(encoding="utf-8") == "Hablaron de cosas."
    assert "deadbeef" in sm_path.name


def test_save_outputs_creates_output_dir(tmp_path: Path):
    diarized = [DiarizedSegment(speaker="SPEAKER_00", start=0.0, end=1.0, text="Hi.")]
    out_dir = tmp_path / "output"
    assert not out_dir.exists()
    _save_outputs("abc", diarized, summary="", base_dir=tmp_path)
    assert out_dir.is_dir()


def test_save_outputs_handles_empty_segments(tmp_path: Path):
    """Even with zero segments we still write an (empty) transcript file."""
    tx_path, sm_path = _save_outputs("empty1", [], summary="", base_dir=tmp_path)
    assert tx_path.exists()
    assert tx_path.read_text(encoding="utf-8") == ""
    assert sm_path is None
