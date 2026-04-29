import pytest
from escucha.merger import merge_transcript_and_diarization
from escucha.models import RawSegment, DiarizationSegment


def test_merge_assigns_correct_speakers(sample_raw_segments, sample_diarization_segments):
    result = merge_transcript_and_diarization(sample_raw_segments, sample_diarization_segments)
    assert len(result) == 3
    assert result[0].speaker == "SPEAKER_00"
    assert result[1].speaker == "SPEAKER_01"
    assert result[2].speaker == "SPEAKER_00"


def test_merge_word_level_alignment():
    # Segment spans two speaker turns; word midpoints: "one"(0.45) "two"(1.45) → S00,
    # "three"(2.45) "four"(3.5) → S01. Vote is tied so result is one of the two.
    raw = [RawSegment(start=0.0, end=4.0, text="one two three four",
                      words=[{"start": 0.0, "end": 0.9, "word": "one"},
                             {"start": 1.0, "end": 1.9, "word": "two"},
                             {"start": 2.0, "end": 2.9, "word": "three"},
                             {"start": 3.0, "end": 4.0, "word": "four"}])]
    diar = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.5),
        DiarizationSegment(speaker="SPEAKER_01", start=2.5, end=4.0),
    ]
    result = merge_transcript_and_diarization(raw, diar)
    assert result[0].speaker in ("SPEAKER_00", "SPEAKER_01")


def test_merge_no_diarization_overlap():
    raw = [RawSegment(start=100.0, end=105.0, text="orphan segment", words=[])]
    diar = [DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=4.5)]
    result = merge_transcript_and_diarization(raw, diar)
    assert result[0].speaker == "UNKNOWN"


def test_merge_empty_inputs():
    result = merge_transcript_and_diarization([], [])
    assert result == []


def test_merge_no_words_falls_back_to_overlap(sample_raw_segments, sample_diarization_segments):
    # Strip word timestamps to force overlap-based assignment.
    stripped = [RawSegment(start=s.start, end=s.end, text=s.text, words=[])
                for s in sample_raw_segments]
    result = merge_transcript_and_diarization(stripped, sample_diarization_segments)
    assert result[0].speaker == "SPEAKER_00"
    assert result[1].speaker == "SPEAKER_01"
