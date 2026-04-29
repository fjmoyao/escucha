from pathlib import Path
import pytest
from escucha.models import RawSegment, DiarizationSegment, DiarizedSegment


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_raw_segments() -> list[RawSegment]:
    """Three transcript segments for merger and export tests."""
    return [
        RawSegment(start=0.0, end=4.5, text="Buenos dias a todos.",
                   words=[{"start": 0.0, "end": 1.0, "word": "Buenos"},
                          {"start": 1.0, "end": 2.0, "word": "dias"},
                          {"start": 2.0, "end": 3.5, "word": "a"},
                          {"start": 3.5, "end": 4.5, "word": "todos."}]),
        RawSegment(start=4.5, end=9.0, text="Tengo los numeros del trimestre.",
                   words=[{"start": 4.5, "end": 5.5, "word": "Tengo"},
                          {"start": 5.5, "end": 6.5, "word": "los"},
                          {"start": 6.5, "end": 7.5, "word": "numeros"},
                          {"start": 7.5, "end": 8.5, "word": "del"},
                          {"start": 8.5, "end": 9.0, "word": "trimestre."}]),
        RawSegment(start=9.0, end=13.0, text="Perfecto, revisemos.",
                   words=[{"start": 9.0, "end": 10.5, "word": "Perfecto,"},
                          {"start": 10.5, "end": 13.0, "word": "revisemos."}]),
    ]


@pytest.fixture
def sample_diarization_segments() -> list[DiarizationSegment]:
    """Two speakers alternating, aligned to sample_raw_segments."""
    return [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=4.5),
        DiarizationSegment(speaker="SPEAKER_01", start=4.5, end=9.0),
        DiarizationSegment(speaker="SPEAKER_00", start=9.0, end=13.0),
    ]


@pytest.fixture
def sample_diarized_segments() -> list[DiarizedSegment]:
    return [
        DiarizedSegment(speaker="SPEAKER_00", start=0.0, end=4.5, text="Buenos dias a todos."),
        DiarizedSegment(speaker="SPEAKER_01", start=4.5, end=9.0, text="Tengo los numeros del trimestre."),
        DiarizedSegment(speaker="SPEAKER_00", start=9.0, end=13.0, text="Perfecto, revisemos."),
    ]
