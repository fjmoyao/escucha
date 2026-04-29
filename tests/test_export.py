from escucha.export import to_txt, to_srt
from escucha.models import DiarizedSegment


def _segs():
    return [
        DiarizedSegment(speaker="SPEAKER_00", start=0.0,    end=4.52,  text="Buenos dias a todos."),
        DiarizedSegment(speaker="SPEAKER_01", start=4.52,   end=9.1,   text="Tengo los numeros."),
        DiarizedSegment(speaker="SPEAKER_00", start=3661.0, end=3665.0, text="Un ejemplo largo."),
    ]


def test_to_txt_format():
    lines = to_txt(_segs()).split("\n")
    assert lines[0] == "[00:00:00] SPEAKER_00: Buenos dias a todos."
    assert lines[1] == "[00:00:04] SPEAKER_01: Tengo los numeros."
    assert lines[2] == "[01:01:01] SPEAKER_00: Un ejemplo largo."


def test_to_srt_format():
    result = to_srt(_segs())
    blocks = [b.strip() for b in result.strip().split("\n\n")]
    assert blocks[0].startswith("1\n")
    assert "-->" in blocks[0]
    assert "[SPEAKER_00]" in blocks[0]
    assert "Buenos dias" in blocks[0]


def test_to_srt_timestamps():
    segs = [DiarizedSegment(speaker="SPEAKER_00", start=4.52, end=9.1, text="test")]
    result = to_srt(segs)
    assert "00:00:04,520" in result
    assert "00:00:09,100" in result


def test_to_txt_empty():
    assert to_txt([]) == ""


def test_to_srt_sequence_numbers():
    blocks = [b.strip() for b in to_srt(_segs()).strip().split("\n\n")]
    assert blocks[0].startswith("1\n")
    assert blocks[1].startswith("2\n")
    assert blocks[2].startswith("3\n")


def test_to_srt_empty():
    assert to_srt([]) == "\n"
