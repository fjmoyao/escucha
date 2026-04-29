from escucha.models import DiarizedSegment


def to_txt(segments: list[DiarizedSegment]) -> str:
    """Format diarized segments as a timestamped plain text transcript.

    Output format per line:
        [HH:MM:SS] SPEAKER_XX: Transcribed text here.

    Args:
        segments: Ordered list of diarized transcript segments.

    Returns:
        The full transcript as a string.
    """
    lines: list[str] = []
    for seg in segments:
        ts = _format_timestamp_txt(seg.start)
        lines.append(f"[{ts}] {seg.speaker}: {seg.text}")
    return "\n".join(lines)


def to_srt(segments: list[DiarizedSegment]) -> str:
    """Format diarized segments as an SRT subtitle file.

    Output format per entry:
        1
        00:00:00,000 --> 00:00:04,520
        [SPEAKER_00] Text here.

    Args:
        segments: Ordered list of diarized transcript segments.

    Returns:
        The full transcript in SRT format.
    """
    blocks: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start = _format_timestamp_srt(seg.start)
        end = _format_timestamp_srt(seg.end)
        blocks.append(f"{i}\n{start} --> {end}\n[{seg.speaker}] {seg.text}")
    return "\n\n".join(blocks) + "\n"


def _format_timestamp_txt(seconds: float) -> str:
    """Convert seconds to HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_timestamp_srt(seconds: float) -> str:
    """Convert seconds to SRT format HH:MM:SS,mmm."""
    total_ms = round(seconds * 1000)
    h = total_ms // 3_600_000
    total_ms %= 3_600_000
    m = total_ms // 60_000
    total_ms %= 60_000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
