from escucha.models import RawSegment, DiarizationSegment, DiarizedSegment


def merge_transcript_and_diarization(
    transcript_segments: list[RawSegment],
    diarization_segments: list[DiarizationSegment],
) -> list[DiarizedSegment]:
    """Assign speaker labels to transcript segments using timestamp overlap.

    Algorithm: For each transcript segment, find the diarization segment
    with the largest temporal overlap and assign its speaker label.
    If a transcript segment has word-level timestamps, use word midpoints
    for finer-grained assignment (majority vote across words).

    Args:
        transcript_segments: Ordered list of transcript segments from Whisper.
        diarization_segments: Ordered list of speaker turns from pyannote.

    Returns:
        List of DiarizedSegment (same length as transcript_segments),
        each with a speaker label assigned.
    """
    results: list[DiarizedSegment] = []

    for tseg in transcript_segments:
        speaker = _assign_speaker_by_words(tseg, diarization_segments)
        if speaker is None:
            speaker = _assign_speaker_by_overlap(tseg, diarization_segments)
        results.append(DiarizedSegment(
            speaker=speaker or "UNKNOWN",
            start=tseg.start,
            end=tseg.end,
            text=tseg.text,
        ))

    return results


def _assign_speaker_by_words(
    tseg: RawSegment,
    diar_segments: list[DiarizationSegment],
) -> str | None:
    """Assign speaker by majority vote over word midpoints.

    Returns None if tseg has no word-level timestamps.
    """
    if not tseg.words:
        return None

    votes: dict[str, int] = {}
    for w in tseg.words:
        midpoint = (w["start"] + w["end"]) / 2
        for dseg in diar_segments:
            if dseg.start <= midpoint <= dseg.end:
                votes[dseg.speaker] = votes.get(dseg.speaker, 0) + 1
                break

    if not votes:
        return None
    return max(votes, key=votes.get)


def _assign_speaker_by_overlap(
    tseg: RawSegment,
    diar_segments: list[DiarizationSegment],
) -> str | None:
    """Assign speaker by largest temporal overlap with the transcript segment."""
    best_speaker: str | None = None
    best_overlap: float = 0.0

    for dseg in diar_segments:
        overlap_start = max(tseg.start, dseg.start)
        overlap_end = min(tseg.end, dseg.end)
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = dseg.speaker

    return best_speaker
