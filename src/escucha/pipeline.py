import asyncio
import logging
from datetime import datetime
from pathlib import Path
from escucha.config import Settings

logger = logging.getLogger("escucha")
from escucha.audio import extract_audio, AudioExtractionError
from escucha.transcriber import load_whisper_model, transcribe, TranscriptionError
from escucha.diarizer import load_diarization_pipeline, diarize, DiarizationError
from escucha.merger import merge_transcript_and_diarization
from escucha.summarizer import (
    summarize_with_ollama,
    summarize_with_claude,
    SummarizationError,
)
from escucha.jobs import Job, JobRegistry, PipelineStep
from escucha.models import RawSegment, DiarizedSegment


def _save_outputs(job_id: str, diarized: list, summary: str, base_dir: Path) -> None:
    """Write transcript and summary files to output/ after a completed job."""
    out_dir = base_dir / "output"
    out_dir.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{stamp}_{job_id}"

    # Transcript: one line per segment
    transcript_lines = [f"[{_fmt(s.start)}] {s.speaker}: {s.text}" for s in diarized]
    transcript_text = "\n".join(transcript_lines)
    (out_dir / f"{stem}_transcript.txt").write_text(transcript_text, encoding="utf-8")

    # Summary
    if summary:
        (out_dir / f"{stem}_summary.txt").write_text(summary, encoding="utf-8")

    logger.info("Saved output files to output/%s_*.txt", stem)


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class PipelineRunner:
    """Stateful pipeline runner. Holds loaded models across jobs to avoid reloading."""

    def __init__(self, settings: Settings, registry: JobRegistry) -> None:
        self._settings = settings
        self._registry = registry
        self._whisper_model = None
        self._diarization_pipeline = None

    async def warm_up(self) -> None:
        """Pre-load models at startup. Run in executor to not block the event loop."""
        loop = asyncio.get_event_loop()
        s = self._settings
        self._whisper_model = await loop.run_in_executor(
            None,
            load_whisper_model,
            s.whisper_model,
            s.resolved_device,
            s.effective_compute_type,
        )
        if s.hf_token:
            try:
                self._diarization_pipeline = await loop.run_in_executor(
                    None,
                    load_diarization_pipeline,
                    s.hf_token,
                    s.resolved_device,
                )
            except DiarizationError as e:
                logger.warning(
                    "Diarization pipeline failed to load — speaker identification will be skipped. "
                    "Reason: %s", e
                )

    async def run(
        self,
        job: Job,
        input_path: Path,
        language: str = "es",
        num_speakers: int | None = None,
        summarize: bool = True,
        use_claude: bool = False,
    ) -> None:
        """Execute the full pipeline for a job.

        This method catches all exceptions and marks the job as failed
        rather than letting them propagate.

        Args:
            job: The Job object to update with progress.
            input_path: Path to the uploaded MP4 file.
            language: ISO 639-1 language code for transcription.
            num_speakers: Optional speaker count hint.
            summarize: Whether to generate a summary.
            use_claude: Use Claude API instead of Ollama for summarization.
        """
        loop = asyncio.get_event_loop()
        s = self._settings
        wav_path = input_path.with_suffix(".wav")

        try:
            # --- Step 1: Extract audio ---
            await self._registry.update_progress(
                job, 5.0, PipelineStep.EXTRACTING_AUDIO, "Converting MP4 to WAV"
            )
            await loop.run_in_executor(
                None, extract_audio, input_path, wav_path, s.ffmpeg_path
            )

            # --- Step 2: Transcribe ---
            await self._registry.update_progress(
                job, 10.0, PipelineStep.TRANSCRIBING, "Starting transcription"
            )
            raw_segments: list[RawSegment] = []

            segments_gen = await loop.run_in_executor(
                None, lambda: list(transcribe(self._whisper_model, wav_path, language))
            )
            total = len(segments_gen)
            logger.info("Transcription produced %d segment(s)", total)
            for i, seg in enumerate(segments_gen):
                raw_segments.append(seg)
                pct = 10.0 + (50.0 * (i + 1) / max(total, 1))
                await self._registry.update_progress(
                    job, pct, PipelineStep.TRANSCRIBING,
                    f"Segment {i + 1}/{total}",
                )

            # --- Step 3: Diarize ---
            diar_segments = []
            if self._diarization_pipeline is not None:
                await self._registry.update_progress(
                    job, 62.0, PipelineStep.DIARIZING, "Identifying speakers"
                )
                try:
                    diar_segments = await loop.run_in_executor(
                        None, diarize, self._diarization_pipeline, wav_path, num_speakers
                    )
                except DiarizationError as e:
                    logger.warning("Diarization failed at inference time, continuing without speaker labels: %s", e)
            else:
                await self._registry.update_progress(
                    job, 62.0, PipelineStep.DIARIZING,
                    "Skipped (no HuggingFace token)",
                )

            # --- Step 4: Merge ---
            await self._registry.update_progress(
                job, 75.0, PipelineStep.MERGING, "Aligning speakers to transcript"
            )
            if diar_segments:
                diarized = merge_transcript_and_diarization(raw_segments, diar_segments)
            else:
                diarized = [
                    DiarizedSegment(
                        speaker="SPEAKER_00", start=seg.start, end=seg.end, text=seg.text
                    )
                    for seg in raw_segments
                ]

            # --- Step 5: Summarize ---
            summary = ""
            if summarize:
                await self._registry.update_progress(
                    job, 80.0, PipelineStep.SUMMARIZING, "Generating summary"
                )
                try:
                    if use_claude and s.anthropic_api_key:
                        summary = await summarize_with_claude(diarized, s.anthropic_api_key)
                    else:
                        summary = await summarize_with_ollama(
                            diarized, s.ollama_base_url, s.ollama_model
                        )
                except SummarizationError as e:
                    summary = f"[Summarization failed: {e}]"

            # --- Done ---
            speakers = sorted(set(seg.speaker for seg in diarized))
            result = {
                "duration_seconds": diarized[-1].end if diarized else 0.0,
                "language": language,
                "speakers": speakers,
                "segments": [
                    {"speaker": seg.speaker, "start": seg.start, "end": seg.end, "text": seg.text}
                    for seg in diarized
                ],
                "summary": summary,
            }

            await self._registry.mark_completed(job, result)
            await loop.run_in_executor(None, _save_outputs, job.job_id, diarized, summary, s.base_dir)

        except (AudioExtractionError, TranscriptionError) as e:
            await self._registry.mark_failed(job, str(e))
        except Exception as e:
            await self._registry.mark_failed(job, f"Unexpected error: {e}")
        finally:
            if wav_path.exists():
                wav_path.unlink(missing_ok=True)
            if input_path.exists():
                input_path.unlink(missing_ok=True)
