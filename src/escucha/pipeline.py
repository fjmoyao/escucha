import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

from escucha.audio import extract_audio, AudioExtractionError
from escucha.config import Settings
from escucha.diarizer import load_diarization_pipeline, diarize, DiarizationError
from escucha.jobs import Job, JobRegistry, PipelineStep
from escucha.merger import merge_transcript_and_diarization
from escucha.models import RawSegment, DiarizedSegment
from escucha.summarizer import summarize_with_ollama, summarize_with_claude, SummarizationError
from escucha.transcriber import load_whisper_model, transcribe, TranscriptionError

logger = logging.getLogger("escucha")


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _save_outputs(
    job_id: str,
    diarized: list[DiarizedSegment],
    summary: str,
    base_dir: Path,
) -> tuple[Path, Path | None]:
    """Write transcript and (if present) summary files to ``output/``.

    Returns the paths of the files written. Raises on any IO error so the
    caller can decide whether to fail the job or just log.
    """
    out_dir = base_dir / "output"
    out_dir.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{stamp}_{job_id}"

    transcript_path = out_dir / f"{stem}_transcript.txt"
    transcript_lines = [
        f"[{_format_timestamp(seg.start)}] {seg.speaker}: {seg.text}"
        for seg in diarized
    ]
    transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")

    summary_path: Path | None = None
    if summary:
        summary_path = out_dir / f"{stem}_summary.txt"
        summary_path.write_text(summary, encoding="utf-8")

    return transcript_path, summary_path


class PipelineRunner:
    """Stateful runner that holds loaded models across jobs to avoid reloading."""

    def __init__(self, settings: Settings, registry: JobRegistry) -> None:
        self._settings = settings
        self._registry = registry
        self._whisper_model = None
        self._diarization_pipeline = None

    @property
    def whisper_ready(self) -> bool:
        return self._whisper_model is not None

    @property
    def diarization_ready(self) -> bool:
        return self._diarization_pipeline is not None

    async def warm_up(self) -> None:
        """Pre-load Whisper and (if HF_TOKEN present) the diarization pipeline.

        Whisper is required — its failure raises ``TranscriptionError`` and
        prevents startup. Diarization failure is logged and ignored so the
        server can still serve transcription-only jobs.
        """
        loop = asyncio.get_running_loop()
        cfg = self._settings

        try:
            self._whisper_model = await loop.run_in_executor(
                None,
                load_whisper_model,
                cfg.whisper_model,
                cfg.resolved_device,
                cfg.effective_compute_type,
            )
        except TranscriptionError:
            logger.exception("Whisper model failed to load — server cannot start.")
            raise

        if cfg.hf_token:
            try:
                self._diarization_pipeline = await loop.run_in_executor(
                    None,
                    load_diarization_pipeline,
                    cfg.hf_token,
                    cfg.resolved_device,
                )
            except DiarizationError as e:
                logger.warning(
                    "Diarization pipeline failed to load — speaker identification "
                    "will be skipped. Reason: %s",
                    e,
                )
        else:
            logger.info("HF_TOKEN not set — diarization disabled.")

    async def run(
        self,
        job: Job,
        input_path: Path,
        language: str | None = "es",
        num_speakers: int | None = None,
        summarize: bool = True,
        use_claude: bool = False,
    ) -> None:
        """Execute the full pipeline for a job.

        Catches all exceptions and marks the job as failed (or completed with
        partial results) rather than letting them propagate to the caller.
        """
        loop = asyncio.get_running_loop()
        cfg = self._settings
        wav_path = input_path.with_suffix(".wav")
        raw_segments: list[RawSegment] = []
        diar_segments: list = []
        summary_text = ""

        try:
            # ── Step 1: Extract audio ─────────────────────────────────────
            await self._registry.update_progress(
                job, 5.0, PipelineStep.EXTRACTING_AUDIO, "Converting MP4 to WAV"
            )
            t0 = time.monotonic()
            await loop.run_in_executor(
                None, extract_audio, input_path, wav_path, cfg.ffmpeg_path
            )
            logger.info("Audio extracted in %.1fs", time.monotonic() - t0)

            # ── Step 2: Transcribe ────────────────────────────────────────
            await self._registry.update_progress(
                job, 10.0, PipelineStep.TRANSCRIBING, "Starting transcription"
            )
            t0 = time.monotonic()
            segments_gen = await loop.run_in_executor(
                None,
                lambda: list(transcribe(self._whisper_model, wav_path, language)),
            )
            total = len(segments_gen)
            logger.info(
                "Transcription produced %d segment(s) in %.1fs",
                total,
                time.monotonic() - t0,
            )

            for i, seg in enumerate(segments_gen):
                raw_segments.append(seg)
                pct = 10.0 + (50.0 * (i + 1) / max(total, 1))
                await self._registry.update_progress(
                    job, pct, PipelineStep.TRANSCRIBING, f"Segment {i + 1}/{total}"
                )

            if total == 0:
                # Push the bar forward even when nothing was found.
                await self._registry.update_progress(
                    job, 60.0, PipelineStep.TRANSCRIBING,
                    "No speech detected by VAD",
                )

            # ── Step 3: Diarize ───────────────────────────────────────────
            if self.diarization_ready and raw_segments:
                await self._registry.update_progress(
                    job, 62.0, PipelineStep.DIARIZING, "Identifying speakers"
                )
                t0 = time.monotonic()
                try:
                    diar_segments = await loop.run_in_executor(
                        None,
                        diarize,
                        self._diarization_pipeline,
                        wav_path,
                        num_speakers,
                    )
                    logger.info(
                        "Diarization produced %d turn(s) in %.1fs",
                        len(diar_segments),
                        time.monotonic() - t0,
                    )
                except DiarizationError as e:
                    logger.warning(
                        "Diarization failed at inference — continuing without "
                        "speaker labels. Reason: %s",
                        e,
                    )
            else:
                detail = (
                    "Skipped (no segments to label)"
                    if not raw_segments
                    else "Skipped (no HuggingFace token / model unavailable)"
                )
                await self._registry.update_progress(
                    job, 62.0, PipelineStep.DIARIZING, detail
                )

            # ── Step 4: Merge ─────────────────────────────────────────────
            await self._registry.update_progress(
                job, 75.0, PipelineStep.MERGING, "Aligning speakers to transcript"
            )
            if diar_segments:
                diarized = merge_transcript_and_diarization(raw_segments, diar_segments)
            else:
                diarized = [
                    DiarizedSegment(
                        speaker="SPEAKER_00",
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                    )
                    for seg in raw_segments
                ]

            # ── Step 5: Summarize (optional, non-fatal) ───────────────────
            if summarize and diarized:
                await self._registry.update_progress(
                    job, 80.0, PipelineStep.SUMMARIZING, "Generating summary"
                )
                t0 = time.monotonic()
                try:
                    if use_claude and cfg.anthropic_api_key:
                        summary_text = await summarize_with_claude(
                            diarized, cfg.anthropic_api_key
                        )
                    else:
                        summary_text = await summarize_with_ollama(
                            diarized, cfg.ollama_base_url, cfg.ollama_model
                        )
                    logger.info("Summary generated in %.1fs", time.monotonic() - t0)
                except SummarizationError as e:
                    logger.warning("Summarization failed: %s", e)
                    summary_text = f"[Summarization failed: {e}]"

            # ── Done ──────────────────────────────────────────────────────
            speakers = sorted(set(seg.speaker for seg in diarized))
            result = {
                "duration_seconds": diarized[-1].end if diarized else 0.0,
                "language": language or "auto",
                "speakers": speakers,
                "segments": [
                    {
                        "speaker": seg.speaker,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                    }
                    for seg in diarized
                ],
                "summary": summary_text,
            }

            await self._registry.mark_completed(job, result)

            # Output saving is non-fatal: if disk is full we still succeed.
            try:
                tx, sm = await loop.run_in_executor(
                    None,
                    _save_outputs,
                    job.job_id,
                    diarized,
                    summary_text,
                    cfg.base_dir,
                )
                logger.info(
                    "Saved %s%s",
                    tx.name,
                    f" + {sm.name}" if sm else "",
                )
            except Exception:
                logger.exception("Failed to save output files (job result kept in memory)")

        except (AudioExtractionError, TranscriptionError) as e:
            logger.error("Pipeline failed: %s", e)
            await self._registry.mark_failed(job, str(e))
        except Exception as e:
            logger.exception("Unexpected pipeline error")
            await self._registry.mark_failed(job, f"Unexpected error: {e}")
        finally:
            for path in (wav_path, input_path):
                try:
                    if path.exists():
                        path.unlink(missing_ok=True)
                except Exception:
                    logger.warning("Could not delete %s", path)
