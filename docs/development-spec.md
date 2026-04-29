# Escucha — Development Specification

> **Version:** 1.0  
> **Date:** 2026-04-24  
> **Prerequisite:** Read `technical-investigation.md` for stack rationale  
> **Target:** A developer reads this document once and builds the complete application without ambiguity.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Dependency Manifest](#2-dependency-manifest)
3. [Backend API Spec](#3-backend-api-spec)
4. [Frontend Spec](#4-frontend-spec)
5. [Module Specs](#5-module-specs)
6. [Data Contracts](#6-data-contracts)
7. [Pipeline Flow](#7-pipeline-flow)
8. [Windows Setup Guide](#8-windows-setup-guide)
9. [Test Plan](#9-test-plan)

---

## 1. Project Structure

```
escucha/
├── bin/
│   └── ffmpeg.exe                  # Bundled FFmpeg binary (not committed to git — .gitignored)
├── docs/
│   ├── technical-investigation.md  # Stack research & rationale
│   └── development-spec.md         # THIS FILE
├── src/
│   └── escucha/
│       ├── __init__.py             # Package marker, exposes __version__
│       ├── main.py                 # FastAPI app factory, startup/shutdown, mounts
│       ├── config.py               # Settings via environment/dotenv, device detection
│       ├── models.py               # All dataclasses / Pydantic models (data contracts)
│       ├── audio.py                # FFmpeg audio extraction from MP4
│       ├── transcriber.py          # faster-whisper transcription engine
│       ├── diarizer.py             # pyannote.audio speaker diarization
│       ├── merger.py               # Merges transcription segments with speaker labels
│       ├── summarizer.py           # Ollama / Claude summarization
│       ├── pipeline.py             # Orchestrates the full processing pipeline
│       ├── jobs.py                 # Job registry, state machine, progress broadcasting
│       ├── export.py               # TXT and SRT formatters
│       └── routes.py               # All FastAPI route definitions
├── static/
│   └── index.html                  # Single-file frontend (HTML + embedded CSS + JS)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures (tmp paths, sample audio, mock models)
│   ├── test_audio.py
│   ├── test_transcriber.py
│   ├── test_diarizer.py
│   ├── test_merger.py
│   ├── test_summarizer.py
│   ├── test_export.py
│   ├── test_pipeline.py
│   └── test_api.py                 # End-to-end HTTP/WebSocket integration tests
├── .env.example                    # Template for user configuration
├── .gitignore
├── requirements.txt                # Pinned production dependencies
├── requirements-dev.txt            # Test/lint dependencies
├── setup.py                        # Minimal setup for editable install
└── README.md
```

### Key design decisions

- **Single-package layout** (`src/escucha/`): avoids import confusion, supports `pip install -e .`.
- **No frontend build step**: `static/index.html` is one self-contained file with embedded `<style>` and `<script>` tags.
- **`bin/ffmpeg.exe` is .gitignored**: users download it during setup or the app auto-detects it on PATH.

---

## 2. Dependency Manifest

### `requirements.txt` — Production

```
# ============================================================
# Escucha — pinned production dependencies
# ============================================================
# INSTALL ORDER MATTERS on Windows with GPU:
#
#   Step 1 — Install PyTorch with CUDA 12.8 first:
#     pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
#
#   Step 2 — Install everything else:
#     pip install -r requirements.txt
#
#   CPU-ONLY alternative for Step 1:
#     pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
#
# ============================================================

# -- Web server -----------------------------------------------
fastapi==0.136.1
uvicorn[standard]==0.46.0
python-multipart==0.0.20          # Required by FastAPI for file uploads

# -- Transcription --------------------------------------------
faster-whisper==1.2.1
# ctranslate2 is pulled in by faster-whisper; do NOT pin separately
# unless debugging CUDA issues (see technical-investigation.md)

# -- Diarization ----------------------------------------------
pyannote-audio==4.0.4
# Requires torch==2.8.0 (exact pin enforced by pyannote).
# This is satisfied by Step 1 above.

# -- Summarization --------------------------------------------
httpx==0.28.1                     # Async HTTP client for Ollama REST API
anthropic==0.52.0                 # Optional: only imported if Claude API key is set

# -- Utilities ------------------------------------------------
python-dotenv==1.1.0              # Loads .env config
pydantic==2.11.3                  # Data validation (also a FastAPI dependency)

# -- NOTE: torch and torchaudio are NOT listed here -----------
# They must be installed in Step 1 with the correct index URL.
# Listing them here would pull CPU-only wheels and break GPU.
```

### `requirements-dev.txt`

```
-r requirements.txt
pytest==8.3.5
pytest-asyncio==0.25.3
httpx==0.28.1                     # Also used at runtime; re-listed for clarity
coverage==7.8.0
ruff==0.11.8                      # Linter + formatter
```

### `.env.example`

```bash
# === Escucha Configuration ===

# HuggingFace token (REQUIRED for pyannote model download)
# Get yours at: https://huggingface.co/settings/tokens
# You must also accept the model terms at:
#   https://huggingface.co/pyannote/speaker-diarization-3.1
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Whisper model size: tiny, base, small, medium, large-v3
# Smaller = faster + less memory, larger = more accurate
# GPU (>=8GB VRAM): large-v3    CPU-only: small or medium
WHISPER_MODEL=large-v3

# Compute type for faster-whisper: int8, float16, float32
# GPU: int8 (fast, low VRAM) or float16 (slightly better quality)
# CPU: int8 or float32 (float16 not supported on CPU)
WHISPER_COMPUTE_TYPE=int8

# Device: cuda or cpu (auto-detected if omitted)
# DEVICE=auto

# Ollama endpoint (default: http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434

# Ollama model for summarization
OLLAMA_MODEL=llama3.2

# Optional: Anthropic API key for Claude-based summarization
# If set, the UI will show a "Use Claude" toggle
# ANTHROPIC_API_KEY=sk-ant-...

# Server
HOST=127.0.0.1
PORT=8000
```

> **Windows Gotcha — Install order:** If you `pip install -r requirements.txt` *before* installing torch with the CUDA index URL, pip will resolve torch from the default PyPI index and install CPU-only wheels. The pyannote-audio pin of `torch==2.8.0` will be satisfied but without CUDA. Always do Step 1 first.

---

## 3. Backend API Spec

Base URL: `http://127.0.0.1:8000`

### 3.1 `GET /` — Serve frontend

Returns `static/index.html`. Mounted via `StaticFiles` or a direct `FileResponse`.

### 3.2 `POST /api/jobs` — Create a transcription job

Accepts an MP4 file upload and optional parameters. Returns a job ID immediately; processing runs in the background.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | binary (MP4) | Yes | The video file to transcribe |
| `num_speakers` | integer | No | Expected speaker count. `null` = auto-detect. |
| `language` | string | No | ISO 639-1 code. Default: `"es"`. |
| `summarize` | boolean | No | Generate summary. Default: `true`. |
| `use_claude` | boolean | No | Use Claude API instead of Ollama. Default: `false`. Ignored if `ANTHROPIC_API_KEY` not set. |

**Response:** `201 Created`

```json
{
  "job_id": "a1b2c3d4",
  "status": "queued",
  "message": "Job created. Connect to /api/jobs/a1b2c3d4/ws for progress."
}
```

**Errors:**

| Status | Condition |
|---|---|
| `400` | File is not an MP4 (checked by extension and MIME sniffing) |
| `400` | File exceeds 2 GB |
| `503` | Another job is already running (single-job concurrency) |

### 3.3 `GET /api/jobs/{job_id}` — Poll job status

Returns current state of a job. This is the polling fallback if WebSocket is unavailable.

**Response:** `200 OK`

```json
{
  "job_id": "a1b2c3d4",
  "status": "processing",
  "progress": 45.2,
  "current_step": "transcribing",
  "step_detail": "Segment 18/40",
  "error": null
}
```

`status` enum: `"queued"` | `"processing"` | `"completed"` | `"failed"`

`current_step` enum: `"extracting_audio"` | `"transcribing"` | `"diarizing"` | `"merging"` | `"summarizing"` | `"done"`

### 3.4 `WebSocket /api/jobs/{job_id}/ws` — Stream progress

The primary real-time channel. The server pushes JSON messages as the pipeline progresses.

**Message types (server → client):**

```json
{"type": "progress", "progress": 12.5, "current_step": "transcribing", "step_detail": "Segment 5/40"}
```

```json
{"type": "completed", "progress": 100.0}
```

```json
{"type": "error", "message": "Ollama is not running. Start it with: ollama serve"}
```

**Client → server:** No messages expected. The client opens the socket and listens.

**Lifecycle:**
1. Client opens socket immediately after `POST /api/jobs` returns.
2. Server sends `progress` messages as each pipeline step advances.
3. On completion, server sends `completed` and closes the socket.
4. On failure, server sends `error` and closes the socket.

### 3.5 `GET /api/jobs/{job_id}/result` — Fetch completed results

**Response:** `200 OK` (only when `status == "completed"`)

```json
{
  "job_id": "a1b2c3d4",
  "duration_seconds": 1847.3,
  "language": "es",
  "speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 4.52,
      "text": "Buenos dias, vamos a comenzar la reunion."
    },
    {
      "speaker": "SPEAKER_01",
      "start": 4.52,
      "end": 9.1,
      "text": "Perfecto, tengo los numeros del trimestre."
    }
  ],
  "summary": "En esta reunion se discutieron los resultados del trimestre..."
}
```

**Errors:**

| Status | Condition |
|---|---|
| `404` | Job ID not found |
| `409` | Job is still processing (not yet completed) |

### 3.6 `GET /api/jobs/{job_id}/export/{format}` — Download transcript file

`format` must be `txt` or `srt`.

**Response:** `200 OK` with `Content-Disposition: attachment; filename="transcription.{format}"`

**TXT format:**

```
[00:00:00] SPEAKER_00: Buenos dias, vamos a comenzar la reunion.
[00:00:04] SPEAKER_01: Perfecto, tengo los numeros del trimestre.
```

**SRT format:**

```
1
00:00:00,000 --> 00:00:04,520
[SPEAKER_00] Buenos dias, vamos a comenzar la reunion.

2
00:00:04,520 --> 00:00:09,100
[SPEAKER_01] Perfecto, tengo los numeros del trimestre.
```

### 3.7 `GET /api/health` — Health check

**Response:** `200 OK`

```json
{
  "status": "ok",
  "ffmpeg": true,
  "ollama": true,
  "cuda": true,
  "device": "cuda",
  "whisper_model": "large-v3",
  "hf_token_set": true
}
```

Each boolean reflects whether the dependency was detected at startup. The frontend uses this to display warnings.

---

## 4. Frontend Spec

### Location: `static/index.html`

A single self-contained HTML file with embedded `<style>` and `<script>` blocks. No external CSS/JS frameworks. No build step.

### 4.1 Layout (top to bottom)

```
┌──────────────────────────────────────────────────┐
│  ESCUCHA — Transcriptor de video                 │  ← Header
├──────────────────────────────────────────────────┤
│  ⚠ Warnings banner (if health check fails)       │  ← Conditional
├──────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────┐  │
│  │  Drag & drop MP4 here, or click to browse  │  │  ← Upload zone
│  │  [Browse...]                                │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  Speakers: [auto ▼]  Language: [es ▼]            │  ← Options row
│  [✓] Generate summary                            │
│  [ ] Use Claude (requires API key)               │  ← Hidden if no key
│                                                  │
│  [ ▶ Start Transcription ]                       │  ← Primary action
├──────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────┐  │
│  │  ████████████░░░░░░░░░  45%                │  │  ← Progress bar
│  │  Transcribing — Segment 18/40              │  │  ← Step label
│  └────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────┤
│  ▸ Summary  (collapsible, default: open)         │  ← Summary panel
│    En esta reunion se discutieron...             │
│                                                  │
│  Transcript                                      │  ← Transcript panel
│  ┌────────────────────────────────────────────┐  │
│  │ 00:00  SPEAKER_00  Buenos dias, vamos a... │  │
│  │ 00:04  SPEAKER_01  Perfecto, tengo los...  │  │
│  │ ...                                        │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  [📄 Download TXT]  [📄 Download SRT]            │  ← Export buttons
└──────────────────────────────────────────────────┘
```

### 4.2 Component Behavior

**Upload zone:**
- Listens for `dragover`, `dragleave`, `drop` events on the zone div.
- Also has a hidden `<input type="file" accept=".mp4,video/mp4">` triggered by clicking the zone.
- On file selection: displays filename and file size. Enables the Start button.
- Rejects files > 2 GB client-side with an inline error message.

**Options row:**
- **Speakers dropdown**: `auto`, `2`, `3`, `4`, `5`, `6`. Default: `auto` (sends `null` to the API).
- **Language dropdown**: `es` (Spanish), `en` (English), `auto`. Default: `es`.
- **Generate summary checkbox**: Default checked.
- **Use Claude checkbox**: Only rendered if `/api/health` returns `anthropic_key_set: true`. Default unchecked.

**Start button:**
- Disabled until a file is selected.
- On click: sends `POST /api/jobs` with `FormData`, then opens WebSocket to `/api/jobs/{job_id}/ws`.
- During processing: button changes to "Processing..." and is disabled.

**Progress bar:**
- Hidden until a job starts.
- Updated by WebSocket `progress` messages.
- Step label shows `current_step` + `step_detail` in human-readable form.
- Step label mapping: `extracting_audio` → "Extracting audio...", `transcribing` → "Transcribing...", `diarizing` → "Identifying speakers...", `merging` → "Aligning transcript...", `summarizing` → "Generating summary...".

**Results panel:**
- Hidden until job completes.
- Fetches data from `GET /api/jobs/{job_id}/result`.
- **Summary section**: collapsible `<details>` element, default open. Renders summary as plain text paragraphs.
- **Transcript section**: a scrollable `<div>` with one row per segment. Each segment shows `[HH:MM:SS]`, speaker label (color-coded), and text.
- **Speaker colors**: assigned from a fixed palette: `#4A90D9` (blue), `#D94A4A` (red), `#4AD97A` (green), `#D9A84A` (amber), `#9B4AD9` (purple), `#4AD9D9` (teal). Mapped by speaker index.

**Export buttons:**
- "Download TXT": triggers `GET /api/jobs/{job_id}/export/txt` which initiates a browser download.
- "Download SRT": triggers `GET /api/jobs/{job_id}/export/srt`.
- Both are simple `<a>` tags with `download` attribute pointing to the API endpoint.

**Error handling:**
- WebSocket `error` messages display a red banner above the progress bar.
- Network failures (WebSocket close without `completed`): show "Connection lost. Retrying..." and fall back to polling `GET /api/jobs/{job_id}` every 3 seconds.
- Upload errors (400, 503): show inline red text below the upload zone.

### 4.3 Startup behavior

On page load:
1. `fetch("/api/health")` to check system status.
2. If `ffmpeg == false`: show warning "FFmpeg not found. See setup instructions."
3. If `hf_token_set == false`: show warning "HuggingFace token not set. Speaker identification will not work."
4. If `ollama == false`: show warning "Ollama not detected. Summarization will be skipped unless Claude API is configured."
5. If `cuda == false`: show info "No GPU detected. Processing will be slower."

---

## 5. Module Specs

### 5.1 `config.py` — Application settings

**Responsibility:** Load and validate all configuration from environment variables / `.env` file.

```python
from dataclasses import dataclass, field
from pathlib import Path
import os
import torch
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Immutable application configuration. Loaded once at startup."""

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    ffmpeg_path: Path = field(default=None)       # Resolved in __post_init__
    upload_dir: Path = field(default=None)         # Resolved in __post_init__

    # Model settings
    whisper_model: str = "large-v3"
    whisper_compute_type: str = "int8"
    device: str = "auto"                           # "auto", "cuda", "cpu"

    # Diarization
    hf_token: str | None = None

    # Summarization
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    anthropic_api_key: str | None = None

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        # Resolve computed defaults using object.__setattr__ (frozen dataclass)
        if self.ffmpeg_path is None:
            bundled = self.base_dir / "bin" / "ffmpeg.exe"
            object.__setattr__(self, "ffmpeg_path", bundled if bundled.exists() else Path("ffmpeg"))
        if self.upload_dir is None:
            d = self.base_dir / "tmp" / "uploads"
            d.mkdir(parents=True, exist_ok=True)
            object.__setattr__(self, "upload_dir", d)

    @property
    def resolved_device(self) -> str:
        """Return 'cuda' or 'cpu'. Resolves 'auto'."""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @property
    def effective_compute_type(self) -> str:
        """Adjust compute type for CPU (float16 not supported on CPU)."""
        if self.resolved_device == "cpu" and self.whisper_compute_type == "float16":
            return "float32"
        return self.whisper_compute_type


def load_settings() -> Settings:
    """Load settings from environment variables and .env file.

    Returns:
        Settings: Frozen configuration object.
    """
    load_dotenv()
    return Settings(
        whisper_model=os.getenv("WHISPER_MODEL", "large-v3"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        device=os.getenv("DEVICE", "auto"),
        hf_token=os.getenv("HF_TOKEN"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )
```

**Error contract:** `load_settings()` never raises. Missing optional values become `None`. The caller (`main.py`) validates critical settings at startup and logs warnings.

---

### 5.2 `audio.py` — Audio extraction

**Responsibility:** Extract 16 kHz mono WAV audio from an MP4 file using FFmpeg.

```python
import subprocess
import sys
from pathlib import Path


class AudioExtractionError(Exception):
    """Raised when FFmpeg fails to extract audio."""


def extract_audio(
    input_path: Path,
    output_path: Path,
    ffmpeg_path: Path = Path("ffmpeg"),
    sample_rate: int = 16000,
) -> Path:
    """Extract audio from an MP4 file as 16kHz mono WAV.

    Args:
        input_path: Path to the source MP4 file.
        output_path: Path where the WAV file will be written.
        ffmpeg_path: Path to the ffmpeg binary.
        sample_rate: Target sample rate in Hz. Default 16000 (Whisper's requirement).

    Returns:
        The output_path on success.

    Raises:
        AudioExtractionError: If FFmpeg returns a non-zero exit code.
        FileNotFoundError: If the input file does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    cmd = [
        str(ffmpeg_path),
        "-y",                   # Overwrite output without asking
        "-i", str(input_path),
        "-vn",                  # Discard video stream
        "-acodec", "pcm_s16le", # 16-bit PCM
        "-ar", str(sample_rate),
        "-ac", "1",             # Mono
        str(output_path),
    ]

    # On Windows: suppress the FFmpeg console popup
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        creationflags=creationflags,
    )

    if result.returncode != 0:
        raise AudioExtractionError(
            f"FFmpeg exited with code {result.returncode}: {result.stderr[:500]}"
        )

    return output_path
```

---

### 5.3 `transcriber.py` — Speech-to-text

**Responsibility:** Transcribe a WAV audio file to timestamped text segments using faster-whisper.

```python
from collections.abc import Generator
from pathlib import Path
from faster_whisper import WhisperModel
from escucha.models import RawSegment


class TranscriptionError(Exception):
    """Raised when transcription fails."""


def load_whisper_model(
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "int8",
) -> WhisperModel:
    """Load and return a faster-whisper model.

    Args:
        model_size: Whisper model name (tiny, base, small, medium, large-v3).
        device: 'cuda' or 'cpu'.
        compute_type: 'int8', 'float16', or 'float32'.
            NOTE: 'float16' is not supported on CPU; caller must
            pass 'int8' or 'float32' for CPU. Use Settings.effective_compute_type.

    Returns:
        A loaded WhisperModel ready for inference.

    Raises:
        TranscriptionError: If model loading fails (e.g., out of memory).
    """
    try:
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        raise TranscriptionError(f"Failed to load Whisper model '{model_size}': {e}") from e


def transcribe(
    model: WhisperModel,
    audio_path: Path,
    language: str = "es",
) -> Generator[RawSegment, None, None]:
    """Transcribe audio and yield segments as they are decoded.

    Each yielded segment represents one Whisper-decoded chunk (typically
    up to 30 seconds). Segments are yielded as they are produced, enabling
    the caller to compute progress incrementally.

    Args:
        model: A loaded WhisperModel instance.
        audio_path: Path to the 16kHz mono WAV file.
        language: ISO 639-1 language code. Default 'es' (Spanish).

    Yields:
        RawSegment with start, end (seconds), and text fields.

    Raises:
        TranscriptionError: If transcription fails.
    """
    try:
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,        # Silero VAD to skip silence
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
            word_timestamps=True,    # Enables finer diarization alignment
        )
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}") from e

    for seg in segments_iter:
        yield RawSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            words=[
                {"start": w.start, "end": w.end, "word": w.word}
                for w in (seg.words or [])
            ],
        )
```

> **CPU vs GPU:** The `model_size` and `compute_type` parameters directly control memory and speed. On CPU with `int8`, `large-v3` is feasible (~10 GB RAM) but slow (~1x real-time). Recommend `medium` + `int8` on CPU for a better speed/quality tradeoff.

---

### 5.4 `diarizer.py` — Speaker diarization

**Responsibility:** Identify who speaks when in an audio file using pyannote.audio.

```python
from pathlib import Path
from pyannote.audio import Pipeline as PyannotePipeline
from escucha.models import DiarizationSegment


class DiarizationError(Exception):
    """Raised when diarization fails."""


def load_diarization_pipeline(
    hf_token: str,
    device: str = "cuda",
) -> PyannotePipeline:
    """Load the pyannote speaker-diarization pipeline.

    Args:
        hf_token: HuggingFace access token.
            User must accept model terms at:
            https://huggingface.co/pyannote/speaker-diarization-3.1
        device: 'cuda' or 'cpu'.

    Returns:
        A loaded pyannote Pipeline ready for inference.

    Raises:
        DiarizationError: If loading fails (e.g., invalid token, terms not accepted).
    """
    try:
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    except Exception as e:
        raise DiarizationError(
            f"Failed to load pyannote pipeline. "
            f"Ensure HF_TOKEN is valid and model terms are accepted: {e}"
        ) from e

    import torch
    pipeline.to(torch.device(device))
    return pipeline


def diarize(
    pipeline: PyannotePipeline,
    audio_path: Path,
    num_speakers: int | None = None,
) -> list[DiarizationSegment]:
    """Run speaker diarization on an audio file.

    Args:
        pipeline: A loaded pyannote Pipeline.
        audio_path: Path to the 16kHz mono WAV file.
        num_speakers: If provided, constrains the expected speaker count.
            None = auto-detect.

    Returns:
        List of DiarizationSegment sorted by start time.

    Raises:
        DiarizationError: If diarization fails.
    """
    try:
        params = {"uri": "escucha", "audio": str(audio_path)}
        if num_speakers is not None:
            diarization = pipeline(params, num_speakers=num_speakers)
        else:
            diarization = pipeline(params)
    except Exception as e:
        raise DiarizationError(f"Diarization failed: {e}") from e

    results: list[DiarizationSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append(DiarizationSegment(
            speaker=speaker,
            start=turn.start,
            end=turn.end,
        ))

    return results
```

> **CPU vs GPU:** Diarization on CPU is ~5-10x slower than GPU. For a 60-min file: ~2 min on GPU, ~15 min on CPU. There is no accuracy difference.

---

### 5.5 `merger.py` — Transcript + diarization alignment

**Responsibility:** Assign a speaker label to each transcription segment by matching timestamps.

```python
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
```

---

### 5.6 `summarizer.py` — Text summarization

**Responsibility:** Generate a structured summary of a diarized transcript using Ollama (default) or Claude (optional).

```python
from escucha.models import DiarizedSegment
import httpx


class SummarizationError(Exception):
    """Raised when summarization fails."""


# -- Constants ---------------------------------------------------------------

_CHUNK_MAX_CHARS = 12000  # ~3000 tokens; leaves room for prompt + response in 8k context

_SYSTEM_PROMPT = (
    "Eres un asistente que resume transcripciones de reuniones en espanol. "
    "Produce un resumen estructurado con: 1) Puntos principales discutidos, "
    "2) Decisiones tomadas, 3) Tareas pendientes con responsable si se menciona. "
    "Responde en espanol. Se conciso."
)

_CHUNK_PROMPT_TEMPLATE = (
    "Resume la siguiente seccion de una transcripcion de reunion:\n\n{text}"
)

_FINAL_PROMPT_TEMPLATE = (
    "A continuacion hay resumenes parciales de diferentes secciones de una reunion. "
    "Combinalos en un unico resumen coherente y estructurado:\n\n{text}"
)


# -- Public API --------------------------------------------------------------

async def summarize_with_ollama(
    segments: list[DiarizedSegment],
    base_url: str = "http://localhost:11434",
    model: str = "llama3.2",
) -> str:
    """Summarize a diarized transcript using a local Ollama model.

    Implements chunked summarization: splits the transcript into chunks
    that fit the model's context window, summarizes each, then produces
    a final combined summary.

    Args:
        segments: The diarized transcript segments.
        base_url: Ollama API base URL.
        model: Ollama model name.

    Returns:
        A structured summary string in Spanish.

    Raises:
        SummarizationError: If Ollama is unreachable or returns an error.
    """
    full_text = _segments_to_text(segments)
    chunks = _split_into_chunks(full_text, _CHUNK_MAX_CHARS)

    if len(chunks) == 1:
        return await _ollama_generate(
            base_url, model,
            _SYSTEM_PROMPT,
            _CHUNK_PROMPT_TEMPLATE.format(text=chunks[0]),
        )

    partial_summaries: list[str] = []
    for chunk in chunks:
        summary = await _ollama_generate(
            base_url, model,
            _SYSTEM_PROMPT,
            _CHUNK_PROMPT_TEMPLATE.format(text=chunk),
        )
        partial_summaries.append(summary)

    combined = "\n\n---\n\n".join(partial_summaries)
    return await _ollama_generate(
        base_url, model,
        _SYSTEM_PROMPT,
        _FINAL_PROMPT_TEMPLATE.format(text=combined),
    )


async def summarize_with_claude(
    segments: list[DiarizedSegment],
    api_key: str,
) -> str:
    """Summarize a diarized transcript using the Anthropic Claude API.

    Sends the full transcript in a single request (Claude supports 200k tokens).

    Args:
        segments: The diarized transcript segments.
        api_key: Anthropic API key.

    Returns:
        A structured summary string in Spanish.

    Raises:
        SummarizationError: If the API call fails.
    """
    try:
        import anthropic
    except ImportError:
        raise SummarizationError(
            "anthropic package is not installed. Run: pip install anthropic"
        )

    full_text = _segments_to_text(segments)
    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": _CHUNK_PROMPT_TEMPLATE.format(text=full_text)}
            ],
        )
        return message.content[0].text
    except Exception as e:
        raise SummarizationError(f"Claude API call failed: {e}") from e


# -- Internal helpers --------------------------------------------------------

def _segments_to_text(segments: list[DiarizedSegment]) -> str:
    """Convert diarized segments to a readable text block."""
    lines: list[str] = []
    for seg in segments:
        lines.append(f"{seg.speaker}: {seg.text}")
    return "\n".join(lines)


def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split text into chunks, breaking at line boundaries."""
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        if current_len + len(line) + 1 > max_chars and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line) + 1

    if current:
        chunks.append("\n".join(current))

    return chunks


async def _ollama_generate(
    base_url: str,
    model: str,
    system: str,
    prompt: str,
) -> str:
    """Send a generate request to the Ollama API.

    Raises:
        SummarizationError: On connection error or non-200 response.
    """
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "system": system,
                    "prompt": prompt,
                    "stream": False,
                },
            )
        except httpx.ConnectError:
            raise SummarizationError(
                "Cannot connect to Ollama. Is it running? Start with: ollama serve"
            )

        if resp.status_code != 200:
            raise SummarizationError(
                f"Ollama returned status {resp.status_code}: {resp.text[:300]}"
            )

        return resp.json()["response"]
```

---

### 5.7 `export.py` — Transcript export formatters

**Responsibility:** Convert diarized segments into downloadable TXT and SRT formats.

```python
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
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
```

---

### 5.8 `jobs.py` — Job state management

**Responsibility:** Track processing jobs, manage their lifecycle, and broadcast progress to WebSocket clients.

```python
import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from fastapi import WebSocket


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStep(str, Enum):
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    MERGING = "merging"
    SUMMARIZING = "summarizing"
    DONE = "done"


@dataclass
class Job:
    """Represents a single transcription job."""
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    current_step: PipelineStep = PipelineStep.EXTRACTING_AUDIO
    step_detail: str = ""
    error: str | None = None
    result: dict | None = None                # Set on completion; holds the full result payload
    listeners: list[WebSocket] = field(default_factory=list, repr=False)


class JobRegistry:
    """In-memory registry of jobs. Single-job concurrency enforced.

    This class is NOT thread-safe by design — all access happens
    on the asyncio event loop.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._active_job_id: str | None = None

    def create(self) -> Job:
        """Create a new job. Raises RuntimeError if a job is already active."""
        if self._active_job_id is not None:
            active = self._jobs[self._active_job_id]
            if active.status == JobStatus.PROCESSING:
                raise RuntimeError("A job is already running")
        job_id = uuid.uuid4().hex[:8]
        job = Job(job_id=job_id)
        self._jobs[job_id] = job
        self._active_job_id = job_id
        return job

    def get(self, job_id: str) -> Job | None:
        """Return a job by ID, or None."""
        return self._jobs.get(job_id)

    async def update_progress(
        self,
        job: Job,
        progress: float,
        step: PipelineStep,
        detail: str = "",
    ) -> None:
        """Update job progress and broadcast to all WebSocket listeners."""
        job.status = JobStatus.PROCESSING
        job.progress = round(progress, 1)
        job.current_step = step
        job.step_detail = detail
        await self._broadcast(job, {
            "type": "progress",
            "progress": job.progress,
            "current_step": step.value,
            "step_detail": detail,
        })

    async def mark_completed(self, job: Job, result: dict) -> None:
        """Mark a job as completed and broadcast."""
        job.status = JobStatus.COMPLETED
        job.progress = 100.0
        job.current_step = PipelineStep.DONE
        job.result = result
        await self._broadcast(job, {"type": "completed", "progress": 100.0})
        await self._close_listeners(job)

    async def mark_failed(self, job: Job, error: str) -> None:
        """Mark a job as failed and broadcast the error."""
        job.status = JobStatus.FAILED
        job.error = error
        await self._broadcast(job, {"type": "error", "message": error})
        await self._close_listeners(job)

    def add_listener(self, job: Job, ws: WebSocket) -> None:
        """Register a WebSocket as a progress listener for a job."""
        job.listeners.append(ws)

    def remove_listener(self, job: Job, ws: WebSocket) -> None:
        """Unregister a WebSocket listener."""
        if ws in job.listeners:
            job.listeners.remove(ws)

    async def _broadcast(self, job: Job, message: dict) -> None:
        """Send a JSON message to all listeners. Remove dead sockets."""
        dead: list[WebSocket] = []
        for ws in job.listeners:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            job.listeners.remove(ws)

    async def _close_listeners(self, job: Job) -> None:
        """Close all WebSocket connections for a job."""
        for ws in job.listeners:
            try:
                await ws.close()
            except Exception:
                pass
        job.listeners.clear()
```

---

### 5.9 `pipeline.py` — Processing orchestrator

**Responsibility:** Run the full extraction → transcription → diarization → merge → summarization pipeline, reporting progress at each stage.

```python
import asyncio
from pathlib import Path
from escucha.config import Settings
from escucha.audio import extract_audio, AudioExtractionError
from escucha.transcriber import load_whisper_model, transcribe, TranscriptionError
from escucha.diarizer import load_diarization_pipeline, diarize, DiarizationError
from escucha.merger import merge_transcript_and_diarization
from escucha.summarizer import (
    summarize_with_ollama,
    summarize_with_claude,
    SummarizationError,
)
from escucha.export import to_txt, to_srt
from escucha.jobs import Job, JobRegistry, PipelineStep
from escucha.models import RawSegment, DiarizedSegment


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
            self._diarization_pipeline = await loop.run_in_executor(
                None,
                load_diarization_pipeline,
                s.hf_token,
                s.resolved_device,
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

            # Transcription yields segments; collect them and report progress.
            # We estimate total segments from audio duration (rough: 1 seg per 10s).
            segments_gen = await loop.run_in_executor(
                None, lambda: list(transcribe(self._whisper_model, wav_path, language))
            )
            total = len(segments_gen)
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
                diar_segments = await loop.run_in_executor(
                    None, diarize, self._diarization_pipeline, wav_path, num_speakers
                )
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
                        speaker="SPEAKER_00", start=s.start, end=s.end, text=s.text
                    )
                    for s in raw_segments
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
                    # Non-fatal: report the error but still return the transcript
                    summary = f"[Summarization failed: {e}]"

            # --- Done ---
            speakers = sorted(set(seg.speaker for seg in diarized))
            result = {
                "duration_seconds": diarized[-1].end if diarized else 0.0,
                "language": language,
                "speakers": speakers,
                "segments": [
                    {"speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text}
                    for s in diarized
                ],
                "summary": summary,
            }

            await self._registry.mark_completed(job, result)

        except (AudioExtractionError, TranscriptionError, DiarizationError) as e:
            await self._registry.mark_failed(job, str(e))
        except Exception as e:
            await self._registry.mark_failed(job, f"Unexpected error: {e}")
        finally:
            # Clean up temp WAV file
            if wav_path.exists():
                wav_path.unlink(missing_ok=True)
```

---

### 5.10 `routes.py` — FastAPI route definitions

**Responsibility:** Define all HTTP and WebSocket endpoints; wire them to the job registry and pipeline.

```python
import asyncio
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
import httpx
import torch

from escucha.config import Settings
from escucha.jobs import JobRegistry, JobStatus
from escucha.pipeline import PipelineRunner
from escucha.export import to_txt, to_srt

router = APIRouter()

# These are injected by main.py at startup
_settings: Settings = None
_registry: JobRegistry = None
_runner: PipelineRunner = None


def init_routes(settings: Settings, registry: JobRegistry, runner: PipelineRunner) -> None:
    """Inject dependencies into the routes module. Called once at startup."""
    global _settings, _registry, _runner
    _settings = settings
    _registry = registry
    _runner = runner


MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB


@router.get("/api/health")
async def health_check() -> dict:
    """Return system status for frontend warnings."""
    ffmpeg_ok = _settings.ffmpeg_path.exists() or shutil.which("ffmpeg") is not None
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{_settings.ollama_base_url}/api/version")
            ollama_ok = r.status_code == 200
    except Exception:
        pass
    return {
        "status": "ok",
        "ffmpeg": ffmpeg_ok,
        "ollama": ollama_ok,
        "cuda": torch.cuda.is_available(),
        "device": _settings.resolved_device,
        "whisper_model": _settings.whisper_model,
        "hf_token_set": _settings.hf_token is not None,
        "anthropic_key_set": _settings.anthropic_api_key is not None,
    }


@router.post("/api/jobs", status_code=201)
async def create_job(
    file: UploadFile = File(...),
    num_speakers: int | None = Form(default=None),
    language: str = Form(default="es"),
    summarize: bool = Form(default=True),
    use_claude: bool = Form(default=False),
) -> dict:
    """Accept an MP4 upload and start processing in the background."""
    # Validate file extension
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(400, "Only MP4 files are accepted.")

    # Create job (raises 503 if another job is active)
    try:
        job = _registry.create()
    except RuntimeError:
        raise HTTPException(503, "A job is already running. Please wait.")

    # Save uploaded file to disk
    upload_path = _settings.upload_dir / f"{job.job_id}.mp4"
    try:
        with open(upload_path, "wb") as f:
            while chunk := await file.read(8 * 1024 * 1024):  # 8 MB chunks
                f.write(chunk)
                if f.tell() > MAX_FILE_SIZE:
                    upload_path.unlink()
                    raise HTTPException(400, "File exceeds 2 GB limit.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to save upload: {e}")

    # Launch pipeline in background
    asyncio.create_task(
        _runner.run(
            job=job,
            input_path=upload_path,
            language=language,
            num_speakers=num_speakers,
            summarize=summarize,
            use_claude=use_claude,
        )
    )

    return {
        "job_id": job.job_id,
        "status": "queued",
        "message": f"Job created. Connect to /api/jobs/{job.job_id}/ws for progress.",
    }


@router.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict:
    """Poll job status (fallback for when WebSocket is unavailable)."""
    job = _registry.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "current_step": job.current_step.value,
        "step_detail": job.step_detail,
        "error": job.error,
    }


@router.websocket("/api/jobs/{job_id}/ws")
async def job_websocket(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for streaming progress updates."""
    job = _registry.get(job_id)
    if job is None:
        await websocket.close(code=4004, reason="Job not found")
        return

    await websocket.accept()
    _registry.add_listener(job, websocket)

    # If job already completed/failed before WS connected, send final state
    if job.status == JobStatus.COMPLETED:
        await websocket.send_json({"type": "completed", "progress": 100.0})
        await websocket.close()
        return
    if job.status == JobStatus.FAILED:
        await websocket.send_json({"type": "error", "message": job.error})
        await websocket.close()
        return

    # Keep connection open until the job finishes (broadcasts close it)
    try:
        while True:
            await websocket.receive_text()  # Blocks until client disconnects
    except WebSocketDisconnect:
        _registry.remove_listener(job, websocket)


@router.get("/api/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> dict:
    """Fetch the full result of a completed job."""
    job = _registry.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, "Job is not yet completed.")
    return {"job_id": job.job_id, **job.result}


@router.get("/api/jobs/{job_id}/export/{fmt}")
async def export_transcript(job_id: str, fmt: str) -> PlainTextResponse:
    """Download transcript as TXT or SRT."""
    if fmt not in ("txt", "srt"):
        raise HTTPException(400, "Format must be 'txt' or 'srt'.")

    job = _registry.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, "Job is not yet completed.")

    from escucha.models import DiarizedSegment
    segments = [DiarizedSegment(**s) for s in job.result["segments"]]

    if fmt == "txt":
        content = to_txt(segments)
    else:
        content = to_srt(segments)

    return PlainTextResponse(
        content=content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="transcription.{fmt}"'},
    )
```

---

### 5.11 `main.py` — Application entry point

**Responsibility:** Create the FastAPI app, load settings, initialize models, mount routes and static files.

```python
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from escucha.config import load_settings
from escucha.jobs import JobRegistry
from escucha.pipeline import PipelineRunner
from escucha.routes import router, init_routes

logger = logging.getLogger("escucha")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load settings, warm up models. Shutdown: cleanup."""
    settings = load_settings()
    registry = JobRegistry()
    runner = PipelineRunner(settings, registry)

    logger.info("Device: %s", settings.resolved_device)
    logger.info("Whisper model: %s (%s)", settings.whisper_model, settings.effective_compute_type)
    logger.info("HF token: %s", "set" if settings.hf_token else "NOT SET")

    await runner.warm_up()
    logger.info("Models loaded. Server ready.")

    init_routes(settings, registry, runner)

    # Store in app.state for access in tests
    app.state.settings = settings
    app.state.registry = registry
    app.state.runner = runner

    yield

    # Shutdown: nothing to clean up beyond GC
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="Escucha",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(router)

    # Serve the frontend
    static_dir = Path(__file__).resolve().parent.parent.parent / "static"

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

    # Serve any additional static assets (future-proof)
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


# Entry point: python -m escucha.main
if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    uvicorn.run(
        "escucha.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )
```

---

## 6. Data Contracts

All shared data structures defined in `src/escucha/models.py`:

```python
"""Data contracts for the Escucha pipeline.

All structures are dataclasses for simplicity and zero-dependency use
in internal pipeline code. Pydantic is used only at the API boundary
(FastAPI handles serialization automatically from dicts).
"""

from dataclasses import dataclass, field


@dataclass
class RawSegment:
    """A single transcript segment from faster-whisper, before speaker assignment.

    Attributes:
        start: Start time in seconds from beginning of audio.
        end: End time in seconds.
        text: Transcribed text for this segment.
        words: Optional word-level timestamps. Each entry is a dict with
               keys 'start' (float), 'end' (float), 'word' (str).
               Empty list if word timestamps were not generated.
    """
    start: float
    end: float
    text: str
    words: list[dict] = field(default_factory=list)


@dataclass
class DiarizationSegment:
    """A speaker turn identified by pyannote.audio.

    Attributes:
        speaker: Speaker label (e.g., 'SPEAKER_00', 'SPEAKER_01').
        start: Start time in seconds.
        end: End time in seconds.
    """
    speaker: str
    start: float
    end: float


@dataclass
class DiarizedSegment:
    """A transcript segment with an assigned speaker label.

    This is the primary output structure: it combines the text from
    RawSegment with the speaker from DiarizationSegment.

    Attributes:
        speaker: Assigned speaker label.
        start: Start time in seconds.
        end: End time in seconds.
        text: Transcribed text.
    """
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class JobState:
    """Serializable snapshot of a job's current state.

    Used by the GET /api/jobs/{job_id} polling endpoint.

    Attributes:
        job_id: Unique 8-character hex identifier.
        status: One of: 'queued', 'processing', 'completed', 'failed'.
        progress: Percentage from 0.0 to 100.0.
        current_step: Pipeline step name.
        step_detail: Human-readable detail string (e.g., 'Segment 5/40').
        error: Error message if status is 'failed', else None.
    """
    job_id: str
    status: str
    progress: float
    current_step: str
    step_detail: str = ""
    error: str | None = None
```

### Relationships

```
MP4 file
  │
  ▼ (audio.extract_audio)
WAV file
  │
  ├──▶ transcriber.transcribe() ──▶ list[RawSegment]
  │                                        │
  └──▶ diarizer.diarize() ──▶ list[DiarizationSegment]
                                           │
                    ┌──────────────────────┘
                    ▼
          merger.merge() ──▶ list[DiarizedSegment]
                                     │
                    ┌────────────────┘
                    ▼
          summarizer.summarize() ──▶ str (summary)
                    │
                    ▼
          export.to_txt() / to_srt() ──▶ str (file content)
```

---

## 7. Pipeline Flow

### Step-by-step execution with progress percentages

```
 0%   ┌─ Job created (status: queued)
      │
 5%   ├─ STEP 1: EXTRACT AUDIO
      │    Action: FFmpeg converts MP4 → 16kHz mono WAV
      │    Input:  /tmp/uploads/{job_id}.mp4
      │    Output: /tmp/uploads/{job_id}.wav
      │    Errors: AudioExtractionError → job FAILED
      │    Progress update: 5% "Converting MP4 to WAV"
      │
10%   ├─ STEP 2: TRANSCRIBE
      │    Action: faster-whisper processes WAV in ~30s chunks
      │    Input:  {job_id}.wav
      │    Output: list[RawSegment] accumulated in memory
      │    Progress: 10% → 60%, updated per segment
      │              Formula: 10 + (50 * segment_index / total_segments)
      │    Errors: TranscriptionError → job FAILED
      │
      │    ┌─ CPU vs GPU decision point ─────────────────────┐
      │    │ GPU (CUDA): ~5-10x real-time. 30 min ≈ 3-6 min │
      │    │ CPU (int8): ~1x real-time. 30 min ≈ 30 min     │
      │    │ CPU recommendation: use 'medium' model          │
      │    └─────────────────────────────────────────────────┘
      │
62%   ├─ STEP 3: DIARIZE
      │    Action: pyannote.audio identifies speaker turns
      │    Input:  {job_id}.wav (reads audio independently)
      │    Output: list[DiarizationSegment]
      │    Decision: if HF_TOKEN is not set → SKIP, assign all to SPEAKER_00
      │    Decision: if num_speakers provided → pass to pipeline as constraint
      │    Progress: 62% "Identifying speakers"
      │    Errors: DiarizationError → job FAILED
      │
      │    ┌─ CPU vs GPU decision point ─────────────────────┐
      │    │ GPU: 60 min audio ≈ 2 min                       │
      │    │ CPU: 60 min audio ≈ 10-15 min                   │
      │    └─────────────────────────────────────────────────┘
      │
75%   ├─ STEP 4: MERGE
      │    Action: Assign speaker labels to transcript segments
      │    Algorithm:
      │      1. For each RawSegment, if word timestamps exist:
      │         - Compute midpoint of each word
      │         - Find which DiarizationSegment contains each midpoint
      │         - Majority vote → speaker label
      │      2. Fallback (no word timestamps):
      │         - Find DiarizationSegment with largest time overlap
      │    Input:  list[RawSegment] + list[DiarizationSegment]
      │    Output: list[DiarizedSegment]
      │    Progress: 75% "Aligning speakers to transcript"
      │    Errors: None expected (pure computation, no I/O)
      │
80%   ├─ STEP 5: SUMMARIZE (optional, skippable)
      │    Decision: if summarize == false → SKIP
      │    Decision: if use_claude AND api_key set → Claude path
      │              else → Ollama path
      │
      │    Ollama path:
      │      1. Convert DiarizedSegments → plain text
      │      2. Split into chunks of ~12,000 chars
      │      3. Summarize each chunk (POST /api/generate)
      │      4. If multiple chunks: summarize the summaries
      │      Error: SummarizationError → NON-FATAL
      │             (transcript is still returned; summary = error message)
      │
      │    Claude path:
      │      1. Convert DiarizedSegments → plain text
      │      2. Single API call with full text (200k context)
      │      Error: SummarizationError → NON-FATAL
      │
      │    Progress: 80% "Generating summary"
      │
100%  ├─ STEP 6: COMPLETE
      │    Action: Build result dict, mark job completed
      │    Broadcast: {"type": "completed", "progress": 100.0}
      │    Close: All WebSocket listeners
      │    Cleanup: Delete {job_id}.wav (keep MP4 for re-processing? → delete it too)
      │
      └─ END
```

### Error recovery summary

| Error | Severity | Recovery |
|---|---|---|
| `AudioExtractionError` | Fatal | Job fails. User fixes input file and re-uploads. |
| `TranscriptionError` | Fatal | Job fails. Likely OOM — user switches to smaller model. |
| `DiarizationError` | Fatal | Job fails. Usually token issue — UI shows setup instructions. |
| `SummarizationError` | Non-fatal | Transcript is returned. Summary field contains error message. |
| WebSocket disconnect | Non-fatal | Job continues. Client can reconnect or poll `/api/jobs/{id}`. |

---

## 8. Windows Setup Guide

### Prerequisites

- Windows 10/11, 64-bit
- Python 3.10, 3.11, or 3.12 (3.10+ required by pyannote-audio)
- (Optional) NVIDIA GPU with CUDA support + drivers installed
- (Optional) Ollama for summarization

### Step-by-step

**Step 1 — Clone the repository**

```powershell
git clone <repo-url> escucha
cd escucha
```

**Step 2 — Create a virtual environment**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Step 3 — Install PyTorch**

Choose ONE of the following:

```powershell
# GPU (NVIDIA CUDA 12.8) — RECOMMENDED if you have an NVIDIA GPU
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# CPU only — slower but works on any machine
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

> **Why this must come first:** If you install `requirements.txt` first, pip will pull `torch` from the default PyPI index which ships CPU-only wheels. The `pyannote-audio` pin of `torch==2.8.0` will be satisfied, but without CUDA support. Always install torch with the correct index URL first.

**Step 4 — Install dependencies**

```powershell
pip install -r requirements.txt
```

**Step 5 — Set up FFmpeg**

Option A (recommended): Download FFmpeg and place it in the project:

```powershell
# Download from https://www.gyan.dev/ffmpeg/builds/
# Extract ffmpeg.exe to escucha\bin\ffmpeg.exe
mkdir bin
# Copy ffmpeg.exe into bin\
```

Option B: Install FFmpeg system-wide and ensure it's on PATH:

```powershell
# Using winget:
winget install Gyan.FFmpeg
# Or using chocolatey:
choco install ffmpeg
```

**Step 6 — Set up HuggingFace (required for speaker identification)**

1. Create a free account at https://huggingface.co/join
2. Go to https://huggingface.co/pyannote/speaker-diarization-3.1
3. Click "Agree and access repository" to accept the model terms
4. Create an access token at https://huggingface.co/settings/tokens
5. Copy `.env.example` to `.env` and set your token:

```powershell
copy .env.example .env
# Edit .env and set HF_TOKEN=hf_your_token_here
```

**Step 7 — Set up Ollama (optional, for summarization)**

1. Download and install from https://ollama.com/download/windows
2. Pull the summarization model:

```powershell
ollama pull llama3.2
```

3. Ollama runs as a background service automatically on Windows.

**Step 8 — Launch the application**

```powershell
python -m escucha.main
```

Open your browser to http://127.0.0.1:8000.

On first launch, models will be downloaded (~3 GB for Whisper large-v3, ~1 GB for pyannote). This only happens once; subsequent launches are fast.

### Expected processing times

| Audio length | GPU (RTX 3060+) | CPU only (i7/Ryzen 7) |
|---|---|---|
| 30 minutes | ~5–8 min total | ~40–60 min total |
| 60 minutes | ~10–15 min total | ~80–120 min total |

GPU times assume `large-v3` + INT8. CPU times assume `medium` + INT8.

---

## 9. Test Plan

### 9.1 Test infrastructure (`tests/conftest.py`)

```python
import asyncio
import tempfile
from pathlib import Path
from dataclasses import dataclass
import pytest
import pytest_asyncio
from escucha.models import RawSegment, DiarizationSegment, DiarizedSegment


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


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
```

### 9.2 Unit tests

#### `test_audio.py`

| Test case | Description |
|---|---|
| `test_extract_audio_produces_wav` | Create a tiny valid MP4 (using FFmpeg to generate a 1-second sine tone video). Call `extract_audio()`. Assert output WAV exists and is > 0 bytes. |
| `test_extract_audio_file_not_found` | Pass a non-existent path. Assert `FileNotFoundError` is raised. |
| `test_extract_audio_invalid_file` | Pass a text file renamed to `.mp4`. Assert `AudioExtractionError` is raised with FFmpeg error message. |
| `test_extract_audio_creates_no_window` | (Windows only) Assert that `subprocess.CREATE_NO_WINDOW` is used in the subprocess call (mock `subprocess.run` and inspect `creationflags`). |

#### `test_transcriber.py`

| Test case | Description |
|---|---|
| `test_transcribe_returns_segments` | Load `tiny` model (fast, ~75 MB). Transcribe a short WAV file (generate 5-second silence with FFmpeg). Assert at least 0 segments returned (silence may produce 0 with VAD). |
| `test_transcribe_spanish_text` | Load `tiny` model. Transcribe a pre-recorded 10-second Spanish WAV fixture. Assert returned text contains at least one Spanish word. |
| `test_transcribe_invalid_audio_raises` | Pass a non-audio file. Assert `TranscriptionError`. |
| `test_load_model_invalid_name_raises` | Call `load_whisper_model("nonexistent")`. Assert `TranscriptionError`. |

> **Note:** Transcription tests use the `tiny` model to keep CI fast (~2s per test). Integration tests may use `small` for higher fidelity.

#### `test_diarizer.py`

| Test case | Description |
|---|---|
| `test_diarize_returns_segments` | Mock the pyannote pipeline to return 3 fixed speaker turns. Call `diarize()`. Assert 3 `DiarizationSegment` returned with correct speaker labels. |
| `test_diarize_with_num_speakers` | Mock pipeline. Call `diarize(num_speakers=2)`. Assert `pipeline()` was called with `num_speakers=2`. |
| `test_load_pipeline_invalid_token` | Call `load_diarization_pipeline(hf_token="invalid")`. Assert `DiarizationError` is raised. |

> **Note:** Diarization tests mock the pyannote pipeline because model download (~1 GB) is too slow for CI. One integration test (below) runs the real pipeline.

#### `test_merger.py`

| Test case | Description |
|---|---|
| `test_merge_assigns_correct_speakers` | Using fixtures: merge 3 raw segments + 3 diarization segments. Assert each `DiarizedSegment.speaker` matches expected value. |
| `test_merge_word_level_alignment` | Create a raw segment spanning two speaker turns (split at word level). Assert the speaker is assigned by majority vote of word midpoints. |
| `test_merge_no_diarization_overlap` | Create a raw segment with timestamps that don't overlap any diarization segment. Assert speaker is `"UNKNOWN"`. |
| `test_merge_empty_inputs` | Pass empty lists. Assert empty list returned. |

#### `test_summarizer.py`

| Test case | Description |
|---|---|
| `test_summarize_ollama_success` | Mock `httpx.AsyncClient.post` to return a valid Ollama response. Call `summarize_with_ollama()`. Assert non-empty summary returned. |
| `test_summarize_ollama_connection_error` | Mock `httpx.AsyncClient.post` to raise `ConnectError`. Assert `SummarizationError` with "Cannot connect to Ollama" message. |
| `test_summarize_ollama_chunking` | Create input with >12,000 chars. Assert that the mock receives multiple generate calls (chunk summaries + final summary). |
| `test_summarize_claude_success` | Mock `anthropic.Anthropic.messages.create`. Assert non-empty summary. |
| `test_summarize_claude_missing_package` | Temporarily make `anthropic` unimportable. Assert `SummarizationError`. |

#### `test_export.py`

| Test case | Description |
|---|---|
| `test_to_txt_format` | Using fixture segments, call `to_txt()`. Assert output matches expected `[HH:MM:SS] SPEAKER: text` format line by line. |
| `test_to_srt_format` | Using fixture segments, call `to_srt()`. Assert output matches SRT format with sequence numbers and `-->` arrows. |
| `test_to_srt_timestamps` | Assert timestamp `4.52` formats as `00:00:04,520`. |
| `test_to_txt_empty` | Pass empty list. Assert empty string. |

### 9.3 Integration tests (`test_api.py`)

| Test case | Description |
|---|---|
| `test_health_endpoint` | Start the app with `TestClient`. `GET /api/health`. Assert 200 and all expected keys present. |
| `test_upload_non_mp4_rejected` | `POST /api/jobs` with a `.txt` file. Assert 400. |
| `test_full_pipeline_e2e` | Generate a 5-second MP4 with FFmpeg (sine tone + blank video). `POST /api/jobs`. Connect WebSocket. Assert progress messages arrive. Assert final `completed` message. `GET /api/jobs/{id}/result`. Assert `segments` is a non-empty list. `GET /api/jobs/{id}/export/txt`. Assert non-empty text. `GET /api/jobs/{id}/export/srt`. Assert non-empty SRT. Uses `tiny` model and mocked diarization + summarization. |
| `test_concurrent_job_rejected` | Start a job (mock pipeline to sleep 5s). Immediately POST another job. Assert 503. |

### 9.4 UI Smoke Test (manual checklist)

Run the app locally and verify each item in a browser:

- [ ] Page loads at `http://127.0.0.1:8000` without console errors
- [ ] Health warnings appear correctly (disconnect Ollama → warning shown)
- [ ] Drag-and-drop an MP4 file onto the upload zone → filename displayed
- [ ] Click "Browse" → file picker opens, selecting a file updates the zone
- [ ] Reject non-MP4 file → error message displayed inline
- [ ] Click "Start Transcription" → progress bar appears and animates
- [ ] Progress steps update in sequence: Extracting → Transcribing → Diarizing → Summarizing
- [ ] On completion: summary panel appears (collapsible, initially open)
- [ ] Transcript panel shows color-coded speaker labels
- [ ] Each segment shows a timestamp, speaker name, and text
- [ ] Click "Download TXT" → browser downloads a `.txt` file with correct content
- [ ] Click "Download SRT" → browser downloads an `.srt` file with correct format
- [ ] If WebSocket disconnects mid-job → "Connection lost" message, polling fallback works
- [ ] Uploading a second file while a job is running → 503 error displayed

---

*End of specification. A developer reading this document has everything needed to implement Escucha without further clarification.*
