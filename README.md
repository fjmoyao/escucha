# Escucha

Transcribe, identify speakers, and summarize Spanish MP4 recordings — fully local, runs on Windows, no cloud required.

Upload a video through the browser, watch it process in real time, and get a timestamped transcript with speaker labels saved to disk.

![Pipeline](https://img.shields.io/badge/Whisper-faster--whisper-blue) ![Diarization](https://img.shields.io/badge/Diarization-pyannote%204.0-green) ![Backend](https://img.shields.io/badge/Backend-FastAPI-teal)

---

## What it does

```
MP4 video
   │
   ├─ FFmpeg extracts 16 kHz mono WAV
   ├─ Whisper transcribes speech → timestamped text segments
   ├─ pyannote identifies who speaks when → speaker labels
   ├─ Merger aligns text + speakers
   ├─ Ollama / Claude generates a structured summary (optional)
   │
   └─ output/YYYYMMDD_HHMMSS_{id}_transcript.txt   ← saved to disk
      output/YYYYMMDD_HHMMSS_{id}_summary.txt       ← saved to disk
```

**Key features**

- Real-time progress in the browser via WebSocket
- Up to 6 speakers identified automatically
- Summary in Spanish (bullet points + decisions + action items)
- Download transcript as TXT or SRT subtitles
- GPU (CUDA) and CPU-only modes
- All processing is local — audio never leaves your machine

---

## Requirements

| Requirement | Notes |
|---|---|
| Windows 10/11 64-bit | |
| Python 3.10 – 3.14 | [python.org](https://www.python.org/downloads/) |
| NVIDIA GPU + CUDA drivers | Optional but strongly recommended |
| FFmpeg full-shared build | See Step 4 below |
| HuggingFace account + token | Free — needed for speaker identification |
| Ollama (optional) | For local summarization |

---

## Setup

### Step 1 — Clone the repository

```powershell
git clone <repo-url> escucha
cd escucha
```

### Step 2 — Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 3 — Install PyTorch

Choose **one** of the following depending on your hardware:

```powershell
# GPU (NVIDIA with CUDA 12.8) — recommended
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# CPU only — slower but works on any machine
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

> **Why first?** If you run `pip install -r requirements.txt` before this step, pip pulls CPU-only PyTorch wheels from PyPI even with a GPU available. Always install torch with the correct index URL first.

### Step 4 — Install FFmpeg (full-shared build)

Speaker diarization (pyannote) requires FFmpeg's shared libraries (DLLs), not just the executable.

```powershell
# Install with winget — adds DLLs to PATH automatically
winget install -e --id Gyan.FFmpeg.Shared
```

Then **open a new terminal** so the updated PATH takes effect.

Verify:
```powershell
ffmpeg -version   # should show "full_build-shared" in the first line
```

> If you skip this step, transcription still works but speaker identification will fail silently and all segments will be labeled `SPEAKER_00`.

### Step 5 — Install Python dependencies

```powershell
pip install -r requirements.txt --prefer-binary
pip install -e .
```

### Step 6 — Set up HuggingFace (for speaker identification)

1. Create a free account at [huggingface.co](https://huggingface.co/join)
2. Accept the terms for these two models (required):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
3. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Step 7 — Configure environment

```powershell
copy .env.example .env
```

Open `.env` and fill in your values:

```bash
# Required for speaker identification
HF_TOKEN=hf_your_token_here

# Model size: tiny/base/small/medium/large-v3
# GPU (>=8GB VRAM): large-v3
# CPU only: small or medium
WHISPER_MODEL=large-v3
```

### Step 8 — Install Ollama (optional, for summarization)

Download from [ollama.com/download/windows](https://ollama.com/download/windows), then pull a model:

```powershell
ollama pull llama3.2
```

Ollama runs as a background service automatically on Windows. Skip this step if you plan to use the Claude API instead.

### Step 9 — Launch

```powershell
python -m escucha.main
```

Open your browser at **http://127.0.0.1:8000**

On first launch, Whisper and pyannote models download automatically (~4 GB total). This only happens once.

---

## Usage

1. Drag an MP4 file onto the upload zone (or click Browse)
2. Choose options:
   - **Hablantes** — number of speakers (auto-detect or 2–6)
   - **Idioma** — Auto, Español, or English
   - **Generar resumen** — enable/disable summarization
   - **Usar Claude** — uses Claude API instead of Ollama (requires `ANTHROPIC_API_KEY` in `.env`)
3. Click **Iniciar Transcripción**
4. Watch the progress bar advance through each stage
5. When done, the transcript and summary appear in the browser
6. Click **Descargar TXT** or **Descargar SRT** to save the transcript
7. Files are also automatically saved to the `output/` folder

---

## Output files

Every completed job writes two files to `output/`:

```
output/
  20260428_195640_b6742983_transcript.txt
  20260428_195640_b6742983_summary.txt
```

**Transcript format (TXT):**
```
[00:00:00] SPEAKER_00: Buenos días, vamos a comenzar la reunión.
[00:00:04] SPEAKER_01: Perfecto, tengo los números del trimestre.
```

**SRT format** (for video players / subtitle editors):
```
1
00:00:00,000 --> 00:00:04,520
[SPEAKER_00] Buenos días, vamos a comenzar la reunión.
```

Files are named by date, time, and job ID — nothing is ever overwritten.

---

## Expected processing times

| Audio length | GPU (RTX 3060+) | CPU only (i7/Ryzen 7) |
|---|---|---|
| 30 minutes | ~5–8 min | ~40–60 min |
| 60 minutes | ~10–15 min | ~80–120 min |

GPU times use `large-v3` + INT8. CPU times use `medium` + INT8.

---

## Configuration reference

All settings are in `.env`. See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace token (required for diarization) |
| `WHISPER_MODEL` | `large-v3` | Model size: tiny, base, small, medium, large-v3 |
| `WHISPER_COMPUTE_TYPE` | `int8` | int8 (fast), float16 (quality), float32 (CPU safe) |
| `DEVICE` | `auto` | auto, cuda, or cpu |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model for summarization |
| `ANTHROPIC_API_KEY` | — | Optional — enables Claude summarization toggle in UI |
| `HOST` | `127.0.0.1` | Server bind address |
| `PORT` | `8000` | Server port |

---

## Project structure

```
escucha/
├── src/escucha/
│   ├── audio.py          # FFmpeg extraction
│   ├── transcriber.py    # faster-whisper
│   ├── diarizer.py       # pyannote speaker ID
│   ├── merger.py         # align text + speakers
│   ├── summarizer.py     # Ollama / Claude
│   ├── export.py         # TXT and SRT formatters
│   ├── pipeline.py       # orchestrates everything
│   ├── jobs.py           # job state + WebSocket broadcast
│   ├── routes.py         # FastAPI endpoints
│   ├── config.py         # settings from .env
│   └── main.py           # app factory + entry point
├── static/
│   └── index.html        # single-file frontend
├── tests/                # 38 unit + integration tests
├── output/               # transcripts saved here (gitignored)
├── bin/                  # optional: bundled ffmpeg.exe
├── .env.example          # config template
└── requirements.txt
```

---

## Running tests

```powershell
.venv\Scripts\activate
pytest --tb=short -q
```

The first run downloads the `tiny` Whisper model (~75 MB). All 38 tests should pass. Diarization tests are fully mocked and require no internet.

---

## Troubleshooting

**Speaker labels all show `SPEAKER_00`**
FFmpeg full-shared DLLs are not on PATH. Run `winget install -e --id Gyan.FFmpeg.Shared` and open a new terminal.

**No speech detected / empty transcript**
Try changing the language dropdown to **Auto** instead of Español. The forced-language mode can fail on recordings with background noise or atypical accents.

**`pydantic-core` fails to install**
On Python 3.14, add `--prefer-binary` to the pip command. A pre-built wheel exists for Python 3.14 starting from pydantic 2.13.3.

**Port 8000 already in use**
Another server process is running. Kill it: `Get-Process python | Stop-Process -Force` or change `PORT=8001` in `.env`.

**Summarization says "Ollama is not running"**
Start Ollama: `ollama serve`. Or set `ANTHROPIC_API_KEY` in `.env` and use the Claude toggle in the UI.

**Models take long to load at startup**
First launch downloads ~4 GB. Subsequent launches use the local HuggingFace cache and take ~20 seconds on GPU.
