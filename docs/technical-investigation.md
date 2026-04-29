# Escucha — Technical Investigation Report

> **Date:** 2026-04-24
> **Scope:** Local Python CLI + web frontend for transcribing Spanish MP4 recordings on Windows
> **Constraints:** Free/open-source only, 30–60 min recordings, speaker diarization + AI summary

---

## 1. Transcription Engine Evaluation

### 1.1 Whisper (openai-whisper)

| Criterion | Assessment |
|---|---|
| **Spanish quality** | Excellent. Large-v3 achieves **3–8 % WER** on Spanish benchmarks (Common Voice). 10–20 % error reduction vs v2 on non-English. |
| **Windows compat** | Good. `pip install openai-whisper` works; requires FFmpeg on PATH. CUDA support via PyTorch wheels (cu121). |
| **Speed CPU / GPU** | Slow on CPU (~1× real-time for `large-v3`). GPU (RTX 3060+): ~5–10× real-time. |
| **Maintenance** | Active repo (openai/whisper), but update cadence has slowed since v3 release. |
| **Memory (60 min)** | `large-v3`: ~10 GB VRAM (GPU) or ~10 GB RAM (CPU). `medium`: ~5 GB. `small`: ~2 GB. |

### 1.2 faster-whisper (CTranslate2)

| Criterion | Assessment |
|---|---|
| **Spanish quality** | Identical to openai-whisper — same model weights, same WER. |
| **Windows compat** | Good. `pip install faster-whisper`. CTranslate2 ships Windows binaries. CUDA 12.x supported. |
| **Speed CPU / GPU** | **Up to 4× faster** than openai-whisper at equal accuracy. INT8 quantization gives additional ~40 % memory savings with minimal accuracy loss. |
| **Maintenance** | Active (SYSTRAN/faster-whisper). Strong community. Regular releases through 2025–2026. |
| **Memory (60 min)** | `large-v3` INT8: ~6 GB VRAM. `medium` INT8: ~3 GB. Processes audio in 30-second chunks, so file length has limited impact on peak RAM beyond model size. |

### 1.3 WhisperX

| Criterion | Assessment |
|---|---|
| **Spanish quality** | Same Whisper models underneath. Adds **word-level timestamps** via wav2vec2 alignment and built-in **speaker diarization** via pyannote. 88–93 % word accuracy on clean audio; 74–83 % on noisy spontaneous speech. |
| **Windows compat** | **Problematic.** Multiple open issues: CUDA wheel platform detection fails on Windows (`AMD64 != x86_64`, issue #1357); cuDNN version mismatches with v3.4.2 (issue #1216). Requires manual dependency pinning. |
| **Speed CPU / GPU** | Uses faster-whisper internally, so transcription speed is similar. Alignment + diarization add overhead (~20–40 % on top). |
| **Maintenance** | Active repo (m-bain/whisperX) but Windows-specific bugs linger. Maintainer bandwidth appears limited for platform-specific fixes. |
| **Memory (60 min)** | Higher than faster-whisper alone: loads Whisper + wav2vec2 + pyannote models simultaneously. Expect ~8–12 GB VRAM for the full pipeline with `large-v3`. |

### 1.4 Vosk

| Criterion | Assessment |
|---|---|
| **Spanish quality** | Moderate. Spanish model available. Adequate for clear single-speaker audio but **noticeably worse** than Whisper on noisy, multi-speaker, or accented speech. No published WER directly comparable to Whisper benchmarks, but community consensus places it well below Whisper-class models. |
| **Windows compat** | Excellent. Pure offline, lightweight. `pip install vosk` + download model. No CUDA dependency. |
| **Speed CPU / GPU** | Very fast on CPU (~10× real-time). CPU-only — no GPU acceleration. |
| **Maintenance** | Active (alphacephei/vosk). 20+ language models. Steady updates. |
| **Memory (60 min)** | Very low: 50–300 MB depending on model size. Best option for constrained hardware. |

### 1.5 DeepSpeech

| Criterion | Assessment |
|---|---|
| **Spanish quality** | Poor. Primarily English-trained. Community Spanish models exist but are low quality. |
| **Windows compat** | Archived. Builds may fail on modern Python/Windows. |
| **Speed CPU / GPU** | N/A — project is dead. |
| **Maintenance** | **Discontinued** by Mozilla in November 2021. Repository archived. **Not recommended.** |
| **Memory (60 min)** | N/A |

### 1.6 Transcription Verdict

**faster-whisper is the clear winner.** Same accuracy as openai-whisper, 4× faster, lower memory with INT8 quantization, and solid Windows support. WhisperX adds integrated diarization but introduces significant Windows installation friction — it is better to compose faster-whisper + pyannote separately. Vosk is a viable fallback for CPU-only machines with limited RAM but cannot match Whisper quality on Spanish. DeepSpeech is dead.

---

## 2. Audio Extraction (MP4 → WAV/PCM)

### 2.1 FFmpeg (via subprocess)

| Aspect | Detail |
|---|---|
| **Approach** | Call `ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav` via `subprocess.run()` |
| **Pros** | Gold standard. Handles every codec. 16 kHz mono WAV is exactly what Whisper expects. Near-zero overhead — remuxes, doesn't transcode if input is already PCM. |
| **Cons** | Requires FFmpeg binary on PATH. On Windows, users must install FFmpeg separately (or bundle it). |
| **Windows** | Pre-built binaries widely available (gyan.dev, BtbN builds). Can be bundled in the project's `bin/` directory. |

### 2.2 MoviePy

| Aspect | Detail |
|---|---|
| **Approach** | `VideoFileClip("input.mp4").audio.write_audiofile("output.wav")` |
| **Pros** | Pythonic API. Auto-downloads FFmpeg on first use. |
| **Cons** | Heavy dependency chain (numpy, imageio, decorator). Loads the video stream unnecessarily for audio-only extraction. Slower than raw FFmpeg. Overkill for this use case. |

### 2.3 PyAV

| Aspect | Detail |
|---|---|
| **Approach** | Python bindings to FFmpeg's C libraries. Direct packet-level access. |
| **Pros** | Faster than MoviePy. No subprocess overhead. Wheels include FFmpeg, so no external binary needed. |
| **Cons** | More complex API for a simple extraction task. Less community knowledge for simple audio extraction recipes. |

### 2.4 Audio Extraction Verdict

**FFmpeg via subprocess** is recommended. It's the simplest, fastest, and most battle-tested approach for this single-purpose task. Bundle the FFmpeg binary inside the project (e.g., `bin/ffmpeg.exe`) to eliminate user installation friction. PyAV is a reasonable alternative if you want to avoid shipping a binary — it bundles FFmpeg in its wheels.

> **Windows Gotcha:** Always use the full path to ffmpeg.exe or ensure it's on PATH. Use `subprocess.run()` with `creationflags=subprocess.CREATE_NO_WINDOW` on Windows to suppress console popups.

---

## 3. Speaker Diarization Evaluation

### 3.1 pyannote.audio

| Criterion | Assessment |
|---|---|
| **Spanish quality** | Best-in-class. DER (Diarization Error Rate) of **~14.3 %** on Spanish benchmarks. Stable performance from 2 to 5+ speakers (DER 6.6–9.9 % on English benchmarks). |
| **Windows CUDA/CPU** | Works on Windows. v3.1 removed onnxruntime dependency — pure PyTorch. Supports CUDA and CPU. |
| **Integration** | Outputs speaker-labeled time segments. Easy to align with faster-whisper transcript segments by timestamp overlap. WhisperX uses pyannote internally, validating this pairing. |
| **License** | **MIT license** for the open-source library and community models. HuggingFace access token required to download models (free account, must accept model terms on HF). |
| **Version note** | `speaker-diarization-3.1` is the recommended pipeline. The newer `speaker-diarization-community-1` (pyannote.audio 4.0) is also MIT and may offer improvements on noisy audio. |

### 3.2 Resemblyzer

| Criterion | Assessment |
|---|---|
| **Spanish quality** | Moderate. Language-agnostic speaker embeddings (d-vector), but clustering is manual and less accurate than pyannote's full pipeline. |
| **Windows CUDA/CPU** | CPU-friendly. PyTorch-based. No special Windows issues. |
| **Integration** | Provides embeddings only — you must implement your own clustering (spectral or agglomerative). More work to integrate. |
| **License** | Apache 2.0. |
| **Maintenance** | Low activity. Side project from Real-Time Voice Cloning repo. Not production-grade. |

### 3.3 simple-diarizer

| Criterion | Assessment |
|---|---|
| **Spanish quality** | Uses SpeechBrain ECAPA-TDNN embeddings + clustering. Reasonable quality but no published Spanish benchmarks. |
| **Windows CUDA/CPU** | Works on Windows. SpeechBrain dependency is heavy. |
| **Integration** | Simple API. Returns speaker labels per segment. |
| **License** | MIT. |
| **Maintenance** | **Effectively abandoned.** No releases in 12+ months. Not recommended for new projects. |

### 3.4 NVIDIA NeMo

| Criterion | Assessment |
|---|---|
| **Spanish quality** | Competitive with pyannote. Sortformer architecture is state-of-the-art on some benchmarks. |
| **Windows CUDA/CPU** | **Poor Windows support.** NeMo is primarily Linux-oriented. Installation on Windows is painful — many users resort to WSL. Heavy dependency tree (Cython, numba, librosa, kaldi). GPU strongly recommended. |
| **Integration** | Full pipeline but tightly coupled to NeMo's ASR ecosystem. Overkill unless also using NeMo for transcription. |
| **License** | Apache 2.0. |
| **Maintenance** | Very active (NVIDIA-backed). But the project's scope and complexity are far beyond what this tool needs. |

### 3.5 Diarization Verdict

**pyannote.audio 3.1** (or the newer `community-1` model) is the clear winner. Best Spanish DER, MIT-licensed, pure PyTorch (no onnxruntime), works on Windows, and already proven to integrate with Whisper-family engines. The only friction is needing a free HuggingFace account to download models.

> **Windows Gotcha:** pyannote downloads models to `~/.cache/huggingface/` on first run. Ensure this path is writable and that the HF token is set via `huggingface-cli login` or `HF_TOKEN` env var.

---

## 4. Summarization

### 4.1 Local LLM via Ollama (Llama 3.2 / Mistral)

| Criterion | Assessment |
|---|---|
| **Quality** | Llama 3.2 (8B) and Mistral 7B produce decent summaries of Spanish text. Llama 3.2 officially supports Spanish. Quality is serviceable for meeting notes but noticeably below Claude/GPT-4 class for nuance, inference, and structure. |
| **Offline** | Fully offline. No API calls, no data leaves the machine. |
| **Hardware** | 8B model: ~6–8 GB RAM/VRAM. Runs on CPU (slow, ~1–3 tokens/sec on modern i7) or GPU (15–30 tokens/sec on RTX 3060). A 60-min transcript (~10k–15k words) may need chunked summarization to fit context window (8k tokens for base Llama 3.2). |
| **Windows** | Ollama has a native Windows installer. Very easy setup. |
| **Integration** | REST API at `localhost:11434`. Simple HTTP calls from Python. |

### 4.2 Anthropic API (Claude)

| Criterion | Assessment |
|---|---|
| **Quality** | **Significantly higher quality** summaries. Claude excels at structured output, nuanced understanding, and Spanish language processing. Can handle full 60-min transcripts in a single context window (200k tokens). |
| **Offline** | No — requires internet and API key. **Violates the "free/no paid APIs" constraint** unless the user has existing API credits. |
| **Hardware** | No local compute needed. |
| **Integration** | `anthropic` Python SDK. Simple and well-documented. |

### 4.3 Summarization Verdict

**Recommended: Ollama + Llama 3.2 (8B) as default, with optional Claude API integration.**

- Primary path: Ollama for fully free, offline summarization. Use Llama 3.2 8B (or Llama 3.1 8B for longer context if needed). Chunk transcripts into ~4k-token segments, summarize each, then produce a final summary of summaries.
- Optional premium path: Allow users to configure an Anthropic API key in settings for higher-quality summaries via Claude. This keeps the tool functional without any paid service while offering an upgrade path.

> **Windows Gotcha:** Ollama runs as a background service on Windows. Ensure port 11434 is available. The first model download (~4.7 GB for Llama 3.2 8B) requires patience.

---

## 5. Local Web Server Options

### 5.1 FastAPI

| Criterion | Assessment |
|---|---|
| **Static files + uploads** | Built-in `StaticFiles` mount for HTML/CSS/JS. `UploadFile` type for multipart file uploads with streaming support. |
| **WebSocket / SSE** | **Native WebSocket support** (`@app.websocket`). SSE via `StreamingResponse`. No extensions needed. Handles ~3,200 concurrent WebSocket connections per instance. |
| **Windows** | Runs on Windows via Uvicorn. `pip install fastapi uvicorn`. No issues. |
| **Async** | Fully async (ASGI). Long-running transcription jobs can run in background tasks without blocking the server. |
| **Ecosystem** | Automatic OpenAPI docs, type validation via Pydantic, modern Python typing. |

### 5.2 Flask

| Criterion | Assessment |
|---|---|
| **Static files + uploads** | Built-in static file serving and `request.files` for uploads. Mature and well-documented. |
| **WebSocket / SSE** | Requires **Flask-SocketIO** extension for WebSockets (~2,100 concurrent connections with eventlet). SSE possible but clunky without async. |
| **Windows** | Works fine. `pip install flask`. |
| **Async** | WSGI — synchronous by default. Long-running tasks block the request unless you add Celery, threading, or subprocess management. Async routes added in Flask 2.0+ but still limited vs FastAPI. |
| **Ecosystem** | Largest community, most tutorials. Every question has a Stack Overflow answer. |

### 5.3 Bottle

| Criterion | Assessment |
|---|---|
| **Static files + uploads** | Basic support. Single-file framework — minimal but functional. |
| **WebSocket / SSE** | No built-in support. Requires gevent-websocket or similar. Poor ecosystem for real-time features. |
| **Windows** | Works fine. Zero dependencies. |
| **Async** | None. Synchronous only. |
| **Ecosystem** | Small community. Not recommended for applications needing real-time progress updates. |

### 5.4 Web Server Verdict

**FastAPI** is the clear recommendation. Native WebSocket/SSE support is essential for streaming transcription progress to the UI (e.g., "Extracting audio... Transcribing segment 12/47... Diarizing..."). Async architecture means the server stays responsive while heavy transcription jobs run in the background. The learning curve vs Flask is minimal, and the developer experience (auto-docs, type safety) is superior.

> **Windows Gotcha:** Use `uvicorn app:app --host 127.0.0.1 --port 8000` to serve. For production-like local use, `uvicorn` works well. Avoid `--reload` in production as file watchers can be flaky on Windows NTFS.

---

## 6. Recommended Stack

### Summary Table

| Layer | Choice | Rationale |
|---|---|---|
| **Audio Extraction** | FFmpeg via `subprocess` (bundled binary) | Fastest, simplest, universal codec support. Bundle `ffmpeg.exe` to remove user installation friction. |
| **Transcription** | **faster-whisper** (`large-v3`, INT8) | 4× faster than openai-whisper, same 3–8 % WER on Spanish, ~6 GB VRAM with INT8. Best speed/accuracy/memory balance. |
| **Diarization** | **pyannote.audio 3.1** | Best Spanish DER (~14.3 %), MIT license, pure PyTorch, proven Whisper integration. |
| **Summarization** | **Ollama + Llama 3.2 8B** (default) / Claude API (optional) | Free and offline by default. Optional Claude upgrade for users with API keys. |
| **Web Server** | **FastAPI + Uvicorn** | Native WebSocket/SSE for progress streaming, async background tasks, modern Python. |
| **Frontend** | Static HTML/CSS/JS (served by FastAPI) | No build tools needed. Vanilla JS + fetch/WebSocket API. |

### Risk / Tradeoff Table

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| **GPU not available** — user has no NVIDIA GPU | Transcription speed drops to ~1× real-time (30–60 min per file). Diarization slows similarly. | Medium | Support CPU mode with `medium` or `small` model. Document expected times. Consider Vosk as a fast-CPU fallback (lower quality). |
| **pyannote HuggingFace token requirement** | Users must create a free HF account and accept model terms. Adds onboarding friction. | High | Provide step-by-step setup guide. Auto-detect missing token and link to setup instructions in the UI. |
| **CUDA version conflicts** | PyTorch, faster-whisper, and pyannote may require different CUDA versions. Dependency hell on Windows. | Medium-High | Pin all dependencies to a tested combination. Ship a `requirements.txt` with exact versions. Consider a Docker/conda environment. |
| **Ollama not installed** — summarization fails | User must install Ollama separately and download a model. | Medium | Detect Ollama availability at startup. Show clear instructions in UI. Offer "skip summarization" option. |
| **Long transcripts exceed LLM context** | 60-min transcript (~15k words) may exceed Llama 3.2's 8k-token context. | High | Implement chunked summarization: summarize each 10-min segment, then summarize the summaries. |
| **WhisperX Windows bugs** (if chosen) | Known platform detection and cuDNN issues on Windows. | High | **Avoided by using faster-whisper + pyannote directly** instead of WhisperX. |
| **FFmpeg not found on PATH** | Audio extraction fails silently or with cryptic error. | Medium | Bundle `ffmpeg.exe` in `bin/`. Check for FFmpeg at startup with clear error messaging. |

### Architecture Diagram (Conceptual)

```
┌─────────────────────────────────────────────────┐
│  Browser (localhost:8000)                       │
│  ┌───────────────────────────────────────────┐  │
│  │  Upload MP4  │  Progress Bar  │  Results  │  │
│  └──────────────┼────────────────┼───────────┘  │
│                 │ WebSocket/SSE  │               │
└─────────────────┼────────────────┼───────────────┘
                  │                │
┌─────────────────┼────────────────┼───────────────┐
│  FastAPI + Uvicorn (Python)                      │
│                                                  │
│  1. Upload MP4 → temp dir                        │
│  2. FFmpeg → extract audio (16kHz mono WAV)      │
│  3. faster-whisper → transcribe (segments + ts)  │
│  4. pyannote.audio → diarize (speaker labels)    │
│  5. Merge: assign speakers to transcript segments│
│  6. Ollama/Claude → summarize diarized text      │
│  7. Return results (JSON → UI renders)           │
└──────────────────────────────────────────────────┘
```

---

## 7. Open Technical Questions for Stage 2

Prioritized from most to least critical:

1. **Dependency matrix validation** — What exact combination of `torch`, `torchaudio`, `faster-whisper`, `pyannote-audio`, and CUDA toolkit versions work together on Windows with both GPU and CPU? This needs a concrete, tested `requirements.txt`.

2. **Diarization ↔ transcription alignment strategy** — What is the optimal method to merge pyannote speaker segments with faster-whisper transcript segments? Options: (a) assign speaker by majority overlap per segment, (b) use word-level timestamps from faster-whisper for finer alignment, (c) use WhisperX-style forced alignment as a separate step.

3. **Chunked summarization design** — How to chunk a 60-min diarized transcript for Llama 3.2 8B's context window? Options: (a) fixed token count, (b) by speaker turn boundaries, (c) by topic shift detection. What prompt template produces the best Spanish meeting summaries?

4. **Progress reporting granularity** — What events does faster-whisper emit during transcription that can be forwarded to the UI? Does pyannote provide progress callbacks, or must we estimate based on audio duration?

5. **File size limits and temp storage** — 60-min MP4 files can be 500 MB–2 GB. What are the upload size limits for FastAPI/Uvicorn by default, and how should temp file cleanup be managed?

6. **Model download UX** — faster-whisper and pyannote models are downloaded on first run (~3–6 GB total). How to handle this in the UI? Progress bar? Pre-download script?

7. **Concurrent jobs** — Should the tool support processing multiple files simultaneously, or enforce a queue? GPU memory constraints likely force single-job processing.

8. **Output format** — What format for the final diarized transcript? Options: plain text, SRT/VTT subtitles, structured JSON, or all of the above?

---

## Sources

- [Best open source STT in 2026 — Northflank](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
- [WhisperX vs Competitors 2026 — BrassTranscripts](https://brasstranscripts.com/blog/whisperx-vs-competitors-accuracy-benchmark)
- [faster-whisper — GitHub](https://github.com/SYSTRAN/faster-whisper)
- [openai/whisper — GitHub](https://github.com/openai/whisper)
- [Whisper large-v3 — HuggingFace](https://huggingface.co/openai/whisper-large-v3)
- [Pyannote vs NeMo comparison — La Javaness R&D](https://lajavaness.medium.com/comparing-state-of-the-art-speaker-diarization-frameworks-pyannote-vs-nemo-31a191c6300)
- [Best Speaker Diarization Models 2026 — BrassTranscripts](https://brasstranscripts.com/blog/speaker-diarization-models-comparison)
- [pyannote/speaker-diarization-3.1 — HuggingFace](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [Top open source STT options 2026 — AssemblyAI](https://www.assemblyai.com/blog/top-open-source-stt-options-for-voice-applications)
- [Vosk — alphacephei](https://alphacephei.com/vosk/)
- [WhisperX Windows GPU issues — GitHub #1216](https://github.com/m-bain/whisperX/issues/1216)
- [WhisperX Windows platform detection — GitHub #1357](https://github.com/m-bain/whisperX/issues/1357)
- [FastAPI vs Flask 2025 — Strapi](https://strapi.io/blog/fastapi-vs-flask-python-framework-comparison)
- [Flask vs FastAPI — Better Stack](https://betterstack.com/community/guides/scaling-python/flask-vs-fastapi/)
- [Ollama — GitHub](https://github.com/ollama/ollama)
- [Llama 3.2 — Ollama](https://ollama.com/library/llama3.2)
- [PyAV — GitHub](https://github.com/PyAV-Org/PyAV)
- [Whisper-large-v3-turbo-es (Spanish fine-tune) — HuggingFace](https://huggingface.co/adriszmar/whisper-large-v3-turbo-es)
