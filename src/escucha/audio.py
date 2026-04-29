import subprocess
import sys
from pathlib import Path


class AudioExtractionError(Exception):
    """Raised when FFmpeg fails to extract audio."""


# 30 minutes is plenty even for very long videos (FFmpeg is fast at audio).
_FFMPEG_TIMEOUT_SECONDS = 1800


def extract_audio(
    input_path: Path,
    output_path: Path,
    ffmpeg_path: Path = Path("ffmpeg"),
    sample_rate: int = 16000,
) -> Path:
    """Extract audio from any FFmpeg-readable container as 16 kHz mono WAV.

    Args:
        input_path: Path to source media (MP4, MKV, WAV, etc.).
        output_path: Where to write the WAV. Parent directory is created if missing.
        ffmpeg_path: Path to the FFmpeg binary. Default 'ffmpeg' (system PATH).
        sample_rate: Target sample rate. 16 000 Hz is what Whisper expects.

    Returns:
        ``output_path`` on success.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        AudioExtractionError: If FFmpeg returns a non-zero exit code or times out.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(ffmpeg_path),
        "-y",                      # overwrite output without prompting
        "-i", str(input_path),
        "-vn",                     # discard video
        "-acodec", "pcm_s16le",    # 16-bit PCM
        "-ar", str(sample_rate),
        "-ac", "1",                # mono
        str(output_path),
    ]

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=creationflags,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        raise AudioExtractionError(
            f"FFmpeg timed out after {_FFMPEG_TIMEOUT_SECONDS}s on {input_path.name}"
        ) from e
    except FileNotFoundError as e:
        raise AudioExtractionError(
            f"FFmpeg binary not found at {ffmpeg_path}. "
            "Install it or set the bundled bin/ffmpeg.exe."
        ) from e

    if result.returncode != 0:
        # Truncate stderr to keep error messages short and serialisable.
        stderr_tail = result.stderr[-500:].strip() if result.stderr else "(no stderr)"
        raise AudioExtractionError(
            f"FFmpeg exited with code {result.returncode}: {stderr_tail}"
        )

    return output_path
