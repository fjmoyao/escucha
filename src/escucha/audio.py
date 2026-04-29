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
