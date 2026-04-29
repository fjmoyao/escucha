from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Immutable application configuration. Loaded once at startup."""

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    ffmpeg_path: Path = field(default=None)
    upload_dir: Path = field(default=None)

    # Model settings
    whisper_model: str = "large-v3"
    whisper_compute_type: str = "int8"
    device: str = "auto"

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
            import torch
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
