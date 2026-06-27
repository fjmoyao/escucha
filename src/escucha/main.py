import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from escucha.config import load_settings
from escucha.jobs import JobRegistry
from escucha.pipeline import PipelineRunner
from escucha.routes import init_routes, router
from escucha.transcriber import TranscriptionError

logger = logging.getLogger("escucha")

# Static directory is two levels up from src/escucha/main.py (project root /static).
_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load settings, warm up models, wire routes. Shutdown: log only."""
    settings = load_settings()
    registry = JobRegistry()
    runner = PipelineRunner(settings, registry)

    logger.info("Device: %s", settings.resolved_device)
    logger.info("Whisper: %s (%s)", settings.whisper_model, settings.effective_compute_type)
    logger.info("HF token: %s", "set" if settings.hf_token else "NOT SET")

    # Wire routes BEFORE warm-up so /api/health responds even during loading.
    init_routes(settings, registry, runner)
    app.state.settings = settings
    app.state.registry = registry
    app.state.runner = runner

    try:
        await runner.warm_up()
    except TranscriptionError as e:
        logger.error("=" * 60)
        logger.error("FATAL: Whisper model failed to load.")
        logger.error("Reason: %s", e)
        logger.error(
            "Hint: try a smaller model in .env (WHISPER_MODEL=small) or "
            "switch to CPU (DEVICE=cpu)."
        )
        logger.error("=" * 60)
        raise

    logger.info(
        "Server ready. whisper=%s diarization=%s",
        runner.whisper_ready,
        runner.diarization_ready,
    )

    yield

    logger.info("Shutting down.")


def create_app() -> FastAPI:
    """Application factory used by uvicorn."""
    app = FastAPI(title="Escucha", version="0.1.0", lifespan=lifespan)
    app.include_router(router)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    return app


def main() -> None:
    """Console entry point: ``python -m escucha.main``."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings()
    try:
        uvicorn.run(
            "escucha.main:create_app",
            factory=True,
            host=settings.host,
            port=settings.port,
            log_level="info",
        )
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
