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

    app.state.settings = settings
    app.state.registry = registry
    app.state.runner = runner

    yield

    logger.info("Shutting down.")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="Escucha",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(router)

    static_dir = Path(__file__).resolve().parent.parent.parent / "static"

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


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
