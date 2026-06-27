import shutil
import subprocess
from pathlib import Path
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from escucha.main import create_app
from escucha.jobs import JobStatus

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FFMPEG = str(_REPO_ROOT / "bin" / "ffmpeg.exe") if (_REPO_ROOT / "bin" / "ffmpeg.exe").exists() else shutil.which("ffmpeg") or "ffmpeg"


@pytest.fixture
def client():
    """Fresh test client per test — model warm-up mocked out."""
    with patch("escucha.pipeline.PipelineRunner.warm_up", new_callable=AsyncMock):
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture
def tiny_mp4(tmp_path: Path) -> Path:
    out = tmp_path / "test.mp4"
    subprocess.run(
        [_FFMPEG, "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-f", "lavfi", "-i", "color=c=black:size=64x64:rate=1:duration=2",
         "-shortest", str(out)],
        check=True, capture_output=True,
    )
    return out


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    for key in (
        "status", "ffmpeg", "ollama", "cuda", "device", "whisper_model",
        "hf_token_set", "anthropic_key_set",
        "whisper_ready", "diarization_ready",
    ):
        assert key in data, f"missing field: {key}"


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Escucha" in r.text


def test_upload_non_mp4_rejected(client):
    r = client.post("/api/jobs", files={"file": ("test.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_upload_mp4_accepted_returns_job_id(client, tiny_mp4):
    with patch("escucha.pipeline.PipelineRunner.run", new_callable=AsyncMock):
        with open(tiny_mp4, "rb") as f:
            r = client.post("/api/jobs", files={"file": ("test.mp4", f, "video/mp4")})
    assert r.status_code == 201
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_get_job_status(client, tiny_mp4):
    with patch("escucha.pipeline.PipelineRunner.run", new_callable=AsyncMock):
        with open(tiny_mp4, "rb") as f:
            post = client.post("/api/jobs", files={"file": ("test.mp4", f, "video/mp4")})
    job_id = post.json()["job_id"]
    r = client.get(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["job_id"] == job_id
    assert "status" in data
    assert "progress" in data


def test_get_unknown_job_returns_404(client):
    r = client.get("/api/jobs/doesnotexist")
    assert r.status_code == 404


def test_result_on_incomplete_job_returns_409(client, tiny_mp4):
    with patch("escucha.pipeline.PipelineRunner.run", new_callable=AsyncMock):
        with open(tiny_mp4, "rb") as f:
            post = client.post("/api/jobs", files={"file": ("test.mp4", f, "video/mp4")})
    job_id = post.json()["job_id"]
    r = client.get(f"/api/jobs/{job_id}/result")
    assert r.status_code == 409


def test_concurrent_job_rejected(client, tiny_mp4):
    # Directly mark the active job as PROCESSING to simulate a running job
    # without depending on background task scheduling.
    registry = client.app.state.registry
    active_job = registry.create()
    active_job.status = JobStatus.PROCESSING

    with open(tiny_mp4, "rb") as f:
        r = client.post("/api/jobs", files={"file": ("test.mp4", f, "video/mp4")})
    assert r.status_code == 503


def test_export_unknown_format_rejected(client, tiny_mp4):
    with patch("escucha.pipeline.PipelineRunner.run", new_callable=AsyncMock):
        with open(tiny_mp4, "rb") as f:
            post = client.post("/api/jobs", files={"file": ("test.mp4", f, "video/mp4")})
    job_id = post.json()["job_id"]
    r = client.get(f"/api/jobs/{job_id}/export/xml")
    assert r.status_code == 400


def test_list_jobs_empty(client):
    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert r.json() == []


def test_list_jobs_returns_recent_first(client, tiny_mp4):
    with patch("escucha.pipeline.PipelineRunner.run", new_callable=AsyncMock):
        ids = []
        for _ in range(2):
            with open(tiny_mp4, "rb") as f:
                r = client.post("/api/jobs", files={"file": ("test.mp4", f, "video/mp4")})
            ids.append(r.json()["job_id"])

    r = client.get("/api/jobs")
    assert r.status_code == 200
    listed = [j["job_id"] for j in r.json()]
    # Newest job first.
    assert listed[0] == ids[-1]
    assert set(ids).issubset(set(listed))


def test_pipeline_exception_does_not_silently_disappear(client, tiny_mp4, caplog):
    """If the pipeline coroutine raises, the route handler's done_callback
    must surface it to the logger instead of swallowing it."""
    import logging

    async def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    caplog.set_level(logging.ERROR, logger="escucha")
    with patch("escucha.pipeline.PipelineRunner.run", new_callable=AsyncMock, side_effect=boom):
        with open(tiny_mp4, "rb") as f:
            r = client.post("/api/jobs", files={"file": ("test.mp4", f, "video/mp4")})
        assert r.status_code == 201
        # Give the event loop a tick to process the done callback.
        import time
        time.sleep(0.5)
    assert any("kaboom" in rec.message or "kaboom" in str(rec.exc_info or "")
               for rec in caplog.records)
