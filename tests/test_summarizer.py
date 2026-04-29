import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from escucha.summarizer import (
    summarize_with_ollama,
    summarize_with_claude,
    SummarizationError,
    _split_into_chunks,
    _segments_to_text,
)
from escucha.models import DiarizedSegment


@pytest.fixture
def three_segments():
    return [
        DiarizedSegment(speaker="SPEAKER_00", start=0.0, end=4.5, text="Buenos dias a todos."),
        DiarizedSegment(speaker="SPEAKER_01", start=4.5, end=9.0, text="Tengo los numeros."),
        DiarizedSegment(speaker="SPEAKER_00", start=9.0, end=13.0, text="Perfecto, revisemos."),
    ]


@pytest.mark.asyncio
async def test_summarize_ollama_success(three_segments):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Este fue un resumen exitoso."}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await summarize_with_ollama(three_segments, model="llama3.2")

    assert len(result) > 0


@pytest.mark.asyncio
async def test_summarize_ollama_connection_error(three_segments):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
               side_effect=httpx.ConnectError("refused")):
        with pytest.raises(SummarizationError, match="Cannot connect to Ollama"):
            await summarize_with_ollama(three_segments)


@pytest.mark.asyncio
async def test_summarize_ollama_chunking():
    long_text = "palabra " * 2000  # ~16 000 chars — forces chunking
    segments = [
        DiarizedSegment(speaker="SPEAKER_00", start=float(i), end=float(i + 1), text=long_text)
        for i in range(2)
    ]
    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = {"response": f"chunk summary {call_count}"}
        return r

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=mock_post):
        await summarize_with_ollama(segments)

    assert call_count > 1


@pytest.mark.asyncio
async def test_summarize_claude_success(three_segments):
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Resumen generado por Claude.")]

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = mock_message
        result = await summarize_with_claude(three_segments, api_key="sk-ant-fake")

    assert len(result) > 0


def test_split_into_chunks_single():
    assert _split_into_chunks("short text", max_chars=1000) == ["short text"]


def test_split_into_chunks_multiple():
    lines = ["line " * 100 for _ in range(10)]
    text = "\n".join(lines)
    chunks = _split_into_chunks(text, max_chars=600)
    assert len(chunks) > 1


def test_segments_to_text(three_segments):
    text = _segments_to_text(three_segments)
    assert "SPEAKER_00" in text
    assert "Buenos dias" in text
