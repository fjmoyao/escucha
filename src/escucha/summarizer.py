from escucha.models import DiarizedSegment
import httpx


class SummarizationError(Exception):
    """Raised when summarization fails."""


_CHUNK_MAX_CHARS = 12000

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
