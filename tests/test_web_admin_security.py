"""Browser-origin checks protect local administrative APIs."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rag_web_server import BrowserOriginGate, _configured_cors_origins  # noqa: E402


async def _call_gate(
    origin: str | None, allowed: tuple[str, ...] = ()
) -> tuple[int, bool]:
    called = False

    async def downstream(_scope, _receive, send):
        nonlocal called
        called = True
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    headers = [(b"host", b"127.0.0.1:8765")]
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": "/api/setup/clear-knowledge-base",
        "headers": headers,
    }
    messages: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await BrowserOriginGate(downstream, allowed)(scope, receive, send)
    status = next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )
    return status, called


@pytest.mark.asyncio
async def test_cross_origin_admin_request_is_rejected() -> None:
    assert await _call_gate("https://evil.example") == (403, False)


@pytest.mark.asyncio
async def test_same_origin_and_non_browser_requests_continue() -> None:
    assert await _call_gate("http://127.0.0.1:8765") == (204, True)
    assert await _call_gate(None) == (204, True)


@pytest.mark.asyncio
async def test_explicit_cross_origin_can_be_allowed() -> None:
    assert await _call_gate(
        "https://trusted.example", ("https://trusted.example",)
    ) == (204, True)


def test_wildcard_cors_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_WEB_CORS_ORIGINS", "*")
    with pytest.raises(ValueError, match="unsafe"):
        _configured_cors_origins()
