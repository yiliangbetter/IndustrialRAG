"""Clarification selections are consumed only after a completed answer."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from fastapi import HTTPException

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import query_progress_hooks
import rag_web_server
from raganything.clarify_gate import ClarifyBypass


@asynccontextmanager
async def _noop_progress_hooks():
    yield asyncio.Queue()


@pytest.fixture()
def _query_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    gate_result = ClarifyBypass("use_candidate")
    finalized: list[ClarifyBypass] = []

    rag_web_server.state.ready = True
    rag_web_server.state.rag = SimpleNamespace()
    rag_web_server.state.parser_output_dir = str(tmp_path)
    rag_web_server.state.query_mode = "mix"

    async def _gate(*_args, **_kwargs):
        return gate_result, 0.0

    monkeypatch.setattr(rag_web_server, "_evaluate_clarify_gate_timed", _gate)
    monkeypatch.setattr(
        rag_web_server,
        "_inject_clarify_bundle_for_bypass",
        lambda *_args, **_kwargs: _async_false(),
    )
    monkeypatch.setattr(rag_web_server, "_clarify_bypass_meta", lambda *_args: None)
    monkeypatch.setattr(
        rag_web_server, "_begin_web_query_trace", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        rag_web_server, "_finalize_web_timing", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        rag_web_server, "_persist_query_debug_dump", lambda **_kwargs: None
    )
    monkeypatch.setattr(rag_web_server, "_clear_clarify_injection_if_set", lambda: None)
    monkeypatch.setattr(rag_web_server, "_release_rerank_after_web_query", lambda: None)
    monkeypatch.setattr(
        rag_web_server,
        "_finalize_clarify_bypass",
        lambda _body, result: finalized.append(result),
    )
    monkeypatch.setattr(
        query_progress_hooks, "query_progress_hooks", _noop_progress_hooks
    )
    monkeypatch.setattr(
        query_progress_hooks, "set_query_media_roots", lambda _roots: None
    )
    monkeypatch.setattr(
        query_progress_hooks, "set_query_text_for_images", lambda _query: None
    )
    monkeypatch.setattr(
        query_progress_hooks,
        "finalize_inline_images",
        lambda **_kwargs: {"images": [], "placements": []},
    )

    body = rag_web_server.QueryBody(
        query="recommended question",
        clarify_choice="use_candidate",
        clarification_id="clarification-id",
        candidate_id="c1",
    )
    return body, finalized


async def _async_false() -> bool:
    return False


@pytest.mark.asyncio
async def test_failed_answer_does_not_consume_clarification(
    monkeypatch: pytest.MonkeyPatch,
    _query_server,
) -> None:
    body, finalized = _query_server

    async def _fail(*_args, **_kwargs):
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(rag_web_server, "_run_aquery", _fail)

    with pytest.raises(HTTPException, match="Query failed"):
        await rag_web_server.api_query(body)

    assert finalized == []


@pytest.mark.asyncio
async def test_successful_answer_consumes_clarification(
    monkeypatch: pytest.MonkeyPatch,
    _query_server,
) -> None:
    body, finalized = _query_server

    async def _succeed(*_args, **_kwargs):
        return "answer"

    monkeypatch.setattr(rag_web_server, "_run_aquery", _succeed)

    payload = await rag_web_server.api_query(body)

    assert payload["answer"] == "answer"
    assert len(finalized) == 1


@pytest.mark.asyncio
async def test_failed_stream_does_not_consume_clarification(
    monkeypatch: pytest.MonkeyPatch,
    _query_server,
) -> None:
    body, finalized = _query_server

    async def failing_stream():
        raise RuntimeError("simulated stream failure")
        yield "unreachable"

    async def _stream(*_args, **_kwargs):
        return failing_stream()

    monkeypatch.setattr(rag_web_server, "_run_aquery", _stream)

    events = [
        event
        async for event in rag_web_server._query_stream_events(body.query, "mix", body)
    ]

    assert any('"type": "error"' in event for event in events)
    assert any("simulated stream failure" in event for event in events)
    assert finalized == []


@pytest.mark.asyncio
async def test_successful_stream_consumes_clarification(
    monkeypatch: pytest.MonkeyPatch,
    _query_server,
) -> None:
    body, finalized = _query_server

    async def answer_stream():
        yield "answer"

    async def _stream(*_args, **_kwargs):
        return answer_stream()

    monkeypatch.setattr(rag_web_server, "_run_aquery", _stream)

    events = [
        event
        async for event in rag_web_server._query_stream_events(body.query, "mix", body)
    ]

    assert any('"type": "done"' in event for event in events)
    assert len(finalized) == 1
