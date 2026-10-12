"""Deterministic latency regressions in the Web response orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import time

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rag_web_server  # noqa: E402
from rag_web_server import (  # noqa: E402
    _iter_hook_events_until_task_done,
    _mark_first_delta,
)


@pytest.mark.asyncio
async def test_completed_gate_does_not_wait_for_poll_timeout() -> None:
    queue: asyncio.Queue = asyncio.Queue()

    async def quick_gate() -> None:
        await asyncio.sleep(0)

    task = asyncio.create_task(quick_gate())
    started = time.perf_counter()
    events = [event async for event in _iter_hook_events_until_task_done(queue, task)]
    elapsed = time.perf_counter() - started

    assert events == []
    assert elapsed < 0.04


def test_first_delta_metric_is_recorded_once(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(rag_web_server.time, "perf_counter", lambda: 12.5)
    monkeypatch.setattr(
        "raganything.query_timing_trace.trace_event",
        lambda kind, **detail: events.append((kind, detail)),
    )

    first = _mark_first_delta(10.0, None, "first_response")
    repeated = _mark_first_delta(10.0, first, "first_response")

    assert first == repeated == 2.5
    assert events == [("first_response_delta", {"first_response_s": 2.5})]


@pytest.mark.asyncio
async def test_gate_progress_tail_is_drained_before_return() -> None:
    queue: asyncio.Queue = asyncio.Queue()

    async def gate_with_tail() -> None:
        await queue.put({"phase": "retrieve"})
        await queue.put({"phase": "rerank"})

    task = asyncio.create_task(gate_with_tail())
    events = [event async for event in _iter_hook_events_until_task_done(queue, task)]

    assert events == [{"phase": "retrieve"}, {"phase": "rerank"}]
