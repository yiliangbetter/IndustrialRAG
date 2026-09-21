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

from rag_web_server import _iter_hook_events_until_task_done  # noqa: E402


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


@pytest.mark.asyncio
async def test_gate_progress_tail_is_drained_before_return() -> None:
    queue: asyncio.Queue = asyncio.Queue()

    async def gate_with_tail() -> None:
        await queue.put({"phase": "retrieve"})
        await queue.put({"phase": "rerank"})

    task = asyncio.create_task(gate_with_tail())
    events = [event async for event in _iter_hook_events_until_task_done(queue, task)]

    assert events == [{"phase": "retrieve"}, {"phase": "rerank"}]
