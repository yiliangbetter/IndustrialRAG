"""Timing events from response worker tasks must reach the parent trace."""

from __future__ import annotations

import asyncio

import pytest

from raganything.query_timing_trace import (
    begin_query_trace,
    clear_query_trace,
    finish_query_trace,
    trace_event,
)


@pytest.fixture(autouse=True)
def _enable_trace(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAG_QUERY_TIMING_DUMP", "1")
    clear_query_trace()
    yield
    clear_query_trace()


@pytest.mark.asyncio
async def test_child_task_events_are_retained() -> None:
    begin_query_trace(query="question")

    async def worker() -> None:
        trace_event("child_event")

    await asyncio.create_task(worker())
    payload = finish_query_trace()

    assert [event["kind"] for event in payload["events"]] == ["child_event"]


@pytest.mark.asyncio
async def test_worker_thread_events_are_retained() -> None:
    begin_query_trace(query="question")

    await asyncio.to_thread(trace_event, "thread_event")
    payload = finish_query_trace()

    assert [event["kind"] for event in payload["events"]] == ["thread_event"]
