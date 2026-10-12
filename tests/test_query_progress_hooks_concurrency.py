"""Concurrency regression tests for request-scoped LightRAG progress hooks."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import lightrag.operate as lightrag_operate  # noqa: E402
import query_progress_hooks as hooks  # noqa: E402


@pytest.mark.asyncio
async def test_overlapping_contexts_keep_one_installed_dispatcher() -> None:
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()
    release_first = asyncio.Event()
    release_second = asyncio.Event()
    installed: list[object] = []

    async def first_request() -> None:
        async with hooks.query_progress_hooks():
            installed.append(lightrag_operate._build_query_context)
            first_entered.set()
            await release_first.wait()

    async def second_request() -> None:
        await first_entered.wait()
        async with hooks.query_progress_hooks():
            installed.append(lightrag_operate._build_query_context)
            second_entered.set()
            await release_second.wait()

    first_task = asyncio.create_task(first_request())
    second_task = asyncio.create_task(second_request())
    await second_entered.wait()

    assert installed[0] is installed[1]
    dispatcher = installed[0]

    release_first.set()
    await first_task
    assert lightrag_operate._build_query_context is dispatcher

    release_second.set()
    await second_task
    assert lightrag_operate._build_query_context is dispatcher


@pytest.mark.asyncio
async def test_overlapping_requests_keep_debug_snapshots_isolated() -> None:
    both_ready = asyncio.Event()
    ready_count = 0
    ready_lock = asyncio.Lock()

    async def request(owner: str) -> str:
        nonlocal ready_count
        async with hooks.query_progress_hooks():
            hooks._llm_input.set({"owner": owner})
            hooks._sync_query_debug_snapshot()
            async with ready_lock:
                ready_count += 1
                if ready_count == 2:
                    both_ready.set()
            await both_ready.wait()
            state = hooks.get_query_debug_state()
            return str((state.get("llm_input") or {}).get("owner"))

    owners = await asyncio.gather(request("first"), request("second"))

    assert owners == ["first", "second"]
