"""RAGAnything.close must call the real finalize_storages method.

tests/test_close_event_loop.py exercises a copied control-flow helper. These
tests call RAGAnything.close so a rewrite that no longer finalizes storages
fails even if that copy stays unchanged.
"""

import asyncio

import pytest

from raganything.raganything import RAGAnything


def _bare_rag():
    """Skip __post_init__ so close() can be tested without LightRAG startup."""
    return object.__new__(RAGAnything)


def test_close_outside_a_loop_finalizes_storages():
    rag = _bare_rag()
    calls = []

    async def finalize_storages():
        calls.append("finalized")

    rag.finalize_storages = finalize_storages

    rag.close()

    assert calls == ["finalized"]


def test_close_finalizes_when_the_thread_loop_is_already_closed():
    rag = _bare_rag()
    calls = []

    async def finalize_storages():
        calls.append("finalized")

    rag.finalize_storages = finalize_storages
    stale = asyncio.new_event_loop()
    asyncio.set_event_loop(stale)
    stale.close()
    try:
        rag.close()
    finally:
        asyncio.set_event_loop(None)

    assert calls == ["finalized"]


def test_close_swallows_finalize_errors():
    rag = _bare_rag()

    async def finalize_storages():
        raise RuntimeError("storage error")

    rag.finalize_storages = finalize_storages

    rag.close()


@pytest.mark.asyncio
async def test_close_schedules_finalize_inside_a_running_loop():
    rag = _bare_rag()
    started = asyncio.Event()
    release = asyncio.Event()

    async def finalize_storages():
        started.set()
        await release.wait()

    rag.finalize_storages = finalize_storages

    rag.close()
    await asyncio.wait_for(started.wait(), timeout=1)
    release.set()
    await asyncio.sleep(0)
