"""Regression tests for RAGAnything.close() wiring to finalize_storages.

``tests/test_close_event_loop.py`` only exercises a copied control-flow replica.
These tests bind the real ``RAGAnything.close`` method to ``finalize_storages``
so atexit/shutdown cannot silently stop cleaning storages.
"""

from __future__ import annotations

import asyncio
import atexit
from unittest.mock import AsyncMock

import pytest


def _make_rag(monkeypatch, tmp_path):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module
    from raganything.config import RAGAnythingConfig

    class StubParser:
        def check_installation(self):
            return True

    monkeypatch.setattr(rag_module, "get_parser", lambda _name: StubParser())
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **kwargs: None)

    config = RAGAnythingConfig(
        working_dir=str(tmp_path / "rag_workdir"),
        parser="mineru",
    )
    rag = rag_module.RAGAnything(config=config)
    # Defensive: avoid atexit noise if registration was not stubbed.
    atexit.unregister(rag.close)
    return rag


class TestRAGAnythingCloseWiring:
    def test_close_without_running_loop_awaits_finalize(self, monkeypatch, tmp_path):
        rag = _make_rag(monkeypatch, tmp_path)
        rag.finalize_storages = AsyncMock()

        rag.close()

        rag.finalize_storages.assert_awaited_once()

    def test_close_swallows_finalize_errors(self, monkeypatch, tmp_path):
        rag = _make_rag(monkeypatch, tmp_path)
        rag.finalize_storages = AsyncMock(side_effect=RuntimeError("storage boom"))

        # Must not raise during interpreter / atexit teardown.
        rag.close()
        rag.finalize_storages.assert_awaited_once()

    def test_close_inside_running_loop_schedules_finalize(self, monkeypatch, tmp_path):
        rag = _make_rag(monkeypatch, tmp_path)
        called = {"n": 0}

        async def finalize():
            called["n"] += 1

        rag.finalize_storages = finalize

        async def run_close():
            rag.close()
            # Allow the scheduled task to run.
            await asyncio.sleep(0.05)
            return called["n"]

        assert asyncio.run(run_close()) == 1
