"""RAGAnything.close() must actually finalize storages.

tests/test_close_event_loop.py replicates the control flow in isolation.
This file exercises the production method so atexit / FastAPI shutdown
cannot drift from that replica: missing finalize leaves parse cache and
LightRAG storages unflushed.
"""

from __future__ import annotations

import asyncio

import pytest


class StubParser:
    def check_installation(self):
        return True


def _make_rag(monkeypatch, tmp_path):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module
    from raganything.config import RAGAnythingConfig

    monkeypatch.setattr(rag_module, "get_parser", lambda _name: StubParser())
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **k: None)
    monkeypatch.setattr(rag_module.atexit, "unregister", lambda *args, **k: None)

    config = RAGAnythingConfig(
        working_dir=str(tmp_path / "rag_workdir"),
        parser="mineru",
    )
    return rag_module.RAGAnything(config=config)


def test_close_runs_finalize_when_no_running_loop(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path)
    called = {"n": 0}

    async def fake_finalize():
        called["n"] += 1

    rag.finalize_storages = fake_finalize
    rag.close()
    assert called["n"] == 1


def test_close_swallows_finalize_errors(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path)

    async def boom():
        raise RuntimeError("storage teardown failed")

    rag.finalize_storages = boom
    rag.close()


@pytest.mark.asyncio
async def test_close_schedules_finalize_inside_running_loop(monkeypatch, tmp_path):
    rag = _make_rag(monkeypatch, tmp_path)
    called = {"n": 0}

    async def fake_finalize():
        called["n"] += 1

    rag.finalize_storages = fake_finalize
    rag.close()
    await asyncio.sleep(0)
    assert called["n"] == 1
