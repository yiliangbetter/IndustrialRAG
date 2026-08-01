"""Regression: finalize_storages before query must not leave Neo4j/PG dead."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.asyncio
async def test_reopen_storages_if_finalized_resets_and_reinitializes():
    from lightrag.base import StoragesStatus

    from raganything.raganything import RAGAnything

    rag = object.__new__(RAGAnything)
    rag.logger = MagicMock()
    rag.parse_cache = AsyncMock()

    lightrag = MagicMock()
    lightrag._storages_status = StoragesStatus.FINALIZED
    lightrag.initialize_storages = AsyncMock(
        side_effect=lambda: setattr(
            lightrag, "_storages_status", StoragesStatus.INITIALIZED
        )
    )
    rag.lightrag = lightrag

    await rag._reopen_storages_if_finalized()

    assert lightrag._storages_status == StoragesStatus.INITIALIZED
    lightrag.initialize_storages.assert_awaited_once()
    rag.parse_cache.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_reopen_storages_if_finalized_noop_when_initialized():
    from lightrag.base import StoragesStatus

    from raganything.raganything import RAGAnything

    rag = object.__new__(RAGAnything)
    rag.logger = MagicMock()
    rag.parse_cache = AsyncMock()

    lightrag = MagicMock()
    lightrag._storages_status = StoragesStatus.INITIALIZED
    lightrag.initialize_storages = AsyncMock()
    rag.lightrag = lightrag

    await rag._reopen_storages_if_finalized()

    lightrag.initialize_storages.assert_not_awaited()
    rag.parse_cache.initialize.assert_not_awaited()


@pytest.mark.asyncio
async def test_aquery_reopens_finalized_storages_before_query():
    from lightrag.base import StoragesStatus

    from raganything.query import QueryMixin

    class _Q(QueryMixin):
        pass

    q = _Q()
    q.logger = MagicMock()
    q.vision_model_func = None
    q.callback_manager = None
    q.lightrag = MagicMock()
    q.lightrag._storages_status = StoragesStatus.FINALIZED
    q.lightrag.aquery = AsyncMock(return_value="ok")
    q._reopen_storages_if_finalized = AsyncMock()

    result = await q.aquery("hello", mode="naive", vlm_enhanced=False)

    assert result == "ok"
    q._reopen_storages_if_finalized.assert_awaited_once()
    q.lightrag.aquery.assert_awaited_once()


def _load_pipeline_module():
    name = "rag_pipeline_parse_graph_chat_finalize_under_test"
    path = REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.asyncio
async def test_pipeline_finalizes_after_query_not_before(tmp_path: Path):
    mod = _load_pipeline_module()
    input_folder = tmp_path / "in"
    input_folder.mkdir()
    (input_folder / "a.pdf").write_bytes(b"%PDF")

    order: list[str] = []

    rag = MagicMock()
    rag.finalize_storages = AsyncMock(side_effect=lambda: order.append("finalize"))
    rag.aquery = AsyncMock(
        side_effect=lambda *a, **k: order.append("query") or "answer"
    )
    config = MagicMock()
    config.parser = "mineru"
    logger = MagicMock()

    async def _ingest(*_a, **_k):
        order.append("ingest")
        return 1, 0

    argv = [
        "rag_pipeline_parse_graph_chat.py",
        "--input-folder",
        str(input_folder),
        "--working-dir",
        str(tmp_path / "wd"),
        "--parser-output-dir",
        str(tmp_path / "parse"),
        "--no-recursive",
        "--query",
        "what is tested?",
    ]

    with (
        patch.object(mod.sys, "argv", argv),
        patch.object(
            mod, "_build_rag", AsyncMock(return_value=(rag, config, logger))
        ),
        patch.object(mod, "_mineru_parse_kwargs", return_value={}),
        patch.object(mod, "_ingest_folder", side_effect=_ingest),
    ):
        await mod.async_main()

    assert order == ["ingest", "query", "finalize"]
    rag.aquery.assert_awaited_once()
    rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_finalizes_on_ingest_only(tmp_path: Path):
    mod = _load_pipeline_module()
    input_folder = tmp_path / "in"
    input_folder.mkdir()
    (input_folder / "a.pdf").write_bytes(b"%PDF")

    order: list[str] = []
    rag = MagicMock()
    rag.finalize_storages = AsyncMock(side_effect=lambda: order.append("finalize"))
    rag.aquery = AsyncMock()
    config = MagicMock()
    config.parser = "mineru"
    logger = MagicMock()

    async def _ingest(*_a, **_k):
        order.append("ingest")
        return 1, 0

    argv = [
        "rag_pipeline_parse_graph_chat.py",
        "--input-folder",
        str(input_folder),
        "--working-dir",
        str(tmp_path / "wd"),
        "--parser-output-dir",
        str(tmp_path / "parse"),
        "--no-recursive",
        "--ingest-only",
    ]

    with (
        patch.object(mod.sys, "argv", argv),
        patch.object(
            mod, "_build_rag", AsyncMock(return_value=(rag, config, logger))
        ),
        patch.object(mod, "_mineru_parse_kwargs", return_value={}),
        patch.object(mod, "_ingest_folder", side_effect=_ingest),
    ):
        await mod.async_main()

    assert order == ["ingest", "finalize"]
    rag.aquery.assert_not_awaited()
