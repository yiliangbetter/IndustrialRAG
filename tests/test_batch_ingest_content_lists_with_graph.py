"""Unit tests for ``scripts/batch_ingest_content_lists_with_graph.py``."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "batch_ingest_content_lists_with_graph.py"


def _load_script_module():
    name = "batch_ingest_content_lists_with_graph_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bicl_graph():
    return _load_script_module()


@pytest.fixture(autouse=True)
def _restore_environ_after_each_test():
    before = {
        "EMBEDDING_BACKEND": os.environ.get("EMBEDDING_BACKEND"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "LLM_BINDING_API_KEY": os.environ.get("LLM_BINDING_API_KEY"),
        "EMBEDDING_API_KEY": os.environ.get("EMBEDDING_API_KEY"),
        "EMBEDDING_BINDING_HOST": os.environ.get("EMBEDDING_BINDING_HOST"),
    }
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def test_main_raises_system_exit_with_async_main_return_value(bicl_graph, monkeypatch):
    def fake_run(coro):
        coro.close()
        return 7

    monkeypatch.setattr(bicl_graph.asyncio, "run", fake_run)
    with pytest.raises(SystemExit) as exc_info:
        bicl_graph.main()
    assert exc_info.value.code == 7


@pytest.mark.asyncio
async def test_async_main_returns_1_when_ingest_raises(
    bicl_graph, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    repo = tmp_path / "repo"
    out = repo / "output" / "data_upload_test_v3"
    out.mkdir(parents=True)
    (out / "doc_content_list_v2.json").write_text(
        json.dumps([{"type": "text", "text": "hello"}]), encoding="utf-8"
    )
    wd = tmp_path / "wd"
    wd.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_with_graph",
            "-w",
            str(wd),
            "--data-repo-root",
            str(repo),
        ],
    )

    fake_emb = MagicMock()
    mock_lr = MagicMock()
    mock_lr.initialize_storages = AsyncMock()
    mock_rag = MagicMock()
    mock_rag.insert_content_list = AsyncMock(side_effect=RuntimeError("boom"))
    mock_rag.finalize_storages = AsyncMock()

    with patch(
        "raganything.local_hf_embedding.make_local_hf_embedding_func",
        return_value=fake_emb,
    ):
        with patch("lightrag.LightRAG", return_value=mock_lr):
            with patch("raganything.RAGAnything", return_value=mock_rag):
                code = await bicl_graph.async_main()

    assert code == 1
    mock_rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_main_returns_1_when_json_is_not_a_list(
    bicl_graph, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    repo = tmp_path / "repo"
    out = repo / "output" / "data_upload_test_v3"
    out.mkdir(parents=True)
    (out / "bad_content_list_v2.json").write_text(
        json.dumps({"not": "a list"}), encoding="utf-8"
    )
    wd = tmp_path / "wd"
    wd.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_with_graph",
            "-w",
            str(wd),
            "--data-repo-root",
            str(repo),
        ],
    )

    fake_emb = MagicMock()
    mock_lr = MagicMock()
    mock_lr.initialize_storages = AsyncMock()
    mock_rag = MagicMock()
    mock_rag.insert_content_list = AsyncMock()
    mock_rag.finalize_storages = AsyncMock()

    with patch(
        "raganything.local_hf_embedding.make_local_hf_embedding_func",
        return_value=fake_emb,
    ):
        with patch("lightrag.LightRAG", return_value=mock_lr):
            with patch("raganything.RAGAnything", return_value=mock_rag):
                code = await bicl_graph.async_main()

    assert code == 1
    mock_rag.insert_content_list.assert_not_awaited()
    mock_rag.finalize_storages.assert_awaited_once()
