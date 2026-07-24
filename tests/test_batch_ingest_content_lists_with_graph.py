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


@pytest.mark.asyncio
async def test_async_main_exits_when_data_root_missing(
    bicl_graph, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("LLM_BINDING_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_with_graph",
            "-w",
            str(tmp_path / "wd"),
            "--data-repo-root",
            str(tmp_path / "missing-repo"),
        ],
    )

    with pytest.raises(SystemExit, match="Data root does not exist"):
        await bicl_graph.async_main()


@pytest.mark.asyncio
async def test_async_main_exits_when_llm_api_key_missing(
    bicl_graph, monkeypatch, tmp_path
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BINDING_API_KEY", raising=False)
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_with_graph",
            "-w",
            str(tmp_path / "wd"),
            "--data-repo-root",
            str(repo),
        ],
    )

    with pytest.raises(SystemExit, match="OPENAI_API_KEY or LLM_BINDING_API_KEY"):
        await bicl_graph.async_main()


@pytest.mark.asyncio
async def test_async_main_exits_when_embedding_host_lacks_api_key(
    bicl_graph, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_with_graph",
            "-w",
            str(tmp_path / "wd"),
            "--data-repo-root",
            str(repo),
        ],
    )

    with pytest.raises(SystemExit, match="EMBEDDING_API_KEY"):
        await bicl_graph.async_main()


@pytest.mark.asyncio
async def test_async_main_exits_when_no_content_list_json(
    bicl_graph, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
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

    fake_emb = MagicMock(name="embedding_func")
    mock_lr = MagicMock()
    mock_lr.initialize_storages = AsyncMock()

    with patch(
        "raganything.local_hf_embedding.make_local_hf_embedding_func",
        return_value=fake_emb,
    ):
        with patch("lightrag.LightRAG", return_value=mock_lr):
            with patch("raganything.RAGAnything") as rag_cls:
                with pytest.raises(SystemExit, match="No files matching"):
                    await bicl_graph.async_main()

    rag_cls.assert_called_once()
    mock_lr.initialize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_main_ingests_with_skip_multimodal_default(
    bicl_graph, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    repo = tmp_path / "repo"
    out = repo / "output" / "data_upload_test_v3"
    out.mkdir(parents=True)
    json_path = out / "doc_content_list_v2.json"
    payload = [{"type": "text", "text": "kg hello"}]
    json_path.write_text(json.dumps(payload), encoding="utf-8")
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

    fake_emb = MagicMock(name="embedding_func")
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
                await bicl_graph.async_main()

    mock_rag.insert_content_list.assert_awaited_once()
    call = mock_rag.insert_content_list.await_args
    assert call.args[0] == payload
    assert call.kwargs.get("file_path", "").endswith("doc_content_list_v2.json")
    assert call.kwargs.get("skip_multimodal_processing") is True
    mock_rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_main_skips_non_list_json_and_finalizes(
    bicl_graph, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    repo = tmp_path / "repo"
    out = repo / "output" / "data_upload_test_v3"
    out.mkdir(parents=True)
    (out / "notalist_content_list_v2.json").write_text(
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
                await bicl_graph.async_main()

    mock_rag.insert_content_list.assert_not_called()
    mock_rag.finalize_storages.assert_awaited_once()
