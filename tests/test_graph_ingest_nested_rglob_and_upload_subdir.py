"""Graph ingest must scan nested MinerU trees and honor the upload subdir.

``scripts/batch_ingest_content_lists_with_graph.py`` uses ``Path.rglob`` under
``output/<upload-subdir>/``. OCR re-ingest writes v4 trees; a ``glob``-only
scan or a hardcoded v3 subdir would silently skip nested manuals. Distinct
from #127 (top-level v3 files, RAG_DATA_REPO, --limit) and #179 (OCR
``Path.glob`` skipping nested PDFs).
"""

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


@pytest.fixture(scope="module")
def bigraph():
    spec = importlib.util.spec_from_file_location(
        "batch_ingest_graph_nested_rglob_1c11", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_environ_after_each_test():
    keys = (
        "OPENAI_API_KEY",
        "LLM_BINDING_API_KEY",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BACKEND",
        "EMBEDDING_BINDING_HOST",
        "RAG_DATA_REPO",
        "RAG_DATA_UPLOAD_SUBDIR",
        "HF_HOME",
    )
    before = {key: os.environ.get(key) for key in keys}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


class _IngestStack:
    def __init__(self):
        self.embedding = MagicMock(name="embedding_func")
        self.lightrag = MagicMock()
        self.lightrag.initialize_storages = AsyncMock()
        self.rag = MagicMock()
        self.rag.insert_content_list = AsyncMock()
        self.rag.finalize_storages = AsyncMock()

    def rag_ctor(self, *args, **kwargs):
        return self.rag


def _write_list(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_graph_ingest_rglob_finds_nested_v4_and_ignores_v3_sibling(
    bigraph, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    nested_payload = [{"type": "text", "text": "nested v4"}]
    nested = _write_list(
        repo
        / "output"
        / "data_upload_test_v4"
        / "plant-a"
        / "manual_content_list_v2.json",
        nested_payload,
    )
    _write_list(
        repo / "output" / "data_upload_test_v3" / "ignored_content_list_v2.json",
        [{"type": "text", "text": "should not ingest"}],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-graph")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_with_graph",
            "-w",
            str(tmp_path / "wd"),
            "--data-repo-root",
            str(repo),
            "--data-upload-subdir",
            "data_upload_test_v4",
        ],
    )

    stack = _IngestStack()
    with patch("lightrag.utils.EmbeddingFunc", return_value=stack.embedding):
        with patch("lightrag.LightRAG", return_value=stack.lightrag):
            with patch("raganything.RAGAnything", side_effect=stack.rag_ctor):
                with patch("lightrag.llm.openai.openai_embed") as openai_embed:
                    openai_embed.func = MagicMock()
                    await bigraph.async_main()

    stack.rag.insert_content_list.assert_awaited_once()
    call = stack.rag.insert_content_list.await_args
    assert call.args[0] == nested_payload
    assert Path(call.kwargs["file_path"]) == nested.resolve().relative_to(
        repo.resolve()
    )
    stack.rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_graph_ingest_upload_subdir_env_selects_tree_when_cli_omitted(
    bigraph, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    payload = [{"type": "text", "text": "from env subdir"}]
    json_path = _write_list(
        repo / "output" / "ocr_tree" / "doc_content_list_v2.json",
        payload,
    )
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-graph")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("RAG_DATA_UPLOAD_SUBDIR", "ocr_tree")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
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

    stack = _IngestStack()
    with patch("lightrag.utils.EmbeddingFunc", return_value=stack.embedding):
        with patch("lightrag.LightRAG", return_value=stack.lightrag):
            with patch("raganything.RAGAnything", side_effect=stack.rag_ctor):
                with patch("lightrag.llm.openai.openai_embed") as openai_embed:
                    openai_embed.func = MagicMock()
                    await bigraph.async_main()

    call = stack.rag.insert_content_list.await_args
    assert call.args[0] == payload
    assert Path(call.kwargs["file_path"]) == json_path.resolve().relative_to(
        repo.resolve()
    )
