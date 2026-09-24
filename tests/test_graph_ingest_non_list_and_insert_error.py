"""Graph ingest must skip non-list JSON and continue after one insert error.

``scripts/batch_ingest_content_lists_with_graph.py`` walks every
``*_content_list_v2.json`` under the upload tree. A dict payload is not a
content list and must not be inserted. An exception on one file must not drop
the rest of the corpus, and storage still finalizes. Distinct from #127
(limit, empty glob, missing keys) and from the local-HF script's exit code.
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

_ENV_KEYS = (
    "OPENAI_API_KEY",
    "LLM_BINDING_API_KEY",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BACKEND",
    "EMBEDDING_BINDING_HOST",
    "LLM_BINDING_HOST",
    "OPENAI_BASE_URL",
    "RAG_DATA_REPO",
    "RAG_DATA_UPLOAD_SUBDIR",
    "HF_HOME",
)


@pytest.fixture(scope="module")
def bigraph():
    spec = importlib.util.spec_from_file_location(
        "batch_ingest_graph_non_list_74ef", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_environ_after_each_test():
    before = {key: os.environ.get(key) for key in _ENV_KEYS}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


class _IngestStack:
    def __init__(self, insert_side_effect=None):
        self.embedding = MagicMock(name="embedding_func")
        self.lightrag = MagicMock()
        self.lightrag.initialize_storages = AsyncMock()
        self.rag = MagicMock()
        self.rag.insert_content_list = AsyncMock(side_effect=insert_side_effect)
        self.rag.finalize_storages = AsyncMock()

    def rag_ctor(self, *args, **kwargs):
        return self.rag


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


async def _ingest(bigraph, monkeypatch, tmp_path, repo, insert_side_effect=None):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-graph")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("RAG_DATA_REPO", raising=False)
    monkeypatch.delenv("RAG_DATA_UPLOAD_SUBDIR", raising=False)
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
    stack = _IngestStack(insert_side_effect)
    with patch("lightrag.utils.EmbeddingFunc", return_value=stack.embedding):
        with patch("lightrag.LightRAG", return_value=stack.lightrag):
            with patch("raganything.RAGAnything", side_effect=stack.rag_ctor):
                with patch("lightrag.llm.openai.openai_embed") as openai_embed:
                    openai_embed.func = MagicMock()
                    await bigraph.async_main()
    return stack


@pytest.mark.asyncio
async def test_non_list_json_is_skipped_and_sibling_list_is_inserted(
    bigraph, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    tree = repo / "output" / "data_upload_test_v3"
    _write_json(tree / "aaa_object_content_list_v2.json", {"type": "not-a-list"})
    list_payload = [{"type": "text", "text": "keep"}]
    _write_json(tree / "bbb_list_content_list_v2.json", list_payload)

    stack = await _ingest(bigraph, monkeypatch, tmp_path, repo)

    stack.rag.insert_content_list.assert_awaited_once()
    call = stack.rag.insert_content_list.await_args
    assert call.args[0] == list_payload
    assert call.kwargs["file_path"].endswith("bbb_list_content_list_v2.json")
    assert "aaa_object" not in call.kwargs["file_path"]
    stack.rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_insert_error_does_not_drop_later_files(bigraph, monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    tree = repo / "output" / "data_upload_test_v3"
    _write_json(
        tree / "aaa_raises_content_list_v2.json",
        [{"type": "text", "text": "boom"}],
    )
    ok_payload = [{"type": "text", "text": "later"}]
    _write_json(tree / "bbb_ok_content_list_v2.json", ok_payload)

    async def insert(raw, **kwargs):
        if "aaa_raises" in kwargs.get("file_path", ""):
            raise RuntimeError("ingest failed")

    stack = await _ingest(
        bigraph, monkeypatch, tmp_path, repo, insert_side_effect=insert
    )

    calls = stack.rag.insert_content_list.await_args_list
    assert [call.kwargs["file_path"] for call in calls] == [
        "output/data_upload_test_v3/aaa_raises_content_list_v2.json",
        "output/data_upload_test_v3/bbb_ok_content_list_v2.json",
    ]
    assert calls[1].args[0] == ok_payload
    stack.rag.finalize_storages.assert_awaited_once()
