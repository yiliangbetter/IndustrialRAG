"""Graph JSON ingest must apply ``--limit`` to the sorted file prefix.

``--limit`` only counted inserts. A nested content list whose basename sorts
early, or a directory whose name matches the glob, could take a slot that
belongs to another manual. ``--limit 0`` means every list, which is the
contract this script documents.
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


def _load_script_module():
    name = "batch_ingest_content_lists_with_graph_limit_order"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bigraph():
    return _load_script_module()


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
    )
    before = {key: os.environ.get(key) for key in keys}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _write_tree(repo: Path) -> dict[str, list]:
    """Create lists out of sorted-path order, plus a decoy directory and v1 file."""
    root = repo / "output" / "data_upload_test_v3"
    payloads = {
        "zeta_content_list_v2.json": [{"type": "text", "text": "zeta valve"}],
        "sub/b_content_list_v2.json": [{"type": "text", "text": "bearing spec"}],
        "m_content_list_v2.json": [{"type": "text", "text": "manual m"}],
    }
    # Write the later sorted name first so an unsorted walk is not already sorted.
    (root).mkdir(parents=True, exist_ok=True)
    (root / "zeta_content_list_v2.json").write_text(
        json.dumps(payloads["zeta_content_list_v2.json"]), encoding="utf-8"
    )
    (root / "aaa_content_list_v2.json").mkdir()
    (root / "sub").mkdir()
    (root / "sub" / "b_content_list_v2.json").write_text(
        json.dumps(payloads["sub/b_content_list_v2.json"]), encoding="utf-8"
    )
    (root / "m_content_list_v2.json").write_text(
        json.dumps(payloads["m_content_list_v2.json"]), encoding="utf-8"
    )
    (root / "legacy_content_list.json").write_text(
        json.dumps([{"type": "text", "text": "v1 should skip"}]),
        encoding="utf-8",
    )
    return payloads


def _rel(name: str) -> str:
    return str(Path("output") / "data_upload_test_v3" / name)


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


async def _run(bigraph, monkeypatch, repo: Path, limit: str):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_with_graph",
            "-w",
            str(repo.parent / "wd"),
            "--data-repo-root",
            str(repo),
            "--limit",
            limit,
        ],
    )
    stack = _IngestStack()
    with patch("lightrag.utils.EmbeddingFunc", return_value=stack.embedding):
        with patch("lightrag.LightRAG", return_value=stack.lightrag):
            with patch("raganything.RAGAnything", side_effect=stack.rag_ctor):
                await bigraph.async_main()
    return stack


def _inserted(stack: _IngestStack) -> list[tuple[str, list]]:
    rows = []
    for call in stack.rag.insert_content_list.await_args_list:
        assert call.kwargs["skip_multimodal_processing"] is True
        rows.append((call.kwargs["file_path"], call.args[0]))
    return rows


@pytest.mark.asyncio
async def test_limit_keeps_sorted_file_prefix(bigraph, monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    payloads = _write_tree(repo)
    stack = await _run(bigraph, monkeypatch, repo, "2")

    assert _inserted(stack) == [
        (_rel("m_content_list_v2.json"), payloads["m_content_list_v2.json"]),
        (_rel("sub/b_content_list_v2.json"), payloads["sub/b_content_list_v2.json"]),
    ]
    stack.rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_limit_zero_ingests_every_list_in_sorted_order(
    bigraph, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    payloads = _write_tree(repo)
    stack = await _run(bigraph, monkeypatch, repo, "0")

    assert _inserted(stack) == [
        (_rel("m_content_list_v2.json"), payloads["m_content_list_v2.json"]),
        (_rel("sub/b_content_list_v2.json"), payloads["sub/b_content_list_v2.json"]),
        (_rel("zeta_content_list_v2.json"), payloads["zeta_content_list_v2.json"]),
    ]
    stack.rag.finalize_storages.assert_awaited_once()
