"""Blank RAG_DATA_UPLOAD_SUBDIR must fall back to data_upload_test_v3.

scripts/batch_ingest_content_lists_with_graph.py uses
``os.getenv(...).strip() or "data_upload_test_v3"``. An empty or
whitespace env value must not glob ``output/`` itself — that would ingest
v3 and v4 trees together and mix OCR generations into one KG.
Distinct from #180 (named RAG_DATA_UPLOAD_SUBDIR tree selection) and #181
(v3 pin on the local-HF path / content-list glob).
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
    "RAG_DATA_REPO",
    "RAG_DATA_UPLOAD_SUBDIR",
    "HF_HOME",
)


@pytest.fixture(scope="module")
def bigraph():
    spec = importlib.util.spec_from_file_location(
        "batch_ingest_graph_blank_upload_subdir_8bd3", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _restore_environ_after_each_test():
    before = {key: os.environ.get(key) for key in _ENV_KEYS}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _write_json(root: Path, name: str, payload) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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


def _ingested_names(stack) -> list[str]:
    return [
        Path(call.kwargs["file_path"]).name
        for call in stack.rag.insert_content_list.await_args_list
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("subdir_env", ["", "   "])
async def test_blank_upload_subdir_env_stays_on_v3_and_ignores_v4(
    bigraph, monkeypatch, tmp_path, subdir_env
):
    repo = tmp_path / "repo"
    v3 = repo / "output" / "data_upload_test_v3"
    v4 = repo / "output" / "data_upload_test_v4"
    _write_json(v3, "v3_content_list_v2.json", [{"type": "text", "text": "v3"}])
    _write_json(v4, "v4_content_list_v2.json", [{"type": "text", "text": "v4"}])

    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("RAG_DATA_UPLOAD_SUBDIR", subdir_env)
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
    with (
        patch("lightrag.utils.EmbeddingFunc", return_value=stack.embedding),
        patch("lightrag.LightRAG", return_value=stack.lightrag),
        patch("raganything.RAGAnything", side_effect=stack.rag_ctor),
    ):
        await bigraph.async_main()

    assert _ingested_names(stack) == ["v3_content_list_v2.json"]
    call = stack.rag.insert_content_list.await_args
    assert call.args[0] == [{"type": "text", "text": "v3"}]
    assert "data_upload_test_v3" in call.kwargs["file_path"]
    assert "data_upload_test_v4" not in call.kwargs["file_path"]
    stack.rag.finalize_storages.assert_awaited_once()
