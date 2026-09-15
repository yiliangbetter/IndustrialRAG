"""Graph ingest must honor --content-list-glob and keep patterns relative.

``scripts/batch_ingest_content_lists_with_graph.py`` defaults to
``*_content_list_v2.json``. Export/MinerU also emit ``*_content_list.json``.
A wrong glob silently skips the OCR tree or ingests the wrong JSON generation.
Leading ``/`` must be stripped so rglob stays under the data subdir.
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
        "batch_ingest_graph_content_list_glob_be19", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


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


def _run_patches(stack):
    return (
        patch("lightrag.utils.EmbeddingFunc", return_value=stack.embedding),
        patch("lightrag.LightRAG", return_value=stack.lightrag),
        patch("raganything.RAGAnything", side_effect=stack.rag_ctor),
    )


def _ingested_names(stack) -> list[str]:
    return [
        Path(call.kwargs["file_path"]).name
        for call in stack.rag.insert_content_list.await_args_list
    ]


@pytest.mark.asyncio
async def test_default_glob_ingests_v2_and_ignores_v1_json(
    bigraph, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    out = repo / "output" / "data_upload_test_v3"
    v2 = _write_json(out, "manual_content_list_v2.json", [{"type": "text", "text": "v2"}])
    _write_json(out, "manual_content_list.json", [{"type": "text", "text": "v1"}])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
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
    p_emb, p_lr, p_rag = _run_patches(stack)
    with p_emb, p_lr, p_rag:
        await bigraph.async_main()

    assert _ingested_names(stack) == ["manual_content_list_v2.json"]
    call = stack.rag.insert_content_list.await_args
    assert call.args[0] == [{"type": "text", "text": "v2"}]
    assert Path(call.kwargs["file_path"]) == v2.resolve().relative_to(repo.resolve())


@pytest.mark.asyncio
async def test_custom_glob_selects_v1_json_and_skips_v2(
    bigraph, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    out = repo / "output" / "data_upload_test_v3"
    v1 = _write_json(out, "manual_content_list.json", [{"type": "text", "text": "v1"}])
    _write_json(out, "manual_content_list_v2.json", [{"type": "text", "text": "v2"}])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_with_graph",
            "-w",
            str(tmp_path / "wd"),
            "--data-repo-root",
            str(repo),
            "--content-list-glob",
            "*_content_list.json",
        ],
    )

    stack = _IngestStack()
    p_emb, p_lr, p_rag = _run_patches(stack)
    with p_emb, p_lr, p_rag:
        await bigraph.async_main()

    assert _ingested_names(stack) == ["manual_content_list.json"]
    call = stack.rag.insert_content_list.await_args
    assert call.args[0] == [{"type": "text", "text": "v1"}]
    assert Path(call.kwargs["file_path"]) == v1.resolve().relative_to(repo.resolve())


@pytest.mark.asyncio
async def test_leading_slash_glob_stays_relative_to_data_subdir(
    bigraph, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    out = repo / "output" / "data_upload_test_v3"
    nested = out / "plant-a"
    _write_json(
        nested, "nested_content_list_v2.json", [{"type": "text", "text": "nested"}]
    )
    outside = tmp_path / "outside_content_list_v2.json"
    outside.write_text(
        json.dumps([{"type": "text", "text": "outside"}]), encoding="utf-8"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_with_graph",
            "-w",
            str(tmp_path / "wd"),
            "--data-repo-root",
            str(repo),
            "--content-list-glob",
            "/*_content_list_v2.json",
        ],
    )

    stack = _IngestStack()
    p_emb, p_lr, p_rag = _run_patches(stack)
    with p_emb, p_lr, p_rag:
        await bigraph.async_main()

    names = _ingested_names(stack)
    assert names == ["nested_content_list_v2.json"]
    rel = Path(stack.rag.insert_content_list.await_args.kwargs["file_path"])
    assert rel == Path("output/data_upload_test_v3/plant-a/nested_content_list_v2.json")
