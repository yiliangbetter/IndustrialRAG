"""Unit tests for ``scripts/batch_ingest_content_lists_with_graph.py``.

Covers fail-closed credential/data gates and ingest orchestration used by the
text-first knowledge-graph path. LightRAG/RAGAnything are mocked so tests stay
deterministic and do not call network or LLM backends.
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
    name = "batch_ingest_content_lists_with_graph_under_test"
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
        "HF_HOME",
        "LLM_MODEL",
        "EMBEDDING_DIM",
        "EMBEDDING_MODEL",
    )
    before = {key: os.environ.get(key) for key in keys}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _argv(script_name, working_dir, data_repo=None, extra=None):
    args = [script_name, "-w", str(working_dir)]
    if data_repo is not None:
        args.extend(["--data-repo-root", str(data_repo)])
    if extra:
        args.extend(extra)
    return args


def _write_content_list(repo: Path, name: str, payload) -> Path:
    out = repo / "output" / "data_upload_test_v3"
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
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
        self.rag_ctor_kwargs = {}

    def rag_ctor(self, *args, **kwargs):
        self.rag_ctor_kwargs = kwargs
        return self.rag


def test_main_invokes_async_main(bigraph, monkeypatch):
    called = {}

    def fake_run(coro):
        called["coro"] = coro
        coro.close()
        return None

    monkeypatch.setattr(bigraph.asyncio, "run", fake_run)
    bigraph.main()
    assert "coro" in called


@pytest.mark.asyncio
async def test_async_main_exits_when_data_root_missing(bigraph, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        sys,
        "argv",
        _argv("batch_ingest_content_lists_with_graph", tmp_path / "wd", tmp_path),
    )
    with pytest.raises(SystemExit, match="Data root does not exist"):
        await bigraph.async_main()


@pytest.mark.asyncio
async def test_async_main_exits_when_llm_api_key_missing(bigraph, monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BINDING_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        _argv("batch_ingest_content_lists_with_graph", tmp_path / "wd", repo),
    )
    with pytest.raises(SystemExit, match="OPENAI_API_KEY or LLM_BINDING_API_KEY"):
        await bigraph.async_main()


@pytest.mark.asyncio
async def test_async_main_exits_when_embedding_host_set_without_key(
    bigraph, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "http://embed.local/v1")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        _argv("batch_ingest_content_lists_with_graph", tmp_path / "wd", repo),
    )
    with pytest.raises(SystemExit, match="EMBEDDING_API_KEY"):
        await bigraph.async_main()


@pytest.mark.asyncio
async def test_hf_backend_skips_embedding_host_key_gate(bigraph, monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "http://embed.local/v1")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        _argv("batch_ingest_content_lists_with_graph", tmp_path / "wd", repo),
    )
    stack = _IngestStack()
    with patch(
        "raganything.local_hf_embedding.make_local_hf_embedding_func",
        return_value=stack.embedding,
    ):
        with patch("lightrag.LightRAG", return_value=stack.lightrag):
            with patch("raganything.RAGAnything", side_effect=stack.rag_ctor):
                with pytest.raises(SystemExit, match="No files matching"):
                    await bigraph.async_main()

    stack.lightrag.initialize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_main_ingests_list_json_and_skips_multimodal_by_default(
    bigraph, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    payload = [{"type": "text", "text": "graph ingest"}]
    json_path = _write_content_list(repo, "doc_content_list_v2.json", payload)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setattr(
        sys,
        "argv",
        _argv("batch_ingest_content_lists_with_graph", tmp_path / "wd", repo),
    )

    stack = _IngestStack()
    with patch("lightrag.utils.EmbeddingFunc", return_value=stack.embedding):
        with patch("lightrag.LightRAG", return_value=stack.lightrag):
            with patch("raganything.RAGAnything", side_effect=stack.rag_ctor):
                await bigraph.async_main()

    config = stack.rag_ctor_kwargs["config"]
    assert config.allow_embedding_only_ingestion is False
    stack.rag.insert_content_list.assert_awaited_once()
    call = stack.rag.insert_content_list.await_args
    assert call.args[0] == payload
    assert Path(call.kwargs["file_path"]) == json_path.resolve().relative_to(
        repo.resolve()
    )
    assert call.kwargs["skip_multimodal_processing"] is True
    stack.rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_skip_multimodal_flag_is_forwarded(bigraph, monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    _write_content_list(repo, "doc_content_list_v2.json", [{"type": "text", "text": "x"}])
    monkeypatch.setenv("LLM_BINDING_API_KEY", "sk-from-binding")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            "batch_ingest_content_lists_with_graph",
            tmp_path / "wd",
            repo,
            extra=["--no-skip-multimodal"],
        ),
    )

    stack = _IngestStack()
    with patch("lightrag.utils.EmbeddingFunc", return_value=stack.embedding):
        with patch("lightrag.LightRAG", return_value=stack.lightrag):
            with patch("raganything.RAGAnything", side_effect=stack.rag_ctor):
                await bigraph.async_main()

    call = stack.rag.insert_content_list.await_args
    assert call.kwargs["skip_multimodal_processing"] is False


@pytest.mark.asyncio
async def test_non_list_json_is_counted_as_failure_and_skipped(
    bigraph, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    _write_content_list(repo, "bad_content_list_v2.json", {"not": "a list"})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.setattr(
        sys,
        "argv",
        _argv("batch_ingest_content_lists_with_graph", tmp_path / "wd", repo),
    )

    stack = _IngestStack()
    with patch("lightrag.utils.EmbeddingFunc", return_value=stack.embedding):
        with patch("lightrag.LightRAG", return_value=stack.lightrag):
            with patch("raganything.RAGAnything", side_effect=stack.rag_ctor):
                await bigraph.async_main()

    stack.rag.insert_content_list.assert_not_called()
    stack.rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_exception_continues_and_still_finalizes(
    bigraph, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    _write_content_list(repo, "a_content_list_v2.json", [{"type": "text", "text": "a"}])
    _write_content_list(repo, "b_content_list_v2.json", [{"type": "text", "text": "b"}])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.setattr(
        sys,
        "argv",
        _argv("batch_ingest_content_lists_with_graph", tmp_path / "wd", repo),
    )

    stack = _IngestStack()
    stack.rag.insert_content_list = AsyncMock(
        side_effect=[RuntimeError("boom"), None]
    )
    with patch("lightrag.utils.EmbeddingFunc", return_value=stack.embedding):
        with patch("lightrag.LightRAG", return_value=stack.lightrag):
            with patch("raganything.RAGAnything", side_effect=stack.rag_ctor):
                await bigraph.async_main()

    assert stack.rag.insert_content_list.await_count == 2
    stack.rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_limit_truncates_ingest_list(bigraph, monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    _write_content_list(repo, "a_content_list_v2.json", [{"type": "text", "text": "a"}])
    _write_content_list(repo, "b_content_list_v2.json", [{"type": "text", "text": "b"}])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            "batch_ingest_content_lists_with_graph",
            tmp_path / "wd",
            repo,
            extra=["--limit", "1"],
        ),
    )

    stack = _IngestStack()
    with patch("lightrag.utils.EmbeddingFunc", return_value=stack.embedding):
        with patch("lightrag.LightRAG", return_value=stack.lightrag):
            with patch("raganything.RAGAnything", side_effect=stack.rag_ctor):
                await bigraph.async_main()

    assert stack.rag.insert_content_list.await_count == 1


@pytest.mark.asyncio
async def test_data_repo_env_used_when_cli_root_omitted(bigraph, monkeypatch, tmp_path):
    repo = tmp_path / "from-env"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.setenv("RAG_DATA_REPO", str(repo))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_with_graph",
            "-w",
            str(tmp_path / "wd"),
        ],
    )

    with pytest.raises(SystemExit, match="No files matching"):
        with patch("lightrag.utils.EmbeddingFunc", return_value=MagicMock()):
            with patch("lightrag.LightRAG") as mock_lr:
                mock_lr.return_value.initialize_storages = AsyncMock()
                with patch("raganything.RAGAnything"):
                    await bigraph.async_main()
