"""Local HF batch ingest path resolution and per-file isolation.

Industrial embedding-only jobs often set RAG_DATA_REPO instead of passing
--data-repo-root. A one-file failure must not abort the rest of the tree or
skip finalize.
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
SCRIPT_PATH = REPO_ROOT / "scripts" / "batch_ingest_content_lists_local_hf.py"


def _load_script_module():
    name = "batch_ingest_content_lists_local_hf_repo_env_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bicl():
    return _load_script_module()


@pytest.fixture(autouse=True)
def _restore_environ_after_each_test():
    keys = (
        "ALLOW_EMBEDDING_ONLY_INGESTION",
        "EMBEDDING_BACKEND",
        "RAG_DATA_REPO",
    )
    before = {key: os.environ.get(key) for key in keys}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _write_content_list(repo: Path, name: str, payload) -> Path:
    out = repo / "output" / "data_upload_test_v3"
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _stack():
    fake_emb = MagicMock(name="embedding_func")
    mock_lr = MagicMock()
    mock_lr.initialize_storages = AsyncMock()
    mock_lr.finalize_storages = AsyncMock()
    mock_rag = MagicMock()
    mock_rag.insert_content_list = AsyncMock()
    mock_rag.finalize_storages = AsyncMock()
    return fake_emb, mock_lr, mock_rag


@pytest.mark.asyncio
async def test_data_repo_env_used_when_cli_root_omitted(bicl, monkeypatch, tmp_path):
    repo = tmp_path / "from-env"
    _write_content_list(
        repo, "doc_content_list_v2.json", [{"type": "text", "text": "ok"}]
    )
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.setenv("RAG_DATA_REPO", str(repo))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_local_hf",
            "-w",
            str(tmp_path / "wd"),
        ],
    )

    fake_emb, mock_lr, mock_rag = _stack()
    with patch(
        "raganything.local_hf_embedding.make_local_hf_embedding_func",
        return_value=fake_emb,
    ):
        with patch("lightrag.LightRAG", return_value=mock_lr):
            with patch("raganything.RAGAnything", return_value=mock_rag):
                code = await bicl.async_main()

    assert code == 0
    mock_rag.insert_content_list.assert_awaited_once()
    call = mock_rag.insert_content_list.await_args
    assert call.args[0] == [{"type": "text", "text": "ok"}]
    mock_rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_one_ingest_failure_still_ingests_sibling_and_finalizes(
    bicl, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    _write_content_list(repo, "a_content_list_v2.json", [{"type": "text", "text": "a"}])
    _write_content_list(repo, "b_content_list_v2.json", [{"type": "text", "text": "b"}])
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_local_hf",
            "-w",
            str(tmp_path / "wd"),
            "--data-repo-root",
            str(repo),
        ],
    )

    fake_emb, mock_lr, mock_rag = _stack()
    mock_rag.insert_content_list = AsyncMock(side_effect=[RuntimeError("boom"), None])

    with patch(
        "raganything.local_hf_embedding.make_local_hf_embedding_func",
        return_value=fake_emb,
    ):
        with patch("lightrag.LightRAG", return_value=mock_lr):
            with patch("raganything.RAGAnything", return_value=mock_rag):
                code = await bicl.async_main()

    assert code == 1
    assert mock_rag.insert_content_list.await_count == 2
    mock_rag.finalize_storages.assert_awaited_once()
