"""Unit tests for ``scripts/batch_ingest_content_lists_local_hf.py`` (main / async_main)."""

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
    name = "batch_ingest_content_lists_local_hf_under_test"
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
    """async_main mutates os.environ; avoid leaking into other test modules."""
    before = {
        "ALLOW_EMBEDDING_ONLY_INGESTION": os.environ.get(
            "ALLOW_EMBEDDING_ONLY_INGESTION"
        ),
        "EMBEDDING_BACKEND": os.environ.get("EMBEDDING_BACKEND"),
    }
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def test_main_raises_system_exit_with_async_main_return_value(bicl, monkeypatch):
    def fake_run(coro):
        coro.close()
        return 43

    monkeypatch.setattr(bicl.asyncio, "run", fake_run)
    with pytest.raises(SystemExit) as exc_info:
        bicl.main()
    assert exc_info.value.code == 43


@pytest.mark.asyncio
async def test_async_main_returns_2_when_embedding_backend_not_hf(
    bicl, monkeypatch, tmp_path
):
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_local_hf",
            "-w",
            str(tmp_path / "wd"),
            "--data-repo-root",
            str(tmp_path),
        ],
    )
    code = await bicl.async_main()
    assert code == 2


@pytest.mark.asyncio
async def test_async_main_returns_1_when_no_content_list_json(
    bicl, monkeypatch, tmp_path
):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    wd = tmp_path / "wd"
    wd.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_local_hf",
            "-w",
            str(wd),
            "--data-repo-root",
            str(repo),
        ],
    )

    fake_emb = MagicMock(name="embedding_func")

    mock_lr_instance = MagicMock()
    mock_lr_instance.initialize_storages = AsyncMock()
    mock_lr_instance.finalize_storages = AsyncMock()

    mock_rag = MagicMock()
    mock_rag.insert_content_list = AsyncMock()
    mock_rag.finalize_storages = AsyncMock()

    with patch(
        "raganything.local_hf_embedding.make_local_hf_embedding_func",
        return_value=fake_emb,
    ):
        with patch("lightrag.LightRAG", return_value=mock_lr_instance):
            with patch("raganything.RAGAnything", return_value=mock_rag):
                code = await bicl.async_main()

    assert code == 1
    mock_lr_instance.initialize_storages.assert_awaited_once()
    mock_rag.insert_content_list.assert_not_called()
    mock_rag.finalize_storages.assert_not_called()


@pytest.mark.asyncio
async def test_async_main_returns_0_on_successful_ingest(bicl, monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    repo = tmp_path / "repo"
    out = repo / "output" / "data_upload_test_v3"
    out.mkdir(parents=True)
    json_path = out / "doc_content_list_v2.json"
    json_path.write_text(
        json.dumps([{"type": "text", "text": "hello"}]), encoding="utf-8"
    )
    wd = tmp_path / "wd"
    wd.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_local_hf",
            "-w",
            str(wd),
            "--data-repo-root",
            str(repo),
        ],
    )

    fake_emb = MagicMock(name="embedding_func")
    mock_lr_instance = MagicMock()
    mock_lr_instance.initialize_storages = AsyncMock()
    mock_lr_instance.finalize_storages = AsyncMock()
    mock_rag = MagicMock()
    mock_rag.insert_content_list = AsyncMock()
    mock_rag.finalize_storages = AsyncMock()

    with patch(
        "raganything.local_hf_embedding.make_local_hf_embedding_func",
        return_value=fake_emb,
    ):
        with patch("lightrag.LightRAG", return_value=mock_lr_instance):
            with patch("raganything.RAGAnything", return_value=mock_rag):
                code = await bicl.async_main()

    assert code == 0
    mock_rag.insert_content_list.assert_awaited_once()
    call = mock_rag.insert_content_list.await_args
    assert call.args[0] == [{"type": "text", "text": "hello"}]
    assert call.kwargs.get("file_path", "").endswith("doc_content_list_v2.json")
    mock_rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_main_returns_1_when_ingest_raises(bicl, monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    repo = tmp_path / "repo"
    out = repo / "output" / "data_upload_test_v3"
    out.mkdir(parents=True)
    (out / "bad_content_list_v2.json").write_text(
        json.dumps([{"type": "text", "text": "x"}]), encoding="utf-8"
    )
    wd = tmp_path / "wd"
    wd.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_local_hf",
            "-w",
            str(wd),
            "--data-repo-root",
            str(repo),
        ],
    )

    fake_emb = MagicMock()
    mock_lr = MagicMock()
    mock_lr.initialize_storages = AsyncMock()
    mock_lr.finalize_storages = AsyncMock()
    mock_rag = MagicMock()
    mock_rag.insert_content_list = AsyncMock(side_effect=RuntimeError("boom"))
    mock_rag.finalize_storages = AsyncMock()

    with patch(
        "raganything.local_hf_embedding.make_local_hf_embedding_func",
        return_value=fake_emb,
    ):
        with patch("lightrag.LightRAG", return_value=mock_lr):
            with patch("raganything.RAGAnything", return_value=mock_rag):
                code = await bicl.async_main()

    assert code == 1


@pytest.mark.asyncio
async def test_async_main_skips_non_list_json_counts_as_failure(
    bicl, monkeypatch, tmp_path
):
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
            "batch_ingest_content_lists_local_hf",
            "-w",
            str(wd),
            "--data-repo-root",
            str(repo),
        ],
    )

    fake_emb = MagicMock()
    mock_lr = MagicMock()
    mock_lr.initialize_storages = AsyncMock()
    mock_lr.finalize_storages = AsyncMock()
    mock_rag = MagicMock()
    mock_rag.insert_content_list = AsyncMock()
    mock_rag.finalize_storages = AsyncMock()

    with patch(
        "raganything.local_hf_embedding.make_local_hf_embedding_func",
        return_value=fake_emb,
    ):
        with patch("lightrag.LightRAG", return_value=mock_lr):
            with patch("raganything.RAGAnything", return_value=mock_rag):
                code = await bicl.async_main()

    assert code == 1
    mock_rag.insert_content_list.assert_not_called()
