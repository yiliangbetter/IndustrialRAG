"""Local HF batch ingest is pinned to the v3 MinerU JSON tree.

``scripts/batch_ingest_content_lists_local_hf.py`` always scans
``output/data_upload_test_v3``. Graph ingest can target OCR v4 via
``RAG_DATA_UPLOAD_SUBDIR``; this embedding-only path must not silently
follow that env and ingest a different tree (wrong vectors / missed manuals).
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


@pytest.fixture(scope="module")
def bicl():
    spec = importlib.util.spec_from_file_location(
        "batch_ingest_local_hf_ignores_v4_be19", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _restore_environ_after_each_test():
    keys = (
        "ALLOW_EMBEDDING_ONLY_INGESTION",
        "EMBEDDING_BACKEND",
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


def _write_list(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"type": "text", "text": text}]), encoding="utf-8")


@pytest.mark.asyncio
async def test_local_hf_ingests_v3_and_ignores_v4_even_when_subdir_env_set(
    bicl, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    v3 = repo / "output" / "data_upload_test_v3" / "plant" / "v3_content_list_v2.json"
    v4 = repo / "output" / "data_upload_test_v4" / "plant" / "v4_content_list_v2.json"
    _write_list(v3, "from-v3")
    _write_list(v4, "from-v4")
    wd = tmp_path / "wd"
    wd.mkdir()

    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.setenv("RAG_DATA_UPLOAD_SUBDIR", "data_upload_test_v4")
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
                code = await bicl.async_main()

    assert code == 0
    mock_rag.insert_content_list.assert_awaited_once()
    call = mock_rag.insert_content_list.await_args
    assert call.args[0] == [{"type": "text", "text": "from-v3"}]
    rel = Path(call.kwargs["file_path"])
    assert rel == Path("output/data_upload_test_v3/plant/v3_content_list_v2.json")
    mock_rag.finalize_storages.assert_awaited_once()
