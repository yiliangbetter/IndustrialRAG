"""Local HF batch ingest must force embedding-only config flags.

``scripts/batch_ingest_content_lists_local_hf.py`` is the offline vector
path: no LLM extraction, no vision/table/equation processors. Dropping
``allow_embedding_only_ingestion=True`` would send MinerU JSON through
LightRAG entity extraction and fail without an LLM. Re-enabling modal
processors would add vision cost to an embedding-only job.

Existing tests on this script cover CLI exit codes and ingest isolation,
not the RAGAnythingConfig flags.
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
        "batch_ingest_local_hf_embedding_only_config_44c5", SCRIPT_PATH
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
        "EMBEDDING_DIM",
        "EMBEDDING_MODEL",
        "PARSER",
        "HF_HOME",
    )
    before = {key: os.environ.get(key) for key in keys}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


@pytest.mark.asyncio
async def test_async_main_forces_embedding_only_and_disables_modal_processors(
    bicl, monkeypatch, tmp_path
):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    repo = tmp_path / "repo"
    out = repo / "output" / "data_upload_test_v3"
    out.mkdir(parents=True)
    (out / "doc_content_list_v2.json").write_text(
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
    mock_lr = MagicMock()
    mock_lr.initialize_storages = AsyncMock()
    mock_rag = MagicMock()
    mock_rag.insert_content_list = AsyncMock()
    mock_rag.finalize_storages = AsyncMock()
    captured = {}

    def rag_ctor(*args, **kwargs):
        captured["kwargs"] = kwargs
        return mock_rag

    with patch(
        "raganything.local_hf_embedding.make_local_hf_embedding_func",
        return_value=fake_emb,
    ):
        with patch("lightrag.LightRAG", return_value=mock_lr):
            with patch("raganything.RAGAnything", side_effect=rag_ctor):
                code = await bicl.async_main()

    assert code == 0
    config = captured["kwargs"]["config"]
    assert config.allow_embedding_only_ingestion is True
    assert config.enable_image_processing is False
    assert config.enable_table_processing is False
    assert config.enable_equation_processing is False
    assert config.working_dir == str(wd)
    assert os.environ["ALLOW_EMBEDDING_ONLY_INGESTION"] == "true"
    assert os.environ["EMBEDDING_BACKEND"] == "hf"
    mock_rag.insert_content_list.assert_awaited_once()
    mock_rag.finalize_storages.assert_awaited_once()
