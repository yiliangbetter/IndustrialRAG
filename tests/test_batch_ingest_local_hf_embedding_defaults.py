"""Local HF JSON ingest must default to bge-m3 / 1024-d vectors.

``scripts/batch_ingest_content_lists_local_hf.py`` is the offline embedding-only
path. A wrong default ``EMBEDDING_DIM`` or ``EMBEDDING_MODEL`` silently writes
vectors into a store that later queries (bge-m3 1024-d) cannot search.

Existing tests cover CLI exit codes, ingest isolation, and embedding-only
config flags — not the factory arguments passed to
``make_local_hf_embedding_func``. Distinct from #179 (pipeline ``_build_rag``
dims) and #180 (graph-ingest dims).
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
        "batch_ingest_local_hf_embedding_defaults_afad", SCRIPT_PATH
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
        "EMBEDDING_FUNC_MAX_ASYNC",
        "EMBEDDING_BATCH_NUM",
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


def _write_v3_list(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    out = repo / "output" / "data_upload_test_v3"
    out.mkdir(parents=True)
    (out / "doc_content_list_v2.json").write_text(
        json.dumps([{"type": "text", "text": "hello"}]), encoding="utf-8"
    )
    wd = tmp_path / "wd"
    wd.mkdir()
    return repo, wd


def _argv(wd: Path, repo: Path) -> list[str]:
    return [
        "batch_ingest_content_lists_local_hf",
        "-w",
        str(wd),
        "--data-repo-root",
        str(repo),
    ]


class _CapturingLightRAG:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.initialize_storages = AsyncMock()


@pytest.mark.asyncio
async def test_async_main_defaults_to_bge_m3_dim_1024(bicl, monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_FUNC_MAX_ASYNC", raising=False)
    monkeypatch.delenv("EMBEDDING_BATCH_NUM", raising=False)
    repo, wd = _write_v3_list(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(wd, repo))

    hf_calls = []
    mock_rag = MagicMock()
    mock_rag.insert_content_list = AsyncMock()
    mock_rag.finalize_storages = AsyncMock()

    def fake_hf(embedding_dim, embedding_model=None):
        hf_calls.append(
            {"embedding_dim": embedding_dim, "embedding_model": embedding_model}
        )
        return MagicMock(name="hf_embed")

    with patch(
        "raganything.local_hf_embedding.make_local_hf_embedding_func",
        side_effect=fake_hf,
    ):
        with patch("lightrag.LightRAG", _CapturingLightRAG):
            with patch("raganything.RAGAnything", return_value=mock_rag):
                code = await bicl.async_main()

    assert code == 0
    assert hf_calls == [
        {"embedding_dim": 1024, "embedding_model": "BAAI/bge-m3"},
    ]
    assert _CapturingLightRAG.last_kwargs["embedding_func_max_async"] == 1
    assert _CapturingLightRAG.last_kwargs["embedding_batch_num"] == 1
    mock_rag.insert_content_list.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_main_forwards_custom_dim_and_model(bicl, monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    monkeypatch.setenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    repo, wd = _write_v3_list(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(wd, repo))

    hf_calls = []
    mock_rag = MagicMock()
    mock_rag.insert_content_list = AsyncMock()
    mock_rag.finalize_storages = AsyncMock()

    def fake_hf(embedding_dim, embedding_model=None):
        hf_calls.append(
            {"embedding_dim": embedding_dim, "embedding_model": embedding_model}
        )
        return MagicMock(name="hf_embed")

    with patch(
        "raganything.local_hf_embedding.make_local_hf_embedding_func",
        side_effect=fake_hf,
    ):
        with patch("lightrag.LightRAG", _CapturingLightRAG):
            with patch("raganything.RAGAnything", return_value=mock_rag):
                code = await bicl.async_main()

    assert code == 0
    assert hf_calls == [
        {
            "embedding_dim": 768,
            "embedding_model": "intfloat/multilingual-e5-base",
        },
    ]
