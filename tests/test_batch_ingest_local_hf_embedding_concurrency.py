"""Local HF JSON ingest must honor embedding concurrency env overrides.

``scripts/batch_ingest_content_lists_local_hf.py`` defaults LightRAG to
``embedding_func_max_async=1`` / ``embedding_batch_num=1`` (#183). Operators
raise those env vars for local GPU encode, or keep them at 1 to avoid
over-subscribing CPU. Ignoring the env keeps the 1/1 defaults and stalls
overnight embedding-only ingest.

Distinct from #183 (local-HF default 1/1 and dim/model), #188 (pipeline/graph
constructor env), and #189 (demo CLI / OCR re-ingest override).
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

_ENV_KEYS = (
    "ALLOW_EMBEDDING_ONLY_INGESTION",
    "EMBEDDING_BACKEND",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "EMBEDDING_FUNC_MAX_ASYNC",
    "EMBEDDING_BATCH_NUM",
    "PARSER",
    "HF_HOME",
)


@pytest.fixture(scope="module")
def bicl():
    spec = importlib.util.spec_from_file_location(
        "batch_ingest_local_hf_embedding_concurrency_8875", SCRIPT_PATH
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
async def test_async_main_forwards_embedding_concurrency_env(
    bicl, monkeypatch, tmp_path
):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.setenv("EMBEDDING_FUNC_MAX_ASYNC", "4")
    monkeypatch.setenv("EMBEDDING_BATCH_NUM", "8")
    repo, wd = _write_v3_list(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(wd, repo))

    mock_rag = MagicMock()
    mock_rag.insert_content_list = AsyncMock()
    mock_rag.finalize_storages = AsyncMock()
    _CapturingLightRAG.last_kwargs = None

    with patch(
        "raganything.local_hf_embedding.make_local_hf_embedding_func",
        return_value=MagicMock(name="hf_embed"),
    ):
        with patch("lightrag.LightRAG", _CapturingLightRAG):
            with patch("raganything.RAGAnything", return_value=mock_rag):
                code = await bicl.async_main()

    assert code == 0
    assert _CapturingLightRAG.last_kwargs["embedding_func_max_async"] == 4
    assert _CapturingLightRAG.last_kwargs["embedding_batch_num"] == 8
    mock_rag.insert_content_list.assert_awaited_once()
    mock_rag.finalize_storages.assert_awaited_once()
