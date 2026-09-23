"""Local HF ingest omitted ``-w`` must keep the ``rag_storage_wt1536`` store.

``scripts/batch_ingest_content_lists_local_hf.py`` defaults LightRAG persistence
to ``<repo>/rag_storage_wt1536``. Changing that default to ``rag_storage`` would
merge embedding-only JSON ingest into the demo Q&A index. An explicit ``-w``
must still override the default. Distinct from #183 (embedding dim defaults),
#191 (concurrency env), and #192 (``PARSER`` env).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

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
    "WORKING_DIR",
    "HF_HOME",
    "PARSER",
    "PARSE_METHOD",
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "batch_ingest_local_hf_omitted_working_dir_37e0", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bicl():
    return _load_script_module()


@pytest.fixture(autouse=True)
def _restore_environ_after_each_test():
    before = {key: os.environ.get(key) for key in _ENV_KEYS}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


class _CapturingLightRAG:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    async def initialize_storages(self):
        return None


class _CapturingRAG:
    last_config = None

    def __init__(self, **kwargs):
        type(self).last_config = kwargs.get("config")


async def _run_until_empty_tree(bicl, monkeypatch, tmp_path, argv):
    repo = tmp_path / "data"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    monkeypatch.delenv("WORKING_DIR", raising=False)
    monkeypatch.setattr(bicl, "_ROOT", checkout)
    monkeypatch.setattr(sys, "argv", argv + ["--data-repo-root", str(repo)])
    _CapturingLightRAG.last_kwargs = None
    _CapturingRAG.last_config = None

    with (
        patch(
            "raganything.local_hf_embedding.make_local_hf_embedding_func",
            return_value=object(),
        ),
        patch("lightrag.LightRAG", _CapturingLightRAG),
        patch("raganything.RAGAnything", _CapturingRAG),
    ):
        code = await bicl.async_main()

    assert code == 1
    assert _CapturingLightRAG.last_kwargs is not None
    assert _CapturingRAG.last_config is not None
    return checkout


@pytest.mark.asyncio
async def test_omitted_working_dir_uses_repo_rag_storage_wt1536(
    bicl, monkeypatch, tmp_path
):
    checkout = await _run_until_empty_tree(
        bicl,
        monkeypatch,
        tmp_path,
        ["batch_ingest_content_lists_local_hf"],
    )

    expected = str(checkout / "rag_storage_wt1536")
    assert _CapturingLightRAG.last_kwargs["working_dir"] == expected
    assert _CapturingRAG.last_config.working_dir == expected


@pytest.mark.asyncio
async def test_explicit_working_dir_overrides_wt1536_default(
    bicl, monkeypatch, tmp_path
):
    custom = tmp_path / "custom_store"
    await _run_until_empty_tree(
        bicl,
        monkeypatch,
        tmp_path,
        ["batch_ingest_content_lists_local_hf", "-w", str(custom)],
    )

    assert _CapturingLightRAG.last_kwargs["working_dir"] == str(custom)
    assert _CapturingRAG.last_config.working_dir == str(custom)
