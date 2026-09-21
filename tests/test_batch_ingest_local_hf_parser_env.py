"""Local HF JSON ingest must honor PARSER env at RAGAnything construction.

``scripts/batch_ingest_content_lists_local_hf.py`` does not parse files, but
``RAGAnything.__post_init__`` still calls ``get_parser(config.parser)``. Ignoring
``PARSER`` (or defaulting away from mineru) fails embedding-only ingest at
startup when the operator swapped MinerU for Docling/PaddleOCR.

``PARSE_METHOD`` must not leak into this JSON path (``parse_method`` stays
hardcoded ``auto``). Distinct from #185 (pipeline PARSER / reingest PARSER) and
#191 (graph-ingest PARSER).
"""

from __future__ import annotations

import importlib.util
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
    "PARSER",
    "PARSE_METHOD",
    "HF_HOME",
)


@pytest.fixture(scope="module")
def bicl():
    spec = importlib.util.spec_from_file_location(
        "batch_ingest_local_hf_parser_env_d920", SCRIPT_PATH
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


class _CapturingRAG:
    last_config = None

    def __init__(self, **kwargs):
        type(self).last_config = kwargs.get("config")
        self.insert_content_list = AsyncMock()
        self.finalize_storages = AsyncMock()


class _FakeLightRAG:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.initialize_storages = AsyncMock()
        self.finalize_storages = AsyncMock()


def _empty_v3_tree(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    wd = tmp_path / "wd"
    wd.mkdir()
    return repo, wd


async def _run_until_constructed(bicl, monkeypatch, tmp_path):
    repo, wd = _empty_v3_tree(tmp_path)
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
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
    _CapturingRAG.last_config = None
    with (
        patch(
            "raganything.local_hf_embedding.make_local_hf_embedding_func",
            return_value=MagicMock(name="hf_embed"),
        ),
        patch("lightrag.LightRAG", _FakeLightRAG),
        patch("raganything.RAGAnything", _CapturingRAG),
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback"),
    ):
        code = await bicl.async_main()
    assert code == 1
    assert _CapturingRAG.last_config is not None
    return _CapturingRAG.last_config


@pytest.mark.asyncio
async def test_local_hf_parser_defaults_to_mineru(bicl, monkeypatch, tmp_path):
    monkeypatch.delenv("PARSER", raising=False)
    monkeypatch.delenv("PARSE_METHOD", raising=False)

    config = await _run_until_constructed(bicl, monkeypatch, tmp_path)

    assert config.parser == "mineru"
    assert config.parse_method == "auto"


@pytest.mark.asyncio
async def test_local_hf_parser_env_is_forwarded(bicl, monkeypatch, tmp_path):
    monkeypatch.setenv("PARSER", "paddleocr")
    # PARSE_METHOD must not leak into this JSON path (parse_method is hardcoded).
    monkeypatch.setenv("PARSE_METHOD", "ocr")

    config = await _run_until_constructed(bicl, monkeypatch, tmp_path)

    assert config.parser == "paddleocr"
    assert config.parse_method == "auto"
