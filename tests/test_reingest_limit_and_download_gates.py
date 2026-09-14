"""OCR re-ingest must fail closed on empty folders and honor --limit / download skip.

scripts/reingest_uploaded_documents_ocr.py is the production OCR path. A folder
with no supported files must not start LightRAG processing. --limit truncates
the sorted top-level list; nested manuals are ignored. Model download is on by
default and skipped only with --skip-model-download.

Distinct from #130 (missing folder / HF-offline / NO_PROXY) and #177 (pipeline
OCR_LANG + opt-in download).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    path = REPO_ROOT / "scripts" / "reingest_uploaded_documents_ocr.py"
    spec = importlib.util.spec_from_file_location(
        "reingest_uploaded_documents_ocr_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reingest():
    return _load_script()


@pytest.fixture(autouse=True)
def _restore_env():
    keys = (
        "HF_HUB_DOWNLOAD_TIMEOUT",
        "HF_HUB_ETAG_TIMEOUT",
        "NO_PROXY",
        "no_proxy",
        "PATH",
        "OPENAI_API_KEY",
        "EMBEDDING_BACKEND",
        "EMBEDDING_BINDING_HOST",
        "HF_HUB_OFFLINE",
    )
    before = {key: os.environ.get(key) for key in keys}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


class _IngestStack:
    def __init__(self):
        self.lightrag = MagicMock()
        self.lightrag.initialize_storages = AsyncMock()
        self.rag = MagicMock()
        self.rag.process_document_complete = AsyncMock()
        self.processed = []

        async def _capture(file_path, **kwargs):
            self.processed.append(
                {"file_path": file_path, "parse_method": kwargs.get("parse_method")}
            )

        self.rag.process_document_complete.side_effect = _capture


def _argv(folder, extra=None):
    args = [
        "reingest_uploaded_documents_ocr",
        "--folder",
        str(folder),
        "--skip-model-download",
        "--output-dir",
        str(folder / "out"),
    ]
    if extra:
        args.extend(extra)
    return args


def _patch_ingest_stack(stack):
    return (
        patch("lightrag.utils.EmbeddingFunc", return_value=MagicMock()),
        patch("lightrag.LightRAG", return_value=stack.lightrag),
        patch("raganything.RAGAnything", return_value=stack.rag),
    )


def test_ensure_hf_hub_timeouts_setdefault_and_preserve(reingest, monkeypatch):
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.delenv("HF_HUB_ETAG_TIMEOUT", raising=False)
    reingest._ensure_hf_hub_timeouts()
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "600"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "120"

    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "5")
    reingest._ensure_hf_hub_timeouts()
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "30"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "5"


@pytest.mark.asyncio
async def test_main_exits_when_folder_has_no_supported_files(
    reingest, monkeypatch, tmp_path
):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "notes.txt.bak").write_text("not a manual", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setattr(sys, "argv", _argv(folder))

    stack = _IngestStack()
    patches = _patch_ingest_stack(stack)
    with patches[0], patches[1], patches[2]:
        with pytest.raises(SystemExit, match="No supported files"):
            await reingest.main()

    stack.rag.process_document_complete.assert_not_called()


@pytest.mark.asyncio
async def test_limit_truncates_top_level_and_ignores_nested_pdfs(
    reingest, monkeypatch, tmp_path
):
    folder = tmp_path / "docs"
    nested = folder / "plant-a"
    nested.mkdir(parents=True)
    (folder / "a.pdf").write_bytes(b"%PDF")
    (folder / "b.pdf").write_bytes(b"%PDF")
    (nested / "nested.pdf").write_bytes(b"%PDF")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setattr(sys, "argv", _argv(folder, extra=["--limit", "1"]))

    stack = _IngestStack()
    patches = _patch_ingest_stack(stack)
    with patches[0], patches[1], patches[2]:
        await reingest.main()

    assert len(stack.processed) == 1
    assert Path(stack.processed[0]["file_path"]).name == "a.pdf"
    assert stack.processed[0]["parse_method"] == "ocr"


@pytest.mark.asyncio
async def test_skip_model_download_does_not_fetch_weights(
    reingest, monkeypatch, tmp_path
):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(b"%PDF")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setattr(sys, "argv", _argv(folder))

    download_calls = []
    monkeypatch.setattr(
        reingest,
        "_download_mineru_pipeline_models",
        lambda: download_calls.append("download"),
    )

    stack = _IngestStack()
    patches = _patch_ingest_stack(stack)
    with patches[0], patches[1], patches[2]:
        await reingest.main()

    assert download_calls == []
    assert len(stack.processed) == 1


@pytest.mark.asyncio
async def test_download_runs_before_processing_by_default(
    reingest, monkeypatch, tmp_path
):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(b"%PDF")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reingest_uploaded_documents_ocr",
            "--folder",
            str(folder),
            "--output-dir",
            str(folder / "out"),
        ],
    )

    order = []
    monkeypatch.setattr(
        reingest,
        "_download_mineru_pipeline_models",
        lambda: order.append("download"),
    )

    stack = _IngestStack()

    async def _capture(file_path, **kwargs):
        order.append("process")
        stack.processed.append(file_path)

    stack.rag.process_document_complete.side_effect = _capture
    patches = _patch_ingest_stack(stack)
    with patches[0], patches[1], patches[2]:
        await reingest.main()

    assert order == ["download", "process"]
