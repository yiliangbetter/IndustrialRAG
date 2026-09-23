"""OCR re-ingest omitted ``--folder`` must scan ``<repo>/uploaded_documents``.

``scripts/reingest_uploaded_documents_ocr.py`` is the production OCR path.
Running it with no folder argument must not walk a sibling export tree, and a
missing default directory must fail before model download or LightRAG writes.
Distinct from #178 (explicit ``--folder``, ``--limit``, download skip) and
#181 (``WORKING_DIR`` / ``OUTPUT_DIR``).
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
        "reingest_uploaded_documents_ocr_omitted_folder", path
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
        "LLM_BINDING_API_KEY",
        "EMBEDDING_BACKEND",
        "EMBEDDING_BINDING_HOST",
        "EMBEDDING_API_KEY",
        "HF_HUB_OFFLINE",
        "SUPPORTED_FILE_EXTENSIONS",
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
        self.processed = []

        async def _capture(file_path, **kwargs):
            self.processed.append(file_path)

        self.rag.process_document_complete = AsyncMock(side_effect=_capture)


def _patch_ingest_stack(stack):
    return (
        patch("lightrag.utils.EmbeddingFunc", return_value=MagicMock()),
        patch("lightrag.LightRAG", return_value=stack.lightrag),
        patch("raganything.RAGAnything", return_value=stack.rag),
    )


@pytest.mark.asyncio
async def test_omitted_folder_missing_default_exits_before_download(
    reingest, monkeypatch, tmp_path
):
    monkeypatch.setattr(reingest, "_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["reingest_uploaded_documents_ocr"])
    download_calls = []
    monkeypatch.setattr(
        reingest,
        "_download_mineru_pipeline_models",
        lambda: download_calls.append("download"),
    )

    expected = (tmp_path / "uploaded_documents").resolve()
    with pytest.raises(SystemExit, match="Not a directory") as exc_info:
        await reingest.main()

    assert str(expected) in str(exc_info.value)
    assert download_calls == []
    assert not (tmp_path / "output").exists()


@pytest.mark.asyncio
async def test_omitted_folder_scans_only_top_level_uploaded_documents(
    reingest, monkeypatch, tmp_path
):
    uploaded = tmp_path / "uploaded_documents"
    nested = uploaded / "plant-a"
    nested.mkdir(parents=True)
    (uploaded / "b.pdf").write_bytes(b"%PDF")
    (uploaded / "a.pdf").write_bytes(b"%PDF")
    (nested / "nested.pdf").write_bytes(b"%PDF")
    decoy_dir = tmp_path / "output" / "data_upload_test_v4"
    decoy_dir.mkdir(parents=True)
    (decoy_dir / "decoy.pdf").write_bytes(b"%PDF")

    monkeypatch.setattr(reingest, "_ROOT", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-llm")
    monkeypatch.delenv("LLM_BINDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("SUPPORTED_FILE_EXTENSIONS", raising=False)
    out = tmp_path / "parser-out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reingest_uploaded_documents_ocr",
            "--skip-model-download",
            "--output-dir",
            str(out),
        ],
    )
    monkeypatch.setattr(reingest, "_download_mineru_pipeline_models", lambda: None)

    stack = _IngestStack()
    patches = _patch_ingest_stack(stack)
    with patches[0], patches[1], patches[2]:
        await reingest.main()

    processed = [Path(path).resolve() for path in stack.processed]
    assert processed == [
        (uploaded / "a.pdf").resolve(),
        (uploaded / "b.pdf").resolve(),
    ]
