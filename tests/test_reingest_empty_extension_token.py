"""OCR re-ingest extension tokens must not become a match-all.

``scripts/reingest_uploaded_documents_ocr.py`` normalizes
``SUPPORTED_FILE_EXTENSIONS`` itself: a blank token becomes ``.`` (trailing-dot
names only) and a token without a leading dot gets one. Folder ingest uses
``*{token}``, so the same blank token matches every file there.

A regression either way re-OCRs the wrong uploads: match-all would parse
notes and extensionless files, and dropping the blank token would skip names
that end in a dot. Selection stays top-level.

Distinct from folder match-all coverage and from pipeline blank-token
trailing-dot collection.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    path = REPO_ROOT / "scripts" / "reingest_uploaded_documents_ocr.py"
    spec = importlib.util.spec_from_file_location(
        "reingest_empty_extension_token", path
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
        "OPENAI_API_KEY",
        "LLM_BINDING_API_KEY",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BACKEND",
        "EMBEDDING_BINDING_HOST",
        "SUPPORTED_FILE_EXTENSIONS",
        "PARSER",
        "WORKING_DIR",
        "OUTPUT_DIR",
        "HF_HOME",
        "HF_HUB_OFFLINE",
    )
    before = {key: os.environ.get(key) for key in keys}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


class _FakeLightRAG:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def initialize_storages(self):
        return None


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


def _enter_stack(rag_ctor):
    stack = ExitStack()
    stack.enter_context(patch("lightrag.LightRAG", _FakeLightRAG))
    stack.enter_context(
        patch("lightrag.llm.openai.openai_complete_if_cache", MagicMock())
    )
    stack.enter_context(patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()))
    stack.enter_context(patch("lightrag.utils.EmbeddingFunc", MagicMock()))
    stack.enter_context(patch("raganything.RAGAnything", rag_ctor))
    return stack


def _prepare(monkeypatch, folder: Path, output_dir: Path, extensions: str):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ocr")
    monkeypatch.delenv("LLM_BINDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("SUPPORTED_FILE_EXTENSIONS", extensions)
    monkeypatch.setenv("WORKING_DIR", str(folder.parent / "wd"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reingest_uploaded_documents_ocr",
            "--folder",
            str(folder),
            "--output-dir",
            str(output_dir),
            "--skip-model-download",
        ],
    )


def _write_tree(folder: Path) -> dict[str, Path]:
    nested = folder / "plant-a"
    nested.mkdir(parents=True)
    (folder / "dotdir.").mkdir()
    paths = {
        "manual": folder / "manual.pdf",
        "spec": folder / "spec.docx",
        "trailing": folder / "trailing.",
        "notes": folder / "notes.txt",
        "plain": folder / "readme",
        "nested_pdf": nested / "manual.pdf",
        "nested_dot": nested / "trailing.",
    }
    for path in paths.values():
        path.write_bytes(b"doc")
    return paths


@pytest.mark.asyncio
async def test_blank_token_matches_trailing_dot_and_bare_ext_gets_a_dot(
    reingest, monkeypatch, tmp_path
):
    folder = tmp_path / "uploaded_documents"
    folder.mkdir()
    paths = _write_tree(folder)
    output_dir = tmp_path / "out"
    _prepare(monkeypatch, folder, output_dir, " .PDF , , DOCX ")

    processed: list[tuple[Path, dict]] = []

    def rag_ctor(**kwargs):
        rag = MagicMock()
        rag.kwargs = kwargs

        async def _capture(file_path, **call_kwargs):
            processed.append((Path(file_path), call_kwargs))

        rag.process_document_complete = AsyncMock(side_effect=_capture)
        return rag

    with _enter_stack(rag_ctor):
        await reingest.main()

    assert [path for path, _ in processed] == [
        paths["manual"].resolve(),
        paths["spec"].resolve(),
        paths["trailing"].resolve(),
    ]
    for _, call_kwargs in processed:
        assert call_kwargs["parse_method"] == "ocr"
        assert call_kwargs["output_dir"] == str(output_dir.resolve())


@pytest.mark.asyncio
async def test_blank_only_token_does_not_match_every_file(
    reingest, monkeypatch, tmp_path
):
    folder = tmp_path / "uploaded_documents"
    folder.mkdir()
    paths = _write_tree(folder)
    _prepare(monkeypatch, folder, tmp_path / "out", " , ")

    processed: list[Path] = []

    def rag_ctor(**kwargs):
        rag = MagicMock()

        async def _capture(file_path, **call_kwargs):
            processed.append(Path(file_path))

        rag.process_document_complete = AsyncMock(side_effect=_capture)
        return rag

    with _enter_stack(rag_ctor):
        await reingest.main()

    assert processed == [paths["trailing"].resolve()]
