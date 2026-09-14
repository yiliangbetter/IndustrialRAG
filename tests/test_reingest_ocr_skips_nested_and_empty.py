"""OCR re-ingest must stay top-level, OCR-forced, and multimodal-on.

``scripts/reingest_uploaded_documents_ocr.py`` re-parses ``uploaded_documents``
with ``parse_method=ocr``. Nested manuals under subfolders are skipped because
the script uses ``Path.glob``, not ``rglob``. An empty folder must fail closed.
Unlike the parse→graph pipeline, OCR re-ingest keeps image/table/equation
processors enabled so scanned figures are not dropped.

Distinct from #130 (missing-folder / HF-offline / NO_PROXY / API-key gates).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def reingest():
    path = REPO_ROOT / "scripts" / "reingest_uploaded_documents_ocr.py"
    spec = importlib.util.spec_from_file_location(
        "reingest_ocr_top_level_empty_44c5", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_env():
    keys = (
        "OPENAI_API_KEY",
        "LLM_BINDING_API_KEY",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BACKEND",
        "EMBEDDING_BINDING_HOST",
        "EMBEDDING_DIM",
        "EMBEDDING_MODEL",
        "WORKING_DIR",
        "OUTPUT_DIR",
        "PARSER",
        "MAX_CONCURRENT_FILES",
        "HF_HOME",
        "LLM_BINDING_HOST",
        "OPENAI_BASE_URL",
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


class _FakeEmbeddingFunc:
    def __init__(self, embedding_dim, max_token_size, func):
        self.embedding_dim = embedding_dim
        self.max_token_size = max_token_size
        self.func = func


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


class _CapturingRAG:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.process_document_complete = AsyncMock()


def _enter_reingest_stack(rag_ctor):
    stack = ExitStack()
    stack.enter_context(patch("lightrag.LightRAG", _FakeLightRAG))
    stack.enter_context(
        patch("lightrag.llm.openai.openai_complete_if_cache", MagicMock())
    )
    stack.enter_context(
        patch("lightrag.llm.openai.openai_embed", _openai_embed_stub())
    )
    stack.enter_context(patch("lightrag.utils.EmbeddingFunc", _FakeEmbeddingFunc))
    stack.enter_context(
        patch("lightrag.utils.logger", SimpleNamespace(info=lambda *a, **k: None))
    )
    stack.enter_context(patch("raganything.RAGAnything", rag_ctor))
    stack.enter_context(
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback")
    )
    return stack


def _argv(folder: Path, output_dir: Path, extra=None):
    args = [
        "reingest_uploaded_documents_ocr",
        "--folder",
        str(folder),
        "--output-dir",
        str(output_dir),
        "--skip-model-download",
    ]
    if extra:
        args.extend(extra)
    return args


@pytest.mark.asyncio
async def test_reingest_exits_when_folder_has_no_supported_files(
    reingest, monkeypatch, tmp_path
):
    folder = tmp_path / "uploaded_documents"
    folder.mkdir()
    (folder / "notes.xyz").write_text("unsupported", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ocr")
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "wd"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setattr(sys, "argv", _argv(folder, tmp_path / "out"))

    rag_holder = {}

    def rag_ctor(**kwargs):
        rag = _CapturingRAG(**kwargs)
        rag_holder["rag"] = rag
        return rag

    with _enter_reingest_stack(rag_ctor):
        with pytest.raises(SystemExit, match="No supported files"):
            await reingest.main()

    rag_holder["rag"].process_document_complete.assert_not_called()


@pytest.mark.asyncio
async def test_reingest_skips_nested_files_and_forces_ocr_with_modal_processors(
    reingest, monkeypatch, tmp_path
):
    folder = tmp_path / "uploaded_documents"
    nested = folder / "plant-a"
    nested.mkdir(parents=True)
    top = folder / "manual.pdf"
    top.write_bytes(b"%PDF")
    (nested / "nested.pdf").write_bytes(b"%PDF")
    out = tmp_path / "out"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ocr")
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "wd"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setattr(sys, "argv", _argv(folder, out))

    rag_holder = {}

    def rag_ctor(**kwargs):
        rag = _CapturingRAG(**kwargs)
        rag_holder["rag"] = rag
        return rag

    with _enter_reingest_stack(rag_ctor):
        await reingest.main()

    rag = rag_holder["rag"]
    config = rag.kwargs["config"]
    assert config.parse_method == "ocr"
    assert config.enable_image_processing is True
    assert config.enable_table_processing is True
    assert config.enable_equation_processing is True
    assert config.parser_output_dir == str(out.resolve())

    rag.process_document_complete.assert_awaited_once()
    call = rag.process_document_complete.await_args
    assert Path(call.args[0]) == top.resolve()
    assert call.kwargs["parse_method"] == "ocr"
    assert call.kwargs["output_dir"] == str(out.resolve())
    assert call.kwargs.get("backend") == "pipeline"


@pytest.mark.asyncio
async def test_reingest_limit_truncates_top_level_files(
    reingest, monkeypatch, tmp_path
):
    folder = tmp_path / "uploaded_documents"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(b"%PDF")
    (folder / "b.pdf").write_bytes(b"%PDF")
    out = tmp_path / "out"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ocr")
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "wd"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setattr(sys, "argv", _argv(folder, out, extra=["--limit", "1"]))

    rag_holder = {}

    def rag_ctor(**kwargs):
        rag = _CapturingRAG(**kwargs)
        rag_holder["rag"] = rag
        return rag

    with _enter_reingest_stack(rag_ctor):
        await reingest.main()

    rag = rag_holder["rag"]
    assert rag.process_document_complete.await_count == 1
    processed = Path(rag.process_document_complete.await_args.args[0]).name
    assert processed == "a.pdf"
