"""OCR re-ingest must not reuse one LLM history list across documents.

``scripts/reingest_uploaded_documents_ocr.py`` used to default
``history_messages=[]``. A mutated default would leak one file's prompt into
the next completion. Omitted history must be a new empty list each call; an
explicit list must still be forwarded. Distinct from #130 (HF offline gate)
and #186 (LLM_MODEL / OCR_LANG).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "reingest_uploaded_documents_ocr.py"

_ENV_KEYS = (
    "OPENAI_API_KEY",
    "LLM_BINDING_API_KEY",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BACKEND",
    "EMBEDDING_BINDING_HOST",
    "LLM_BINDING_HOST",
    "OPENAI_BASE_URL",
    "LLM_MODEL",
    "VISION_MODEL",
    "HF_HOME",
    "WORKING_DIR",
    "OUTPUT_DIR",
    "HF_HUB_OFFLINE",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_ETAG_TIMEOUT",
    "HF_FORCE_ONLINE",
    "NO_PROXY",
    "no_proxy",
    "PATH",
    "PARSER",
)


@pytest.fixture(scope="module")
def reingest():
    spec = importlib.util.spec_from_file_location(
        "reingest_ocr_history_74ef", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_env():
    before = {key: os.environ.get(key) for key in _ENV_KEYS}
    yield
    for key, val in before.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


class _CapturingLightRAG:
    kwargs = None

    def __init__(self, **kwargs):
        type(self).kwargs = kwargs

    async def initialize_storages(self):
        return None


class _FakeRAGAnything:
    def __init__(self, **kwargs):
        pass


class _RecordingEmbeddingFunc:
    def __init__(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_omitted_llm_history_is_fresh_and_explicit_list_is_forwarded(
    reingest, monkeypatch, tmp_path
):
    folder = tmp_path / "uploaded"
    folder.mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ocr")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-extract")
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "wd"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.delenv("VISION_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reingest_uploaded_documents_ocr",
            "--folder",
            str(folder),
            "--skip-model-download",
        ],
    )

    seen = []

    def complete(*args, **kwargs):
        history = kwargs["history_messages"]
        seen.append((history, list(history)))
        history.append({"role": "user", "content": args[1]})
        return "ok"

    _CapturingLightRAG.kwargs = None
    monkeypatch.setattr("lightrag.LightRAG", _CapturingLightRAG)
    monkeypatch.setattr("raganything.RAGAnything", _FakeRAGAnything)
    monkeypatch.setattr("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc)
    openai_embed = MagicMock()
    openai_embed.func = MagicMock(name="openai_embed_func")
    monkeypatch.setattr("lightrag.llm.openai.openai_embed", openai_embed)
    monkeypatch.setattr("lightrag.llm.openai.openai_complete_if_cache", complete)

    with pytest.raises(SystemExit, match="No supported files"):
        await reingest.main()

    assert _CapturingLightRAG.kwargs is not None
    llm = _CapturingLightRAG.kwargs["llm_model_func"]
    llm("first document")
    llm("second document")
    explicit = [{"role": "user", "content": "keep"}]
    llm("third document", history_messages=explicit)

    first_list, first_snapshot = seen[0]
    second_list, second_snapshot = seen[1]
    third_list, third_snapshot = seen[2]
    assert first_snapshot == []
    assert second_snapshot == []
    assert first_list is not second_list
    assert third_list is explicit
    assert third_snapshot == [{"role": "user", "content": "keep"}]
