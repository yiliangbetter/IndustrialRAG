"""OCR re-ingest vision closures must honor VISION_MODEL vs LLM_MODEL.

``scripts/reingest_uploaded_documents_ocr.py`` runs ``process_document_complete``
with image/table/equation processors enabled. A swapped model id captions
figures with the text LLM (or sends extraction to the vision model).
``messages=`` / ``image_data=`` must use VISION_MODEL; text-only fallback
must keep LLM_MODEL. Distinct from #183 (embed dims / OCR concurrency) and
#179 (glob / processor flags).
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

_ENV_KEYS = (
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
    "LLM_BINDING_HOST",
    "OPENAI_BASE_URL",
    "LLM_MODEL",
    "VISION_MODEL",
    "HF_HOME",
)


@pytest.fixture(scope="module")
def reingest():
    path = REPO_ROOT / "scripts" / "reingest_uploaded_documents_ocr.py"
    spec = importlib.util.spec_from_file_location(
        "reingest_ocr_vision_model_b17b", path
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


class _FakeLightRAG:
    def __init__(self, **kwargs):
        pass

    async def initialize_storages(self):
        return None


class _RecordingEmbeddingFunc:
    def __init__(self, embedding_dim, max_token_size, func):
        pass


class _CapturingRAG:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.process_document_complete = AsyncMock()


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


def _enter_reingest_stack(*, complete_mock):
    stack = ExitStack()
    stack.enter_context(patch("lightrag.LightRAG", _FakeLightRAG))
    stack.enter_context(
        patch("lightrag.llm.openai.openai_complete_if_cache", complete_mock)
    )
    stack.enter_context(patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()))
    stack.enter_context(patch("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc))
    stack.enter_context(
        patch("lightrag.utils.logger", SimpleNamespace(info=lambda *a, **k: None))
    )
    stack.enter_context(patch("raganything.RAGAnything", _CapturingRAG))
    stack.enter_context(
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback")
    )
    stack.enter_context(
        patch("raganything.local_hf_embedding.make_local_hf_embedding_func")
    )
    return stack


def _prepare_folder(monkeypatch, tmp_path) -> None:
    folder = tmp_path / "uploaded_documents"
    folder.mkdir()
    (folder / "manual.pdf").write_bytes(b"%PDF")
    out = tmp_path / "out"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ocr")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "wd"))
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reingest_uploaded_documents_ocr",
            "--folder",
            str(folder),
            "--output-dir",
            str(out),
            "--skip-model-download",
        ],
    )


def _model_id(call):
    return call.args[0]


async def _capture_vision(reingest, monkeypatch, tmp_path, *, complete_mock):
    _prepare_folder(monkeypatch, tmp_path)
    _CapturingRAG.last_kwargs = None
    with _enter_reingest_stack(complete_mock=complete_mock):
        await reingest.main()
    assert _CapturingRAG.last_kwargs is not None
    return _CapturingRAG.last_kwargs["vision_model_func"]


@pytest.mark.asyncio
async def test_vision_model_defaults_to_llm_model(reingest, monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MODEL", "gpt-extract")
    monkeypatch.delenv("VISION_MODEL", raising=False)
    complete_mock = MagicMock(return_value="ok")
    vision = await _capture_vision(
        reingest, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    vision("caption", messages=[{"role": "user", "content": "hi"}])
    assert _model_id(complete_mock.call_args) == "gpt-extract"


@pytest.mark.asyncio
async def test_vision_model_used_for_messages_and_image_data_not_text_fallback(
    reingest, monkeypatch, tmp_path
):
    monkeypatch.setenv("LLM_MODEL", "gpt-extract")
    monkeypatch.setenv("VISION_MODEL", "gpt-vision")
    complete_mock = MagicMock(return_value="ok")
    vision = await _capture_vision(
        reingest, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    vision("ignored", messages=[{"role": "user", "content": "describe"}])
    assert _model_id(complete_mock.call_args) == "gpt-vision"
    assert complete_mock.call_args.args[1] == ""
    assert complete_mock.call_args.kwargs["messages"] == [
        {"role": "user", "content": "describe"}
    ]

    complete_mock.reset_mock()
    vision("what is this", system_prompt="sys", image_data="abc123")
    assert _model_id(complete_mock.call_args) == "gpt-vision"
    messages = complete_mock.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "sys"}
    user_content = messages[1]["content"]
    assert {"type": "text", "text": "what is this"} in user_content
    image_part = next(part for part in user_content if part.get("type") == "image_url")
    assert image_part["image_url"]["url"] == "data:image/jpeg;base64,abc123"

    complete_mock.reset_mock()
    vision("plain text fallback")
    assert _model_id(complete_mock.call_args) == "gpt-extract"
    assert complete_mock.call_args.args[1] == "plain text fallback"
