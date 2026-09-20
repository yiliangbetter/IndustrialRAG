"""Graph batch ingest vision closures must honor VISION_MODEL vs LLM_MODEL.

``scripts/batch_ingest_content_lists_with_graph.py`` still constructs
``vision_model_func`` even though processors are off and skip-multimodal
defaults True. A swapped model id would caption figures with the text LLM
(or send extraction to the vision model) if ``--no-skip-multimodal`` is
used later. ``messages=`` / ``image_data=`` must use VISION_MODEL;
text-only fallback must keep LLM_MODEL.

Distinct from #180 (processor flags / embed dims), #184 (demo/OCR vision;
those CLIs keep processors on), and #185 (LLM_MODEL into the text closure).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "batch_ingest_content_lists_with_graph.py"

_ENV_KEYS = (
    "OPENAI_API_KEY",
    "LLM_BINDING_API_KEY",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BACKEND",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "PARSER",
    "LLM_MODEL",
    "VISION_MODEL",
    "LLM_BINDING_HOST",
    "OPENAI_BASE_URL",
    "HF_HOME",
    "RAG_DATA_REPO",
    "RAG_DATA_UPLOAD_SUBDIR",
)


@pytest.fixture(scope="module")
def bigraph():
    spec = importlib.util.spec_from_file_location(
        "batch_ingest_graph_vision_model_routing_7d2e", SCRIPT_PATH
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
        self.kwargs = kwargs

    async def initialize_storages(self):
        return None


class _CapturingRAG:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


class _RecordingEmbeddingFunc:
    def __init__(self, embedding_dim, max_token_size, func):
        pass


def _openai_embed_stub():
    stub = MagicMock()
    stub.func = MagicMock(name="openai_embed_func")
    return stub


@contextmanager
def _patched_ingest_stack(*, complete_mock):
    with (
        patch("lightrag.LightRAG", _FakeLightRAG),
        patch("lightrag.llm.openai.openai_complete_if_cache", complete_mock),
        patch("lightrag.llm.openai.openai_embed", _openai_embed_stub()),
        patch("lightrag.utils.EmbeddingFunc", _RecordingEmbeddingFunc),
        patch("lightrag.utils.logger", SimpleNamespace()),
        patch("raganything.RAGAnything", _CapturingRAG),
        patch("raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback"),
        patch("raganything.local_hf_embedding.make_local_hf_embedding_func"),
    ):
        yield


def _empty_upload_tree(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "output" / "data_upload_test_v3").mkdir(parents=True)
    return repo


def _prepare_graph_env(monkeypatch, tmp_path) -> None:
    repo = _empty_upload_tree(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-graph")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_ingest_content_lists_with_graph",
            "-w",
            str(tmp_path / "wd"),
            "--data-repo-root",
            str(repo),
        ],
    )


def _model_id(call):
    return call.args[0]


async def _capture_vision(bigraph, monkeypatch, tmp_path, *, complete_mock):
    _prepare_graph_env(monkeypatch, tmp_path)
    _CapturingRAG.last_kwargs = None
    with _patched_ingest_stack(complete_mock=complete_mock):
        with pytest.raises(SystemExit, match="No files matching"):
            await bigraph.async_main()
    assert _CapturingRAG.last_kwargs is not None
    return _CapturingRAG.last_kwargs["vision_model_func"]


@pytest.mark.asyncio
async def test_vision_model_defaults_to_llm_model(bigraph, monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MODEL", "gpt-extract")
    monkeypatch.delenv("VISION_MODEL", raising=False)
    complete_mock = AsyncMock(return_value="ok")
    vision = await _capture_vision(
        bigraph, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    await vision("caption", messages=[{"role": "user", "content": "hi"}])
    assert _model_id(complete_mock.call_args) == "gpt-extract"


@pytest.mark.asyncio
async def test_vision_model_used_for_messages_and_image_data_not_text_fallback(
    bigraph, monkeypatch, tmp_path
):
    monkeypatch.setenv("LLM_MODEL", "gpt-extract")
    monkeypatch.setenv("VISION_MODEL", "gpt-vision")
    complete_mock = AsyncMock(return_value="ok")
    vision = await _capture_vision(
        bigraph, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    await vision("ignored", messages=[{"role": "user", "content": "describe"}])
    assert _model_id(complete_mock.call_args) == "gpt-vision"
    assert complete_mock.call_args.args[1] == ""
    assert complete_mock.call_args.kwargs["messages"] == [
        {"role": "user", "content": "describe"}
    ]

    complete_mock.reset_mock()
    await vision("what is this", system_prompt="sys", image_data="abc123")
    assert _model_id(complete_mock.call_args) == "gpt-vision"
    messages = complete_mock.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "sys"}
    user_content = messages[1]["content"]
    assert {"type": "text", "text": "what is this"} in user_content
    image_part = next(part for part in user_content if part.get("type") == "image_url")
    assert image_part["image_url"]["url"] == "data:image/jpeg;base64,abc123"

    complete_mock.reset_mock()
    await vision("plain text fallback")
    assert _model_id(complete_mock.call_args) == "gpt-extract"
    assert complete_mock.call_args.args[1] == "plain text fallback"
