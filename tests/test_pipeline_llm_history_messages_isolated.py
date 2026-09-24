"""Pipeline chat must not reuse one LLM history list across calls.

``scripts/rag_pipeline_parse_graph_chat.py`` builds its own async LLM and
vision closures. LightRAG appends turns onto the list it is given, so a
shared default would leak the previous prompt into the next completion.
Omitted history must be a new empty list each call; an explicit list must
still be forwarded on the text path. Image and prebuilt-message calls must
not reuse that list or the caller's history.

Distinct from #196 (demo Q&A and OCR re-ingest closures) and from #190
(VISION_MODEL routing).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"

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
)


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_history_a8d6", SCRIPT_PATH
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
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def initialize_storages(self):
        return None


class _CapturingRAG:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _recording_complete(seen):
    async def complete(*args, **kwargs):
        history = kwargs["history_messages"]
        seen.append((history, list(history)))
        history.append({"role": "user", "content": args[1]})
        return "ok"

    return complete


async def _build(pipeline, monkeypatch, tmp_path, complete):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-pipeline")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-extract")
    monkeypatch.delenv("VISION_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    captured = {}

    def lightrag_ctor(**kwargs):
        captured["llm"] = kwargs["llm_model_func"]
        return _CapturingLightRAG(**kwargs)

    def rag_ctor(**kwargs):
        captured["vision"] = kwargs["vision_model_func"]
        return _CapturingRAG(**kwargs)

    with patch("lightrag.LightRAG", side_effect=lightrag_ctor):
        with patch("raganything.RAGAnything", side_effect=rag_ctor):
            with patch("lightrag.utils.EmbeddingFunc", return_value=MagicMock()):
                with patch(
                    "raganything.local_hf_embedding.ensure_hf_home_from_repo_fallback"
                ):
                    with patch(
                        "lightrag.llm.openai.openai_complete_if_cache", complete
                    ):
                        with patch("lightrag.llm.openai.openai_embed") as openai_embed:
                            openai_embed.func = MagicMock()
                            await pipeline._build_rag(
                                tmp_path / "wd", tmp_path / "parse"
                            )
    assert captured["llm"] is not None
    assert captured["vision"] is not None
    return captured


@pytest.mark.asyncio
async def test_text_paths_isolate_omitted_history_and_forward_explicit_list(
    pipeline, monkeypatch, tmp_path
):
    seen = []
    captured = await _build(pipeline, monkeypatch, tmp_path, _recording_complete(seen))
    llm = captured["llm"]
    vision = captured["vision"]

    await llm("first question")
    await llm("second question")
    explicit = [{"role": "user", "content": "keep"}]
    await llm("third question", history_messages=explicit)

    await vision("vision text")
    vision_explicit = [{"role": "user", "content": "keep-vision"}]
    await vision("vision explicit", history_messages=vision_explicit)

    first_list, first_snapshot = seen[0]
    second_list, second_snapshot = seen[1]
    third_list, third_snapshot = seen[2]
    fourth_list, fourth_snapshot = seen[3]
    fifth_list, fifth_snapshot = seen[4]

    assert first_snapshot == []
    assert second_snapshot == []
    assert first_list is not second_list
    assert third_list is explicit
    assert third_snapshot == [{"role": "user", "content": "keep"}]

    assert fourth_snapshot == []
    assert fourth_list is not first_list
    assert fourth_list is not second_list
    assert fifth_list is vision_explicit
    assert fifth_snapshot == [{"role": "user", "content": "keep-vision"}]


@pytest.mark.asyncio
async def test_vision_image_and_messages_paths_do_not_reuse_caller_history(
    pipeline, monkeypatch, tmp_path
):
    seen = []
    captured = await _build(pipeline, monkeypatch, tmp_path, _recording_complete(seen))
    vision = captured["vision"]
    prior = [{"role": "user", "content": "prior"}]

    await vision(
        "caption",
        image_data="abc",
        system_prompt="sys",
        history_messages=prior,
    )
    await vision(
        "caption",
        messages=[{"role": "user", "content": "hi"}],
        history_messages=prior,
    )

    image_list, image_snapshot = seen[0]
    messages_list, messages_snapshot = seen[1]
    assert image_snapshot == []
    assert messages_snapshot == []
    assert image_list is not messages_list
    assert image_list is not prior
    assert messages_list is not prior
    assert prior == [{"role": "user", "content": "prior"}]
