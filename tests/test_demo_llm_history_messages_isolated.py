"""Demo Q&A must not reuse one LLM history list across questions.

``scripts/run_demo_question_bank.py`` used to default ``history_messages=[]``.
LightRAG appends turns to that list, so a later question would see the previous
prompt. Omitted history must be a new empty list each call; an explicit list
must still be forwarded. Distinct from #184 (VISION_MODEL routing) and #186
(LLM_MODEL / query mode).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_demo_question_bank.py"

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
def demo_script():
    spec = importlib.util.spec_from_file_location(
        "run_demo_question_bank_history_74ef", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_env_and_openpyxl():
    before_env = {key: os.environ.get(key) for key in _ENV_KEYS}
    before_openpyxl = sys.modules.get("openpyxl")
    yield
    for key, val in before_env.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val
    if before_openpyxl is None:
        sys.modules.pop("openpyxl", None)
    else:
        sys.modules["openpyxl"] = before_openpyxl


class _Cell:
    def __init__(self, value):
        self.value = value


class _Worksheet:
    def iter_rows(self, min_row=1, max_row=None, values_only=False):
        if min_row == 1:
            yield [_Cell("问题")]

    def cell(self, row, column, value=None):
        return _Cell(value)


class _Workbook:
    def __init__(self):
        self.active = _Worksheet()

    def save(self, path):
        self.saved_to = path


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
    def __init__(self, embedding_dim, max_token_size, func):
        pass


async def _capture_llm(demo_script, monkeypatch, tmp_path, complete):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-demo")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-extract")
    monkeypatch.delenv("VISION_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    fake_openpyxl = types.ModuleType("openpyxl")
    fake_openpyxl.load_workbook = lambda path: _Workbook()
    monkeypatch.setitem(sys.modules, "openpyxl", fake_openpyxl)

    _CapturingLightRAG.kwargs = None
    monkeypatch.setattr(demo_script, "LightRAG", _CapturingLightRAG)
    monkeypatch.setattr(demo_script, "RAGAnything", _FakeRAGAnything)
    monkeypatch.setattr(demo_script, "EmbeddingFunc", _RecordingEmbeddingFunc)
    monkeypatch.setattr(
        demo_script, "ensure_hf_home_from_repo_fallback", lambda *a, **k: None
    )
    monkeypatch.setattr(
        demo_script,
        "openai_embed",
        types.SimpleNamespace(func=MagicMock(name="openai_embed_func")),
    )
    monkeypatch.setattr(demo_script, "openai_complete_if_cache", complete)

    await demo_script.run_batch(
        tmp_path / "in.xlsx",
        tmp_path / "out.xlsx",
        tmp_path / "out.jsonl",
        tmp_path / "wd",
        "mix",
        0.0,
        1,
        1,
        0,
        None,
    )
    assert _CapturingLightRAG.kwargs is not None
    return _CapturingLightRAG.kwargs["llm_model_func"]


@pytest.mark.asyncio
async def test_omitted_llm_history_is_fresh_and_explicit_list_is_forwarded(
    demo_script, monkeypatch, tmp_path
):
    seen = []

    def complete(*args, **kwargs):
        history = kwargs["history_messages"]
        seen.append((history, list(history)))
        history.append({"role": "user", "content": args[1]})
        return "ok"

    llm = await _capture_llm(demo_script, monkeypatch, tmp_path, complete)

    llm("first question")
    llm("second question")
    explicit = [{"role": "user", "content": "keep"}]
    llm("third question", history_messages=explicit)

    first_list, first_snapshot = seen[0]
    second_list, second_snapshot = seen[1]
    third_list, third_snapshot = seen[2]
    assert first_snapshot == []
    assert second_snapshot == []
    assert first_list is not second_list
    assert third_list is explicit
    assert third_snapshot == [{"role": "user", "content": "keep"}]
