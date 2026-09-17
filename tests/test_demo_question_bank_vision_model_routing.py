"""Demo Q&A vision closures must honor VISION_MODEL vs LLM_MODEL.

``scripts/run_demo_question_bank.py`` keeps image/table/equation processors
on, so a wrong vision model id captions figures with the text LLM (or the
reverse). ``messages=`` and ``image_data=`` must call VISION_MODEL; text-only
fallback must keep LLM_MODEL. Distinct from #183 (embed dims) and #182
(gateway host fallback).
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
    "PARSER",
)


@pytest.fixture(scope="module")
def demo_script():
    spec = importlib.util.spec_from_file_location(
        "run_demo_question_bank_vision_model_b17b", SCRIPT_PATH
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
    def __init__(self, value, row=1):
        self.value = value
        self.row = row


class _Worksheet:
    def __init__(self, headers, data_rows):
        self.headers = list(headers)
        self.data_rows = list(data_rows)

    def iter_rows(self, min_row=1, max_row=None, values_only=False):
        if min_row == 1:
            yield [_Cell(h, row=1) for h in self.headers]
            return
        for offset, values in enumerate(self.data_rows):
            row_num = offset + 2
            yield [_Cell(v, row=row_num) for v in values]

    def cell(self, row, column, value=None):
        return _Cell(value, row=row)


class _Workbook:
    def __init__(self, headers, data_rows):
        self.active = _Worksheet(headers, data_rows)

    def save(self, path):
        self.saved_to = path


class _CapturingLightRAG:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    async def initialize_storages(self):
        return None


class _FakeRAGAnything:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    async def aquery(self, question, mode="mix", **kwargs):
        return f"ans:{question}"


class _RecordingEmbeddingFunc:
    def __init__(self, embedding_dim, max_token_size, func):
        pass


def _install_openpyxl(monkeypatch):
    fake = types.ModuleType("openpyxl")
    fake.load_workbook = lambda path: _Workbook(["问题"], [["q1"]])
    monkeypatch.setitem(sys.modules, "openpyxl", fake)


async def _run_and_capture(demo_script, monkeypatch, tmp_path, *, complete_mock):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-demo")
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    _install_openpyxl(monkeypatch)
    _CapturingLightRAG.last_kwargs = None
    _FakeRAGAnything.last_kwargs = None

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
    monkeypatch.setattr(demo_script, "openai_complete_if_cache", complete_mock)

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
    assert _FakeRAGAnything.last_kwargs is not None
    return _FakeRAGAnything.last_kwargs


def _model_id(call):
    return call.args[0]


@pytest.mark.asyncio
async def test_vision_model_defaults_to_llm_model(demo_script, monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MODEL", "gpt-extract")
    monkeypatch.delenv("VISION_MODEL", raising=False)
    complete_mock = MagicMock(return_value="ok")
    kwargs = await _run_and_capture(
        demo_script, monkeypatch, tmp_path, complete_mock=complete_mock
    )

    kwargs["vision_model_func"]("caption", messages=[{"role": "user", "content": "hi"}])
    assert _model_id(complete_mock.call_args) == "gpt-extract"


@pytest.mark.asyncio
async def test_vision_model_used_for_messages_and_image_data_not_text_fallback(
    demo_script, monkeypatch, tmp_path
):
    monkeypatch.setenv("LLM_MODEL", "gpt-extract")
    monkeypatch.setenv("VISION_MODEL", "gpt-vision")
    complete_mock = MagicMock(return_value="ok")
    kwargs = await _run_and_capture(
        demo_script, monkeypatch, tmp_path, complete_mock=complete_mock
    )
    vision = kwargs["vision_model_func"]

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
