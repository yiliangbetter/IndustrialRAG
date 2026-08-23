"""Fail-closed gates for scripts/run_demo_question_bank.py.

Demo batch Q&A must not start without LLM credentials, must not reuse the
LLM key against a separate embedding host, and must abort when the xlsx
sheet is missing the expected question column.
"""

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_demo_question_bank.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_demo_question_bank_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo_script():
    return _load_script()


class TestJsonSafeAndResolveKeys:
    def test_json_safe_nan_and_inf_become_none(self, demo_script):
        assert demo_script._json_safe(math.nan) is None
        assert demo_script._json_safe(math.inf) is None
        assert demo_script._json_safe(-math.inf) is None
        assert demo_script._json_safe(1.5) == 1.5
        assert demo_script._json_safe("ok") == "ok"

    def test_resolve_keys_prefers_openai_then_llm_binding(
        self, demo_script, monkeypatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LLM_BINDING_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        assert demo_script._resolve_keys() == ("", "")

        monkeypatch.setenv("LLM_BINDING_API_KEY", "  llm-key  ")
        llm_key, emb_key = demo_script._resolve_keys()
        assert llm_key == "llm-key"
        assert emb_key == "llm-key"

        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("EMBEDDING_API_KEY", "  emb-key  ")
        llm_key, emb_key = demo_script._resolve_keys()
        assert llm_key == "sk-openai"
        assert emb_key == "emb-key"


def _stub_openpyxl(monkeypatch, workbook=None):
    fake = types.ModuleType("openpyxl")
    fake.load_workbook = lambda path: workbook
    monkeypatch.setitem(sys.modules, "openpyxl", fake)


@pytest.mark.asyncio
async def test_run_batch_exits_when_llm_key_missing(demo_script, monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BINDING_API_KEY", raising=False)
    _stub_openpyxl(monkeypatch)

    with pytest.raises(SystemExit, match="OPENAI_API_KEY or LLM_BINDING_API_KEY"):
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


@pytest.mark.asyncio
async def test_run_batch_exits_when_embed_host_has_no_dedicated_key(
    demo_script, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    _stub_openpyxl(monkeypatch)

    with pytest.raises(SystemExit, match="EMBEDDING_API_KEY"):
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


class _Cell:
    def __init__(self, value, row=1):
        self.value = value
        self.row = row


class _Worksheet:
    def __init__(self, headers):
        self._headers = headers

    def iter_rows(self, min_row=1, max_row=None, values_only=False):
        if min_row == 1:
            yield [_Cell(h) for h in self._headers]


class _Workbook:
    def __init__(self, headers):
        self.active = _Worksheet(headers)

    def save(self, path):
        raise AssertionError("should not save after a schema error")


class _FakeLightRAG:
    def __init__(self, **kwargs):
        pass

    async def initialize_storages(self):
        pass


class _FakeRAGAnything:
    def __init__(self, **kwargs):
        pass

    async def aquery(self, *args, **kwargs):
        raise AssertionError("should not query after a schema error")


@pytest.mark.asyncio
async def test_run_batch_exits_when_question_column_missing(
    demo_script, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("EMBEDDING_BINDING_HOST", raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")

    workbook = _Workbook(["Question", "Answer"])
    fake_openpyxl = types.ModuleType("openpyxl")
    fake_openpyxl.load_workbook = lambda path: workbook
    monkeypatch.setitem(sys.modules, "openpyxl", fake_openpyxl)

    monkeypatch.setattr(demo_script, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(demo_script, "RAGAnything", _FakeRAGAnything)
    monkeypatch.setattr(demo_script, "RAGAnythingConfig", lambda **kwargs: object())
    monkeypatch.setattr(demo_script, "EmbeddingFunc", lambda **kwargs: object())
    monkeypatch.setattr(
        demo_script,
        "openai_embed",
        types.SimpleNamespace(func=lambda *args, **kwargs: None),
    )

    with pytest.raises(SystemExit, match='Column "问题" not found'):
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
