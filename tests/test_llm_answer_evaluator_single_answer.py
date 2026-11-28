"""Regression tests for one LLM answer evaluation call.

The scorer decides whether a generated answer counts as correct, whether a
timeout is retried, and whether prose that merely says "correct" becomes a
score. These tests stub the LLM client and do not call a network.
"""

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MODULE_NAME = "llm_answer_evaluator_single_1128"


def _load_evaluator():
    added = []

    def ensure(name, module):
        if name not in sys.modules:
            sys.modules[name] = module
            added.append(name)

    pandas_mod = types.ModuleType("pandas")
    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = lambda *args, **kwargs: None
    lightrag_mod = types.ModuleType("lightrag")
    llm_mod = types.ModuleType("lightrag.llm")
    openai_mod = types.ModuleType("lightrag.llm.openai")

    async def _unpatched(*args, **kwargs):
        raise AssertionError("openai_complete_if_cache was not patched")

    openai_mod.openai_complete_if_cache = _unpatched
    ensure("pandas", pandas_mod)
    ensure("dotenv", dotenv_mod)
    ensure("lightrag", lightrag_mod)
    ensure("lightrag.llm", llm_mod)
    ensure("lightrag.llm.openai", openai_mod)

    spec = importlib.util.spec_from_file_location(
        _MODULE_NAME, _ROOT / "reproduce" / "llm_answer_evaluator.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name in added:
            sys.modules.pop(name, None)
    return module


evaluator = _load_evaluator()


class _Config:
    def __init__(self, output_dir, request_timeout=5, max_retries=2):
        self.output_dir = str(output_dir)
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.api_key = "test-key"
        self.base_url = "http://llm.invalid"


def _make_evaluator(tmp_path, request_timeout=5, max_retries=2):
    return evaluator.LLMAnswerEvaluator(
        _Config(tmp_path, request_timeout=request_timeout, max_retries=max_retries)
    )


def _install_llm(monkeypatch, func):
    monkeypatch.setattr(evaluator, "openai_complete_if_cache", func)


async def _evaluate(scorer, evaluation_type):
    return await scorer.evaluate_single_answer(
        question="What is the bolt torque?",
        expected_answer="12 N.m",
        generated_answer="50 N.m",
        evidence_pages="4",
        evidence_sources="manual.pdf",
        doc_id="doc-torque",
        evaluation_type=evaluation_type,
    )


def _assert_call_identity(call):
    assert call["model"] == "gpt-4o-mini"
    assert call["api_key"] == "test-key"
    assert call["base_url"] == "http://llm.invalid"
    assert call["temperature"] == 0.1
    assert call["max_tokens"] == 1000
    assert "What is the bolt torque?" in call["prompt"]
    assert "12 N.m" in call["prompt"]
    assert "50 N.m" in call["prompt"]
    assert "manual.pdf" in call["prompt"]


@pytest.mark.asyncio
async def test_accuracy_only_keeps_explicit_zero_score(tmp_path, monkeypatch):
    calls = []

    async def fake_complete(**kwargs):
        calls.append(kwargs)
        return json.dumps({"accuracy": 0, "reasoning": "torque mismatch"})

    _install_llm(monkeypatch, fake_complete)
    result = await _evaluate(_make_evaluator(tmp_path), "accuracy_only")

    assert len(calls) == 1
    _assert_call_identity(calls[0])
    assert "assessing the accuracy" in calls[0]["prompt"]
    assert "Faithfulness" not in calls[0]["prompt"]
    assert result["success"] is True
    assert result["error"] is None
    assert result["accuracy"] == 0
    assert result["reasoning"] == "torque mismatch"
    assert result["evaluation_type"] == "accuracy_only"
    assert result["doc_id"] == "doc-torque"
    assert result["expected_answer"] == "12 N.m"
    assert result["generated_answer"] == "50 N.m"
    assert result["evidence_pages"] == "4"
    assert isinstance(result["timestamp"], str) and result["timestamp"]
    assert result["evaluation_time"] >= 0


@pytest.mark.asyncio
async def test_comprehensive_prompt_is_not_the_accuracy_rubric(tmp_path, monkeypatch):
    calls = []

    async def fake_complete(**kwargs):
        calls.append(kwargs)
        return json.dumps(
            {
                "accuracy": 1,
                "relevance": 1.0,
                "completeness": 1.0,
                "faithfulness": 1.0,
                "clarity": 1.0,
                "consistency": 1.0,
                "overall_score": 1.0,
                "reasoning": "matches",
            }
        )

    _install_llm(monkeypatch, fake_complete)
    result = await _evaluate(_make_evaluator(tmp_path), "comprehensive")

    assert len(calls) == 1
    _assert_call_identity(calls[0])
    assert "multiple dimensions" in calls[0]["prompt"]
    assert "Faithfulness" in calls[0]["prompt"]
    assert "assessing the accuracy" not in calls[0]["prompt"]
    assert result["success"] is True
    assert result["accuracy"] == 1
    assert result["overall_score"] == 1.0
    assert result["evaluation_type"] == "comprehensive"


@pytest.mark.asyncio
async def test_fenced_json_score_is_accepted(tmp_path, monkeypatch):
    async def fake_complete(**kwargs):
        return '\n```json\n{"accuracy": 1, "reasoning": "matches"}\n```\n'

    _install_llm(monkeypatch, fake_complete)
    result = await _evaluate(_make_evaluator(tmp_path), "accuracy_only")

    assert result["success"] is True
    assert result["accuracy"] == 1
    assert result["reasoning"] == "matches"
    assert result["error"] is None


@pytest.mark.asyncio
async def test_timeout_is_retried_then_recorded_without_a_guessed_score(
    tmp_path, monkeypatch
):
    calls = []

    async def always_timeout(**kwargs):
        calls.append(kwargs)
        raise asyncio.TimeoutError()

    _install_llm(monkeypatch, always_timeout)
    result = await _evaluate(
        _make_evaluator(tmp_path, request_timeout=7, max_retries=2),
        "accuracy_only",
    )

    assert len(calls) == 2
    assert result["success"] is False
    assert result["accuracy"] is None
    assert result["reasoning"] is None
    assert "timed out after 7 seconds" in result["error"]
    assert result["doc_id"] == "doc-torque"
    assert result["evaluation_type"] == "accuracy_only"


@pytest.mark.asyncio
async def test_timeout_then_json_keeps_the_later_score(tmp_path, monkeypatch):
    calls = []

    async def timeout_once(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise asyncio.TimeoutError()
        return json.dumps({"accuracy": 0, "reasoning": "still wrong"})

    _install_llm(monkeypatch, timeout_once)
    result = await _evaluate(_make_evaluator(tmp_path, max_retries=2), "accuracy_only")

    assert len(calls) == 2
    assert result["success"] is True
    assert result["accuracy"] == 0
    assert result["reasoning"] == "still wrong"


@pytest.mark.asyncio
async def test_non_timeout_error_is_not_retried(tmp_path, monkeypatch):
    calls = []

    async def fail_once(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("upstream 500")

    _install_llm(monkeypatch, fail_once)
    result = await _evaluate(_make_evaluator(tmp_path, max_retries=3), "comprehensive")

    assert len(calls) == 1
    assert result["success"] is False
    assert result["accuracy"] is None
    assert result["reasoning"] is None
    assert "Error during evaluation: upstream 500" in result["error"]


@pytest.mark.asyncio
async def test_prose_without_json_is_not_scored_as_accurate(tmp_path, monkeypatch):
    async def prose(**kwargs):
        return "The answer is correct and accurate."

    _install_llm(monkeypatch, prose)
    result = await _evaluate(_make_evaluator(tmp_path), "accuracy_only")

    assert result["success"] is False
    assert result["accuracy"] is None
    assert "Could not extract valid JSON" in result["error"]
