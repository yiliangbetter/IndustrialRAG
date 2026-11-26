"""Resume identity and error classification for the answer evaluator.

A completed score of 0 must not be retried, a failed run must be, and
report summaries must keep document-missing and token-limit failures
distinct from generic query errors.
"""

import importlib.util
import logging
import sys
import types
from pathlib import Path


def _load_evaluator_module():
    """Load the evaluator without pulling LightRAG, pandas, or dotenv."""
    stubs = {
        "pandas": types.ModuleType("pandas"),
        "lightrag": types.ModuleType("lightrag"),
        "lightrag.llm": types.ModuleType("lightrag.llm"),
        "lightrag.llm.openai": types.ModuleType("lightrag.llm.openai"),
        "dotenv": types.ModuleType("dotenv"),
    }
    stubs["lightrag.llm.openai"].openai_complete_if_cache = None
    stubs["dotenv"].load_dotenv = lambda *args, **kwargs: False

    present = set(sys.modules)
    for name, module in stubs.items():
        sys.modules.setdefault(name, module)

    module_path = (
        Path(__file__).resolve().parents[1] / "reproduce" / "llm_answer_evaluator.py"
    )
    path_before = list(sys.path)
    spec = importlib.util.spec_from_file_location(
        "llm_answer_evaluator_resume_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = path_before
        for name in list(sys.modules):
            if name in present:
                continue
            if (
                name == "pandas"
                or name == "dotenv"
                or name.startswith(("lightrag", "dotenv."))
            ):
                sys.modules.pop(name, None)
    return module


EVAL = _load_evaluator_module()


def _evaluator():
    evaluator = EVAL.LLMAnswerEvaluator.__new__(EVAL.LLMAnswerEvaluator)
    evaluator.logger = logging.getLogger("llm_evaluator_resume_under_test")
    evaluator.evaluation_results = []
    evaluator.existing_evaluations = {}
    return evaluator


def test_completed_zero_score_is_not_retried_and_failures_are():
    evaluator = _evaluator()
    question = "What is the bearing clearance?"
    evaluator.existing_evaluations = {
        ("manual.pdf", question, "hybrid"): {"success": True, "accuracy": 0},
        ("manual.pdf", question, "mix"): {"success": True, "accuracy": None},
        ("manual.pdf", question, "naive"): {"success": False, "accuracy": 1},
        ("manual.pdf", "Which seal kit?", ""): {"success": True, "accuracy": 1},
    }

    assert evaluator.is_evaluation_completed("manual.pdf", question, "hybrid") is True
    assert evaluator.is_evaluation_completed("manual.pdf", question, "mix") is False
    assert evaluator.is_evaluation_completed("manual.pdf", question, "naive") is False
    assert evaluator.is_evaluation_completed("manual.pdf", question, "qa") is False
    assert (
        evaluator.is_evaluation_completed("manual.pdf", "Which seal kit?", "") is True
    )
    assert (
        evaluator.get_existing_evaluation("manual.pdf", question, "hybrid")["accuracy"]
        == 0
    )


def test_update_replaces_same_method_and_keeps_other_methods():
    evaluator = _evaluator()
    question = "What torque is specified?"
    hybrid = {
        "doc_id": "manual.pdf",
        "question": question,
        "method": "hybrid",
        "accuracy": 0,
    }
    mix = {
        "doc_id": "manual.pdf",
        "question": question,
        "method": "mix",
        "accuracy": 1,
    }
    omitted = {"doc_id": "manual.pdf", "question": "Which gasket?", "accuracy": 1}
    evaluator.evaluation_results = [hybrid, mix, omitted]

    replacement = {
        "doc_id": "manual.pdf",
        "question": question,
        "method": "hybrid",
        "accuracy": 1,
    }
    omitted_replacement = {
        "doc_id": "manual.pdf",
        "question": "Which gasket?",
        "accuracy": 0,
    }
    evaluator.update_evaluation_results_list([replacement, omitted_replacement])

    assert evaluator.evaluation_results == [mix, replacement, omitted_replacement]


def test_error_detection_uses_each_query_channel():
    evaluator = _evaluator()

    assert evaluator.has_any_error({}) is False
    assert evaluator.has_any_error({"vlm_result": {}}) is False
    assert evaluator.has_any_error({"error": "parser crashed"}) is True
    assert evaluator.has_any_error({"vlm_result": {"success": False}}) is True
    assert (
        evaluator.has_any_error(
            {"hybrid_result": {"success": True, "error": "host reset"}}
        )
        is True
    )
    assert evaluator.has_any_error({"mix_result": {"success": False}}) is True
    assert evaluator.has_any_error({"naive_result": {"success": False}}) is True
    assert (
        evaluator.has_any_error(
            {
                "vlm_result": {"success": True},
                "hybrid_result": {"success": True},
                "mix_result": {"success": True},
                "naive_result": {"success": True},
            }
        )
        is False
    )


def test_error_summary_classifies_missing_docs_and_token_limits():
    evaluator = _evaluator()
    long_error = "embedding host reset while scoring the housing manual"
    summary = evaluator.get_error_summary(
        {
            "error": "Document not found: housing-manual.pdf",
            "vlm_result": {"error": "context_length_exceeded for page 4"},
            "hybrid_result": {"error": long_error},
            "mix_result": {"success": False},
            "naive_result": {"success": False},
        }
    )

    assert summary == (
        "Document not found; "
        "VLM token limit exceeded; "
        f"Hybrid error: {long_error[:50]}...; "
        "Mix error: Mix query failed...; "
        "Naive error: Naive query failed..."
    )
    assert evaluator.get_error_summary({}) == "Unknown error"


def test_fallback_keeps_explicit_accuracy_and_repairs_json():
    evaluator = _evaluator()
    scored = evaluator.create_fallback_evaluation(
        'The figure is present. {"accuracy": 0, "relevance": 0.5}'
    )
    assert scored["accuracy"] == 0
    assert scored["relevance"] == 0.5

    unscored = evaluator.create_fallback_evaluation("The model returned a paragraph.")
    assert unscored["accuracy"] == 0
    assert unscored["relevance"] is None

    fenced = evaluator.fix_json_format(
        '```json\n{"accuracy": 1, "reasoning": "kept", "reasoning": "dropped"}\n```'
    )
    assert fenced == '{"accuracy": 1, "reasoning": "kept"}'

    closed = evaluator.fix_json_format('{\n"accuracy": 1')
    assert closed == '{"accuracy": 1}'
