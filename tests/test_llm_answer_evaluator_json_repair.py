"""JSON repair edges for stored answer-evaluation scores.

Duplicate plain reasoning fields and a single missing brace are already
covered. These tests lock escaped quotes, a truncated string value, and
control characters that would otherwise be stored as the score payload.
"""

import importlib.util
import json
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
        "llm_answer_evaluator_json_repair_1127", module_path
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
    evaluator.logger = logging.getLogger("llm_evaluator_json_repair_1127")
    return evaluator


def test_escaped_quote_in_kept_reasoning_survives_duplicate_removal():
    evaluator = _evaluator()
    repaired = evaluator.fix_json_format(
        '{"accuracy": 1, "reasoning": "keep \\"this\\"", "reasoning": "drop"}'
    )

    assert repaired == '{"accuracy": 1, "reasoning": "keep \\"this\\""}'
    assert json.loads(repaired) == {"accuracy": 1, "reasoning": 'keep "this"'}


def test_truncated_string_and_nested_object_are_closed():
    evaluator = _evaluator()

    odd_quote = evaluator.fix_json_format('{"accuracy": 1, "reasoning": "cut off')
    assert odd_quote == '{"accuracy": 1, "reasoning": "cut off"}'
    assert json.loads(odd_quote)["reasoning"] == "cut off"

    closed_string = evaluator.fix_json_format('{"accuracy": 1, "reasoning": "done"')
    assert closed_string == '{"accuracy": 1, "reasoning": "done"}'

    nested = evaluator.fix_json_format('{"outer": {"accuracy": 1')
    assert nested == '{"outer": {"accuracy": 1}}'
    assert json.loads(nested) == {"outer": {"accuracy": 1}}


def test_control_characters_are_stripped_from_a_closed_object():
    evaluator = _evaluator()
    repaired = evaluator.fix_json_format('{"accuracy": 1, "note": "a\x00b\x7fc"}')

    assert repaired == '{"accuracy": 1, "note": "abc"}'
    assert json.loads(repaired) == {"accuracy": 1, "note": "abc"}


def test_single_reasoning_with_escaped_quote_is_unchanged():
    evaluator = _evaluator()
    original = '{"reasoning": "keep \\"this\\""}'

    assert evaluator.fix_json_format(original) == original
