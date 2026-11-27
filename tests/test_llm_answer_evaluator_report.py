"""Accuracy totals and document labels in the evaluation report.

A stored accuracy of 0 is a wrong answer, not a missing score. The report
must keep that zero in the rate, sort plant folders numerically, and label
manuals, filings, surveys, and papers from the document id and evidence.
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
        "llm_answer_evaluator_report_under_test", module_path
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


def _evaluator(output_dir):
    evaluator = EVAL.LLMAnswerEvaluator.__new__(EVAL.LLMAnswerEvaluator)
    evaluator.logger = logging.getLogger("llm_evaluator_report_under_test")
    evaluator.evaluation_results = []
    evaluator.existing_evaluations = {}
    evaluator.config = types.SimpleNamespace(output_dir=str(output_dir))
    return evaluator


def _row(
    doc_id,
    method,
    accuracy,
    source_file="",
    evidence_sources="",
    success=True,
    **scores,
):
    row = {
        "doc_id": doc_id,
        "question": f"Question for {doc_id}",
        "method": method,
        "source_file": source_file,
        "evidence_sources": evidence_sources,
        "success": success,
        "accuracy": accuracy,
    }
    row.update(scores)
    return row


def test_failed_only_results_do_not_write_a_report(tmp_path):
    evaluator = _evaluator(tmp_path)
    evaluator.evaluation_results = [
        _row("seal-kit.pdf", "hybrid", 1, success=False),
    ]

    evaluator.generate_evaluation_report(4.0)

    assert not (tmp_path / "llm_evaluation_report.md").exists()


def test_report_keeps_zero_scores_and_sorts_document_labels(tmp_path):
    evaluator = _evaluator(tmp_path)
    dimensions = {
        "relevance": 0.5,
        "completeness": 0.5,
        "faithfulness": 0.5,
        "clarity": 0.5,
        "consistency": 0.5,
        "overall_score": 0.5,
    }
    long_doc = "bearing-clearance-housing-manual.pdf"
    evaluator.evaluation_results = [
        _row("seal-kit.pdf", "hybrid", 1, success=False),
        _row(
            "torque-spec.pdf",
            "hybrid",
            0,
            source_file="/plants/reports/answers.json",
        ),
        _row(
            "torque-spec.pdf",
            "mix",
            1,
            source_file="/plants/reports/answers.json",
        ),
        _row(
            long_doc,
            "vlm",
            1,
            source_file="/evals/2/out.json",
            **dimensions,
        ),
        _row(
            "survey-notes.pdf",
            "vlm",
            0,
            source_file="/evals/10/out.json",
            **dimensions,
        ),
        _row("gasket.pdf", "vlm", 0, **dimensions),
        _row("notes.pdf", "vlm", None, **dimensions),
        _row(
            "10k-filing.pdf",
            "vlm",
            1,
            source_file="/evals/misc/out.json",
            **dimensions,
        ),
        _row(
            "attention.pdf",
            "vlm",
            1,
            source_file="/evals/papers/out.json",
            evidence_sources="arXiv:2401.00001",
            **dimensions,
        ),
        _row("naive-only.pdf", "naive", 1),
    ]

    evaluator.generate_evaluation_report(8.0)

    report = (tmp_path / "llm_evaluation_report.md").read_text(encoding="utf-8")
    long_display = f"2/{long_doc}"
    assert long_display not in report
    assert f"{long_display[:30]}..." in report

    assert "- Total evaluations: 10" in report
    assert "- Successful evaluations: 9" in report
    assert "- Failed evaluations: 1" in report
    assert "- Total elapsed time: 8.00 seconds" in report
    assert "- Average time per evaluation: 0.80 seconds" in report
    assert "seal-kit.pdf" not in report

    assert (
        """## HYBRID Method Results

### Accuracy Statistics
- Evaluated samples: 1
- Correct answers: 0
- Accuracy rate: 0.0%
"""
        in report
    )
    assert (
        """## MIX Method Results

### Accuracy Statistics
- Evaluated samples: 1
- Correct answers: 1
- Accuracy rate: 100.0%
"""
        in report
    )
    assert (
        """## VLM Method Results

### Accuracy Statistics
- Evaluated samples: 6
- Correct answers: 3
- Accuracy rate: 60.0%
"""
        in report
    )
    assert "- Average relevance: 0.500" in report
    assert "- Average overall score: 0.500" in report
    assert (
        """## NAIVE Method Results

### Accuracy Statistics
- Evaluated samples: 1
- Correct answers: 1
- Accuracy rate: 100.0%
"""
        in report
    )

    assert "| hybrid | 1 | 0.0% | 0.000 |" in report
    assert "| mix | 1 | 100.0% | 0.000 |" in report
    assert "| vlm | 6 | 60.0% | 0.500 |" in report
    assert "| naive | 1 | 100.0% | 0.000 |" in report
    assert "| reports/torque-spec.pdf | hybrid | 0/1 | 0.0% |" in report
    assert "| reports/torque-spec.pdf | mix | 1/1 | 100.0% |" in report

    document_section = report.split("## Document-Level Results (VLM Method)", 1)[1]
    assert "naive-only.pdf" not in report

    rows = [
        f"| {long_display[:30]}... | 1/1 | 100.0% | Manual / guide |",
        "| 10/survey-notes.pdf | 0/1 | 0.0% | Research report |",
        "| gasket.pdf | 0/1 | 0.0% | Other |",
        "| misc/10k-filing.pdf | 1/1 | 100.0% | Financial report |",
        "| notes.pdf | 0/0 | 0.0% | Other |",
        "| papers/attention.pdf | 1/1 | 100.0% | Academic paper |",
    ]
    positions = [document_section.index(row) for row in rows]
    assert positions == sorted(positions)
    assert "| Other | 2 | 1 | 0 | 0.0% |" in document_section
    assert "| Manual / guide | 1 | 1 | 1 | 100.0% |" in document_section
    assert "| Research report | 1 | 1 | 0 | 0.0% |" in document_section
