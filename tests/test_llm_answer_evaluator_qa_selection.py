"""QA-file selection for stored answer evaluations.

The resume helper, error summaries, report scores, and JSON repair are
already covered. These tests lock which QA result files are scored, which
rows are skipped, and which document id is attached before a question is
sent to the evaluator.
"""

import asyncio
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
        "llm_answer_evaluator_qa_selection_under_test", module_path
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


def _bind_qa_evaluator(output_dir):
    evaluator = EVAL.LLMAnswerEvaluator.__new__(EVAL.LLMAnswerEvaluator)
    evaluator.logger = logging.getLogger("llm_evaluator_qa_selection")
    evaluator.evaluation_results = []
    evaluator.existing_evaluations = {}
    evaluator.config = types.SimpleNamespace(output_dir=str(output_dir))
    calls = []
    saves = []
    reports = []

    async def evaluate_single_answer(**kwargs):
        calls.append(kwargs)
        return {
            "success": True,
            "accuracy": 1,
            "question": kwargs["question"],
            "doc_id": kwargs["doc_id"],
        }

    def save_evaluation_results():
        saves.append(len(evaluator.evaluation_results))

    def generate_evaluation_report(total_time):
        reports.append(total_time)

    evaluator.evaluate_single_answer = evaluate_single_answer
    evaluator.save_evaluation_results = save_evaluation_results
    evaluator.generate_evaluation_report = generate_evaluation_report
    return evaluator, calls, saves, reports


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_qa_directory_dedupes_sorted_result_files_and_ignores_absent_dirs(tmp_path):
    root = tmp_path / "manuals"
    exact = root / "bearing" / "qa_results_naive_mm.json"
    rerun = root / "bearing" / "qa_results_naive_mm_rerun.json"
    nested = root / "seal" / "nested" / "qa_results_naive_mm.json"
    _write_json(exact, [])
    _write_json(rerun, [])
    _write_json(nested, [])
    _write_json(root / "seal" / "qa_results_other.json", [])
    _write_json(root / "bearing" / "QA_RESULTS_NAIVE_MM.json", [])
    (root / "seal" / "notes.txt").write_text("not a qa file", encoding="utf-8")
    (root / "bearing" / "qa_results_naive_mm.json.bak").write_text(
        "[]", encoding="utf-8"
    )

    evaluator, calls, saves, reports = _bind_qa_evaluator(tmp_path)
    seen = []

    async def evaluate_qa_files(
        qa_files, evaluation_type="comprehensive", max_evaluations=None
    ):
        seen.append((list(qa_files), evaluation_type, max_evaluations))

    evaluator.evaluate_qa_files = evaluate_qa_files

    asyncio.run(evaluator.evaluate_qa_directory(str(tmp_path / "missing-manuals")))
    (root / "empty-folder").mkdir()
    asyncio.run(
        evaluator.evaluate_qa_directory(
            str(root / "empty-folder"),
            evaluation_type="accuracy_only",
            max_evaluations=2,
        )
    )
    assert seen == []
    assert calls == []
    assert saves == []
    assert reports == []

    asyncio.run(
        evaluator.evaluate_qa_directory(
            str(root),
            evaluation_type="accuracy_only",
            max_evaluations=2,
        )
    )

    assert seen == [
        (
            [str(exact), str(rerun), str(nested)],
            "accuracy_only",
            2,
        )
    ]


def test_qa_rows_skip_invalid_items_and_use_the_pdf_name(tmp_path):
    folder = tmp_path / "housing"
    folder.mkdir()
    (folder / "Housing-Manual.PDF").write_bytes(b"%PDF-1.4 housing")
    valid = folder / "qa_results_naive_mm.json"
    _write_json(
        valid,
        [
            {
                "question": "  What torque is specified?  ",
                "correct_answer": " 40 Nm ",
                "answer": " 40 Nm ",
            },
            {"question": "   ", "correct_answer": "ignored", "answer": "ignored"},
            {"correct_answer": "orphan", "answer": "orphan"},
            {"question": "Which gasket?", "correct_answer": "", "answer": ""},
        ],
    )
    _write_json(folder / "qa_results_naive_mm_object.json", {"question": "not a list"})
    (folder / "qa_results_naive_mm_broken.json").write_text("{", encoding="utf-8")

    evaluator, calls, saves, reports = _bind_qa_evaluator(tmp_path)
    asyncio.run(
        evaluator.evaluate_qa_files(
            [
                str(valid),
                str(folder / "qa_results_naive_mm_object.json"),
                str(folder / "qa_results_naive_mm_broken.json"),
                str(folder / "qa_results_naive_mm_missing.json"),
            ],
            evaluation_type="accuracy_only",
        )
    )

    assert [call["question"] for call in calls] == [
        "What torque is specified?",
        "Which gasket?",
    ]
    assert calls[0]["expected_answer"] == "40 Nm"
    assert calls[0]["generated_answer"] == "40 Nm"
    assert calls[1]["expected_answer"] == ""
    assert calls[1]["generated_answer"] == ""
    assert {call["doc_id"] for call in calls} == {"Housing-Manual.PDF"}
    assert {call["evidence_pages"] for call in calls} == {""}
    assert {call["evidence_sources"] for call in calls} == {""}
    assert {call["evaluation_type"] for call in calls} == {"accuracy_only"}
    assert [row["question"] for row in evaluator.evaluation_results] == [
        "What torque is specified?",
        "Which gasket?",
    ]
    assert {row["method"] for row in evaluator.evaluation_results} == {"qa"}
    assert {row["source_file"] for row in evaluator.evaluation_results} == {str(valid)}
    assert saves == [1, 2, 2]
    assert len(reports) == 1
    assert reports[0] >= 0


def test_qa_row_cap_is_applied_separately_to_each_file(tmp_path):
    folder = tmp_path / "housing"
    folder.mkdir()
    (folder / "Housing-Manual.PDF").write_bytes(b"%PDF-1.4 housing")
    clearance = folder / "qa_results_naive_mm.json"
    seal = folder / "qa_results_naive_mm_seal.json"
    _write_json(
        clearance,
        [
            {"question": "   ", "correct_answer": "0.05 mm", "answer": "0.05 mm"},
            {
                "question": "What is the bearing clearance?",
                "correct_answer": "0.05 mm",
                "answer": "0.05 mm",
            },
        ],
    )
    _write_json(
        seal,
        [
            {
                "question": "Which seal kit?",
                "correct_answer": "green",
                "answer": "green",
            },
            {
                "question": "Which gasket?",
                "correct_answer": "fiber",
                "answer": "fiber",
            },
        ],
    )

    evaluator, calls, saves, reports = _bind_qa_evaluator(tmp_path)
    asyncio.run(
        evaluator.evaluate_qa_files(
            [str(clearance), str(seal)],
            max_evaluations=1,
        )
    )

    assert calls == [
        {
            "question": "Which seal kit?",
            "expected_answer": "green",
            "generated_answer": "green",
            "evidence_pages": "",
            "evidence_sources": "",
            "doc_id": "Housing-Manual.PDF",
            "evaluation_type": "comprehensive",
        }
    ]
    assert evaluator.evaluation_results[0]["source_file"] == str(seal)
    assert evaluator.evaluation_results[0]["method"] == "qa"
    assert saves == [1, 1]
    assert len(reports) == 1


def test_folder_name_doc_id_skips_a_completed_qa_question(tmp_path):
    folder = tmp_path / "seal-kit"
    qa_file = folder / "qa_results_naive_mm.json"
    _write_json(
        qa_file,
        [
            {
                "question": "  Which seal kit?  ",
                "correct_answer": "green",
                "answer": "green",
            },
            {
                "question": "Which gasket?",
                "correct_answer": "fiber",
                "answer": "blue",
            },
        ],
    )

    evaluator, calls, saves, reports = _bind_qa_evaluator(tmp_path)
    evaluator.existing_evaluations[("seal-kit", "Which seal kit?", "qa")] = {
        "success": True,
        "accuracy": 1,
        "doc_id": "seal-kit",
        "question": "Which seal kit?",
        "method": "qa",
    }

    asyncio.run(evaluator.evaluate_qa_files([str(qa_file)]))

    assert [call["question"] for call in calls] == ["Which gasket?"]
    assert calls[0]["doc_id"] == "seal-kit"
    assert calls[0]["expected_answer"] == "fiber"
    assert calls[0]["generated_answer"] == "blue"
    assert [row["question"] for row in evaluator.evaluation_results] == [
        "Which gasket?"
    ]
    assert ("seal-kit", "Which seal kit?", "qa") in evaluator.existing_evaluations
    assert saves == [1, 1]
    assert len(reports) == 1
