"""Production-path collection behavior for the RAGAS adapter."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import eval_ragas_webpath as evaluation


def test_default_dataset_is_tracked_canonical_shili17() -> None:
    rows = evaluation._load_dataset(evaluation._DEFAULT_DATASET)

    assert len(rows) == 17
    assert rows[0]["id"] == 1
    assert rows[0]["ground_truth"]


@pytest.mark.asyncio
async def test_collect_row_preserves_trace_and_non_answer_outcome() -> None:
    web = SimpleNamespace(QueryBody=SimpleNamespace)

    async def execute_query_with_trace(body):
        assert body.query == "ambiguous question"
        assert body.mode == "mix"
        return {
            "outcome": "clarify",
            "answer": "",
            "contexts": [],
            "duration_ms": 12,
            "error": None,
            "clarification": {"clarification_id": "c1"},
        }

    web.execute_query_with_trace = execute_query_with_trace
    row = await evaluation._collect_webpath_row(
        web,
        case={
            "id": 7,
            "question": "ambiguous question",
            "ground_truth": "reference",
        },
        mode="mix",
    )

    assert row["outcome"] == "clarify"
    assert row["answer"] == ""
    assert row["context_count"] == 0
    assert row["trace"]["clarification"] == {"clarification_id": "c1"}


@pytest.mark.asyncio
async def test_collect_rows_wires_web_state_and_restores_environment(
    monkeypatch, tmp_path
) -> None:
    calls: list[str] = []

    class FakeRag:
        async def finalize_storages(self):
            calls.append("finalized")

    fake_rag = FakeRag()
    fake_config = SimpleNamespace(name="config")

    async def build_rag(wd, pod, *, enable_llm_cache):
        assert wd == tmp_path / "storage"
        assert pod == tmp_path / "parser"
        assert enable_llm_cache is False
        assert "EVAL_IMPORT_MUTATION" not in os.environ
        return fake_rag, fake_config, SimpleNamespace()

    fake_rpc = SimpleNamespace(_build_rag=build_rag)

    def load_rpc():
        assert os.environ["RAG_WEB_WORKING_DIR"] == str(tmp_path / "storage")
        assert os.environ["RAG_WEB_PARSER_OUTPUT_DIR"] == str(tmp_path / "parser")
        os.environ["EVAL_IMPORT_MUTATION"] = "from dynamic import"
        return fake_rpc

    monkeypatch.setattr(evaluation, "_load_rpc", load_rpc)

    fake_web = ModuleType("rag_web_server")
    fake_web.QueryBody = SimpleNamespace
    fake_web.state = SimpleNamespace()

    async def execute_query_with_trace(body):
        calls.append(body.query)
        assert fake_web.state.rag is fake_rag
        assert fake_web.state.config is fake_config
        return {
            "outcome": "answer",
            "answer": f"answer:{body.query}",
            "contexts": [f"context:{body.query}"],
            "duration_ms": 5,
            "error": None,
        }

    fake_web.execute_query_with_trace = execute_query_with_trace
    monkeypatch.setitem(sys.modules, "rag_web_server", fake_web)
    monkeypatch.setenv("EVAL_ENV_SENTINEL", "before")
    before = dict(os.environ)

    rows = await evaluation.collect_rows(
        [
            {"id": 1, "question": "q1", "ground_truth": "a1"},
            {"id": 2, "question": "q2", "ground_truth": "a2"},
        ],
        mode="mix",
        wd=tmp_path / "storage",
        pod=tmp_path / "parser",
    )

    assert calls == ["q1", "q2", "finalized"]
    assert [row["contexts"] for row in rows] == [["context:q1"], ["context:q2"]]
    assert dict(os.environ) == before


@pytest.mark.asyncio
async def test_collect_rows_checkpoints_errors_and_records_finalize_failure(
    monkeypatch, tmp_path
) -> None:
    class FakeRag:
        async def finalize_storages(self):
            raise RuntimeError("finalize broke")

    async def build_rag(*_args, **_kwargs):
        return FakeRag(), SimpleNamespace(), SimpleNamespace()

    monkeypatch.setattr(
        evaluation,
        "_load_rpc",
        lambda: SimpleNamespace(_build_rag=build_rag),
    )
    fake_web = ModuleType("rag_web_server")
    fake_web.QueryBody = SimpleNamespace
    fake_web.state = SimpleNamespace()

    async def execute_query_with_trace(body):
        if body.query == "bad":
            raise ValueError("provider broke")
        return {
            "outcome": "answer",
            "answer": "ok",
            "contexts": ["source"],
            "duration_ms": 1,
            "error": None,
        }

    fake_web.execute_query_with_trace = execute_query_with_trace
    monkeypatch.setitem(sys.modules, "rag_web_server", fake_web)
    checkpoint_sizes: list[int] = []
    diagnostics: dict[str, object] = {}

    rows = await evaluation.collect_rows(
        [
            {"id": 1, "question": "ok", "ground_truth": "a"},
            {"id": 2, "question": "bad", "ground_truth": "b"},
        ],
        mode="mix",
        wd=tmp_path / "storage",
        pod=tmp_path / "parser",
        on_progress=lambda partial: checkpoint_sizes.append(len(partial)),
        diagnostics=diagnostics,
    )

    assert checkpoint_sizes == [1, 2]
    assert rows[0]["outcome"] == "answer"
    assert rows[1]["outcome"] == "error"
    assert rows[1]["error"] == "ValueError: provider broke"
    assert diagnostics == {"finalization_error": "RuntimeError: finalize broke"}


def test_ragas_row_eligibility_requires_successful_completed_answer() -> None:
    valid = {
        "outcome": "answer",
        "answer": "complete answer",
        "contexts": ["source"],
        "error": None,
        "trace": {
            "terminal_event": {"type": "done"},
            "contexts_complete": True,
        },
    }
    partial_error = {
        **valid,
        "outcome": "error",
        "answer": "partial answer",
        "error": "provider disconnected",
        "trace": {"terminal_event": {"type": "done", "error": True}},
    }
    unfinished = {**valid, "trace": {"terminal_event": None}}
    legacy_without_terminal_field = {**valid, "trace": {}}
    incomplete_context = {
        **valid,
        "trace": {
            "terminal_event": {"type": "done"},
            "contexts_complete": False,
        },
    }

    assert evaluation._unevaluated_reason(valid) is None
    assert (
        evaluation._unevaluated_reason(partial_error) == "partial_answer_ended_in_error"
    )
    assert evaluation._unevaluated_reason(unfinished) == "terminal_not_done"
    assert evaluation._unevaluated_reason(legacy_without_terminal_field) is None
    assert (
        evaluation._unevaluated_reason(incomplete_context)
        == "incomplete_context_capture"
    )


def test_zero_usable_rows_returns_zero_coverage_result() -> None:
    rows = [
        {
            "id": 1,
            "outcome": "error",
            "answer": "partial answer",
            "contexts": ["source"],
            "error": "stream failed",
            "trace": {"terminal_event": {"type": "done", "error": True}},
        },
        {
            "id": 2,
            "outcome": "clarify",
            "answer": "",
            "contexts": [],
            "error": None,
            "trace": {"terminal_event": {"type": "done"}},
        },
    ]

    result = evaluation.run_ragas_eval(rows)

    assert result["evaluated"] == 0
    assert result["skipped"] == 2
    assert result["coverage"] == 0.0
    assert result["per_case"] == []
    assert [item["reason"] for item in result["unevaluated"]] == [
        "partial_answer_ended_in_error",
        "outcome_not_answer",
    ]


def test_judge_nan_is_not_reported_as_full_ragas_coverage() -> None:
    rows = [
        {"id": 1, "question": "q1"},
        {"id": 2, "question": "q2"},
    ]
    result = evaluation._finalize_ragas_scores(
        rows=rows,
        usable=rows,
        unevaluated=[],
        score_rows=[
            {"faithfulness": 0.9, "answer_correctness": 0.8},
            {"faithfulness": "nan", "answer_correctness": 0.7},
        ],
        metric_cols=["faithfulness", "answer_correctness"],
        llm_model="judge",
        embedding_model="embed",
    )

    assert result["eligible"] == 2
    assert result["eligibility_coverage"] == 1.0
    assert result["evaluated"] == 1
    assert result["partially_evaluated"] == 1
    assert result["coverage"] == 0.5
    assert result["metric_coverage"]["faithfulness"] == {
        "evaluated": 1,
        "missing": 1,
        "coverage": 0.5,
        "eligible_coverage": 0.5,
    }
    assert result["judge_incomplete"] == [
        {"id": 2, "missing_metrics": ["faithfulness"]}
    ]
    assert result["summary"]["faithfulness"] == 0.9


def test_report_renders_coverage_and_unevaluated_reasons(tmp_path) -> None:
    report = evaluation._format_report(
        stamp="test",
        rows=[{"id": 1}, {"id": 2}],
        ragas={
            "eligible": 0,
            "evaluated": 0,
            "partially_evaluated": 0,
            "skipped": 2,
            "coverage": 0.0,
            "eligibility_coverage": 0.0,
            "summary": {},
            "per_case": [],
            "raw_columns": [],
            "metric_coverage": {},
            "judge_incomplete": [],
            "unevaluated": [
                {
                    "id": 1,
                    "outcome": "error",
                    "error": "stream failed",
                    "reason": "partial_answer_ended_in_error",
                },
                {
                    "id": 2,
                    "outcome": "clarify",
                    "error": None,
                    "reason": "outcome_not_answer",
                },
            ],
        },
        meta={
            "time": "now",
            "working_dir": "/wd",
            "parser_output_dir": "/pod",
            "mode": "mix",
            "dataset": "/dataset.json",
        },
        run_path=tmp_path / "run.jsonl",
        csv_path=tmp_path / "ragas_scores.csv",
    )

    assert "evaluated: 0" in report
    assert "skipped: 2" in report
    assert "coverage: 0.0" in report
    assert "1: partial_answer_ended_in_error" in report
    assert "outcome=error, error=stream failed" in report
    assert "2: outcome_not_answer (outcome=clarify)" in report
