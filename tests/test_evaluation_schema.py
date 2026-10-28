"""Focused tests for the versioned, dependency-free evaluation schemas."""

from __future__ import annotations

import json

import pytest

from raganything.evaluation_schema import (
    CASE_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    EvaluationCase,
    EvaluationDataset,
    EvaluationRun,
    EvaluationTraceResult,
    ExpectedAction,
    RunManifest,
    SchemaValidationError,
    load_evaluation_cases,
    load_evaluation_dataset,
    load_evaluation_run,
)


def _case_dict(case_id: str = "maintenance-001") -> dict:
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "query": "主轴润滑的周期是什么？",
        "conversation": [
            {"role": "user", "content": "我正在检查设备保养计划。"},
            {"role": "assistant", "content": "请告诉我需要检查的部件。"},
        ],
        "expected_action": "answer",
        "reference_answer": "主轴应每周润滑一次。",
        "vital_claims": ["润滑周期是每周一次"],
        "optional_claims": ["使用指定润滑脂"],
        "forbidden_claims": ["每天润滑一次"],
        "evidence": [
            {
                "doc_id": "manual-a",
                "hash": "sha256:abc",
                "page": 12,
                "span": {"start": 42, "end": 61, "quote": "每周润滑一次"},
                "bbox": {
                    "x0": 0.1,
                    "y0": 0.2,
                    "x1": 0.8,
                    "y1": 0.3,
                    "coordinate_space": "normalized",
                },
                "modality": "table",
                "chunk_id": "chunk-runtime-9",
            }
        ],
        "expected_images": [
            {
                "image_id": "lubrication-point",
                "description": "主轴润滑点示意图",
                "required": True,
                "evidence": [
                    {
                        "doc_id": "manual-a",
                        "hash": "sha256:abc",
                        "page": 13,
                        "bbox": [10, 20, 200, 300],
                        "modality": "image",
                    }
                ],
            }
        ],
        "severity": "critical",
        "tags": ["maintenance", "numeric", "image"],
        "split": "holdout",
        "metadata": {"owner": "domain-team", "reviewed": True},
    }


def _manifest_dict() -> dict:
    return {
        "run_id": "run-2026-10-26-001",
        "dataset_id": "industrial-gold",
        "dataset_version": "2026.10",
        "dataset_hash": "sha256:dataset",
        "started_at": "2026-10-26T09:00:00+09:00",
        "runner": "web-production-path",
        "corpus_id": "manuals-prod-snapshot",
        "corpus_hash": "sha256:corpus",
        "code_revision": "abc123",
        "random_seed": 7,
        "config": {"top_k": 10},
        "models": {"generator": "example/model"},
        "environment": {"python": "3.10"},
    }


def _trace_dict(case_id: str = "maintenance-001") -> dict:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "case_id": case_id,
        "status": "completed",
        "observed_action": "answer",
        "answer": "主轴应每周润滑一次。",
        "started_at": "2026-10-26T09:01:00+09:00",
        "completed_at": "2026-10-26T09:01:02+09:00",
        "retrieved_evidence": [
            {
                "rank": 1,
                "score": 0.93,
                "text": "主轴：每周润滑一次",
                "anchor": {
                    "doc_id": "manual-a",
                    "hash": "sha256:abc",
                    "page": 12,
                    "span": "主轴：每周润滑一次",
                    "modality": "table",
                    "chunk_id": "chunk-runtime-9",
                },
            }
        ],
        "images": [
            {
                "image_id": "lubrication-point",
                "uri": "artifacts/lubrication.png",
                "evidence": [
                    {
                        "doc_id": "manual-a",
                        "page": 13,
                        "bbox": [10, 20, 200, 300],
                        "modality": "image",
                    }
                ],
                "metadata": {"placement": "after-supporting-paragraph"},
            }
        ],
        "metrics": {"vital_claim_recall": 1.0, "hard_gate_passed": True},
        "timings_ms": {"retrieval": 90.2, "total": 2000},
        "token_usage": {"prompt": 800, "completion": 50},
        "error": None,
        "trace": {"events": [{"kind": "retrieval_done"}]},
    }


def _write_jsonl(path, records) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def test_case_round_trip_preserves_typed_fields() -> None:
    case = EvaluationCase.from_dict(_case_dict())

    assert case.expected_action is ExpectedAction.ANSWER
    assert case.evidence[0].span.quote == "每周润滑一次"
    assert case.expected_images[0].evidence[0].bbox.coordinate_space == "page"
    assert case.to_dict() == EvaluationCase.from_dict(case.to_dict()).to_dict()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(expected_action="guess"), "expected one of"),
        (lambda row: row.update(schema_version="v0"), "unsupported version"),
        (
            lambda row: row["evidence"].__setitem__(
                0, {"doc_id": "manual-a", "modality": "text"}
            ),
            "provide at least one locator",
        ),
        (
            lambda row: row.update(forbidden_claims=["润滑周期是每周一次"]),
            "claims cannot also be expected",
        ),
        (lambda row: row.update(unexpected=True), "unknown field"),
    ],
)
def test_case_validation_rejects_invalid_documents(mutate, message: str) -> None:
    row = _case_dict()
    mutate(row)

    with pytest.raises(SchemaValidationError, match=message):
        EvaluationCase.from_dict(row)


def test_load_dataset_from_json_and_jsonl(tmp_path) -> None:
    dataset = EvaluationDataset(
        dataset_id="industrial-gold",
        dataset_version="2026.10",
        description="Reviewed release cases",
        created_at="2026-10-26T08:00:00Z",
        metadata={"reviewers": 2},
        cases=(EvaluationCase.from_dict(_case_dict()),),
    )
    json_path = tmp_path / "dataset.json"
    json_path.write_text(
        json.dumps(dataset.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    jsonl_path = tmp_path / "dataset.jsonl"
    _write_jsonl(jsonl_path, dataset.to_jsonl_records())

    loaded_json = load_evaluation_dataset(json_path)
    loaded_jsonl = load_evaluation_dataset(jsonl_path)

    assert loaded_json.to_dict() == dataset.to_dict()
    assert loaded_jsonl.to_dict() == dataset.to_dict()
    assert load_evaluation_cases(json_path) == loaded_json.cases
    assert load_evaluation_cases(jsonl_path) == loaded_jsonl.cases


def test_case_loader_accepts_bare_jsonl_records(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    _write_jsonl(path, [_case_dict("case-a"), _case_dict("case-b")])

    cases = load_evaluation_cases(path)

    assert [case.case_id for case in cases] == ["case-a", "case-b"]


def test_dataset_rejects_duplicate_case_ids() -> None:
    with pytest.raises(SchemaValidationError, match="case_id values must be unique"):
        EvaluationDataset(
            dataset_id="duplicate-set",
            dataset_version="1",
            cases=(
                EvaluationCase.from_dict(_case_dict()),
                EvaluationCase.from_dict(_case_dict()),
            ),
        )


def test_run_round_trip_and_jsonl_loader(tmp_path) -> None:
    run = EvaluationRun(
        manifest=RunManifest.from_dict(_manifest_dict()),
        traces=(EvaluationTraceResult.from_dict(_trace_dict()),),
        completed_at="2026-10-26T09:02:00+09:00",
        summary={"completed": 1, "failed": 0},
    )
    json_path = tmp_path / "run.json"
    json_path.write_text(
        json.dumps(run.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    jsonl_path = tmp_path / "run.jsonl"
    _write_jsonl(jsonl_path, run.to_jsonl_records())

    assert run.schema_version == RUN_SCHEMA_VERSION
    assert load_evaluation_run(json_path).to_dict() == run.to_dict()
    assert load_evaluation_run(jsonl_path).to_dict() == run.to_dict()


def test_trace_requires_error_details_and_ordered_timestamps() -> None:
    row = _trace_dict()
    row.update(status="error", error=None)
    with pytest.raises(SchemaValidationError, match="required when status is 'error'"):
        EvaluationTraceResult.from_dict(row)

    row = _trace_dict()
    row["completed_at"] = "2026-10-26T08:59:00+09:00"
    with pytest.raises(SchemaValidationError, match="must not precede started_at"):
        EvaluationTraceResult.from_dict(row)


def test_jsonl_parse_error_reports_source_line(tmp_path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps(_case_dict(), ensure_ascii=False) + "\n{" + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SchemaValidationError) as exc_info:
        load_evaluation_cases(path)

    assert f"{path}:2:" in str(exc_info.value)


def test_dataset_schema_version_is_explicit() -> None:
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": "industrial-gold",
        "dataset_version": "1",
        "cases": [_case_dict()],
    }
    assert EvaluationDataset.from_dict(payload).schema_version == DATASET_SCHEMA_VERSION
