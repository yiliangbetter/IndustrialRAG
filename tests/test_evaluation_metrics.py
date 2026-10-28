"""Deterministic scorer tests that require no model or external service."""

from __future__ import annotations

import json

import pytest

from raganything.evaluation_metrics import (
    DeterministicEvaluationScore,
    normalize_lexical_text,
    score_evaluation_case,
)
from raganything.evaluation_schema import (
    CASE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    EvaluationCase,
    EvaluationTraceResult,
)


def _anchor(doc_id: str, page: int = 1) -> dict:
    return {
        "doc_id": doc_id,
        "hash": f"sha256:{doc_id}",
        "page": page,
        "modality": "text",
    }


def _case_dict(**overrides) -> dict:
    value = {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": "case-001",
        "query": "主轴的润滑要求是什么？",
        "expected_action": "answer",
        "vital_claims": ["主轴每周润滑一次", "使用2号润滑脂"],
        "optional_claims": ["停机后操作"],
        "forbidden_claims": ["每天润滑一次"],
        "evidence": [_anchor("manual-a", 12), _anchor("manual-b", 3)],
        "expected_images": [
            {
                "image_id": "figure-required",
                "evidence": [_anchor("manual-a", 13)],
                "required": True,
            },
            {
                "image_id": "figure-optional",
                "evidence": [_anchor("manual-a", 14)],
                "required": False,
            },
        ],
    }
    value.update(overrides)
    return value


def _retrieved(doc_id: str, rank: int, page: int = 1, **anchor_overrides) -> dict:
    anchor = _anchor(doc_id, page)
    anchor.update(anchor_overrides)
    return {"anchor": anchor, "rank": rank, "score": 0.9}


def _trace_dict(**overrides) -> dict:
    value = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "case_id": "case-001",
        "status": "completed",
        "observed_action": "answer",
        "answer": "主轴：每 周，润滑一次；使用２号润滑脂。",
        "retrieved_evidence": [
            _retrieved("manual-b", 1, 3),
            _retrieved("manual-a", 2, 12),
            _retrieved("irrelevant", 3),
        ],
        "images": [
            {
                "image_id": "figure-required",
                "evidence": [_anchor("manual-a", 13)],
            }
        ],
        "trace": {
            "contexts_complete": True,
            "terminal_event": {"type": "done"},
        },
    }
    value.update(overrides)
    return value


def test_unicode_normalization_removes_spacing_punctuation_and_formatting() -> None:
    text = " Ａ＿b，C。\u200b\n"

    assert normalize_lexical_text(text) == "a_bc"


@pytest.mark.parametrize(
    ("technical_token", "collapsed_token"),
    [
        ("0.6", "06"),
        ("5-10", "510"),
        ("AB-12", "AB12"),
        ("part_number", "partnumber"),
    ],
)
def test_normalization_preserves_meaningful_intra_token_punctuation(
    technical_token: str, collapsed_token: str
) -> None:
    assert normalize_lexical_text(technical_token) != normalize_lexical_text(
        collapsed_token
    )


def test_normalization_still_ignores_spacing_and_sentence_punctuation() -> None:
    assert normalize_lexical_text(" 版本 1 . 2，已发布。 ") == "版本1.2已发布"
    assert normalize_lexical_text("5 – 10") == "5-10"


def test_perfect_safety_gate_with_lexical_claim_matching() -> None:
    score = score_evaluation_case(_case_dict(), _trace_dict())

    assert score.metrics == {
        "status_completed": True,
        "terminal_event_complete": True,
        "action_match": True,
        "vital_claim_recall": 1.0,
        "all_vital_claims_present": True,
        "optional_claim_recall": 0.0,
        "forbidden_claim_hit_rate": 0.0,
        "no_forbidden_claims_present": True,
        "answer_evidence_present": True,
        "answer_has_gold_evidence": True,
        "context_capture_complete": True,
        "answer_has_graded_content": True,
        "gold_document_recall": 1.0,
        "all_gold_documents_retrieved": True,
        "gold_anchor_recall": 1.0,
        "all_gold_anchors_retrieved": True,
        "required_image_recall": 1.0,
        "all_required_images_retrieved": True,
        "required_image_evidence_recall": 1.0,
        "required_image_support_recall": 1.0,
        "all_required_images_supported": True,
        "hard_gate_passed": True,
    }
    assert score.diagnostics["claims"]["method"] == "normalized_lexical_substring"
    assert score.diagnostics["claims"]["optional"]["missing"] == ["停机后操作"]
    assert score.diagnostics["evidence"]["matched_document_ids"] == [
        "manual-a",
        "manual-b",
    ]


@pytest.mark.parametrize(
    ("trace_changes", "failed_condition"),
    [
        ({"status": "skipped"}, "status_completed"),
        (
            {"trace": {"contexts_complete": True}},
            "terminal_event_complete",
        ),
        ({"observed_action": "clarify"}, "action_match"),
        (
            {"answer": "主轴每周润滑一次。"},
            "all_vital_claims_present",
        ),
        (
            {"answer": ("主轴每周润滑一次，使用2号润滑脂，也可以每天润滑一次。")},
            "no_forbidden_claims_present",
        ),
        (
            {"retrieved_evidence": [_retrieved("manual-a", 1, 12)]},
            "all_gold_documents_retrieved",
        ),
        ({"images": []}, "all_required_images_retrieved"),
    ],
)
def test_each_safety_condition_can_fail_gate(
    trace_changes: dict, failed_condition: str
) -> None:
    score = score_evaluation_case(_case_dict(), _trace_dict(**trace_changes))

    assert score.metrics[failed_condition] is False
    assert score.metrics["hard_gate_passed"] is False
    assert failed_condition in score.diagnostics["hard_gate"]["failed_conditions"]


def test_empty_expectation_groups_use_none_rates_and_vacuous_gate_checks() -> None:
    case = _case_dict(
        expected_action="abstain",
        vital_claims=[],
        optional_claims=[],
        forbidden_claims=[],
        evidence=[],
        expected_images=[
            {
                "image_id": "optional-only",
                "evidence": [_anchor("manual-a")],
                "required": False,
            }
        ],
    )
    trace = _trace_dict(
        observed_action="abstain",
        answer=None,
        retrieved_evidence=[],
        images=[],
    )

    score = score_evaluation_case(case, trace)

    for name in (
        "vital_claim_recall",
        "optional_claim_recall",
        "forbidden_claim_hit_rate",
        "gold_document_recall",
        "gold_anchor_recall",
        "required_image_recall",
        "required_image_evidence_recall",
        "required_image_support_recall",
    ):
        assert score.metrics[name] is None
    assert score.metrics["hard_gate_passed"] is True


def test_document_recall_uses_unique_exact_document_ids() -> None:
    case = _case_dict(
        evidence=[
            _anchor("manual-a", 1),
            _anchor("manual-a", 2),
            _anchor("Manual-B", 1),
        ]
    )
    trace = _trace_dict(
        retrieved_evidence=[
            _retrieved("manual-a", 1),
            _retrieved("manual-b", 2),
        ]
    )

    score = score_evaluation_case(case, trace)

    assert score.metrics["gold_document_recall"] == 0.5
    assert score.metrics["gold_anchor_recall"] == pytest.approx(1 / 3)
    assert score.diagnostics["evidence"]["missing_document_ids"] == ["Manual-B"]
    assert score.metrics["hard_gate_passed"] is False


def test_optional_images_do_not_affect_required_image_recall_or_gate() -> None:
    trace = _trace_dict(
        images=[
            {
                "image_id": "figure-required",
                "evidence": [_anchor("manual-a", 13)],
            }
        ]
    )

    score = score_evaluation_case(_case_dict(), trace)

    assert score.metrics["required_image_recall"] == 1.0
    assert score.metrics["all_required_images_retrieved"] is True
    assert score.metrics["all_required_images_supported"] is True
    assert score.diagnostics["images"]["optional_image_ids"] == ["figure-optional"]


def test_answer_requires_retrieved_evidence_even_without_gold_anchors() -> None:
    case = _case_dict(evidence=[], expected_images=[])
    trace = _trace_dict(retrieved_evidence=[], images=[])

    score = score_evaluation_case(case, trace)

    assert score.metrics["all_gold_anchors_retrieved"] is True
    assert score.metrics["answer_evidence_present"] is False
    assert score.metrics["answer_has_gold_evidence"] is False
    assert score.metrics["hard_gate_passed"] is False
    assert (
        "answer_evidence_present" in score.diagnostics["hard_gate"]["failed_conditions"]
    )


def test_answer_requires_typed_gold_evidence() -> None:
    case = _case_dict(evidence=[], expected_images=[])

    score = score_evaluation_case(case, _trace_dict(images=[]))

    assert score.metrics["answer_evidence_present"] is True
    assert score.metrics["answer_has_gold_evidence"] is False
    assert score.metrics["hard_gate_passed"] is False
    assert (
        "answer_has_gold_evidence"
        in score.diagnostics["hard_gate"]["failed_conditions"]
    )


def test_answer_requires_complete_model_context_capture() -> None:
    trace = _trace_dict(
        trace={
            "contexts_complete": False,
            "terminal_event": {"type": "done"},
        }
    )

    score = score_evaluation_case(_case_dict(), trace)

    assert score.metrics["answer_evidence_present"] is True
    assert score.metrics["context_capture_complete"] is False
    assert score.metrics["hard_gate_passed"] is False
    assert (
        "context_capture_complete"
        in score.diagnostics["hard_gate"]["failed_conditions"]
    )


def test_answer_requires_a_correctness_bearing_vital_claim() -> None:
    case = _case_dict(
        reference_answer="主轴每周润滑一次。",
        vital_claims=[],
        optional_claims=[],
        forbidden_claims=[],
        expected_images=[],
    )
    trace = _trace_dict(answer="完全错误的答案", images=[])

    score = score_evaluation_case(case, trace)

    assert score.metrics["answer_evidence_present"] is True
    assert score.metrics["answer_has_graded_content"] is False
    assert score.metrics["hard_gate_passed"] is False
    assert (
        "answer_has_graded_content"
        in score.diagnostics["hard_gate"]["failed_conditions"]
    )


def _fully_located_anchor(**overrides) -> dict:
    anchor = {
        "doc_id": "manual-exact",
        "hash": "sha256:exact",
        "page": 7,
        "span": {"start": 10, "end": 20, "quote": "supporting text"},
        "bbox": {
            "x0": 0.1,
            "y0": 0.2,
            "x1": 0.8,
            "y1": 0.9,
            "coordinate_space": "normalized",
        },
        "modality": "table",
        "chunk_id": "chunk-007",
    }
    anchor.update(overrides)
    return anchor


@pytest.mark.parametrize(
    "anchor_change",
    [
        {"doc_id": "other-manual"},
        {"hash": "sha256:other"},
        {"page": 8},
        {"span": {"start": 10, "end": 21, "quote": "supporting text"}},
        {
            "bbox": {
                "x0": 0.1,
                "y0": 0.2,
                "x1": 0.7,
                "y1": 0.9,
                "coordinate_space": "normalized",
            }
        },
        {"modality": "text"},
        {"chunk_id": "chunk-008"},
    ],
)
def test_gold_anchor_matching_requires_every_specified_locator(
    anchor_change: dict,
) -> None:
    case = _case_dict(evidence=[_fully_located_anchor()], expected_images=[])
    returned_anchor = _fully_located_anchor(**anchor_change)
    trace = _trace_dict(
        retrieved_evidence=[{"anchor": returned_anchor, "rank": 1, "score": 0.9}],
        images=[],
    )

    score = score_evaluation_case(case, trace)

    assert score.metrics["gold_anchor_recall"] == 0.0
    assert score.metrics["all_gold_anchors_retrieved"] is False
    assert score.diagnostics["evidence"]["anchor_matches"][0]["matched"] is False
    assert score.metrics["hard_gate_passed"] is False


def test_gold_anchor_matching_allows_additional_runtime_span_detail() -> None:
    gold_anchor = _fully_located_anchor(span={"quote": "supporting text"})
    returned_anchor = _fully_located_anchor()
    case = _case_dict(evidence=[gold_anchor], expected_images=[])
    trace = _trace_dict(
        retrieved_evidence=[{"anchor": returned_anchor, "rank": 1}], images=[]
    )

    score = score_evaluation_case(case, trace)

    assert score.metrics["gold_anchor_recall"] == 1.0
    assert score.metrics["all_gold_anchors_retrieved"] is True


def test_required_image_id_without_gold_evidence_fails_support_gate() -> None:
    trace = _trace_dict(images=[{"image_id": "figure-required"}])

    score = score_evaluation_case(_case_dict(), trace)

    assert score.metrics["required_image_recall"] == 1.0
    assert score.metrics["all_required_images_retrieved"] is True
    assert score.metrics["required_image_evidence_recall"] == 0.0
    assert score.metrics["required_image_support_recall"] == 0.0
    assert score.metrics["all_required_images_supported"] is False
    assert score.metrics["hard_gate_passed"] is False


def test_required_image_must_have_all_anchors_on_one_trace_image() -> None:
    case = _case_dict(
        expected_images=[
            {
                "image_id": "figure-required",
                "evidence": [
                    _anchor("manual-a", 13),
                    _anchor("manual-b", 4),
                ],
                "required": True,
            }
        ]
    )
    trace = _trace_dict(
        images=[
            {
                "image_id": "figure-required",
                "evidence": [_anchor("manual-a", 13)],
            },
            {
                "image_id": "figure-required",
                "evidence": [_anchor("manual-b", 4)],
            },
        ]
    )

    score = score_evaluation_case(case, trace)

    assert score.metrics["required_image_evidence_recall"] == 1.0
    assert score.metrics["required_image_support_recall"] == 0.0
    assert score.metrics["all_required_images_supported"] is False
    image_diagnostics = score.diagnostics["images"]["required_image_matches"][0]
    assert image_diagnostics["fully_supporting_returned_indexes"] == []
    assert score.metrics["hard_gate_passed"] is False


def test_punctuation_only_claim_never_matches_empty_normalized_text() -> None:
    case = _case_dict(vital_claims=["……"], optional_claims=[])

    score = score_evaluation_case(case, _trace_dict(answer="任意答案"))

    assert score.metrics["vital_claim_recall"] == 0.0
    assert score.metrics["hard_gate_passed"] is False
    assert score.diagnostics["claims"]["vital"]["items"][0] == {
        "claim": "……",
        "normalized": "",
        "matched": False,
    }


def test_metrics_are_trace_schema_compatible_and_diagnostics_are_json() -> None:
    score = score_evaluation_case(_case_dict(), _trace_dict())
    trace_payload = _trace_dict(metrics=dict(score.metrics))

    parsed_trace = EvaluationTraceResult.from_dict(trace_payload)

    assert parsed_trace.metrics == score.metrics
    assert json.loads(json.dumps(score.diagnostics, allow_nan=False))["case_id"] == (
        "case-001"
    )
    assert isinstance(score, DeterministicEvaluationScore)


def test_case_and_trace_instances_are_accepted() -> None:
    case = EvaluationCase.from_dict(_case_dict())
    trace = EvaluationTraceResult.from_dict(_trace_dict())

    assert score_evaluation_case(case, trace).metrics["hard_gate_passed"] is True


def test_mismatched_case_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="mismatched case IDs"):
        score_evaluation_case(
            _case_dict(),
            _trace_dict(case_id="different-case"),
        )


def test_score_object_rejects_non_schema_metric_values() -> None:
    with pytest.raises(TypeError, match="finite float, bool, or None"):
        DeterministicEvaluationScore(metrics={"count": 1}, diagnostics={})
