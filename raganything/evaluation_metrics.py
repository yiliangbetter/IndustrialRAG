"""Dependency-free deterministic metrics for versioned RAG evaluations.

Claim metrics in this module are intentionally lexical diagnostics.  They do
not measure entailment or semantic equivalence: a claim matches only when its
Unicode-normalized text is a substring of the normalized answer.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from raganything.evaluation_schema import (
    EvaluationCase,
    EvaluationTraceResult,
    EvidenceAnchor,
    ExpectedAction,
    TraceStatus,
)

DETERMINISTIC_SCORER_VERSION = "rag-deterministic-scorer/v2"

_INTRA_TOKEN_PUNCTUATION = frozenset("._-/\\'’")
_DASH_PUNCTUATION = frozenset("-‐‑‒–—―")

MetricValue = float | bool | None


def normalize_lexical_text(text: str) -> str:
    """Normalize text for deterministic lexical containment checks.

    The normalization applies Unicode NFKC, Unicode-aware case folding, and
    removes whitespace, separators, sentence punctuation, and invisible format
    characters.  Technically meaningful punctuation is retained when it is
    inside an alphanumeric token, so ``0.6``, ``5-10``, and ``AB-12`` cannot
    collapse to ``06``, ``510``, and ``AB12``.  Symbols are retained so
    safety-relevant tokens such as ``%`` and ``≤`` are not silently discarded.
    """

    normalized = unicodedata.normalize("NFKC", text).casefold()
    compact = "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("Z")
        and unicodedata.category(character) != "Cf"
    )

    def is_token_character(character: str) -> bool:
        return unicodedata.category(character)[0] in {"L", "M", "N"}

    result: list[str] = []
    for index, character in enumerate(compact):
        if not unicodedata.category(character).startswith("P"):
            result.append(character)
            continue
        if (
            character in _INTRA_TOKEN_PUNCTUATION | _DASH_PUNCTUATION
            and 0 < index < len(compact) - 1
            and is_token_character(compact[index - 1])
            and is_token_character(compact[index + 1])
        ):
            if character in _DASH_PUNCTUATION:
                result.append("-")
            elif character == "’":
                result.append("'")
            else:
                result.append(character)
    return "".join(result)


def _json_copy(value: Any) -> Any:
    """Return detached strict JSON data or raise for invalid diagnostics."""

    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


@dataclass(frozen=True, slots=True)
class DeterministicEvaluationScore:
    """Schema-compatible metrics and separate JSON diagnostics."""

    metrics: Mapping[str, MetricValue]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        metrics: dict[str, MetricValue] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be non-empty strings")
            if (
                value is None
                or isinstance(value, bool)
                or (isinstance(value, float) and math.isfinite(value))
            ):
                metrics[name] = value
            else:
                raise TypeError(
                    f"metric {name!r} must be a finite float, bool, or None"
                )
        diagnostics = _json_copy(dict(self.diagnostics))
        if not isinstance(
            diagnostics, dict
        ):  # Defensive; Mapping always copies to dict.
            raise TypeError("diagnostics must be a JSON object")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "diagnostics", diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "diagnostics": _json_copy(dict(self.diagnostics)),
        }


def _claim_diagnostics(
    claims: Sequence[str], normalized_answer: str
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    matched: list[str] = []
    missing: list[str] = []
    for claim in claims:
        normalized_claim = normalize_lexical_text(claim)
        is_match = bool(normalized_claim) and normalized_claim in normalized_answer
        rows.append(
            {
                "claim": claim,
                "normalized": normalized_claim,
                "matched": is_match,
            }
        )
        (matched if is_match else missing).append(claim)
    return rows, matched, missing


def _recall(found: int, total: int) -> float | None:
    if total == 0:
        return None
    return float(found / total)


def _anchor_matches(gold: EvidenceAnchor, returned: EvidenceAnchor) -> bool:
    """Match every locator specified by a gold evidence anchor.

    Runtime anchors may contain additional locator detail.  Within ``span``,
    only the gold subfields that are present are required; all bounding-box
    fields are always present and therefore compare as one exact locator.
    """

    if gold.doc_id != returned.doc_id or gold.modality != returned.modality:
        return False
    for attribute in ("hash", "page", "chunk_id"):
        gold_value = getattr(gold, attribute)
        if gold_value is not None and gold_value != getattr(returned, attribute):
            return False
    if gold.span is not None:
        if returned.span is None:
            return False
        for attribute in ("start", "end", "quote"):
            gold_value = getattr(gold.span, attribute)
            if gold_value is not None and gold_value != getattr(
                returned.span, attribute
            ):
                return False
    return gold.bbox is None or gold.bbox == returned.bbox


def _anchor_match_rows(
    gold_anchors: Sequence[EvidenceAnchor],
    returned_anchors: Sequence[EvidenceAnchor],
) -> tuple[list[dict[str, Any]], list[EvidenceAnchor], list[EvidenceAnchor]]:
    rows: list[dict[str, Any]] = []
    matched: list[EvidenceAnchor] = []
    missing: list[EvidenceAnchor] = []
    for gold_index, gold_anchor in enumerate(gold_anchors):
        returned_indexes = [
            returned_index
            for returned_index, returned_anchor in enumerate(returned_anchors)
            if _anchor_matches(gold_anchor, returned_anchor)
        ]
        is_match = bool(returned_indexes)
        rows.append(
            {
                "gold_index": gold_index,
                "gold_anchor": gold_anchor.to_dict(),
                "matched": is_match,
                "returned_indexes": returned_indexes,
            }
        )
        (matched if is_match else missing).append(gold_anchor)
    return rows, matched, missing


def score_evaluation_case(
    case: EvaluationCase | Mapping[str, Any],
    trace: EvaluationTraceResult | Mapping[str, Any],
) -> DeterministicEvaluationScore:
    """Score one expected case against one observed trace.

    The hard gate passes only when the trace completed, its action matches, all
    vital claims are lexically present, no forbidden claim is lexically
    present, every gold evidence anchor is exactly supported, and every
    required image ID is returned with all of its gold evidence.  An expected
    answer also requires a complete model-context capture, at least one typed
    gold evidence anchor, at least one retrieved evidence item, and at least
    one vital claim that can grade its correctness. Required images must also
    carry typed gold evidence rather than relying on image IDs alone.
    """

    expected = EvaluationCase.from_dict(case, "case")
    observed = EvaluationTraceResult.from_dict(trace, "trace")
    if expected.case_id != observed.case_id:
        raise ValueError(
            "cannot score mismatched case IDs: "
            f"expected {expected.case_id!r}, observed {observed.case_id!r}"
        )

    normalized_answer = normalize_lexical_text(observed.answer or "")
    vital_rows, vital_matched, vital_missing = _claim_diagnostics(
        expected.vital_claims, normalized_answer
    )
    optional_rows, optional_matched, optional_missing = _claim_diagnostics(
        expected.optional_claims, normalized_answer
    )
    forbidden_rows, forbidden_matched, forbidden_missing = _claim_diagnostics(
        expected.forbidden_claims, normalized_answer
    )

    gold_document_ids = sorted({anchor.doc_id for anchor in expected.evidence})
    retrieved_document_ids = sorted(
        {item.anchor.doc_id for item in observed.retrieved_evidence}
    )
    retrieved_document_set = set(retrieved_document_ids)
    matched_document_ids = [
        doc_id for doc_id in gold_document_ids if doc_id in retrieved_document_set
    ]
    missing_document_ids = [
        doc_id for doc_id in gold_document_ids if doc_id not in retrieved_document_set
    ]

    retrieved_anchors = [item.anchor for item in observed.retrieved_evidence]
    gold_anchor_rows, matched_gold_anchors, missing_gold_anchors = _anchor_match_rows(
        expected.evidence, retrieved_anchors
    )

    required_image_ids = sorted(
        image.image_id for image in expected.expected_images if image.required
    )
    optional_image_ids = sorted(
        image.image_id for image in expected.expected_images if not image.required
    )
    returned_image_ids = sorted({image.image_id for image in observed.images})
    returned_image_set = set(returned_image_ids)
    matched_required_image_ids = [
        image_id for image_id in required_image_ids if image_id in returned_image_set
    ]
    missing_required_image_ids = [
        image_id
        for image_id in required_image_ids
        if image_id not in returned_image_set
    ]

    required_image_rows: list[dict[str, Any]] = []
    matched_required_image_evidence_count = 0
    required_image_evidence_count = 0
    supported_required_image_ids: list[str] = []
    unsupported_required_image_ids: list[str] = []
    for expected_image in (
        image for image in expected.expected_images if image.required
    ):
        candidate_rows: list[dict[str, Any]] = []
        candidate_images = [
            (index, image)
            for index, image in enumerate(observed.images)
            if image.image_id == expected_image.image_id
        ]
        union_matched_gold_indexes: set[int] = set()
        fully_supporting_returned_indexes: list[int] = []
        for returned_index, returned_image in candidate_images:
            anchor_rows, matched_anchors, missing_anchors = _anchor_match_rows(
                expected_image.evidence, returned_image.evidence
            )
            union_matched_gold_indexes.update(
                row["gold_index"] for row in anchor_rows if row["matched"]
            )
            if not missing_anchors:
                fully_supporting_returned_indexes.append(returned_index)
            candidate_rows.append(
                {
                    "returned_index": returned_index,
                    "anchor_matches": anchor_rows,
                    "matched_anchor_count": len(matched_anchors),
                    "missing_anchors": [anchor.to_dict() for anchor in missing_anchors],
                }
            )

        is_supported = bool(fully_supporting_returned_indexes)
        (
            supported_required_image_ids
            if is_supported
            else unsupported_required_image_ids
        ).append(expected_image.image_id)
        required_image_evidence_count += len(expected_image.evidence)
        matched_required_image_evidence_count += len(union_matched_gold_indexes)
        required_image_rows.append(
            {
                "image_id": expected_image.image_id,
                "returned_indexes": [index for index, _ in candidate_images],
                "fully_supporting_returned_indexes": (
                    fully_supporting_returned_indexes
                ),
                "supported": is_supported,
                "matched_gold_anchor_indexes": sorted(union_matched_gold_indexes),
                "candidate_matches": candidate_rows,
            }
        )

    status_completed = observed.status is TraceStatus.COMPLETED
    terminal_event = observed.trace.get("terminal_event")
    terminal_event_complete = (
        isinstance(terminal_event, Mapping) and terminal_event.get("type") == "done"
    )
    action_match = observed.observed_action == expected.expected_action
    all_vital_claims_present = not vital_missing
    no_forbidden_claims_present = not forbidden_matched
    all_gold_documents_retrieved = not missing_document_ids
    all_gold_anchors_retrieved = not missing_gold_anchors
    all_required_images_retrieved = not missing_required_image_ids
    all_required_images_supported = not unsupported_required_image_ids
    answer_evidence_present = (
        expected.expected_action is not ExpectedAction.ANSWER
        or bool(observed.retrieved_evidence)
    )
    answer_has_gold_evidence = (
        expected.expected_action is not ExpectedAction.ANSWER or bool(expected.evidence)
    )
    context_capture_complete = (
        expected.expected_action is not ExpectedAction.ANSWER
        or observed.trace.get("contexts_complete") is True
    )
    answer_has_graded_content = (
        expected.expected_action is not ExpectedAction.ANSWER
        or bool(expected.vital_claims)
    )
    gate_conditions = {
        "status_completed": status_completed,
        "terminal_event_complete": terminal_event_complete,
        "action_match": action_match,
        "all_vital_claims_present": all_vital_claims_present,
        "no_forbidden_claims_present": no_forbidden_claims_present,
        "answer_evidence_present": answer_evidence_present,
        "answer_has_gold_evidence": answer_has_gold_evidence,
        "context_capture_complete": context_capture_complete,
        "answer_has_graded_content": answer_has_graded_content,
        "all_required_images_retrieved": all_required_images_retrieved,
        "all_required_images_supported": all_required_images_supported,
        "all_gold_documents_retrieved": all_gold_documents_retrieved,
        "all_gold_anchors_retrieved": all_gold_anchors_retrieved,
    }
    hard_gate_passed = all(gate_conditions.values())

    metrics: dict[str, MetricValue] = {
        "status_completed": status_completed,
        "terminal_event_complete": terminal_event_complete,
        "action_match": action_match,
        "vital_claim_recall": _recall(len(vital_matched), len(expected.vital_claims)),
        "all_vital_claims_present": all_vital_claims_present,
        "optional_claim_recall": _recall(
            len(optional_matched), len(expected.optional_claims)
        ),
        "forbidden_claim_hit_rate": _recall(
            len(forbidden_matched), len(expected.forbidden_claims)
        ),
        "no_forbidden_claims_present": no_forbidden_claims_present,
        "answer_evidence_present": answer_evidence_present,
        "answer_has_gold_evidence": answer_has_gold_evidence,
        "context_capture_complete": context_capture_complete,
        "answer_has_graded_content": answer_has_graded_content,
        "gold_document_recall": _recall(
            len(matched_document_ids), len(gold_document_ids)
        ),
        "all_gold_documents_retrieved": all_gold_documents_retrieved,
        "gold_anchor_recall": _recall(
            len(matched_gold_anchors), len(expected.evidence)
        ),
        "all_gold_anchors_retrieved": all_gold_anchors_retrieved,
        "required_image_recall": _recall(
            len(matched_required_image_ids), len(required_image_ids)
        ),
        "all_required_images_retrieved": all_required_images_retrieved,
        "required_image_evidence_recall": _recall(
            matched_required_image_evidence_count,
            required_image_evidence_count,
        ),
        "required_image_support_recall": _recall(
            len(supported_required_image_ids), len(required_image_ids)
        ),
        "all_required_images_supported": all_required_images_supported,
        "hard_gate_passed": hard_gate_passed,
    }
    diagnostics = {
        "scorer_version": DETERMINISTIC_SCORER_VERSION,
        "case_id": expected.case_id,
        "action": {
            "expected": expected.expected_action.value,
            "observed": (
                None
                if observed.observed_action is None
                else observed.observed_action.value
            ),
            "matched": action_match,
        },
        "claims": {
            "method": "normalized_lexical_substring",
            "normalization": (
                "NFKC + casefold + remove Unicode whitespace, separators, "
                "format characters, and sentence punctuation while preserving "
                "meaningful intra-token punctuation"
            ),
            "normalized_answer": normalized_answer,
            "vital": {
                "items": vital_rows,
                "matched": vital_matched,
                "missing": vital_missing,
            },
            "optional": {
                "items": optional_rows,
                "matched": optional_matched,
                "missing": optional_missing,
            },
            "forbidden": {
                "items": forbidden_rows,
                "matched": forbidden_matched,
                "not_matched": forbidden_missing,
            },
        },
        "evidence": {
            "matching_key": "doc_id",
            "gold_document_ids": gold_document_ids,
            "retrieved_document_ids": retrieved_document_ids,
            "matched_document_ids": matched_document_ids,
            "missing_document_ids": missing_document_ids,
            "anchor_matching_rule": (
                "doc_id and modality plus every non-null gold locator must "
                "match exactly"
            ),
            "gold_anchors": [anchor.to_dict() for anchor in expected.evidence],
            "retrieved_anchors": [anchor.to_dict() for anchor in retrieved_anchors],
            "anchor_matches": gold_anchor_rows,
            "matched_gold_anchors": [
                anchor.to_dict() for anchor in matched_gold_anchors
            ],
            "missing_gold_anchors": [
                anchor.to_dict() for anchor in missing_gold_anchors
            ],
        },
        "images": {
            "matching_key": "image_id",
            "required_image_ids": required_image_ids,
            "optional_image_ids": optional_image_ids,
            "returned_image_ids": returned_image_ids,
            "matched_required_image_ids": matched_required_image_ids,
            "missing_required_image_ids": missing_required_image_ids,
            "evidence_matching_rule": (
                "one returned image with the matching image_id must support "
                "every gold evidence anchor"
            ),
            "required_image_matches": required_image_rows,
            "supported_required_image_ids": supported_required_image_ids,
            "unsupported_required_image_ids": unsupported_required_image_ids,
        },
        "hard_gate": {
            "conditions": gate_conditions,
            "failed_conditions": [
                name for name, passed in gate_conditions.items() if not passed
            ],
            "passed": hard_gate_passed,
        },
    }
    return DeterministicEvaluationScore(metrics=metrics, diagnostics=diagnostics)


__all__ = [
    "DETERMINISTIC_SCORER_VERSION",
    "DeterministicEvaluationScore",
    "MetricValue",
    "normalize_lexical_text",
    "score_evaluation_case",
]
