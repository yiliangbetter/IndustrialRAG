"""Load canonical evaluation fixtures and reconstruct legacy case shapes.

The canonical fixtures use :mod:`raganything.evaluation_schema`, while older
benchmark scripts consume several unrelated dictionaries.  This module keeps
that compatibility boundary explicit: migrations store the original grader
payloads in case metadata, and these adapters return detached legacy values.

Legacy runners import these adapters at their fixture-loading boundary. Parity
tests prove that moving their defaults to the canonical files does not change
grader semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .evaluation_schema import (
    EvaluationCase,
    EvaluationDataset,
    load_evaluation_dataset,
)

_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_FIXTURE_DIR = _ROOT / "tests" / "fixtures" / "evaluation"

_DATASET_PATHS = {
    "shili17": CANONICAL_FIXTURE_DIR / "shili17.json",
    "nanxing-latency": CANONICAL_FIXTURE_DIR / "nanxing_latency.json",
    "voice29": CANONICAL_FIXTURE_DIR / "voice29.json",
}
_DATASET_ALIASES = {
    "nanxing": "nanxing-latency",
    "nanxing_latency": "nanxing-latency",
    "voice": "voice29",
}


class LegacyFixtureError(ValueError):
    """Raised when canonical metadata cannot reproduce a legacy fixture."""


def canonical_fixture_path(dataset_id: str) -> Path:
    """Return the tracked canonical fixture path for a dataset identifier."""

    normalized = _DATASET_ALIASES.get(dataset_id, dataset_id)
    try:
        return _DATASET_PATHS[normalized]
    except KeyError:
        choices = ", ".join(sorted(_DATASET_PATHS))
        raise LegacyFixtureError(
            f"unknown canonical evaluation dataset {dataset_id!r}; choose {choices}"
        ) from None


def load_canonical_fixture(dataset_id: str) -> EvaluationDataset:
    """Load and validate one of the tracked canonical evaluation datasets."""

    path = canonical_fixture_path(dataset_id)
    dataset = load_evaluation_dataset(path)
    expected_id = _DATASET_ALIASES.get(dataset_id, dataset_id)
    if dataset.dataset_id != expected_id:
        raise LegacyFixtureError(
            f"{path}: dataset_id {dataset.dataset_id!r} does not match "
            f"requested {expected_id!r}"
        )
    return dataset


def _expect_dataset(dataset: EvaluationDataset, dataset_id: str) -> None:
    if dataset.dataset_id != dataset_id:
        raise LegacyFixtureError(
            f"expected dataset_id {dataset_id!r}, got {dataset.dataset_id!r}"
        )


def _legacy_metadata(case: EvaluationCase, suite: str) -> Mapping[str, Any]:
    legacy = case.metadata.get("legacy")
    if not isinstance(legacy, Mapping):
        raise LegacyFixtureError(f"{case.case_id}: missing metadata.legacy object")
    if legacy.get("suite") != suite:
        raise LegacyFixtureError(
            f"{case.case_id}: legacy suite {legacy.get('suite')!r} is not {suite!r}"
        )
    return legacy


def _mapping(value: Any, *, case_id: str, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LegacyFixtureError(f"{case_id}: {field} must be an object")
    return value


def shili17_legacy_cases(
    dataset: EvaluationDataset | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct ``test_cases_shili17.json`` case rows exactly."""

    dataset = dataset or load_canonical_fixture("shili17")
    _expect_dataset(dataset, "shili17")
    rows: list[dict[str, Any]] = []
    for case in dataset.cases:
        legacy = _legacy_metadata(case, "shili17")
        payloads = _mapping(
            legacy.get("grader_payloads"),
            case_id=case.case_id,
            field="metadata.legacy.grader_payloads",
        )
        text = _mapping(
            payloads.get("text"),
            case_id=case.case_id,
            field="metadata.legacy.grader_payloads.text",
        )
        image = _mapping(
            payloads.get("image"),
            case_id=case.case_id,
            field="metadata.legacy.grader_payloads.image",
        )
        rows.append(
            {
                "id": deepcopy(legacy.get("case_id")),
                "query": case.query,
                "text": deepcopy(dict(text)),
                "image": deepcopy(dict(image)),
            }
        )
    return rows


def shili17_legacy_ref(
    dataset: EvaluationDataset | None = None,
) -> dict[int, dict[str, Any]]:
    """Reconstruct the flattened ``REF`` mapping used by the Web-path grader."""

    result: dict[int, dict[str, Any]] = {}
    for row in shili17_legacy_cases(dataset):
        case_id = row["id"]
        if isinstance(case_id, bool) or not isinstance(case_id, int):
            raise LegacyFixtureError(f"invalid Shili17 legacy id: {case_id!r}")
        spec: dict[str, Any] = {"query": row["query"]}
        spec.update(row["text"])
        spec.update(row["image"])
        result[case_id] = spec
    return result


def ragas_shili17_cases(
    dataset: EvaluationDataset | None = None,
) -> list[dict[str, Any]]:
    """Emit the legacy Ragas ground-truth view from canonical Shili17 cases."""

    dataset = dataset or load_canonical_fixture("shili17")
    _expect_dataset(dataset, "shili17")
    rows: list[dict[str, Any]] = []
    for case in dataset.cases:
        legacy = _legacy_metadata(case, "shili17")
        case_id = legacy.get("case_id")
        if isinstance(case_id, bool) or not isinstance(case_id, int):
            raise LegacyFixtureError(f"invalid Shili17 legacy id: {case_id!r}")
        if not case.reference_answer:
            raise LegacyFixtureError(f"{case.case_id}: missing reference_answer")
        rows.append(
            {
                "id": case_id,
                "question": case.query,
                "ground_truth": case.reference_answer,
            }
        )
    return rows


def nanxing_latency_legacy_cases(
    dataset: EvaluationDataset | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct ``nanxing_latency_cases.json`` case rows exactly."""

    dataset = dataset or load_canonical_fixture("nanxing-latency")
    _expect_dataset(dataset, "nanxing-latency")
    rows: list[dict[str, Any]] = []
    for case in dataset.cases:
        legacy = _legacy_metadata(case, "nanxing-latency")
        payloads = _mapping(
            legacy.get("grader_payloads"),
            case_id=case.case_id,
            field="metadata.legacy.grader_payloads",
        )
        answer = _mapping(
            payloads.get("answer"),
            case_id=case.case_id,
            field="metadata.legacy.grader_payloads.answer",
        )
        source = payloads.get("source")
        if not isinstance(source, str) or not source.strip():
            raise LegacyFixtureError(
                f"{case.case_id}: metadata.legacy.grader_payloads.source "
                "must be a non-empty string"
            )
        row: dict[str, Any] = {
            "id": deepcopy(legacy.get("case_id")),
            "question": case.query,
            "source": source,
        }
        row.update(deepcopy(dict(answer)))
        rows.append(row)
    return rows


def voice29_legacy_cases(
    dataset: EvaluationDataset | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct all recoverable Voice29 rows without Git or the source XLSX."""

    dataset = dataset or load_canonical_fixture("voice29")
    _expect_dataset(dataset, "voice29")
    rows: list[dict[str, Any]] = []
    for case in dataset.cases:
        legacy = _legacy_metadata(case, "voice29")
        fields = _mapping(
            legacy.get("case_fields"),
            case_id=case.case_id,
            field="metadata.legacy.case_fields",
        )
        payloads = _mapping(
            legacy.get("grader_payloads"),
            case_id=case.case_id,
            field="metadata.legacy.grader_payloads",
        )
        response = _mapping(
            payloads.get("response"),
            case_id=case.case_id,
            field="metadata.legacy.grader_payloads.response",
        )
        row: dict[str, Any] = {
            "id": deepcopy(legacy.get("case_id")),
            **deepcopy(dict(fields)),
            "standard_question": case.query,
            "grade": deepcopy(dict(response)),
        }
        rows.append(row)
    return rows


__all__ = [
    "CANONICAL_FIXTURE_DIR",
    "LegacyFixtureError",
    "canonical_fixture_path",
    "load_canonical_fixture",
    "nanxing_latency_legacy_cases",
    "ragas_shili17_cases",
    "shili17_legacy_cases",
    "shili17_legacy_ref",
    "voice29_legacy_cases",
]
