"""Versioned, dependency-free schemas for offline RAG evaluation.

The module deliberately uses only the Python standard library so evaluation
fixtures can be validated in lightweight CI jobs.  JSON is the canonical
envelope format.  The canonical JSONL representation starts with a manifest
record and is followed by one case or trace record per line.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

CASE_SCHEMA_VERSION = "rag-evaluation-case/v1"
DATASET_SCHEMA_VERSION = "rag-evaluation-dataset/v1"
TRACE_SCHEMA_VERSION = "rag-evaluation-trace/v1"
RUN_SCHEMA_VERSION = "rag-evaluation-run/v1"


class SchemaValidationError(ValueError):
    """Raised when an evaluation document does not match its schema."""


class ExpectedAction(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConversationRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TraceStatus(str, Enum):
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


def _fail(path: str, message: str) -> SchemaValidationError:
    return SchemaValidationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(path, "expected an object")
    for key in value:
        if not isinstance(key, str):
            raise _fail(path, "object keys must be strings")
    return value


def _fields(
    value: Any,
    *,
    path: str,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> Mapping[str, Any]:
    data = _mapping(value, path)
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(data))
    if missing:
        raise _fail(path, f"missing required field(s): {', '.join(missing)}")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise _fail(path, f"unknown field(s): {', '.join(unknown)}")
    return data


def _string(value: Any, path: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _fail(path, "expected a non-empty string")
    return value.strip()


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fail(path, "expected an array of strings")
    result = tuple(
        _string(item, f"{path}[{index}]") for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise _fail(path, "entries must be unique")
    return result  # type: ignore[return-value]


def _integer(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    optional: bool = False,
) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(path, "expected an integer")
    if minimum is not None and value < minimum:
        raise _fail(path, f"must be at least {minimum}")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    optional: bool = False,
) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(path, "expected a number")
    result = float(value)
    if not math.isfinite(result):
        raise _fail(path, "must be finite")
    if minimum is not None and result < minimum:
        raise _fail(path, f"must be at least {minimum}")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(path, "expected a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], path: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        choices = ", ".join(repr(item.value) for item in enum_type)
        raise _fail(path, f"expected one of: {choices}") from None


def _timestamp(value: Any, path: str, *, optional: bool = False) -> str | None:
    result = _string(value, path, optional=optional)
    if result is None:
        return None
    iso_value = result[:-1] + "+00:00" if result.endswith("Z") else result
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError:
        raise _fail(path, "expected an ISO-8601 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _fail(path, "timestamp must include a timezone offset")
    return result


def _parsed_timestamp(value: str) -> datetime:
    iso_value = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(iso_value)


def _json_value(value: Any, path: str) -> Any:
    """Validate and detach arbitrary metadata as strict JSON data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _fail(path, "JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise _fail(path, "object keys must be strings")
            result[key] = _json_value(child, f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _json_value(child, f"{path}[{index}]") for index, child in enumerate(value)
        ]
    raise _fail(path, f"value of type {type(value).__name__} is not JSON-compatible")


def _json_object(value: Any, path: str) -> dict[str, Any]:
    result = _json_value(value, path)
    if not isinstance(result, dict):
        raise _fail(path, "expected an object")
    return result


@dataclass(frozen=True, slots=True)
class TextSpan:
    """A document character range and/or exact supporting quote."""

    start: int | None = None
    end: int | None = None
    quote: str | None = None

    def __post_init__(self) -> None:
        start = _integer(self.start, "span.start", minimum=0, optional=True)
        end = _integer(self.end, "span.end", minimum=1, optional=True)
        quote = _string(self.quote, "span.quote", optional=True)
        if (start is None) != (end is None):
            raise _fail("span", "start and end must be provided together")
        if start is not None and end is not None and end <= start:
            raise _fail("span.end", "must be greater than start")
        if start is None and quote is None:
            raise _fail("span", "provide offsets, a quote, or both")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "quote", quote)

    @classmethod
    def from_value(cls, value: Any, path: str = "span") -> TextSpan:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(quote=_string(value, path))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 2:
                raise _fail(path, "array form must be [start, end]")
            return cls(
                start=_integer(value[0], f"{path}[0]", minimum=0),
                end=_integer(value[1], f"{path}[1]", minimum=1),
            )
        data = _fields(
            value,
            path=path,
            required=(),
            optional=("start", "end", "quote"),
        )
        return cls(
            start=_integer(
                data.get("start"), f"{path}.start", minimum=0, optional=True
            ),
            end=_integer(data.get("end"), f"{path}.end", minimum=1, optional=True),
            quote=_string(data.get("quote"), f"{path}.quote", optional=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "quote": self.quote}


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An axis-aligned region in page or normalized coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float
    coordinate_space: str = "page"

    def __post_init__(self) -> None:
        x0 = _number(self.x0, "bbox.x0")
        y0 = _number(self.y0, "bbox.y0")
        x1 = _number(self.x1, "bbox.x1")
        y1 = _number(self.y1, "bbox.y1")
        coordinate_space = _string(self.coordinate_space, "bbox.coordinate_space")
        assert x0 is not None and y0 is not None and x1 is not None and y1 is not None
        assert coordinate_space is not None
        if x1 <= x0 or y1 <= y0:
            raise _fail("bbox", "x1/y1 must be greater than x0/y0")
        if coordinate_space == "normalized" and not all(
            0.0 <= coordinate <= 1.0 for coordinate in (x0, y0, x1, y1)
        ):
            raise _fail("bbox", "normalized coordinates must be between 0 and 1")
        object.__setattr__(self, "x0", x0)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "y1", y1)
        object.__setattr__(self, "coordinate_space", coordinate_space)

    @classmethod
    def from_value(cls, value: Any, path: str = "bbox") -> BoundingBox:
        if isinstance(value, cls):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 4:
                raise _fail(path, "array form must be [x0, y0, x1, y1]")
            return cls(*value)
        data = _fields(
            value,
            path=path,
            required=("x0", "y0", "x1", "y1"),
            optional=("coordinate_space",),
        )
        return cls(
            x0=data["x0"],
            y0=data["y0"],
            x1=data["x1"],
            y1=data["y1"],
            coordinate_space=data.get("coordinate_space", "page"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "coordinate_space": self.coordinate_space,
        }


@dataclass(frozen=True, slots=True)
class EvidenceAnchor:
    """Stable evidence identity plus an optional runtime chunk identifier."""

    doc_id: str
    hash: str | None = None
    page: int | None = None
    span: TextSpan | None = None
    bbox: BoundingBox | None = None
    modality: str = "text"
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        doc_id = _string(self.doc_id, "evidence.doc_id")
        digest = _string(self.hash, "evidence.hash", optional=True)
        page = _integer(self.page, "evidence.page", minimum=1, optional=True)
        span = (
            None
            if self.span is None
            else TextSpan.from_value(self.span, "evidence.span")
        )
        bbox = (
            None
            if self.bbox is None
            else BoundingBox.from_value(self.bbox, "evidence.bbox")
        )
        modality = _string(self.modality, "evidence.modality")
        chunk_id = _string(self.chunk_id, "evidence.chunk_id", optional=True)
        if all(item is None for item in (page, span, bbox, chunk_id)):
            raise _fail(
                "evidence",
                "provide at least one locator: page, span, bbox, or chunk_id",
            )
        if bbox is not None and page is None:
            raise _fail("evidence.bbox", "a bounding box requires page")
        assert doc_id is not None and modality is not None
        object.__setattr__(self, "doc_id", doc_id)
        object.__setattr__(self, "hash", digest)
        object.__setattr__(self, "page", page)
        object.__setattr__(self, "span", span)
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "chunk_id", chunk_id)

    @classmethod
    def from_dict(cls, value: Any, path: str = "evidence") -> EvidenceAnchor:
        if isinstance(value, cls):
            return value
        data = _fields(
            value,
            path=path,
            required=("doc_id",),
            optional=("hash", "page", "span", "bbox", "modality", "chunk_id"),
        )
        return cls(
            doc_id=data["doc_id"],
            hash=data.get("hash"),
            page=data.get("page"),
            span=(
                None
                if data.get("span") is None
                else TextSpan.from_value(data["span"], f"{path}.span")
            ),
            bbox=(
                None
                if data.get("bbox") is None
                else BoundingBox.from_value(data["bbox"], f"{path}.bbox")
            ),
            modality=data.get("modality", "text"),
            chunk_id=data.get("chunk_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "hash": self.hash,
            "page": self.page,
            "span": None if self.span is None else self.span.to_dict(),
            "bbox": None if self.bbox is None else self.bbox.to_dict(),
            "modality": self.modality,
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    role: ConversationRole
    content: str

    def __post_init__(self) -> None:
        role = _enum(self.role, ConversationRole, "conversation.role")
        content = _string(self.content, "conversation.content")
        assert content is not None
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", content)

    @classmethod
    def from_dict(cls, value: Any, path: str = "conversation") -> ConversationTurn:
        if isinstance(value, cls):
            return value
        data = _fields(value, path=path, required=("role", "content"))
        return cls(role=data["role"], content=data["content"])

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True, slots=True)
class ExpectedImage:
    image_id: str
    evidence: tuple[EvidenceAnchor, ...]
    description: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        image_id = _string(self.image_id, "expected_image.image_id")
        if isinstance(self.evidence, (str, bytes)) or not isinstance(
            self.evidence, Sequence
        ):
            raise _fail("expected_image.evidence", "expected an array")
        evidence = tuple(
            EvidenceAnchor.from_dict(item, f"expected_image.evidence[{index}]")
            for index, item in enumerate(self.evidence)
        )
        if not evidence:
            raise _fail("expected_image.evidence", "must contain at least one anchor")
        description = _string(
            self.description, "expected_image.description", optional=True
        )
        required = _boolean(self.required, "expected_image.required")
        assert image_id is not None
        object.__setattr__(self, "image_id", image_id)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "required", required)

    @classmethod
    def from_dict(cls, value: Any, path: str = "expected_image") -> ExpectedImage:
        if isinstance(value, cls):
            return value
        data = _fields(
            value,
            path=path,
            required=("image_id", "evidence"),
            optional=("description", "required"),
        )
        return cls(
            image_id=data["image_id"],
            evidence=data["evidence"],
            description=data.get("description"),
            required=data.get("required", True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "evidence": [item.to_dict() for item in self.evidence],
            "description": self.description,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    query: str
    expected_action: ExpectedAction
    conversation: tuple[ConversationTurn, ...] = ()
    reference_answer: str | None = None
    vital_claims: tuple[str, ...] = ()
    optional_claims: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    evidence: tuple[EvidenceAnchor, ...] = ()
    expected_images: tuple[ExpectedImage, ...] = ()
    severity: Severity = Severity.MEDIUM
    tags: tuple[str, ...] = ()
    split: str = "dev"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CASE_SCHEMA_VERSION:
            raise _fail(
                "case.schema_version",
                f"unsupported version {self.schema_version!r}; expected {CASE_SCHEMA_VERSION!r}",
            )
        case_id = _string(self.case_id, "case.case_id")
        query = _string(self.query, "case.query")
        expected_action = _enum(
            self.expected_action, ExpectedAction, "case.expected_action"
        )
        conversation = tuple(
            ConversationTurn.from_dict(item, f"case.conversation[{index}]")
            for index, item in enumerate(self.conversation)
        )
        reference_answer = _string(
            self.reference_answer, "case.reference_answer", optional=True
        )
        vital_claims = _string_tuple(self.vital_claims, "case.vital_claims")
        optional_claims = _string_tuple(self.optional_claims, "case.optional_claims")
        forbidden_claims = _string_tuple(self.forbidden_claims, "case.forbidden_claims")
        expected = set(vital_claims) | set(optional_claims)
        overlap = sorted(expected & set(forbidden_claims))
        if overlap:
            raise _fail(
                "case.forbidden_claims",
                f"claims cannot also be expected: {overlap!r}",
            )
        evidence = tuple(
            EvidenceAnchor.from_dict(item, f"case.evidence[{index}]")
            for index, item in enumerate(self.evidence)
        )
        expected_images = tuple(
            ExpectedImage.from_dict(item, f"case.expected_images[{index}]")
            for index, item in enumerate(self.expected_images)
        )
        image_ids = [item.image_id for item in expected_images]
        if len(image_ids) != len(set(image_ids)):
            raise _fail("case.expected_images", "image_id values must be unique")
        severity = _enum(self.severity, Severity, "case.severity")
        tags = _string_tuple(self.tags, "case.tags")
        split = _string(self.split, "case.split")
        metadata = _json_object(self.metadata, "case.metadata")
        assert case_id is not None and query is not None and split is not None
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "expected_action", expected_action)
        object.__setattr__(self, "conversation", conversation)
        object.__setattr__(self, "reference_answer", reference_answer)
        object.__setattr__(self, "vital_claims", vital_claims)
        object.__setattr__(self, "optional_claims", optional_claims)
        object.__setattr__(self, "forbidden_claims", forbidden_claims)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "expected_images", expected_images)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_dict(cls, value: Any, path: str = "case") -> EvaluationCase:
        if isinstance(value, cls):
            return value
        data = _fields(
            value,
            path=path,
            required=("schema_version", "case_id", "query", "expected_action"),
            optional=(
                "conversation",
                "reference_answer",
                "vital_claims",
                "optional_claims",
                "forbidden_claims",
                "evidence",
                "expected_images",
                "severity",
                "tags",
                "split",
                "metadata",
            ),
        )
        return cls(
            schema_version=data["schema_version"],
            case_id=data["case_id"],
            query=data["query"],
            expected_action=data["expected_action"],
            conversation=data.get("conversation", ()),
            reference_answer=data.get("reference_answer"),
            vital_claims=data.get("vital_claims", ()),
            optional_claims=data.get("optional_claims", ()),
            forbidden_claims=data.get("forbidden_claims", ()),
            evidence=data.get("evidence", ()),
            expected_images=data.get("expected_images", ()),
            severity=data.get("severity", Severity.MEDIUM.value),
            tags=data.get("tags", ()),
            split=data.get("split", "dev"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "query": self.query,
            "conversation": [item.to_dict() for item in self.conversation],
            "expected_action": self.expected_action.value,
            "reference_answer": self.reference_answer,
            "vital_claims": list(self.vital_claims),
            "optional_claims": list(self.optional_claims),
            "forbidden_claims": list(self.forbidden_claims),
            "evidence": [item.to_dict() for item in self.evidence],
            "expected_images": [item.to_dict() for item in self.expected_images],
            "severity": self.severity.value,
            "tags": list(self.tags),
            "split": self.split,
            "metadata": _json_object(self.metadata, "case.metadata"),
        }


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    dataset_id: str
    dataset_version: str
    cases: tuple[EvaluationCase, ...]
    description: str | None = None
    created_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_SCHEMA_VERSION:
            raise _fail(
                "dataset.schema_version",
                f"unsupported version {self.schema_version!r}; expected {DATASET_SCHEMA_VERSION!r}",
            )
        dataset_id = _string(self.dataset_id, "dataset.dataset_id")
        dataset_version = _string(self.dataset_version, "dataset.dataset_version")
        cases = tuple(
            EvaluationCase.from_dict(item, f"dataset.cases[{index}]")
            for index, item in enumerate(self.cases)
        )
        if not cases:
            raise _fail("dataset.cases", "must contain at least one case")
        case_ids = [case.case_id for case in cases]
        if len(case_ids) != len(set(case_ids)):
            raise _fail("dataset.cases", "case_id values must be unique")
        description = _string(self.description, "dataset.description", optional=True)
        created_at = _timestamp(self.created_at, "dataset.created_at", optional=True)
        metadata = _json_object(self.metadata, "dataset.metadata")
        assert dataset_id is not None and dataset_version is not None
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "dataset_version", dataset_version)
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_dict(cls, value: Any, path: str = "dataset") -> EvaluationDataset:
        if isinstance(value, cls):
            return value
        data = _fields(
            value,
            path=path,
            required=("schema_version", "dataset_id", "dataset_version", "cases"),
            optional=("description", "created_at", "metadata"),
        )
        return cls(
            schema_version=data["schema_version"],
            dataset_id=data["dataset_id"],
            dataset_version=data["dataset_version"],
            cases=data["cases"],
            description=data.get("description"),
            created_at=data.get("created_at"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "description": self.description,
            "created_at": self.created_at,
            "metadata": _json_object(self.metadata, "dataset.metadata"),
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_jsonl_records(self) -> tuple[dict[str, Any], ...]:
        manifest = self.to_dict()
        manifest.pop("cases")
        records: list[dict[str, Any]] = [
            {"record_type": "dataset_manifest", **manifest}
        ]
        records.extend(
            {"record_type": "case", "case": case.to_dict()} for case in self.cases
        )
        return tuple(records)


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    dataset_id: str
    dataset_version: str
    dataset_hash: str
    started_at: str
    runner: str
    corpus_id: str | None = None
    corpus_hash: str | None = None
    code_revision: str | None = None
    random_seed: int | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    models: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "dataset_id",
            "dataset_version",
            "dataset_hash",
            "runner",
        ):
            value = _string(getattr(self, name), f"manifest.{name}")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "started_at", _timestamp(self.started_at, "manifest.started_at")
        )
        for name in ("corpus_id", "corpus_hash", "code_revision"):
            value = _string(getattr(self, name), f"manifest.{name}", optional=True)
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "random_seed",
            _integer(
                self.random_seed,
                "manifest.random_seed",
                minimum=0,
                optional=True,
            ),
        )
        for name in ("config", "models", "environment"):
            object.__setattr__(
                self,
                name,
                _json_object(getattr(self, name), f"manifest.{name}"),
            )

    @classmethod
    def from_dict(cls, value: Any, path: str = "manifest") -> RunManifest:
        if isinstance(value, cls):
            return value
        data = _fields(
            value,
            path=path,
            required=(
                "run_id",
                "dataset_id",
                "dataset_version",
                "dataset_hash",
                "started_at",
                "runner",
            ),
            optional=(
                "corpus_id",
                "corpus_hash",
                "code_revision",
                "random_seed",
                "config",
                "models",
                "environment",
            ),
        )
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "dataset_hash": self.dataset_hash,
            "started_at": self.started_at,
            "runner": self.runner,
            "corpus_id": self.corpus_id,
            "corpus_hash": self.corpus_hash,
            "code_revision": self.code_revision,
            "random_seed": self.random_seed,
            "config": _json_object(self.config, "manifest.config"),
            "models": _json_object(self.models, "manifest.models"),
            "environment": _json_object(self.environment, "manifest.environment"),
        }


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    anchor: EvidenceAnchor
    rank: int
    score: float | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "anchor", EvidenceAnchor.from_dict(self.anchor, "retrieved.anchor")
        )
        object.__setattr__(
            self, "rank", _integer(self.rank, "retrieved.rank", minimum=1)
        )
        object.__setattr__(
            self,
            "score",
            _number(self.score, "retrieved.score", optional=True),
        )
        object.__setattr__(
            self, "text", _string(self.text, "retrieved.text", optional=True)
        )

    @classmethod
    def from_dict(cls, value: Any, path: str = "retrieved") -> RetrievedEvidence:
        if isinstance(value, cls):
            return value
        data = _fields(
            value,
            path=path,
            required=("anchor", "rank"),
            optional=("score", "text"),
        )
        return cls(
            anchor=EvidenceAnchor.from_dict(data["anchor"], f"{path}.anchor"),
            rank=data["rank"],
            score=data.get("score"),
            text=data.get("text"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor": self.anchor.to_dict(),
            "rank": self.rank,
            "score": self.score,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class TraceImage:
    image_id: str
    uri: str | None = None
    evidence: tuple[EvidenceAnchor, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        image_id = _string(self.image_id, "trace_image.image_id")
        uri = _string(self.uri, "trace_image.uri", optional=True)
        evidence = tuple(
            EvidenceAnchor.from_dict(item, f"trace_image.evidence[{index}]")
            for index, item in enumerate(self.evidence)
        )
        metadata = _json_object(self.metadata, "trace_image.metadata")
        assert image_id is not None
        object.__setattr__(self, "image_id", image_id)
        object.__setattr__(self, "uri", uri)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_dict(cls, value: Any, path: str = "trace_image") -> TraceImage:
        if isinstance(value, cls):
            return value
        data = _fields(
            value,
            path=path,
            required=("image_id",),
            optional=("uri", "evidence", "metadata"),
        )
        return cls(
            image_id=data["image_id"],
            uri=data.get("uri"),
            evidence=data.get("evidence", ()),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "uri": self.uri,
            "evidence": [item.to_dict() for item in self.evidence],
            "metadata": _json_object(self.metadata, "trace_image.metadata"),
        }


def _metric_object(value: Any, path: str) -> dict[str, float | bool | None]:
    data = _mapping(value, path)
    result: dict[str, float | bool | None] = {}
    for name, metric in data.items():
        if metric is None or isinstance(metric, bool):
            result[name] = metric
        else:
            result[name] = _number(metric, f"{path}.{name}")
    return result


def _nonnegative_number_object(value: Any, path: str) -> dict[str, float]:
    data = _mapping(value, path)
    return {
        name: _number(number, f"{path}.{name}", minimum=0.0)  # type: ignore[misc]
        for name, number in data.items()
    }


def _nonnegative_integer_object(value: Any, path: str) -> dict[str, int]:
    data = _mapping(value, path)
    return {
        name: _integer(number, f"{path}.{name}", minimum=0)  # type: ignore[misc]
        for name, number in data.items()
    }


@dataclass(frozen=True, slots=True)
class EvaluationTraceResult:
    case_id: str
    status: TraceStatus
    observed_action: ExpectedAction | None = None
    answer: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    retrieved_evidence: tuple[RetrievedEvidence, ...] = ()
    images: tuple[TraceImage, ...] = ()
    metrics: Mapping[str, float | bool | None] = field(default_factory=dict)
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    token_usage: Mapping[str, int] = field(default_factory=dict)
    error: str | None = None
    trace: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise _fail(
                "trace.schema_version",
                f"unsupported version {self.schema_version!r}; expected {TRACE_SCHEMA_VERSION!r}",
            )
        case_id = _string(self.case_id, "trace.case_id")
        status = _enum(self.status, TraceStatus, "trace.status")
        observed_action = (
            None
            if self.observed_action is None
            else _enum(self.observed_action, ExpectedAction, "trace.observed_action")
        )
        answer = _string(self.answer, "trace.answer", optional=True)
        started_at = _timestamp(self.started_at, "trace.started_at", optional=True)
        completed_at = _timestamp(
            self.completed_at, "trace.completed_at", optional=True
        )
        if (
            started_at is not None
            and completed_at is not None
            and _parsed_timestamp(completed_at) < _parsed_timestamp(started_at)
        ):
            raise _fail("trace.completed_at", "must not precede started_at")
        retrieved_evidence = tuple(
            RetrievedEvidence.from_dict(item, f"trace.retrieved_evidence[{index}]")
            for index, item in enumerate(self.retrieved_evidence)
        )
        ranks = [item.rank for item in retrieved_evidence]
        if len(ranks) != len(set(ranks)):
            raise _fail("trace.retrieved_evidence", "rank values must be unique")
        images = tuple(
            TraceImage.from_dict(item, f"trace.images[{index}]")
            for index, item in enumerate(self.images)
        )
        metrics = _metric_object(self.metrics, "trace.metrics")
        timings_ms = _nonnegative_number_object(self.timings_ms, "trace.timings_ms")
        token_usage = _nonnegative_integer_object(self.token_usage, "trace.token_usage")
        error = _string(self.error, "trace.error", optional=True)
        if status is TraceStatus.ERROR and error is None:
            raise _fail("trace.error", "is required when status is 'error'")
        if status is TraceStatus.COMPLETED and error is not None:
            raise _fail("trace.error", "must be absent when status is 'completed'")
        trace = _json_object(self.trace, "trace.trace")
        assert case_id is not None
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "observed_action", observed_action)
        object.__setattr__(self, "answer", answer)
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "completed_at", completed_at)
        object.__setattr__(self, "retrieved_evidence", retrieved_evidence)
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "timings_ms", timings_ms)
        object.__setattr__(self, "token_usage", token_usage)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "trace", trace)

    @classmethod
    def from_dict(cls, value: Any, path: str = "trace") -> EvaluationTraceResult:
        if isinstance(value, cls):
            return value
        data = _fields(
            value,
            path=path,
            required=("schema_version", "case_id", "status"),
            optional=(
                "observed_action",
                "answer",
                "started_at",
                "completed_at",
                "retrieved_evidence",
                "images",
                "metrics",
                "timings_ms",
                "token_usage",
                "error",
                "trace",
            ),
        )
        return cls(
            schema_version=data["schema_version"],
            case_id=data["case_id"],
            status=data["status"],
            observed_action=data.get("observed_action"),
            answer=data.get("answer"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            retrieved_evidence=data.get("retrieved_evidence", ()),
            images=data.get("images", ()),
            metrics=data.get("metrics", {}),
            timings_ms=data.get("timings_ms", {}),
            token_usage=data.get("token_usage", {}),
            error=data.get("error"),
            trace=data.get("trace", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "status": self.status.value,
            "observed_action": (
                None if self.observed_action is None else self.observed_action.value
            ),
            "answer": self.answer,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retrieved_evidence": [item.to_dict() for item in self.retrieved_evidence],
            "images": [item.to_dict() for item in self.images],
            "metrics": dict(self.metrics),
            "timings_ms": dict(self.timings_ms),
            "token_usage": dict(self.token_usage),
            "error": self.error,
            "trace": _json_object(self.trace, "trace.trace"),
        }


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    manifest: RunManifest
    traces: tuple[EvaluationTraceResult, ...]
    completed_at: str | None = None
    summary: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUN_SCHEMA_VERSION:
            raise _fail(
                "run.schema_version",
                f"unsupported version {self.schema_version!r}; expected {RUN_SCHEMA_VERSION!r}",
            )
        manifest = RunManifest.from_dict(self.manifest, "run.manifest")
        traces = tuple(
            EvaluationTraceResult.from_dict(item, f"run.traces[{index}]")
            for index, item in enumerate(self.traces)
        )
        case_ids = [trace.case_id for trace in traces]
        if len(case_ids) != len(set(case_ids)):
            raise _fail("run.traces", "case_id values must be unique")
        completed_at = _timestamp(self.completed_at, "run.completed_at", optional=True)
        if completed_at is not None and _parsed_timestamp(
            completed_at
        ) < _parsed_timestamp(manifest.started_at):
            raise _fail("run.completed_at", "must not precede manifest.started_at")
        summary = _json_object(self.summary, "run.summary")
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "traces", traces)
        object.__setattr__(self, "completed_at", completed_at)
        object.__setattr__(self, "summary", summary)

    @classmethod
    def from_dict(cls, value: Any, path: str = "run") -> EvaluationRun:
        if isinstance(value, cls):
            return value
        data = _fields(
            value,
            path=path,
            required=("schema_version", "manifest", "traces"),
            optional=("completed_at", "summary"),
        )
        return cls(
            schema_version=data["schema_version"],
            manifest=RunManifest.from_dict(data["manifest"], f"{path}.manifest"),
            traces=data["traces"],
            completed_at=data.get("completed_at"),
            summary=data.get("summary", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest": self.manifest.to_dict(),
            "completed_at": self.completed_at,
            "summary": _json_object(self.summary, "run.summary"),
            "traces": [trace.to_dict() for trace in self.traces],
        }

    def to_jsonl_records(self) -> tuple[dict[str, Any], ...]:
        records: list[dict[str, Any]] = [
            {
                "record_type": "run_manifest",
                "schema_version": self.schema_version,
                "manifest": self.manifest.to_dict(),
                "completed_at": self.completed_at,
                "summary": _json_object(self.summary, "run.summary"),
            }
        ]
        records.extend(
            {"record_type": "trace", "trace": trace.to_dict()} for trace in self.traces
        )
        return tuple(records)


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(
                    f"{path}:{line_number}:{exc.colno}: invalid JSON: {exc.msg}"
                ) from exc
            records.append(_mapping(value, f"{path}:{line_number}"))
    if not records:
        raise SchemaValidationError(f"{path}: JSONL file contains no records")
    return records


def load_evaluation_cases(path: str | Path) -> tuple[EvaluationCase, ...]:
    """Load case records from a dataset JSON, case-array JSON, or JSONL file."""

    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        records = _read_jsonl(source)
        if records[0].get("record_type") == "dataset_manifest":
            return load_evaluation_dataset(source).cases
        return tuple(
            EvaluationCase.from_dict(record, f"{source}:{index}")
            for index, record in enumerate(records, start=1)
        )

    value = _read_json(source)
    if isinstance(value, Mapping) and "cases" in value:
        return EvaluationDataset.from_dict(value, str(source)).cases
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(
            EvaluationCase.from_dict(item, f"{source}[{index}]")
            for index, item in enumerate(value)
        )
    return (EvaluationCase.from_dict(value, str(source)),)


def load_evaluation_dataset(path: str | Path) -> EvaluationDataset:
    """Load a versioned dataset envelope from canonical JSON or JSONL."""

    source = Path(path)
    if source.suffix.lower() != ".jsonl":
        return EvaluationDataset.from_dict(_read_json(source), str(source))

    records = _read_jsonl(source)
    header = _fields(
        records[0],
        path=f"{source}:1",
        required=(
            "record_type",
            "schema_version",
            "dataset_id",
            "dataset_version",
        ),
        optional=("description", "created_at", "metadata"),
    )
    if header["record_type"] != "dataset_manifest":
        raise _fail(f"{source}:1.record_type", "expected 'dataset_manifest'")
    cases: list[EvaluationCase] = []
    for line_number, record in enumerate(records[1:], start=2):
        row = _fields(
            record,
            path=f"{source}:{line_number}",
            required=("record_type", "case"),
        )
        if row["record_type"] != "case":
            raise _fail(f"{source}:{line_number}.record_type", "expected 'case'")
        cases.append(
            EvaluationCase.from_dict(row["case"], f"{source}:{line_number}.case")
        )
    return EvaluationDataset(
        schema_version=header["schema_version"],
        dataset_id=header["dataset_id"],
        dataset_version=header["dataset_version"],
        description=header.get("description"),
        created_at=header.get("created_at"),
        metadata=header.get("metadata", {}),
        cases=tuple(cases),
    )


def load_evaluation_run(path: str | Path) -> EvaluationRun:
    """Load a versioned run envelope from canonical JSON or JSONL."""

    source = Path(path)
    if source.suffix.lower() != ".jsonl":
        return EvaluationRun.from_dict(_read_json(source), str(source))

    records = _read_jsonl(source)
    header = _fields(
        records[0],
        path=f"{source}:1",
        required=("record_type", "schema_version", "manifest"),
        optional=("completed_at", "summary"),
    )
    if header["record_type"] != "run_manifest":
        raise _fail(f"{source}:1.record_type", "expected 'run_manifest'")
    traces: list[EvaluationTraceResult] = []
    for line_number, record in enumerate(records[1:], start=2):
        row = _fields(
            record,
            path=f"{source}:{line_number}",
            required=("record_type", "trace"),
        )
        if row["record_type"] != "trace":
            raise _fail(f"{source}:{line_number}.record_type", "expected 'trace'")
        traces.append(
            EvaluationTraceResult.from_dict(
                row["trace"], f"{source}:{line_number}.trace"
            )
        )
    return EvaluationRun(
        schema_version=header["schema_version"],
        manifest=RunManifest.from_dict(header["manifest"], f"{source}:1.manifest"),
        traces=tuple(traces),
        completed_at=header.get("completed_at"),
        summary=header.get("summary", {}),
    )


__all__ = [
    "CASE_SCHEMA_VERSION",
    "DATASET_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "TRACE_SCHEMA_VERSION",
    "BoundingBox",
    "ConversationRole",
    "ConversationTurn",
    "EvaluationCase",
    "EvaluationDataset",
    "EvaluationRun",
    "EvaluationTraceResult",
    "EvidenceAnchor",
    "ExpectedAction",
    "ExpectedImage",
    "RetrievedEvidence",
    "RunManifest",
    "SchemaValidationError",
    "Severity",
    "TextSpan",
    "TraceImage",
    "TraceStatus",
    "load_evaluation_cases",
    "load_evaluation_dataset",
    "load_evaluation_run",
]
