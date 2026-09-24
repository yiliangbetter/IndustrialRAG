"""Domain Schema loader — externalized domain vocabulary for RAG pipeline.

Reads ``config/domain_schema.json`` (or ``RAG_DOMAIN_SCHEMA`` env override)
and exposes a reload-aware ``schema`` proxy with typed accessors. Missing or
malformed domain policy fails loudly instead of silently changing behaviour.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PATH = _ROOT / "config" / "domain_schema.json"
_LIST_FIELDS = (
    "section_markers",
    "structural_field_keys",
    "action_prefixes",
    "footnote_labels",
    "paragraph_connectors",
    "filename_truncate_markers",
    "machine_class_suffixes",
    "image_block_fields",
)


class DomainSchemaError(RuntimeError):
    """The configured domain schema cannot be loaded safely."""


class DomainSchema:
    """Typed accessor over the domain schema dict."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        # Pre-build sets for O(1) lookup.
        self._section_markers_set = set(data.get("section_markers") or [])
        self._field_keys_set = set(data.get("structural_field_keys") or [])
        self._action_prefixes_tuple = tuple(data.get("action_prefixes") or [])
        self._footnote_labels_set = set(data.get("footnote_labels") or [])
        self._connectors_set = set(data.get("paragraph_connectors") or [])
        self._truncate_markers = list(data.get("filename_truncate_markers") or [])
        self._machine_class_suffixes = list(data.get("machine_class_suffixes") or [])

    # --- List properties ---

    @property
    def domain(self) -> str:
        return str(self._data.get("domain") or "unknown")

    @property
    def section_markers(self) -> list[str]:
        return list(self._data.get("section_markers") or [])

    @property
    def structural_field_keys(self) -> frozenset[str]:
        return frozenset(self._field_keys_set)

    @property
    def action_prefixes(self) -> tuple[str, ...]:
        return self._action_prefixes_tuple

    @property
    def footnote_labels(self) -> frozenset[str]:
        return frozenset(self._footnote_labels_set)

    @property
    def paragraph_connectors(self) -> frozenset[str]:
        return frozenset(self._connectors_set)

    @property
    def filename_truncate_markers(self) -> list[str]:
        return list(self._truncate_markers)

    @property
    def machine_class_suffixes(self) -> list[str]:
        return list(self._machine_class_suffixes)

    @property
    def catalog_page_marker(self) -> str:
        return str(self._data.get("catalog_page_marker") or "")

    @property
    def image_block_fields(self) -> list[str]:
        return list(self._data.get("image_block_fields") or [])

    # --- Convenience predicates ---

    def is_section_marker(self, text: str) -> bool:
        return text in self._section_markers_set

    def is_structural_field_key(self, text: str) -> bool:
        return text in self._field_keys_set

    def is_footnote_label(self, text: str) -> bool:
        return text in self._footnote_labels_set

    def is_paragraph_connector(self, text: str) -> bool:
        return text in self._connectors_set

    def truncate_filename(self, title: str) -> str:
        """Strip domain suffix markers from a PDF title to get the machine name."""
        result = title
        for marker in self._truncate_markers:
            result = result.split(marker)[0]
        return result.split(".pdf")[0].split(".PDF")[0].strip()


def _schema_path() -> Path:
    raw = (os.getenv("RAG_DOMAIN_SCHEMA") or "").strip()
    path = Path(raw).expanduser() if raw else _DEFAULT_PATH
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


def _validate_schema(data: Any, path: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise DomainSchemaError(f"Domain schema must be a JSON object: {path}")
    for key in ("domain", "catalog_page_marker"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise DomainSchemaError(f"Domain schema field {key!r} is required: {path}")
    for key in _LIST_FIELDS:
        values = data.get(key)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value.strip() for value in values)
        ):
            raise DomainSchemaError(
                f"Domain schema field {key!r} must be a non-empty string list: {path}"
            )
    return data


@lru_cache(maxsize=8)
def _load_schema_file(path_text: str, mtime_ns: int) -> DomainSchema:
    del mtime_ns  # part of the cache key
    path = Path(path_text)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DomainSchemaError(f"Cannot load domain schema {path}: {exc}") from exc
    return DomainSchema(_validate_schema(data, path))


def get_domain_schema() -> DomainSchema:
    path = _schema_path()
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError as exc:
        raise DomainSchemaError(f"Domain schema is missing: {path}") from exc
    return _load_schema_file(str(path), mtime_ns)


class _DomainSchemaProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(get_domain_schema(), name)


schema = _DomainSchemaProxy()
