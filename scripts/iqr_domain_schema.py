"""Domain Schema loader — externalized domain vocabulary for RAG pipeline.

Reads ``config/domain_schema.json`` (or ``RAG_DOMAIN_SCHEMA`` env override)
and exposes a cached singleton ``schema`` with typed accessors.

If the file is absent, falls back to built-in defaults identical to the
former hard-coded values, ensuring zero-config backward compatibility.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PATH = _ROOT / "config" / "domain_schema.json"

# Built-in fallback (= current industrial maintenance domain).
_FALLBACK: dict[str, Any] = {
    "domain": "industrial_equipment_maintenance",
    "version": "0.0-fallback",
    "section_markers": ["保养步骤", "保养内容", "保养周期"],
    "structural_field_keys": [
        "周期", "步骤", "内容", "方式", "部位",
        "部件", "工具", "标准", "依据", "要求", "方法", "说明",
    ],
    "action_prefixes": ["清理", "检查", "更换", "调整", "清洁"],
    "footnote_labels": ["注", "备注"],
    "paragraph_connectors": ["此外", "另外", "同时", "除此之外"],
    "filename_truncate_markers": ["维护保养"],
    "special_query_patterns": ["开机前"],
    "machine_class_suffixes": ["封边机", "钻", "中心"],
    "catalog_page_marker": "本手册适用产品型号",
    "image_block_fields": ["图片路径", "页码", "关联正文", "图注", "脚注"],
}


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
        self._special_patterns = list(data.get("special_query_patterns") or [])
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
    def special_query_patterns(self) -> list[str]:
        return list(self._special_patterns)

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

    def matches_special_pattern(self, query: str) -> str | None:
        """Return the first special pattern found in query, or None."""
        for pat in self._special_patterns:
            if pat in query:
                return pat
        return None

    def truncate_filename(self, title: str) -> str:
        """Strip domain suffix markers from a PDF title to get the machine name."""
        result = title
        for marker in self._truncate_markers:
            result = result.split(marker)[0]
        return result.split(".pdf")[0].split(".PDF")[0].strip()


@lru_cache(maxsize=1)
def _load_schema() -> DomainSchema:
    path_str = os.environ.get("RAG_DOMAIN_SCHEMA", "")
    path = Path(path_str) if path_str else _DEFAULT_PATH
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return DomainSchema(data)
        except (json.JSONDecodeError, OSError):
            pass
    return DomainSchema(_FALLBACK)


# Module-level singleton for convenient import.
schema: DomainSchema = _load_schema()
