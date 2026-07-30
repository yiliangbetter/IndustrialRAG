"""image_query_refs submodule ``iqr_query_intent``.

Query intent classifiers and shared compiled constants.  This module is a
pure leaf — it imports only from external utilities and ``iqr_domain_schema``,
never from other ``iqr_*`` siblings (avoiding circular imports).
"""
from __future__ import annotations

import re

from raganything.utils import discriminative_terms, substantive_bigrams
from iqr_domain_schema import schema as _domain_schema


# ---------------------------------------------------------------------------
# Compiled constants (schema-driven + static)
# ---------------------------------------------------------------------------

def _build_section_marker_re() -> re.Pattern[str]:
    alt = "|".join(re.escape(m) for m in _domain_schema.section_markers)
    return re.compile(rf"(?:{alt})[：:]\s*([^\n]{{2,48}})", re.IGNORECASE)


def _build_cycle_label_re() -> re.Pattern[str]:
    alt = "|".join(re.escape(m) for m in _domain_schema.section_markers)
    return re.compile(rf"^(?:{alt})[：:].+$")


_MAINT_TOPIC_RE = _build_section_marker_re()

# Ingest / manual section templates (document structure, not product vocabulary).
_GENERIC_CYCLE_LABEL_RE = _build_cycle_label_re()

_MAINT_CYCLE_VALUE_RE = re.compile(r"^每.{1,16}(?:一次|保养一次|/次|一遍)$")

# Section-marker based split / content extraction (schema-driven).
_SECTION_ALT = "|".join(re.escape(m) for m in _domain_schema.section_markers)
_SECTION_MARKER_SPLIT_RE = re.compile(_SECTION_ALT)
_SECTION_MARKER_CONTENT_RE = re.compile(rf"(?:{_SECTION_ALT})[：:]\s*([^\n]+)")

# --- Frequently-used inline patterns (compiled once) ---
_WS_RE = re.compile(r"\s+")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_PAREN_SPLIT_RE = re.compile(r"[（(]")
_CONJUNCTION_RE = re.compile(r"[与和、及]")
_REFERENCES_SPLIT_RE = re.compile(r"\n###\s*References\b", re.IGNORECASE)
_NEWLINES_RE = re.compile(r"[\n\r]+")
_BULLET_PREFIX_RE = re.compile(r"^[-*•]\s+")
_PERIOD_NL_SPLIT_RE = re.compile(r"[。\n]")

_STRUCTURAL_LABEL_SUFFIXES = tuple(_domain_schema.structural_field_keys)

try:
    from query_doc_steering import (  # noqa: WPS433
        _CATALOG_MODEL_MARKER,
        detect_table_filter_signal,
    )
except ImportError:
    _CATALOG_MODEL_MARKER = _domain_schema.catalog_page_marker

    def detect_table_filter_signal(query: str, text: str) -> bool:  # type: ignore[misc]
        return False


# ---------------------------------------------------------------------------
# Query intent classifiers
# ---------------------------------------------------------------------------

def _retrieval_prefers_catalog_field(query: str, text: str) -> bool:
    """Top retrieval lines are foreword catalog rows, not procedure topics."""
    from iqr_store import _ranked_retrieval_lines

    ranked = _ranked_retrieval_lines(query, text, limit=8)
    if not ranked:
        return False
    catalog_best = 0.0
    procedure_best = 0.0
    for overlap, line in ranked:
        if _CATALOG_MODEL_MARKER in line:
            catalog_best = max(catalog_best, overlap)
        if _MAINT_TOPIC_RE.search(line):
            procedure_best = max(procedure_best, overlap)
    if catalog_best <= 0:
        return False
    return catalog_best >= procedure_best


def _topic_overlaps_query(topic: str, query: str) -> bool:
    """Topic relates to the query via multiple terms or substantive bigrams."""
    topic = (topic or "").strip()
    if not topic:
        return False
    short_terms = discriminative_terms(query, min_len=2)
    if sum(1 for term in short_terms if term in topic) >= 2:
        return True
    if any(
        len(term) >= 3 and term in topic
        for term in discriminative_terms(query, min_len=3)
    ):
        return True
    qb = substantive_bigrams(query)
    tb = substantive_bigrams(topic)
    return bool(qb and tb and len(qb & tb) >= 2)


def _is_catalog_or_model_listing_query(query: str) -> bool:
    """Product catalog / model list questions (Q1/Q14), not maintenance item listings."""
    q = (query or "").strip()
    if not q:
        return False
    return bool(re.search(r"型号|产品型号|适用于哪些|一共有多少|有多少|几种产品", q))


def _is_listing_scope_query(query: str) -> bool:
    """Multi-item scope questions (哪些/几种…), not single how-to steps."""
    q = (query or "").strip()
    if not q:
        return False
    return bool(
        re.search(
            r"哪些|有几种|几种|列举|清单|多少个|多少种|一共有多少|总共|全部|有哪些|一共",
            q,
        )
    )


def _is_maintenance_cycle_query(query: str) -> bool:
    """Cadence / frequency questions (not component listings)."""
    q = (query or "").strip()
    if not q:
        return False
    return bool(re.search(r"多长时间|多久|周期|频率|多少次|几次", q))


def _is_multi_machine_comparison_query(query: str) -> bool:
    """Cross-manual answers comparing several machine lines (e.g. Q15)."""
    q = (query or "").strip()
    if not q:
        return False
    return bool(
        re.search(
            r"四种|多种机型|所有机型|各.*机型|分别.*多久|分别.*周期|几种封边机",
            q,
        )
    )


def _is_component_listing_across_machines(query: str) -> bool:
    """Cross-manual questions listing parts/components (e.g. Q17).

    A query that names a *specific* known machine is a single-machine component
    question and must NOT enter the cross-manual listing path.
    """
    from iqr_machine import resolve_machine_name

    q = (query or "").strip()
    if not q:
        return False
    if not (
        re.search(r"部件|零件|组件", q) and re.search(r"哪些|有什么|有哪|各自", q)
    ):
        return False
    # Single-machine query → not a cross-manual listing.
    if resolve_machine_name(q):
        return False
    return True


def _strict_object_image_gate(query: str) -> bool:
    """Single-item maintenance questions get strict object matching; listings stay permissive."""
    from iqr_align import _trust_llm_chunk_images
    from iqr_terms import _query_primary_object_term

    if _trust_llm_chunk_images():
        return False
    if _is_listing_scope_query(query):
        return False
    primary = _query_primary_object_term(query)
    if not primary or re.search(r"什么|哪些|多少|如何|怎样|怎么|哪个", primary):
        return False
    return len(primary) >= 4
