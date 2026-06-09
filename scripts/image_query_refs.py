"""Query-time image resolution for RAG Q&A (Plan B).

Ingest co-locates ``[图片]`` blocks with anchor text in vector chunks (Route A).
This module parses inline refs from rerank+steering ``primary_text`` at finalize.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from raganything.utils import (
    build_image_ref_block,
    discriminative_terms,
    flatten_image_refs_for_skip_multimodal,
    image_label_for_item,
    image_label_text,
    short_label_bag_aligns,
    substantive_bigrams,
    text_term_alignment_symmetric,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_image_ref_block",
    "flatten_image_refs_for_skip_multimodal",
    "extract_image_refs_from_context",
    # focus_text_for_images, supplement_refs_from_content_lists — legacy, see EOF
    "images_for_api",
    "resolve_query_images",
    "query_wants_kb_images",
    "retrieval_supports_images",
    "encode_media_token",
    "decode_media_token",
    "is_safe_media_path",
    "normalize_context_for_image_parse",
]

_IMAGE_EXT_GROUP = r"(?:jpg|jpeg|png|gif|webp|bmp|tif|tiff)"
_IMAGE_PATH_RE = re.compile(
    rf"图片路径[：:]\s*([^\n]+?\.{_IMAGE_EXT_GROUP})"
    rf"|Image Path:\s*([^\n]+?\.{_IMAGE_EXT_GROUP})",
    re.IGNORECASE,
)
_PAGE_RE = re.compile(r"页码[：:]\s*(\d+)", re.IGNORECASE)
_CAPTION_RE = re.compile(r"图注[：:]\s*(.+?)(?:\n|$)", re.IGNORECASE)
_FOOTNOTE_RE = re.compile(r"脚注[：:]\s*(.+?)(?:\n|$)", re.IGNORECASE)
_CONTEXT_RE = re.compile(r"关联正文[：:]\s*(.+?)(?:\n\n|\Z)", re.IGNORECASE | re.DOTALL)
_MAINT_TOPIC_RE = re.compile(r"保养内容[：:]\s*([^\n]{2,48})", re.IGNORECASE)
_SECTION_HEADING_RE = re.compile(r"^[\d\.]+\s*\S")
_SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)")
# Ingest / manual section templates (document structure, not product vocabulary).
_GENERIC_CYCLE_LABEL_RE = re.compile(r"^保养周期[：:].+$")

try:
    from query_doc_steering import (  # noqa: WPS433
        _CATALOG_MODEL_MARKER,
        detect_table_filter_signal,
    )
except ImportError:
    _CATALOG_MODEL_MARKER = "本手册适用产品型号"

    def detect_table_filter_signal(query: str, text: str) -> bool:  # type: ignore[misc]
        return False

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"})

_MIN_SUBSTANTIVE_TERM_LEN = 3
_MIN_QUERY_CHARS_FOR_IMAGES = 6
_PDF_NAME_RE = re.compile(r"[^\s\\n/]+\.pdf", re.IGNORECASE)
_REF_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s*(.+?)\s*$", re.MULTILINE)


def _min_substantive_term_len() -> int:
    raw = os.getenv("RAG_IMAGE_MIN_TERM_LEN") or str(_MIN_SUBSTANTIVE_TERM_LEN)
    try:
        return max(2, int(raw))
    except ValueError:
        return _MIN_SUBSTANTIVE_TERM_LEN


def _min_query_chars_for_images() -> int:
    raw = os.getenv("RAG_IMAGE_MIN_QUERY_CHARS") or str(_MIN_QUERY_CHARS_FOR_IMAGES)
    try:
        return max(2, int(raw))
    except ValueError:
        return _MIN_QUERY_CHARS_FOR_IMAGES


def _normalize_query_for_match(query: str) -> str:
    q = (query or "").strip()
    q = re.sub(r"[\s!！?？。.，,~、；;：:""''\"']+", "", q, flags=re.I)
    return q


def _query_terms(query: str) -> list[str]:
    """Extract discriminative terms from the query (length-based, no phrase lists)."""
    return discriminative_terms(query, min_len=_min_substantive_term_len())


def _unique_retrieved_doc_count(retrieved_docs: list[dict[str, Any]] | None) -> int:
    paths: set[str] = set()
    for doc in retrieved_docs or []:
        if not isinstance(doc, dict):
            continue
        fp = str(doc.get("file_path") or "").strip()
        if fp:
            paths.add(fp)
    return len(paths)


def _is_multi_source_retrieval(retrieved_docs: list[dict[str, Any]] | None) -> bool:
    return _unique_retrieved_doc_count(retrieved_docs) >= 2


def _query_subject_needles(query: str) -> list[str]:
    """Substantive query spans (discriminative_terms + subject clauses), no QA phrase lists."""
    q = (query or "").strip()
    if not q:
        return []
    needles: list[str] = []
    seen: set[str] = set()

    def add(span: str) -> None:
        span = span.strip()
        if len(span) < _min_substantive_term_len() or span in seen:
            return
        seen.add(span)
        needles.append(span)

    for term in _query_terms(q):
        add(term)
    for clause in _subject_action_clauses(q):
        for term in _query_terms(q):
            if len(term) >= _min_substantive_term_len() and term in clause:
                add(term)
        cjk = "".join(re.findall(r"[\u4e00-\u9fff]", clause))
        if len(cjk) >= _min_substantive_term_len():
            add(cjk[-min(8, len(cjk)) :])
    return sorted(needles, key=len, reverse=True)[:12]


def _line_has_query_subject_hit(query: str, line: str) -> bool:
    """True when a substantive query span appears in a retrieval line."""
    line = (line or "").strip()
    if not line:
        return False
    return any(needle in line for needle in _query_subject_needles(query))


def _retrieval_prefers_catalog_field(query: str, text: str) -> bool:
    """Top retrieval lines are foreword catalog rows, not procedure ``保养内容`` topics."""
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


def _multi_figure_image_limit() -> int:
    raw = os.getenv("RAG_IMAGE_MULTI_LIMIT") or "4"
    try:
        return max(2, min(6, int(raw)))
    except ValueError:
        return 4


def _normalize_label_key(label: str) -> str:
    return re.sub(r"\s+", "", (label or "").strip())


def _figure_label_key(ref: dict[str, Any]) -> str:
    return _normalize_label_key(_ref_effective_label(ref))


def _topic_overlaps_query(topic: str, query: str) -> bool:
    """Topic relates to the query via multiple terms or substantive bigrams (no single 2-char hit)."""
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


def _maintenance_topics_in_text(text: str, query: str) -> list[str]:
    """All ``保养内容：`` topics in retrieval bodies (ingest template, not phrase lists)."""
    topics: list[str] = []
    seen: set[str] = set()
    for match in _MAINT_TOPIC_RE.finditer(text or ""):
        topic = match.group(1).strip()
        topic = re.split(r"保养步骤|保养周期", topic, maxsplit=1)[0].strip()
        topic = re.split(r"[。\n]", topic, maxsplit=1)[0].strip()[:32]
        if len(topic) < 4:
            continue
        key = _normalize_label_key(topic)
        if key in seen:
            continue
        if not _topic_overlaps_query(topic, query):
            continue
        seen.add(key)
        topics.append(topic)
    return topics


def _listing_target_phrases(query: str, retrieved_text: str) -> list[str]:
    """Distinct maintenance phrases in retrieval that anchor separate figures."""
    text = (retrieved_text or "").strip()
    if not text:
        return []
    terms = _query_terms(query)
    phrases: list[str] = []
    seen: set[str] = set()

    def add(phrase: str) -> None:
        phrase = phrase.strip()
        if len(phrase) < 4:
            return
        key = _normalize_label_key(phrase)
        if key in seen:
            return
        seen.add(key)
        phrases.append(phrase)

    for topic in _maintenance_topics_in_text(text, query):
        add(topic)

    for line in re.split(r"[\n\r]+", text):
        line = line.strip()
        if len(line) < 4 or _is_image_metadata_line(line) or _is_toc_or_directory_line(line):
            continue
        short_terms = discriminative_terms(query, min_len=2)
        if short_terms and not any(term in line for term in short_terms):
            continue
        topic = _maintenance_topic_from_text(line)
        if topic and _topic_overlaps_query(topic, query):
            add(topic)

    return phrases[:12]


def _listing_targets_anchored_in_text(
    query: str,
    source_text: str,
    *,
    anchor_text: str,
) -> list[str]:
    """Listing targets that also appear in focus / rerank bodies (not manual-wide labels)."""
    targets = _listing_target_phrases(query, source_text)
    anchor = (anchor_text or "").strip()
    if len(targets) < 2 or not anchor:
        return targets
    lines = [line.strip() for line in re.split(r"[\n\r]+", anchor) if line.strip()]
    return [
        t
        for t in targets
        if any(
            t in line or _label_matches_listing_target(line, t) for line in lines
        )
    ]


# --- LEGACY (disabled): _focus_has_query_* — see EOF ``if False:`` block ---


def _listing_targets_with_query_line_overlap(
    query: str, full_text: str, focus: str
) -> list[str]:
    """Listing targets whose focus lines also share substantive query terms."""
    targets = _listing_targets_anchored_in_text(query, full_text, anchor_text=focus)
    if not targets:
        return []
    lines = [
        line.strip()
        for line in re.split(r"[\n\r]+", normalize_context_for_image_parse(focus))
        if line.strip()
    ]
    return [
        t
        for t in targets
        if any(
            (t in line or _label_matches_listing_target(line, t))
            and _line_has_specific_query_overlap(query, line, "")
            for line in lines
        )
    ]


# --- LEGACY (disabled): _focus_qualifies_for_image_lookup — see EOF ---


def _label_matches_listing_target(label: str, target: str) -> bool:
    label = (label or "").strip()
    target = (target or "").strip()
    if not label or not target:
        return False
    if short_label_bag_aligns(target, label):
        return True
    if _figure_label_matches_query(target, label):
        return True
    lk = _normalize_label_key(label)
    tk = _normalize_label_key(target)
    if len(tk) >= 3 and tk in lk:
        return True
    if len(lk) >= 4 and lk in tk:
        return True
    return False


def _ref_aligns_for_multi_figure_listing(
    query: str,
    ref: dict[str, Any],
    *,
    threshold: float,
    retrieved_text: str | None,
    listing_source_text: str | None = None,
) -> bool:
    label = _ref_effective_label(ref)
    if len(label) < _min_substantive_term_len():
        return False
    source = (listing_source_text or retrieved_text or "").strip()
    anchor = (retrieved_text or "").strip()
    for target in _listing_targets_with_query_line_overlap(query, source, anchor):
        if not _label_matches_listing_target(label, target):
            continue
        core = _strip_section_prefix(label) if _is_section_number_heading(label) else label
        if _figure_label_matches_query(query, core) or _topic_overlaps_query(core, query):
            return True
    return _ref_aligns_with_query_label(
        query, ref, threshold=threshold, retrieved_text=retrieved_text
    )


def _ref_blob(ref: dict[str, Any]) -> str:
    return " ".join(
        str(ref.get(key) or "") for key in ("label", "caption", "context")
    ).strip()


def _is_generic_cycle_only_label(label: str) -> bool:
    stripped = (label or "").strip()
    return bool(stripped) and bool(_GENERIC_CYCLE_LABEL_RE.match(stripped))


def _ref_matches_figure_focus(query: str, ref: dict[str, Any]) -> bool:
    """Reject ingest ``保养周期：``-only captions unless the figure mentions query subjects."""
    label = _ref_effective_label(ref)
    if not _is_generic_cycle_only_label(label):
        return True
    needles = _query_subject_needles(query)
    if not needles:
        return False
    blob = _ref_blob(ref)
    if any(needle in blob for needle in needles):
        return True
    object_bgs = _subject_object_bigrams(query)
    return bool(object_bgs and (object_bgs & substantive_bigrams(blob)))


def _source_hints_from_retrieved_docs(
    retrieved_docs: list[dict[str, Any]] | None,
) -> set[str]:
    hints: set[str] = set()
    for doc in retrieved_docs or []:
        if not isinstance(doc, dict):
            continue
        fp = str(doc.get("file_path") or "").strip()
        if not fp:
            continue
        stem = Path(fp).stem[:80]
        hints.add(stem)
        compact = re.sub(r"\s+", "", stem)
        if compact:
            hints.add(compact[:80])
    return hints


def _merged_source_hints(
    text: str,
    retrieved_docs: list[dict[str, Any]] | None = None,
) -> set[str]:
    hints = _source_hints_from_text(text)
    hints |= _source_hints_from_retrieved_docs(retrieved_docs)
    return {h for h in hints if h and len(h) >= 4}


def _ref_matches_source_hints(ref: dict[str, Any], hints: set[str]) -> bool:
    if not hints:
        return True
    path = str(ref.get("path") or "")
    return any(hint in path for hint in hints)


def _ref_inline_in_retrieved_context(
    ref: dict[str, Any], retrieved_text: str | None
) -> bool:
    """True when this figure is cited inside rerank-filtered chunk bodies."""
    text = (retrieved_text or "").strip()
    if not text:
        return False
    path_name = Path(str(ref.get("path") or "")).name
    if path_name and path_name in text:
        return True
    label = _ref_effective_label(ref)
    if label and len(label) >= _min_substantive_term_len() and label in text:
        return True
    ctx = str(ref.get("context") or "").strip()
    if len(ctx) >= 12 and ctx in text:
        return True
    return False


def _ref_inline_matches_query_subject(query: str, ref: dict[str, Any]) -> bool:
    """Inline chunk figures must match query subject bigrams, not a shared verb alone."""
    label = _ref_effective_label(ref)
    if not label:
        return False
    object_bgs = _subject_object_bigrams(query)
    label_bgs = substantive_bigrams(label)
    if object_bgs and object_bgs & label_bgs:
        return True
    for term in _query_terms(query):
        if len(term) >= 3 and term in label:
            return True
    return False


def _ref_anchored_in_retrieved_text(
    ref: dict[str, Any],
    retrieved_text: str | None,
    query: str,
) -> bool:
    """Figures must come from answer retrieval text, not manual-wide caption grep."""
    text = (retrieved_text or "").strip()
    if not text:
        return True
    if _ref_inline_in_retrieved_context(ref, text):
        return True
    ctx = str(ref.get("context") or "").strip()
    if len(ctx) >= 8 and ctx in text:
        return True
    label = _ref_effective_label(ref)
    if label and _ref_aligns_with_retrieval_focus(query, ref, text):
        return True
    return False


def _ref_passes_image_align_gate(
    query: str,
    ref: dict[str, Any],
    *,
    retrieved_text: str | None,
    source_hints: set[str] | None = None,
    listing_source_text: str | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
) -> bool:
    if source_hints and not _ref_matches_source_hints(ref, source_hints):
        return False
    text = (retrieved_text or "").strip()
    list_source = (listing_source_text or text).strip()
    anchor_sections: list[str] = []
    if _image_anchor_mode() not in ("", "off", "none", "0", "false"):
        anchor_sections = _pick_anchor_sections(query, list_source, retrieved_docs)
        if anchor_sections and _ref_conflicts_anchor_sections(
            anchor_sections,
            ref,
            query=query,
            primary_text=list_source,
        ):
            return False
        ref_maint = _maintenance_content_spans(str(ref.get("context") or ""))
        anchor_maint = _anchor_maintenance_spans(list_source, anchor_sections)
        if ref_maint and anchor_maint and not any(
            text_term_alignment_symmetric(rm, am) >= 0.42
            for rm in ref_maint
            for am in anchor_maint
        ):
            return False
    listing_targets = _listing_targets_with_query_line_overlap(
        query, list_source, text
    )
    listing_label_ok = len(listing_targets) >= 2 and any(
        _label_matches_listing_target(_ref_effective_label(ref), target)
        and (
            _figure_label_matches_query(query, _ref_effective_label(ref))
            or _topic_overlaps_query(_ref_effective_label(ref), query)
        )
        for target in listing_targets
    )
    if retrieved_text and not listing_label_ok and not _ref_anchored_in_retrieved_text(
        ref, retrieved_text, query
    ):
        ctx = str(ref.get("context") or "").strip()
        if not (
            anchor_sections
            and ctx
            and _ref_context_subject_aligns(
                query,
                ctx,
                retrieved_text,
                anchor_sections=anchor_sections,
                ref=ref,
                align_text=list_source,
            )
        ):
            return False
    if not _ref_matches_figure_focus(query, ref):
        return False
    if text:
        ranked = _ranked_retrieval_lines(query, text, limit=6)
        if ranked and ranked[0][0] >= _image_retrieval_focus_min_overlap():
            if len(listing_targets) >= 2:
                pass
            elif _ref_inline_in_retrieved_context(ref, text) and _ref_inline_matches_query_subject(
                query, ref
            ):
                pass
            elif not _ref_aligns_with_retrieval_focus(query, ref, text):
                ctx = str(ref.get("context") or "").strip()
                if not (
                    anchor_sections
                    and ctx
                    and _ref_context_subject_aligns(
                        query,
                        ctx,
                        text,
                        anchor_sections=anchor_sections,
                        ref=ref,
                        align_text=list_source,
                    )
                ):
                    return False
    threshold = _image_min_ref_align()
    if len(listing_targets) >= 2:
        return _ref_aligns_for_multi_figure_listing(
            query,
            ref,
            threshold=threshold,
            retrieved_text=retrieved_text,
            listing_source_text=list_source,
        )
    return _ref_aligns_with_query_label(
        query,
        ref,
        threshold=threshold,
        retrieved_text=retrieved_text,
        anchor_sections=anchor_sections,
        align_source_text=list_source,
    )


def query_wants_kb_images(query: str | None) -> bool:
    """False when the query has no substantive terms for KB-style image lookup."""
    q = (query or "").strip()
    if not q:
        return False
    if _query_terms(q):
        return True
    qn = _normalize_query_for_match(q)
    return len(qn) >= _min_query_chars_for_images()


def _max_rerank_score(retrieved_docs: list[dict[str, Any]] | None) -> float | None:
    scores: list[float] = []
    for doc in retrieved_docs or []:
        raw = doc.get("rerank_score")
        if raw is None:
            continue
        try:
            scores.append(float(raw))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else None


def _term_overlap_ratio(query: str, text: str) -> float:
    terms = _query_terms(query)
    if not terms or not text.strip():
        return 0.0
    hits = sum(1 for term in terms if term in text)
    return hits / len(terms)


def _image_min_rerank_score() -> float:
    raw = (
        os.getenv("RAG_IMAGE_MIN_RERANK_SCORE")
        or os.getenv("MIN_RERANK_SCORE")
        or "0.28"
    )
    try:
        return float(raw)
    except ValueError:
        return 0.28


def _image_min_term_overlap() -> float:
    raw = os.getenv("RAG_IMAGE_MIN_TERM_OVERLAP") or "0.28"
    try:
        return float(raw)
    except ValueError:
        return 0.34


def _image_min_ref_align() -> float:
    raw = os.getenv("RAG_IMAGE_MIN_REF_ALIGN") or "0.28"
    try:
        return float(raw)
    except ValueError:
        return 0.28


def _text_alignment(left: str, right: str) -> float:
    """Symmetric overlap of discriminative terms between two text snippets."""
    return text_term_alignment_symmetric(
        left, right, min_len=_min_substantive_term_len()
    )


_IMAGE_METADATA_MARKERS = (
    "[图片]",
    "图片路径",
    "Image Path:",
    "页码：",
    "页码:",
    "关联正文",
    "图注：",
    "图注:",
    "脚注：",
    "脚注:",
)


def _is_image_metadata_line(line: str) -> bool:
    """Skip parser/KB image blocks and path lines when aligning figures."""
    stripped = line.strip()
    if not stripped:
        return True
    return any(marker in stripped for marker in _IMAGE_METADATA_MARKERS)


def _is_cover_page_ref(ref: dict[str, Any]) -> bool:
    """Decorative cover figures (page 0) are never shown as query illustrations."""
    page = ref.get("page")
    if page is None:
        return False
    try:
        return int(page) == 0
    except (TypeError, ValueError):
        return False


def _line_has_specific_query_overlap(
    query: str, line: str, ref_text: str
) -> bool:
    """True when substantive query terms hit the answer line but not figure metadata."""
    blob = ref_text or ""
    long_hits = [
        term
        for term in discriminative_terms(query, min_len=3)
        if len(term) >= 3 and term in line and term not in blob
    ]
    if long_hits:
        return True
    short_hits = [
        term
        for term in discriminative_terms(query, min_len=2)
        if len(term) == 2 and term in line and term not in blob
    ]
    return len(set(short_hits)) >= 2


def _ref_label_aligns_with_line(ref: dict[str, Any], line: str) -> bool:
    label = str(ref.get("label") or ref.get("caption") or "").strip()
    if len(label) < _min_substantive_term_len():
        return False
    if label in line:
        return True
    return _text_alignment(label, line) >= _image_min_ref_align()


def _ranked_retrieval_lines(
    query: str, text: str, *, limit: int = 8
) -> list[tuple[float, str]]:
    """Lines from retrieved context ranked by query term overlap."""
    if not text.strip():
        return []
    scored: list[tuple[float, str]] = []
    for line in re.split(r"[\n\r]+", text):
        line = line.strip()
        if len(line) < _min_substantive_term_len():
            continue
        if _is_image_metadata_line(line) or _is_toc_or_directory_line(line):
            continue
        if _PDF_NAME_RE.search(line) or ".jpg" in line.lower() or ".png" in line.lower():
            continue
        cjk = "".join(re.findall(r"[\u4e00-\u9fff]", line))
        if len(cjk) < _min_substantive_term_len():
            continue
        overlap = _term_overlap_ratio(query, line)
        if overlap <= 0:
            continue
        cjk_only = "".join(re.findall(r"[\u4e00-\u9fff]", line))
        is_short_title = (
            not _SECTION_NUM_RE.match(line) and len(cjk_only) <= 10
        )
        if not is_short_title:
            for needle in _query_subject_needles(query):
                if len(needle) >= 4 and needle in line:
                    overlap += 0.22
            if _SECTION_NUM_RE.match(line):
                overlap += 0.08
        if is_short_title and overlap < 0.35:
            overlap *= 0.45
        scored.append((overlap, line))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:limit]


def _doc_content(doc: dict[str, Any]) -> str:
    for key in ("content", "text", "chunk_content", "page_content"):
        val = doc.get(key)
        if isinstance(val, str):
            return val
    return ""


def _primary_section_id(text: str) -> str | None:
    for line in (text or "").splitlines():
        line = line.strip()
        if not _SECTION_HEADING_RE.match(line):
            continue
        match = _SECTION_NUM_RE.match(line)
        if match:
            return match.group(1)
    return None


def _section_heading_before_line(text: str, target_line: str) -> str | None:
    pos = text.find(target_line)
    if pos < 0:
        return None
    found: str | None = None
    for line in text[:pos].splitlines():
        line = line.strip()
        if not _SECTION_HEADING_RE.match(line):
            continue
        match = _SECTION_NUM_RE.match(line)
        if match:
            found = match.group(1)
    return found


def _sections_compatible(anchor: str, candidate: str) -> bool:
    if not anchor or not candidate:
        return False
    if anchor == candidate:
        return True
    if candidate.startswith(anchor + ".") or anchor.startswith(candidate + "."):
        return True
    anchor_parts = anchor.split(".")
    candidate_parts = candidate.split(".")
    depth = min(len(anchor_parts), len(candidate_parts))
    return anchor_parts[:depth] == candidate_parts[:depth]


def _section_ids_in_text(text: str) -> list[str]:
    """Manual section numbers embedded in chunk or figure context."""
    found: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)+)(?![\d.])", text or ""):
        section_id = match.group(1)
        if not _valid_manual_section_id(section_id) or section_id in seen:
            continue
        seen.add(section_id)
        found.append(section_id)
    return found


def _ref_inferred_section_id(
    query: str,
    ref: dict[str, Any],
    primary_text: str,
) -> str | None:
    """Best-effort section id when figure context omits the heading number."""
    ctx = str(ref.get("context") or "").strip()
    label = _ref_effective_label(ref)
    blob = f"{ctx} {label}".strip()
    if not blob or not primary_text.strip():
        return None
    explicit = _section_ids_in_text(blob)
    if explicit:
        return explicit[0]
    maint = _maintenance_content_spans(blob)
    if not maint and label:
        maint = [label]
    best_sid: str | None = None
    best_score = 0.0
    for overlap, line in _ranked_retrieval_lines(query, primary_text, limit=12):
        if overlap < 0.08:
            break
        sid = _section_id_from_line_or_context(primary_text, line)
        if not sid:
            continue
        line_maint = _maintenance_content_spans(line)
        if maint and line_maint:
            for mc in maint:
                for lm in line_maint:
                    score = text_term_alignment_symmetric(mc, lm)
                    if score > best_score:
                        best_score = score
                        best_sid = sid
            continue
        if maint:
            title_match = re.match(
                r"^\d+(?:\.\d+)+\s+(\S.+?)(?:\s+保养|$)", line.strip()
            )
            if title_match:
                title = title_match.group(1).strip()
                for mc in maint:
                    score = text_term_alignment_symmetric(mc, title)
                    if score > best_score:
                        best_score = score
                        best_sid = sid
            continue
        score = text_term_alignment_symmetric(label or ctx, line)
        if score > best_score:
            best_score = score
            best_sid = sid
    return best_sid if best_score >= 0.42 else None


def _ref_maint_section_id(
    query: str,
    ref: dict[str, Any],
    primary_text: str,
) -> str | None:
    """Map figure ``保养内容`` to the best matching section line in retrieval."""
    maint = _maintenance_content_spans(str(ref.get("context") or ""))
    if not maint or not primary_text.strip():
        return None
    best_sid: str | None = None
    best_score = 0.0
    for overlap, line in _ranked_retrieval_lines(query, primary_text, limit=12):
        if overlap < 0.08:
            break
        sid = _section_id_from_line_or_context(primary_text, line)
        if not sid:
            continue
        for lm in _maintenance_content_spans(line):
            for mc in maint:
                score = text_term_alignment_symmetric(mc, lm)
                if score > best_score:
                    best_score = score
                    best_sid = sid
    return best_sid if best_score >= 0.42 else None


def _ref_conflicts_anchor_sections(
    anchor_sections: list[str],
    ref: dict[str, Any],
    *,
    query: str | None = None,
    primary_text: str | None = None,
) -> bool:
    """True when a figure cites an explicit section incompatible with answer anchors."""
    if not anchor_sections:
        return False
    blob = " ".join(
        str(ref.get(key) or "") for key in ("context", "caption", "label")
    ).strip()
    ref_sections = _section_ids_in_text(blob)
    if not ref_sections and query and primary_text:
        inferred = _ref_maint_section_id(query, ref, primary_text)
        if inferred:
            ref_sections = [inferred]
    if not ref_sections:
        return False
    return not any(
        _sections_compatible(anchor, ref_section)
        for anchor in anchor_sections
        for ref_section in ref_sections
    )


def _maintenance_content_spans(text: str) -> list[str]:
    spans: list[str] = []
    for match in re.finditer(r"保养内容[：:]\s*([^\n\r。；;]+)", text or ""):
        span = match.group(1).strip()
        if len(span) >= 3:
            spans.append(span)
    return spans


def _anchor_maintenance_spans(primary: str, anchor_sections: list[str]) -> list[str]:
    """``保养内容`` fields under answer anchor section ids in retrieval text."""
    spans: list[str] = []
    seen: set[str] = set()
    text = primary or ""
    for anchor in anchor_sections:
        start = 0
        while True:
            idx = text.find(anchor, start)
            if idx < 0:
                break
            window = text[idx : min(len(text), idx + 600)]
            for span in _maintenance_content_spans(window):
                if span not in seen:
                    seen.add(span)
                    spans.append(span)
            start = idx + max(1, len(anchor))
    return spans


def _ref_context_subject_aligns(
    query: str,
    ctx: str,
    retrieved_text: str | None,
    *,
    min_sym: float = 0.15,
    anchor_sections: list[str] | None = None,
    ref: dict[str, Any] | None = None,
    align_text: str | None = None,
) -> bool:
    """Context carries answer substance when parser footnotes truncate (e.g. 床外部清洁)."""
    ctx = (ctx or "").strip()
    if not ctx:
        return False
    label = _ref_effective_label(ref) if ref else ""
    maint = _maintenance_content_spans(ctx)
    if not maint and label:
        maint = [label]
    rt = (align_text or retrieved_text or "").strip()
    if anchor_sections and rt and maint:
        for mc in maint:
            for am in _anchor_maintenance_spans(rt, anchor_sections):
                if text_term_alignment_symmetric(mc, am) >= 0.42:
                    return True
    if not rt:
        return False
    for overlap, line in _ranked_retrieval_lines(query, rt, limit=8):
        if overlap < 0.12:
            break
        sid = _section_id_from_line_or_context(rt, line)
        if anchor_sections and sid and not any(
            _sections_compatible(anchor, sid) for anchor in anchor_sections
        ):
            continue
        line_maint = _maintenance_content_spans(line)
        if maint and line_maint:
            if any(
                text_term_alignment_symmetric(mc, lm) >= 0.42
                for mc in maint
                for lm in line_maint
            ):
                return True
            continue
        if anchor_sections:
            continue
        if text_term_alignment_symmetric(ctx, line) >= min_sym:
            return True
    return False


def _image_anchor_mode() -> str:
    return (os.getenv("RAG_IMAGE_ANCHOR_MODE") or "off").strip().lower()


def _image_anchor_min_line_overlap() -> float:
    raw = os.getenv("RAG_IMAGE_ANCHOR_MIN_LINE_OVERLAP") or "0.15"
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.15


def _image_anchor_max_sections() -> int:
    raw = os.getenv("RAG_IMAGE_ANCHOR_MAX_SECTIONS") or "1"
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _valid_manual_section_id(section_id: str) -> bool:
    section_id = (section_id or "").strip()
    if not section_id or not section_id[0].isdigit():
        return False
    parts = section_id.split(".")
    if len(parts) > 4:
        return False
    if len(parts) == 1:
        try:
            return 1 <= int(parts[0]) <= 9
        except ValueError:
            return False
    return all(part.isdigit() for part in parts)


def _section_id_from_line_or_context(text: str, line: str) -> str | None:
    match = _SECTION_NUM_RE.search(line)
    if match and _valid_manual_section_id(match.group(1)):
        return match.group(1)
    heading = _section_heading_before_line(text, line)
    if heading and _valid_manual_section_id(heading):
        return heading
    return None


def _chunk_is_toc_heavy(content: str) -> bool:
    lines = [line.strip() for line in content.splitlines() if len(line.strip()) >= 4]
    if len(lines) < 4:
        return False
    toc_count = sum(1 for line in lines if _is_toc_or_directory_line(line))
    return toc_count / len(lines) >= 0.35


def _chunk_subject_score(query: str, content: str) -> float:
    needles = _query_subject_needles(query)
    if not content.strip() or not needles:
        return 0.0
    score = 0.0
    for needle in needles:
        if len(needle) >= 3 and needle in content:
            score += min(len(needle), 12) * 0.12
    for line in content.splitlines():
        line = line.strip()
        if _is_toc_or_directory_line(line) or _is_image_metadata_line(line):
            continue
        if not _SECTION_HEADING_RE.match(line):
            continue
        hits = sum(1 for needle in needles if len(needle) >= 3 and needle in line)
        if hits:
            score += 1.0 + hits * 0.25
    return score


def _best_section_id_from_content(query: str, content: str) -> str | None:
    """Prefer a section heading that contains query subject terms (e.g. 2.1.1 机床床身清洁)."""
    needles = _query_subject_needles(query)
    best: tuple[float, str] | None = None
    for line in content.splitlines():
        line = line.strip()
        if _is_toc_or_directory_line(line) or _is_image_metadata_line(line):
            continue
        match = _SECTION_NUM_RE.match(line)
        if not match:
            continue
        section_id = match.group(1)
        if not _valid_manual_section_id(section_id):
            continue
        hits = sum(1 for needle in needles if len(needle) >= 3 and needle in line)
        score = float(hits)
        if hits:
            score += 0.5
        if best is None or score > best[0]:
            best = (score, section_id)
    if best and best[0] > 0:
        return best[1]
    if _chunk_is_toc_heavy(content):
        return None
    return _primary_section_id(content)


def _section_title_from_heading_line(line: str) -> str:
    stripped = line.strip()
    match = re.match(r"^\d+(?:\.\d+)+\s+(.+)$", stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _section_line_query_boost(query: str, line: str) -> float:
    """Boost section headings whose title overlaps query clauses (partial spans ok)."""
    title = _section_title_from_heading_line(line)
    if len(title) < _min_substantive_term_len():
        return 0.0
    boost = 0.0
    shared = substantive_bigrams(query) & substantive_bigrams(title)
    if len(shared) >= 2:
        boost += 0.12 + 0.05 * len(shared)
    for term in _query_terms(query):
        if len(term) >= 3 and term in title:
            boost += 0.08
    for clause in _subject_action_clauses(query):
        cjk = "".join(re.findall(r"[\u4e00-\u9fff]", clause))
        if len(cjk) < 4:
            continue
        for tail in range(min(12, len(cjk)), 3, -1):
            span = cjk[-tail:]
            if len(span) >= 4 and span in title:
                boost += 0.28 + 0.02 * min(tail, 8)
                break
    return boost


def _pick_anchor_sections(
    query: str,
    primary: str,
    retrieved_docs: list[dict[str, Any]] | None,
) -> list[str]:
    """Choose one (or few) manual section ids for inline-image scan."""
    min_overlap = _image_anchor_min_line_overlap()
    max_sections = _image_anchor_max_sections()
    needles = _query_subject_needles(query)
    line_scored: list[tuple[float, str]] = []

    for overlap, line in _ranked_retrieval_lines(query, primary, limit=8):
        if overlap < min_overlap or _is_toc_or_directory_line(line):
            continue
        section_id = _section_id_from_line_or_context(primary, line)
        if not section_id:
            continue
        boost = _section_line_query_boost(query, line)
        for needle in needles:
            if len(needle) >= 4 and needle in line:
                boost += 0.35
        total = overlap + boost
        if _SECTION_NUM_RE.match(line.strip()) and len(line.strip()) <= 56:
            total += 0.08
        line_scored.append((total, section_id))

    doc_scored: list[tuple[float, str]] = []
    docs = retrieved_docs or []
    for doc in docs:
        content = _doc_content(doc).strip()
        if not content or _chunk_is_toc_heavy(content):
            continue
        subject_score = _chunk_subject_score(query, content)
        section_id = _best_section_id_from_content(query, content) or _primary_section_id(
            content
        )
        if not section_id:
            continue
        rerank = float(doc.get("rerank_score") or doc.get("score") or 0)
        score = subject_score * 2.0 + rerank * 0.25
        if subject_score <= 0:
            score = rerank * 0.4
        doc_scored.append((score, section_id))

    line_scored.sort(key=lambda pair: pair[0], reverse=True)
    doc_scored.sort(key=lambda pair: pair[0], reverse=True)

    picked: list[str] = []
    seen: set[str] = set()
    line_best = line_scored[0][0] if line_scored else 0.0

    if line_scored and line_best >= min_overlap:
        for score, section_id in line_scored:
            if section_id in seen:
                continue
            seen.add(section_id)
            picked.append(section_id)
            if len(picked) >= max_sections:
                return picked

    for score, section_id in doc_scored:
        if section_id in seen:
            continue
        if line_scored and score < line_best * 0.85:
            continue
        seen.add(section_id)
        picked.append(section_id)
        if len(picked) >= max_sections:
            break

    if not picked and doc_scored:
        picked.append(doc_scored[0][1])
    return picked[:max_sections]


def _chunk_belongs_to_anchor_sections(
    query: str,
    primary: str,
    content: str,
    anchor_sections: list[str],
) -> bool:
    """True when chunk text is tied to answer anchor section ids (not TOC-wide grep)."""
    if not content.strip() or not anchor_sections:
        return False
    section_id = _best_section_id_from_content(query, content) or _primary_section_id(
        content
    )
    if section_id and any(
        _sections_compatible(anchor, section_id) for anchor in anchor_sections
    ):
        return True
    for anchor in anchor_sections:
        if re.search(rf"(?<![\d.]){re.escape(anchor)}\s+\S", content):
            return True
    for overlap, line in _ranked_retrieval_lines(query, primary, limit=8):
        if overlap < 0.12:
            break
        sid = _section_id_from_line_or_context(primary, line)
        if sid not in anchor_sections:
            continue
        snippet = line[: min(36, len(line))]
        if snippet and snippet in content:
            return True
    return False


def _context_for_image_scan(
    query: str,
    primary_text: str,
    retrieved_docs: list[dict[str, Any]] | None,
) -> tuple[str, dict[str, Any]]:
    """Narrow inline-image scan to answer lines / same-section chunks when enabled."""
    mode = _image_anchor_mode()
    primary = (primary_text or "").strip()
    meta: dict[str, Any] = {
        "mode": mode,
        "primary_chars": len(primary),
        "anchor_sections": [],
        "anchor_chars": 0,
        "anchor_chunks": 0,
    }
    if mode in ("", "off", "none", "0", "false"):
        meta["mode"] = "off"
        return primary, meta

    q = (query or "").strip()
    anchor_sections = _pick_anchor_sections(q, primary, retrieved_docs)

    meta["anchor_sections"] = anchor_sections
    if not anchor_sections:
        meta["reason"] = "no_anchor"
        # Multi-part listing spans several sections; fall back to full primary when
        # it already contains inline figures (avoid empty scan_text → no images).
        if extract_image_refs_from_context(primary):
            meta["fallback"] = "primary_no_anchor"
            meta["anchor_chars"] = len(primary)
            return primary, meta
        return "", meta

    docs = retrieved_docs or []
    matching_parts: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        content = _doc_content(doc).strip()
        if not content or content in seen or _chunk_is_toc_heavy(content):
            continue
        if _chunk_belongs_to_anchor_sections(q, primary, content, anchor_sections):
            seen.add(content)
            matching_parts.append(content)

    anchor_text = "\n\n".join(matching_parts)
    meta["anchor_chars"] = len(anchor_text)
    meta["anchor_chunks"] = len(matching_parts)
    if anchor_text.strip() and not extract_image_refs_from_context(anchor_text):
        meta["fallback"] = "primary_scan"
        meta["anchor_chars"] = len(primary)
        return primary, meta
    if not anchor_text.strip():
        meta["reason"] = "no_matching_chunks"
    return anchor_text, meta


# --- LEGACY (disabled): focus_text_for_images + helpers — see EOF ---


def _image_retrieval_focus_min_overlap() -> float:
    raw = os.getenv("RAG_IMAGE_RETRIEVAL_FOCUS_MIN_OVERLAP") or "0.1"
    try:
        return float(raw)
    except ValueError:
        return 0.1


def _query_tail_text(query: str, *, tail_chars: int = 12) -> str:
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", query or ""))
    if len(cjk) <= tail_chars:
        return cjk
    return cjk[-tail_chars:]


def _best_query_run_in_line(query: str, line: str) -> str:
    """Longest contiguous CJK span from the query that appears in ``line``."""
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", query or ""))
    best = ""
    if len(cjk) < _min_substantive_term_len():
        return best
    for i in range(len(cjk)):
        for j in range(i + _min_substantive_term_len(), len(cjk) + 1):
            sub = cjk[i:j]
            if sub in line and len(sub) > len(best):
                best = sub
    return best


def _suffix_focus_bigrams(text: str, *, min_suffix: int = 4) -> set[str]:
    """Prefer object/action tail bigrams over shared inspection verbs."""
    run = (text or "").strip()
    if len(run) <= min_suffix:
        return substantive_bigrams(run)
    suffix = run[-min_suffix:]
    return substantive_bigrams(suffix)


def _query_clauses(query: str) -> list[str]:
    """Split the query on punctuation only (no phrase stripping)."""
    clauses: list[str] = []
    for part in re.split(r"[，,？?！!；;]", query or ""):
        part = part.strip()
        cjk = "".join(re.findall(r"[\u4e00-\u9fff]", part))
        if len(cjk) >= _min_substantive_term_len():
            clauses.append(part)
    return clauses


def _clause_substance_score(clause: str, query: str) -> int:
    """Prefer clauses whose n-grams overlap the query's substantive terms."""
    score = 0
    for term in _query_terms(query):
        if len(term) >= 4 and term in clause:
            score += len(term)
    return score


def _subject_action_clauses(query: str) -> list[str]:
    """Clause(s) with the strongest substantive-term overlap (not blind tail pick)."""
    clauses = _query_clauses(query)
    if len(clauses) <= 1:
        return clauses
    scored = [(_clause_substance_score(clause, query), clause) for clause in clauses]
    best = max(score for score, _ in scored)
    if best <= 0:
        return clauses
    threshold = max(4, int(best * 0.45))
    picked = [clause for score, clause in scored if score >= threshold]
    return picked or [max(scored, key=lambda pair: pair[0])[1]]


def _subject_object_bigrams(query: str) -> set[str]:
    """Bigrams from the object tail of subject clause(s), not shared inspection verbs."""
    focus: set[str] = set()
    for clause in _subject_action_clauses(query):
        cjk = "".join(re.findall(r"[\u4e00-\u9fff]", clause))
        if len(cjk) >= 4:
            focus |= substantive_bigrams(cjk[-4:])
        elif cjk:
            focus |= substantive_bigrams(cjk)
        terms = sorted(
            (term for term in _query_terms(query) if len(term) >= 4 and term in clause),
            key=len,
            reverse=True,
        )
        for term in terms[:3]:
            focus |= substantive_bigrams(term)
    return focus


def _action_focus_bigrams(
    query: str, retrieved_text: str | None = None
) -> set[str]:
    """Bigrams for the query's concrete subject/action (not the device name echo)."""
    focus: set[str] = set()
    for clause in _subject_action_clauses(query):
        snippet = clause[-16:] if len(clause) > 16 else clause
        focus |= _suffix_focus_bigrams(snippet)
        for term in _query_terms(query):
            if len(term) >= _min_substantive_term_len() and term in clause:
                focus |= substantive_bigrams(term)
    if focus:
        return focus

    text = (retrieved_text or "").strip()
    if text:
        ranked = _ranked_retrieval_lines(query, text, limit=8)
        for overlap, line in ranked:
            if overlap < 0.04:
                break
            action = _best_query_run_in_line(query, line)
            if len(action) >= _min_substantive_term_len() and not _is_toc_or_directory_line(
                line
            ):
                candidate = _suffix_focus_bigrams(action)
                if candidate:
                    return candidate
    return _suffix_focus_bigrams(_query_tail_text(query))


def _focus_bigrams_for_figure_gate(
    query: str, retrieved_text: str | None = None
) -> set[str]:
    """Discriminative bigrams from the query action, anchored by strong retrieval lines."""
    focus: set[str] = set()
    text = (retrieved_text or "").strip()
    if text:
        ranked = _ranked_retrieval_lines(query, text, limit=6)
        if ranked and ranked[0][0] >= _image_retrieval_focus_min_overlap():
            best = ranked[0][0]
            for overlap, line in ranked:
                if overlap < best * 0.45:
                    break
                for term in _query_terms(query):
                    if len(term) >= _min_substantive_term_len() and term in line:
                        focus |= substantive_bigrams(term)
    if focus:
        return focus

    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", query or ""))
    tail = cjk[-12:] if len(cjk) >= 12 else cjk
    for term in _query_terms(query):
        if len(term) >= _min_substantive_term_len() and term in tail:
            focus |= substantive_bigrams(term)
    return focus


def _ref_passes_focus_bigram_gate(
    query: str,
    ref: dict[str, Any],
    retrieved_text: str | None = None,
) -> bool:
    """Reject figures whose label only shares generic inspection bigrams with the query."""
    label = _ref_effective_label(ref)
    if not label:
        return False
    blob = label + str(ref.get("context") or "")
    label_bgs = substantive_bigrams(blob)
    for term in _query_terms(query):
        if len(term) >= 4 and term in blob:
            return True
        if len(term) == 3 and term in label:
            return True
    object_focus = _subject_object_bigrams(query)
    if object_focus and not (object_focus & label_bgs):
        return False
    focus = _action_focus_bigrams(query, retrieved_text)
    if not focus:
        return True
    return bool(focus & label_bgs)


def _ref_aligns_with_retrieval_focus(
    query: str,
    ref: dict[str, Any],
    retrieved_text: str | None,
) -> bool:
    """When retrieval has a clear answer line, figures must align with that line."""
    label = _ref_effective_label(ref)

    text = (retrieved_text or "").strip()
    if not text:
        return True

    ranked = _ranked_retrieval_lines(query, text, limit=8)
    if not ranked or ranked[0][0] < _image_retrieval_focus_min_overlap():
        return True

    line_focus = _subject_object_bigrams(query) or _action_focus_bigrams(query, None)

    for overlap, line in ranked:
        if overlap < 0.03:
            break
        if line_focus and not (line_focus & substantive_bigrams(line)):
            continue
        if _ref_label_aligns_with_line(ref, line):
            return True
        if label and label in line:
            return True
    return False


def _is_toc_or_directory_line(line: str) -> bool:
    """Skip table-of-contents / cover directory lines (structural, not domain words)."""
    stripped = line.strip()
    if not stripped:
        return True
    if re.search(r"\d+\.\d+(?:\.\d+)?\s+.+?\.\s*\d+\s*$", stripped):
        return True
    if re.search(r"\.\s*\.\s*\d+\s*$", stripped):
        return True
    if stripped.count(".") >= 3 and re.search(r"\.\s*\d+\s*$", stripped):
        return True
    return False


def _heading_before_image_block(context: str, path_match_start: int) -> str:
    """Section title immediately preceding an inline ``[图片]`` block."""
    window_start = max(0, path_match_start - 900)
    window = context[window_start:path_match_start]
    lines = [ln.strip() for ln in window.splitlines() if ln.strip()]
    for line in reversed(lines):
        if _is_image_metadata_line(line) or _is_toc_or_directory_line(line):
            continue
        if re.match(r"^\d+\.\d+(?:\.\d+)?\s+\S", line) and len(line) <= 80:
            return line
        if line.startswith("保养") and "：" in line:
            continue
        # Unnumbered section titles (e.g. ``机床床身清洁``) often sit directly above figures.
        if (
            4 <= len(line) <= 48
            and not re.search(r"[。；;，,：:]", line)
            and re.search(r"[\u4e00-\u9fff]", line)
            and not re.match(r"^(地\s*址|电\s*话|传\s*真|邮\s*箱|网\s*址)", line)
        ):
            return line
    return ""


def _strip_section_prefix(label: str) -> str:
    return re.sub(r"^[\d\.\s]+", "", (label or "").strip()).strip()


def _is_section_number_heading(label: str) -> bool:
    stripped = (label or "").strip()
    if not stripped:
        return False
    return bool(_SECTION_HEADING_RE.match(stripped))


def _maintenance_topic_from_text(text: str) -> str:
    if not text:
        return ""
    match = _MAINT_TOPIC_RE.search(text)
    if not match:
        return ""
    topic = match.group(1).strip()
    topic = re.split(r"\s*\d+\.\d+", topic, maxsplit=1)[0].strip()
    topic = re.split(r"保养步骤|保养周期", topic, maxsplit=1)[0].strip()
    topic = re.split(r"[。\n]", topic, maxsplit=1)[0].strip()
    return topic[:32]


def _is_usable_source_figure_label(text: str) -> bool:
    label = (text or "").strip()
    return bool(label) and not _is_section_number_heading(label)


def _source_figure_label(ref: dict[str, Any]) -> str:
    """Label from ingest ``[图片]`` block (footnote / 图注); never inferred text."""
    for key in ("footnote", "caption", "label"):
        val = str(ref.get(key) or "").strip()
        if _is_usable_source_figure_label(val):
            return val
    return ""


def _preserve_source_figure_labels(ref: dict[str, Any]) -> bool:
    """True when MinerU/ingest already gave a figure caption — do not rewrite at query time."""
    src = _source_figure_label(ref)
    if not src:
        return False
    ref["label"] = src
    if not _is_usable_source_figure_label(str(ref.get("caption") or "")):
        ref["caption"] = src
    return True


def _enrich_ref_from_image_block(ref: dict[str, Any], block: str) -> None:
    """Fill missing labels only. Ingest footnote/图注 are authoritative and never replaced."""
    if _preserve_source_figure_labels(ref):
        return

    topic = _maintenance_topic_from_text(block)
    if not topic:
        topic = _maintenance_topic_from_text(str(ref.get("context") or ""))
    if topic:
        ref["label"] = topic
        caption = str(ref.get("caption") or "").strip()
        if not caption or _is_section_number_heading(caption):
            ref["caption"] = topic


def _ref_effective_label(ref: dict[str, Any]) -> str:
    raw = _source_figure_label(ref) or str(ref.get("label") or "").strip()
    if _is_section_number_heading(raw):
        stripped = _strip_section_prefix(raw)
        if stripped:
            return stripped
    return raw


def _figure_label_matches_query(query: str, label: str) -> bool:
    """Match figure captions when word order differs from the question."""
    label = (label or "").strip()
    query = (query or "").strip()
    if not label or not query:
        return False
    core = _strip_section_prefix(label) if _is_section_number_heading(label) else label
    if short_label_bag_aligns(query, core):
        return True
    qb = substantive_bigrams(query)
    lb = substantive_bigrams(core)
    if len(qb & lb) >= 2:
        return True
    object_bgs = _subject_object_bigrams(query)
    for term in _query_terms(query):
        if len(term) >= 3 and term in core:
            if object_bgs and not (object_bgs & lb):
                if len(qb & lb) < 2 and not short_label_bag_aligns(query, core):
                    continue
            return True
    return False


def _ref_aligns_with_query_label(
    query: str,
    ref: dict[str, Any],
    *,
    threshold: float,
    retrieved_text: str | None = None,
    anchor_sections: list[str] | None = None,
    align_source_text: str | None = None,
) -> bool:
    """Parser caption/footnote/section heading aligns with the query."""
    label = _ref_effective_label(ref)
    if len(label) < _min_substantive_term_len():
        return False

    core_label = _strip_section_prefix(label) if _is_section_number_heading(label) else label
    if _is_generic_cycle_only_label(core_label) and not _ref_matches_figure_focus(
        query, ref
    ):
        return False
    label_ok = _figure_label_matches_query(query, core_label)
    if not label_ok and _text_alignment(query, core_label) >= threshold and any(
        len(term) >= 4 and term in core_label for term in _query_terms(query)
    ):
        label_ok = True

    context_label_ok = False
    if not label_ok:
        ctx = str(ref.get("context") or "").strip()
        if ctx and _ref_context_subject_aligns(
            query,
            ctx,
            retrieved_text,
            anchor_sections=anchor_sections,
            ref=ref,
            align_text=align_source_text,
        ):
            label_ok = True
            context_label_ok = True

    if not label_ok:
        return False
    if not _ref_passes_focus_bigram_gate(query, ref, retrieved_text):
        return False
    if context_label_ok:
        return True
    if _ref_inline_in_retrieved_context(ref, retrieved_text):
        return True
    return _ref_aligns_with_retrieval_focus(query, ref, retrieved_text)


# --- LEGACY (disabled): supplement_refs_for_listing_targets — see EOF ---


def _refs_from_retrieved_docs_text(
    context: str,
    media_roots: list[Path],
    *,
    query: str | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
    full_context: str | None = None,
) -> list[dict[str, Any]]:
    """Inline ``[图片]`` refs from rerank-filtered primary text only."""
    ctx = (context or "").strip()
    hint_source = (full_context or ctx).strip()
    hints = _merged_source_hints(hint_source, retrieved_docs)
    refs = extract_image_refs_from_context(ctx)
    if hints:
        refs = [ref for ref in refs if _ref_matches_source_hints(ref, hints)]
    return refs


def _candidate_align_lines(
    query: str,
    ref: dict[str, Any],
    retrieved_text: str,
) -> list[tuple[float, str]]:
    """Lines to test alignment: figure label/context first, then retrieval."""
    candidates: list[tuple[float, str]] = []

    label = str(ref.get("label") or ref.get("caption") or "").strip()
    if len(label) >= _min_substantive_term_len():
        candidates.append((1.0, label))

    ctx = str(ref.get("context") or "").strip()
    if ctx:
        for part in re.split(r"[\n\r。；;]+", ctx):
            part = part.strip()
            if len(part) >= _min_substantive_term_len():
                candidates.append((0.95, part))

    for overlap, line in _ranked_retrieval_lines(query, retrieved_text):
        if _is_toc_or_directory_line(line):
            continue
        candidates.append((overlap, line))

    return candidates


def _ref_aligns_with_retrieval(
    query: str,
    ref: dict[str, Any],
    retrieved_text: str,
    *,
    min_align: float | None = None,
) -> bool:
    """Keep refs aligned with substantive answer lines, not cover/title echoes."""
    if _is_cover_page_ref(ref):
        return False

    threshold = min_align if min_align is not None else _image_min_ref_align()
    if _ref_aligns_with_query_label(
        query, ref, threshold=threshold, retrieved_text=retrieved_text
    ):
        return True

    ranked = [
        (overlap, line)
        for overlap, line in _ranked_retrieval_lines(query, retrieved_text)
        if not _is_image_metadata_line(line) and not _is_toc_or_directory_line(line)
    ]
    ref_text = " ".join(
        str(ref.get(key) or "") for key in ("context", "caption", "label")
    ).strip()

    candidates = _candidate_align_lines(query, ref, retrieved_text)
    if not candidates and not ref_text:
        return False

    top_overlap = ranked[0][0] if ranked else 0.0
    seen_lines: set[str] = set()
    for line_overlap, line in sorted(candidates, key=lambda pair: pair[0], reverse=True):
        if line in seen_lines:
            continue
        seen_lines.add(line)
        if ranked and line_overlap < top_overlap * 0.55 and line_overlap < 0.95:
            continue
        if ref_text and _text_alignment(ref_text, line) < threshold:
            if not _ref_label_aligns_with_line(ref, line):
                shared = [
                    term
                    for term in _query_terms(query)
                    if len(term) >= 4 and term in line and term in ref_text
                ]
                if not shared or _text_alignment(query, line) < threshold * 0.85:
                    continue
        if _line_has_specific_query_overlap(query, line, ref_text):
            return True
        if _ref_label_aligns_with_line(ref, line):
            return True
    return False


def _eligible_figure_refs(context: str) -> list[dict[str, Any]]:
    return [
        ref
        for ref in extract_image_refs_from_context(context)
        if not _is_cover_page_ref(ref)
    ]


def _collect_figure_refs(
    context: str,
    media_roots: list[Path] | None = None,
    *,
    query: str | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
    full_context: str | None = None,
) -> list[dict[str, Any]]:
    """Inline chunk figures from rerank-filtered primary context."""
    if media_roots:
        return _refs_from_retrieved_docs_text(
            context,
            media_roots,
            query=query,
            retrieved_docs=retrieved_docs,
            full_context=full_context,
        )
    return _eligible_figure_refs(context)


def _cross_manual_has_aligned_figures(
    query: str,
    focus_text: str,
    *,
    retrieved_docs: list[dict[str, Any]] | None,
    media_roots: list[Path] | None,
    full_text: str | None = None,
) -> bool:
    if not _is_multi_source_retrieval(retrieved_docs):
        return False
    focus = (focus_text or "").strip()
    if not focus:
        return False
    full = (full_text or focus).strip()
    hints = _merged_source_hints(full, retrieved_docs)
    eligible = _collect_figure_refs(
        focus,
        media_roots,
        query=query,
        retrieved_docs=retrieved_docs,
        full_context=full,
    )
    return any(
        _ref_passes_image_align_gate(
            query,
            ref,
            retrieved_text=focus,
            source_hints=hints,
            listing_source_text=full,
            retrieved_docs=retrieved_docs,
        )
        for ref in eligible
    )


def retrieval_supports_images(
    query: str | None,
    *,
    retrieved_docs: list[dict[str, Any]] | None = None,
    context_text: str | None = None,
    media_roots: list[Path] | None = None,
    focus_text: str | None = None,
) -> bool:
    """True when rerank-filtered primary context has query-aligned inline figures."""
    del focus_text  # legacy kwarg; primary-only pipeline
    if not query_wants_kb_images(query):
        logger.info("Skip related images: non-KB / chitchat query")
        return False
    primary = (context_text or "").strip()
    q = (query or "").strip()
    if primary and detect_table_filter_signal(q, primary):
        logger.info("Skip related images: confirmed table filter listing in context")
        return False

    if not primary:
        logger.info("Skip related images: empty retrieval context")
        return False

    scan_text, anchor_meta = _context_for_image_scan(q, primary, retrieved_docs)
    figure_context = primary if anchor_meta.get("mode") == "off" else scan_text
    if anchor_meta.get("mode") != "off" and not figure_context.strip():
        logger.info(
            "Skip related images: anchor mode (%s) found no same-section context",
            anchor_meta.get("reason") or "empty",
        )
        return False

    if _cross_manual_has_aligned_figures(
        q,
        figure_context if figure_context.strip() else primary,
        retrieved_docs=retrieved_docs,
        media_roots=media_roots,
        full_text=primary,
    ):
        return True

    max_score = _max_rerank_score(retrieved_docs)
    if max_score is not None:
        threshold = _image_min_rerank_score()
        if max_score < threshold:
            logger.info(
                "Skip related images: max rerank_score %.3f < %.3f",
                max_score,
                threshold,
            )
            return False

    hints = _merged_source_hints(primary, retrieved_docs)
    eligible = _collect_figure_refs(
        figure_context,
        media_roots,
        query=query,
        retrieved_docs=retrieved_docs,
        full_context=primary,
    )
    if not eligible:
        logger.info("Skip related images: no non-cover inline figures in scan context")
        return False
    if any(
        _ref_passes_image_align_gate(
            q,
            ref,
            retrieved_text=figure_context,
            source_hints=hints,
            listing_source_text=primary,
            retrieved_docs=retrieved_docs,
        )
        for ref in eligible
    ):
        return True

    terms = _query_terms(q)
    if not terms:
        logger.info("Skip related images: no substantive query terms")
        return False

    overlap = _term_overlap_ratio(q, primary)
    min_overlap = _image_min_term_overlap()
    listing_ok = len(_listing_targets_with_query_line_overlap(q, primary, primary)) >= 2
    cross_ok = _is_multi_source_retrieval(retrieved_docs)
    near_miss = overlap + 0.051 >= min_overlap
    if overlap < min_overlap and not listing_ok and not cross_ok and not (
        near_miss
        and any(
            _figure_label_matches_query(q, _ref_effective_label(ref))
            for ref in eligible
        )
    ):
        logger.info(
            "Skip related images: primary term overlap %.2f < %.2f",
            overlap,
            min_overlap,
        )
        return False
    logger.info("Skip related images: no figure label/heading matches query")
    return False


def normalize_context_for_image_parse(text: str) -> str:
    """LightRAG context may contain literal ``\\n`` instead of real newlines."""
    if not text:
        return ""
    if "\\n" in text:
        text = text.replace("\\n", "\n")
    if "\\t" in text:
        text = text.replace("\\t", "\t")
    return text


def normalize_image_path(path: str) -> str:
    p = path.strip().strip('"').strip("'")
    while "\\\\" in p:
        p = p.replace("\\\\", "\\")
    return p


def _image_block_for_path(context: str, path_match_start: int) -> str:
    """Isolate the ``[图片]`` block that owns this path marker."""
    block_start = context.rfind("[图片]", 0, path_match_start)
    if block_start < 0:
        block_start = max(0, path_match_start - 200)
    next_block = context.find("\n[图片]", path_match_start)
    if next_block < 0:
        next_block = context.find("[图片]", path_match_start + 1)
    if next_block < 0:
        return context[block_start:]
    return context[block_start:next_block]


def _metadata_from_block(block: str) -> dict[str, Any]:
    page = None
    pm = _PAGE_RE.search(block)
    if pm:
        try:
            page = int(pm.group(1))
        except ValueError:
            page = None

    caption = ""
    footnote = ""
    cm = _CAPTION_RE.search(block)
    if cm:
        caption = cm.group(1).strip()
    fm = _FOOTNOTE_RE.search(block)
    if fm:
        footnote = fm.group(1).strip()
    if not caption and not footnote:
        cap_m = re.search(r"标注[：:]\s*(.+?)(?:\n|$)", block)
        if cap_m:
            caption = cap_m.group(1).strip()

    ctx_m = _CONTEXT_RE.search(block)
    context_snippet = ctx_m.group(1).strip()[:300] if ctx_m else ""
    topic = _maintenance_topic_from_text(block) or _maintenance_topic_from_text(
        context_snippet
    )
    meta: dict[str, Any] = {
        "page": page,
        "caption": caption or footnote,
        "footnote": footnote,
        "context": context_snippet,
    }
    if _is_usable_source_figure_label(footnote):
        meta["label"] = footnote
    elif _is_usable_source_figure_label(caption):
        meta["label"] = caption
    elif topic:
        meta["label"] = topic
    return meta


def _extract_context_anchors(query: str | None, text: str) -> list[str]:
    """Phrases/lines linking query terms to retrieved text (no domain regex lists)."""
    if not text.strip():
        return []
    terms = _query_terms(query or "")
    anchors: list[str] = []
    seen: set[str] = set()

    def add(phrase: str) -> None:
        phrase = phrase.strip()
        if len(phrase) < _min_substantive_term_len() or phrase in seen:
            return
        seen.add(phrase)
        anchors.append(phrase)

    q = (query or "").strip()
    needles = _query_subject_needles(q)
    scored: list[tuple[float, str]] = []
    for run in re.findall(r"[\u4e00-\u9fff]+", q):
        if len(run) >= _min_substantive_term_len() and run in text:
            add(run)

    for line in re.split(r"[\n\r]+", text):
        line = line.strip()
        if len(line) < _min_substantive_term_len():
            continue
        if _is_image_metadata_line(line) or _is_toc_or_directory_line(line):
            continue
        if not terms:
            continue
        hits = sum(1 for term in terms if term in line)
        if hits <= 0:
            continue
        overlap = hits / len(terms)
        if _GENERIC_CYCLE_LABEL_RE.match(line) or re.fullmatch(
            r"保养周期[：:].+", line
        ):
            if needles and not any(n in line for n in needles):
                continue
        scored.append((overlap, line[:120]))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    for overlap, line in scored:
        if overlap < scored[0][0] * 0.45:
            break
        if _GENERIC_CYCLE_LABEL_RE.match(line) or re.fullmatch(
            r"保养周期[：:].+", line
        ):
            if needles and not any(n in line for n in needles):
                continue
        add(line)

    for clause in _subject_action_clauses(q):
        cjk = "".join(re.findall(r"[\u4e00-\u9fff]", clause))
        if len(cjk) >= _min_substantive_term_len():
            add(cjk[-10:] if len(cjk) > 10 else cjk)
    for term in _query_terms(q):
        if len(term) >= 4:
            add(term)

    return anchors[:12]


def _source_key_from_path(path_str: str) -> str:
    """Group images by parser output folder (one key per ingested document tree)."""
    path = Path(normalize_image_path(path_str))
    parts = [p for p in path.parts if p.lower() not in {"images", "auto", "figures"}]
    if len(parts) >= 2:
        return parts[-2]
    return path.stem


def _source_hints_from_text(text: str) -> set[str]:
    """Document titles/paths mentioned in retrieved context."""
    hints: set[str] = set()
    for match in _REF_LINE_RE.finditer(text):
        title = match.group(2).strip()
        if title:
            hints.add(title.split(".pdf")[0].split(".PDF")[0][:80])
    for match in _PDF_NAME_RE.finditer(text):
        hints.add(Path(match.group(0)).stem[:80])
    return hints


def _score_ref_for_query(
    ref: dict[str, Any],
    query: str | None,
    *,
    anchor_phrases: list[str] | None = None,
    source_hints: set[str] | None = None,
    retrieved_text: str | None = None,
) -> int:
    if not query:
        return 0
    blob = " ".join(
        str(ref.get(key) or "")
        for key in ("caption", "context", "path", "label")
    )
    q = query.strip()
    score = 0
    ctx = str(ref.get("context") or "")

    for term in _query_terms(q):
        if term in blob:
            score += min(24, len(term) * 4)

    for anchor in anchor_phrases or []:
        if anchor in blob:
            score += min(40, len(anchor) * 2)

    if len(q) >= 4 and q in blob:
        score += 20

    if source_hints:
        path = str(ref.get("path") or "")
        for hint in source_hints:
            if hint and hint in path:
                score += 15

    if retrieved_text:
        ranked = _ranked_retrieval_lines(q, retrieved_text, limit=3)
        if ranked:
            ref_text = " ".join(
                str(ref.get(key) or "")
                for key in ("context", "caption", "label")
            ).strip()
            if ref_text:
                score += int(_text_alignment(ref_text, ranked[0][1]) * 50)
                label = _ref_effective_label(ref)
                if label:
                    score += int(_text_alignment(label, ranked[0][1]) * 40)
            ref_sections = _section_ids_in_text(ctx)
            for overlap, line in ranked:
                line_sid = _section_id_from_line_or_context(retrieved_text, line)
                if not line_sid:
                    continue
                if ref_sections and not any(
                    _sections_compatible(line_sid, rs) for rs in ref_sections
                ):
                    score -= 180
                    break
                if not ref_sections or any(
                    _sections_compatible(line_sid, rs) for rs in ref_sections
                ):
                    score += int(overlap * 90)
                    break

    label = _ref_effective_label(ref)
    if _is_section_number_heading(str(ref.get("caption") or "")):
        score -= 45
    if _is_generic_cycle_only_label(label) and not _ref_matches_figure_focus(q, ref):
        score -= 120
    if label and _figure_label_matches_query(q, label):
        score += 40
    if retrieved_text and label:
        for overlap, line in _ranked_retrieval_lines(q, retrieved_text, limit=5):
            if overlap < 0.12:
                break
            if label in line or _ref_label_aligns_with_line(ref, line):
                score += 75
                break
    if retrieved_text:
        targets = _listing_target_phrases(q, retrieved_text)
        if len(targets) >= 2:
            for target in targets:
                if _label_matches_listing_target(label, target):
                    score += 85
                    break
    for term in _query_terms(q):
        if len(term) >= 3 and term in ctx:
            score += min(28, len(term) * 5)

    return max(0, score)


def extract_image_refs_from_context(context: str) -> list[dict[str, Any]]:
    """Parse image metadata from retrieved LightRAG context text."""
    context = normalize_context_for_image_parse(context)
    if not context.strip():
        return []

    refs: list[dict[str, Any]] = []
    seen: set[str] = set()

    for m in _IMAGE_PATH_RE.finditer(context):
        path = normalize_image_path(m.group(1) or m.group(2) or "")
        if not path:
            continue
        dedupe_key = Path(path).name
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        block = _image_block_for_path(context, m.start())
        meta = _metadata_from_block(block)
        page_idx = meta.get("page")
        if not meta.get("caption") and page_idx != 0:
            heading = _heading_before_image_block(context, m.start())
            if heading:
                meta["caption"] = heading
                if not meta.get("label"):
                    meta["label"] = heading
        elif not meta.get("label") and meta.get("caption"):
            meta["label"] = meta["caption"]
        ref = {"path": path, **meta}
        _enrich_ref_from_image_block(ref, block)
        refs.append(ref)

    return refs


def _merge_refs(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for ref in group:
            key = Path(str(ref.get("path") or "")).name
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(ref)
    return merged


# --- LEGACY (disabled): query-time bbox supplement helpers — see EOF ---


def text_from_retrieved_docs(docs: list[dict[str, Any]] | None) -> str:
    """Concatenate chunk bodies returned by LightRAG rerank / retrieval."""
    parts: list[str] = []
    for doc in docs or []:
        if not isinstance(doc, dict):
            continue
        for key in ("content", "text", "chunk_content", "page_content"):
            val = doc.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
                break
    return "\n\n".join(parts)


def merge_context_for_images(*sources: str | None) -> str:
    parts = [s.strip() for s in sources if isinstance(s, str) and s.strip()]
    return "\n\n".join(parts)


def resolve_media_path(path_str: str, media_roots: list[Path]) -> Path | None:
    """Resolve image path; fall back to filename search under parser output (re-ingest safe)."""
    path = Path(normalize_image_path(path_str))
    if is_safe_media_path(path, media_roots):
        return path.resolve()
    name = path.name
    if not name:
        return None
    for root in media_roots:
        try:
            for candidate in root.rglob(name):
                if is_safe_media_path(candidate, media_roots):
                    return candidate.resolve()
        except OSError:
            continue
    return None


def is_safe_media_path(path: Path, allowed_roots: list[Path]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if not resolved.is_file():
        return False
    if resolved.suffix.lower() not in _IMAGE_EXTS:
        return False
    for root in allowed_roots:
        try:
            root_res = root.resolve()
            if resolved.is_relative_to(root_res):
                return True
        except OSError:
            continue
    return False


def encode_media_token(abs_path: Path, media_root: Path) -> str:
    rel = abs_path.resolve().relative_to(media_root.resolve())
    raw = str(rel).replace("\\", "/").encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_media_token(token: str, media_root: Path) -> Path | None:
    if not token or ".." in token or token.startswith("/"):
        return None
    try:
        pad = "=" * (-len(token) % 4)
        rel = base64.urlsafe_b64decode(token + pad).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    if ".." in Path(rel).parts:
        return None
    return (media_root.resolve() / rel).resolve()


def _select_scored_refs_for_listing(
    scored: list[tuple[int, dict[str, Any]]],
    *,
    query: str,
    retrieved_text: str,
    limit: int,
    min_absolute: int = 12,
) -> list[dict[str, Any]]:
    """One figure per listing target phrase (e.g. three glue-cleaning procedures)."""
    targets = _listing_target_phrases(query, retrieved_text)
    if not targets:
        return _select_scored_refs_by_label(
            scored, limit=limit, min_absolute=min_absolute
        )

    selected: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    for target in targets:
        best_score = 0
        best_ref: dict[str, Any] | None = None
        for score, ref in scored:
            if score < min_absolute:
                continue
            label = _ref_effective_label(ref)
            if not _label_matches_listing_target(label, target):
                continue
            key = _figure_label_key(ref)
            if key in used_keys:
                continue
            if score > best_score:
                best_score = score
                best_ref = ref
        if best_ref is not None:
            used_keys.add(_figure_label_key(best_ref))
            selected.append(best_ref)
        if len(selected) >= limit:
            break
    if selected:
        return selected
    return _select_scored_refs_by_label(
        scored, limit=limit, min_absolute=min_absolute
    )


def _select_scored_refs_cross_manual(
    scored: list[tuple[int, dict[str, Any]]],
    *,
    limit: int,
    min_absolute: int = 12,
) -> list[dict[str, Any]]:
    """One on-topic figure per ingested manual for multi-handbook comparison answers."""
    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    for score, ref in sorted(scored, key=lambda pair: pair[0], reverse=True):
        if score < min_absolute:
            continue
        source = _source_key_from_path(str(ref.get("path") or ""))
        if source in used_sources:
            continue
        used_sources.add(source)
        selected.append(ref)
        if len(selected) >= limit:
            break
    return selected


def _select_scored_refs_by_label(
    scored: list[tuple[int, dict[str, Any]]],
    *,
    limit: int,
    min_absolute: int = 12,
) -> list[dict[str, Any]]:
    """One figure per distinct caption (listing queries, same source document)."""
    best_by_label: dict[str, tuple[int, dict[str, Any]]] = {}
    for score, ref in scored:
        if score <= 0:
            continue
        key = _figure_label_key(ref) or Path(str(ref.get("path") or "")).name
        prev = best_by_label.get(key)
        if prev is None or score > prev[0]:
            best_by_label[key] = (score, ref)

    candidates = sorted(best_by_label.values(), key=lambda pair: pair[0], reverse=True)
    selected: list[dict[str, Any]] = []
    for score, ref in candidates:
        if score < min_absolute:
            continue
        selected.append(ref)
        if len(selected) >= limit:
            break
    return selected


def _select_scored_refs(
    scored: list[tuple[int, dict[str, Any]]],
    *,
    limit: int,
    query: str | None = None,
    retrieved_text: str | None = None,
    min_relative: float = 0.45,
    min_absolute: int = 20,
) -> list[dict[str, Any]]:
    if not scored:
        return []
    scored = [(score, ref) for score, ref in scored if score > 0]
    if not scored:
        return []

    distinct_sources = {
        _source_key_from_path(str(ref.get("path") or ""))
        for score, ref in scored
        if score > 0
    }
    if len(distinct_sources) >= 2:
        cap = min(limit, _multi_figure_image_limit())
        return _select_scored_refs_cross_manual(
            scored, limit=cap, min_absolute=min(12, min_absolute)
        )

    targets = _listing_target_phrases(query or "", retrieved_text or "")
    if len(targets) >= 2:
        cap = _multi_figure_image_limit()
        if retrieved_text:
            return _select_scored_refs_for_listing(
                scored,
                query=query or "",
                retrieved_text=retrieved_text,
                limit=cap,
                min_absolute=min(12, min_absolute),
            )
        return _select_scored_refs_by_label(
            scored,
            limit=cap,
            min_absolute=min(12, min_absolute),
        )

    best_by_source: dict[str, tuple[int, dict[str, Any]]] = {}
    for score, ref in scored:
        source = _source_key_from_path(str(ref.get("path") or ""))
        prev = best_by_source.get(source)
        if prev is None or score > prev[0]:
            best_by_source[source] = (score, ref)

    candidates = sorted(best_by_source.values(), key=lambda pair: pair[0], reverse=True)
    max_score = candidates[0][0]
    threshold = max(min_absolute, int(max_score * min_relative))

    selected: list[dict[str, Any]] = []
    for score, ref in candidates:
        if score < min_absolute:
            continue
        if not selected or score >= threshold:
            selected.append(ref)
        if len(selected) >= limit:
            break
    return selected


def _summarize_ref(ref: dict[str, Any]) -> dict[str, Any]:
    path = str(ref.get("path") or "")
    return {
        "path": path,
        "page": ref.get("page"),
        "caption": ref.get("caption"),
        "label": ref.get("label"),
        "context": _truncate_ref_text(str(ref.get("context") or ""), 240),
    }


def _truncate_ref_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def explain_retrieval_supports_images(
    query: str | None,
    *,
    retrieved_docs: list[dict[str, Any]] | None = None,
    context_text: str | None = None,
    media_roots: list[Path] | None = None,
    focus_text: str | None = None,
) -> dict[str, Any]:
    """Explain image gate decisions for debug dumps."""
    del focus_text  # legacy kwarg; primary-only pipeline
    q = (query or "").strip()
    if not query_wants_kb_images(query):
        return {"ok": False, "reason": "non_kb_query"}

    primary = (context_text or "").strip()
    if not primary:
        return {"ok": False, "reason": "empty_retrieval_context"}
    if not _eligible_figure_refs(primary):
        return {
            "ok": False,
            "reason": "no_inline_figures_in_primary",
            "primary_chars": len(primary),
            "listing_in_primary": len(
                _listing_targets_with_query_line_overlap(q, primary, primary)
            ),
        }

    max_score = _max_rerank_score(retrieved_docs)
    ok = retrieval_supports_images(
        query,
        retrieved_docs=retrieved_docs,
        context_text=primary,
        media_roots=media_roots,
    )
    if ok:
        overlap = _term_overlap_ratio(q, primary)
        reason = "rerank_score_ok" if max_score is not None else "aligned_figures_in_primary"
        payload: dict[str, Any] = {
            "ok": True,
            "reason": reason,
            "primary_chars": len(primary),
        }
        if max_score is not None:
            payload["max_rerank_score"] = max_score
        else:
            payload["overlap"] = overlap
        return payload

    if max_score is not None and max_score < _image_min_rerank_score():
        return {
            "ok": False,
            "reason": "low_rerank_score",
            "max_rerank_score": max_score,
            "threshold": _image_min_rerank_score(),
        }
    overlap = _term_overlap_ratio(q, primary)
    return {
        "ok": False,
        "reason": "no_query_label_match",
        "overlap": overlap,
        "primary_chars": len(primary),
    }


def explain_query_images(
    context: str,
    media_roots: list[Path],
    *,
    query: str | None = None,
    extra_context: str | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
    limit: int = 4,
) -> dict[str, Any]:
    """Full image pipeline trace for debug dumps (no side effects)."""
    primary_text = (context or "").strip()
    scan_text, anchor_scan = _context_for_image_scan(
        query or "", primary_text, retrieved_docs
    )
    figure_context = (
        primary_text if anchor_scan.get("mode") == "off" else scan_text
    )
    gate = explain_retrieval_supports_images(
        query,
        retrieved_docs=retrieved_docs,
        context_text=primary_text,
        media_roots=media_roots,
    )
    debug: dict[str, Any] = {
        "gate": gate,
        "primary_text_preview": _truncate_ref_text(primary_text, 600),
        "anchor_scan": anchor_scan,
        "figure_context_preview": _truncate_ref_text(figure_context, 600),
        "anchor_phrases": [],
        "top_retrieval_lines": [],
        "refs_from_context": 0,
        "refs_from_supplement": 0,
        "refs_from_listing": 0,
        "refs_merged": 0,
        "refs_after_align": [],
        "refs_dropped_align": [],
        "scored": [],
        "selected_paths": [],
    }
    if not gate.get("ok"):
        return debug

    anchor_phrases = _extract_context_anchors(query, primary_text)
    debug["anchor_phrases"] = anchor_phrases[:12]
    debug["top_retrieval_lines"] = [
        {"overlap": overlap, "line": line[:200]}
        for overlap, line in _ranked_retrieval_lines(query or "", primary_text, limit=6)
    ]

    source_hints = _merged_source_hints(primary_text, retrieved_docs)
    debug["source_hints"] = sorted(source_hints)[:8]
    debug["query_subject_needles"] = _query_subject_needles(query or "")

    from_context = extract_image_refs_from_context(figure_context)
    debug["refs_from_context"] = len(from_context)
    debug["listing_targets"] = _listing_target_phrases(query or "", primary_text)

    refs = from_context
    debug["refs_merged"] = len(refs)
    debug["refs_cover_filtered"] = sum(
        1 for ref in refs if _is_cover_page_ref(ref)
    )

    aligned_refs: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for ref in refs:
        summary = _summarize_ref(ref)
        if _is_cover_page_ref(ref):
            dropped.append({**summary, "drop_reason": "cover_page"})
            continue
        if _ref_passes_image_align_gate(
            query or "",
            ref,
            retrieved_text=figure_context,
            source_hints=source_hints,
            listing_source_text=primary_text,
            retrieved_docs=retrieved_docs,
        ):
            aligned_refs.append(ref)
        else:
            dropped.append({**summary, "drop_reason": "query_label_mismatch"})
    debug["refs_after_align"] = [_summarize_ref(ref) for ref in aligned_refs]
    debug["refs_dropped_align"] = dropped

    if not aligned_refs:
        debug["gate"] = {"ok": False, "reason": "no_query_label_match"}
        return debug

    if source_hints:
        aligned_refs = [
            ref
            for ref in aligned_refs
            if _ref_matches_source_hints(ref, source_hints)
        ]
    scored_pairs = [
        (
            _score_ref_for_query(
                ref,
                query,
                anchor_phrases=anchor_phrases,
                source_hints=source_hints,
                retrieved_text=figure_context,
            ),
            ref,
        )
        for ref in aligned_refs
    ]
    debug["scored"] = [
        {"score": score, **_summarize_ref(ref)} for score, ref in scored_pairs
    ]
    selected = _select_scored_refs(
        scored_pairs, limit=limit, query=query, retrieved_text=figure_context
    )
    debug["selected_paths"] = [str(ref.get("path") or "") for ref in selected]
    return debug


def images_for_api(
    context: str,
    media_roots: list[Path],
    *,
    query: str | None = None,
    extra_context: str | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Turn rerank-filtered primary context into Web-safe image descriptors."""
    primary_text = (context or "").strip()
    if not primary_text:
        return []

    if not retrieval_supports_images(
        query,
        retrieved_docs=retrieved_docs,
        context_text=primary_text,
        media_roots=media_roots,
    ):
        return []

    scan_text, anchor_scan = _context_for_image_scan(
        query or "", primary_text, retrieved_docs
    )
    figure_context = (
        primary_text if anchor_scan.get("mode") == "off" else scan_text
    )
    if anchor_scan.get("mode") != "off" and not figure_context.strip():
        logger.info("Skip related images: anchor sections have no inline figures")
        return []

    anchor_phrases = _extract_context_anchors(query, primary_text)
    source_hints = _merged_source_hints(primary_text, retrieved_docs)

    refs = _refs_from_retrieved_docs_text(
        figure_context,
        media_roots,
        query=query,
        retrieved_docs=retrieved_docs,
        full_context=primary_text,
    )
    refs = [
        ref
        for ref in refs
        if not _is_cover_page_ref(ref)
        and _ref_passes_image_align_gate(
            query or "",
            ref,
            retrieved_text=figure_context,
            source_hints=source_hints,
            listing_source_text=primary_text,
            retrieved_docs=retrieved_docs,
        )
    ]
    if not refs:
        logger.info("Skip related images: no figure aligns with primary context")
        return []

    scored = [
        (
            _score_ref_for_query(
                ref,
                query,
                anchor_phrases=anchor_phrases,
                source_hints=source_hints,
                retrieved_text=figure_context,
            ),
            ref,
        )
        for ref in refs
    ]
    selected = _select_scored_refs(
        scored,
        limit=limit,
        query=query,
        retrieved_text=figure_context,
    )

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for ref in selected:
        path_str = ref.get("path") or ""
        path_key = Path(path_str).name
        if not path_str or path_key in seen:
            continue
        resolved = resolve_media_path(path_str, media_roots)
        if resolved is None:
            continue
        root_for_token: Path | None = None
        for root in media_roots:
            try:
                if resolved.is_relative_to(root.resolve()):
                    root_for_token = root.resolve()
                    break
            except OSError:
                continue
        if root_for_token is None:
            continue
        seen.add(path_key)
        token = encode_media_token(resolved, root_for_token)
        caption = _ref_effective_label(ref) or ref.get("caption") or ""
        context_snippet = ref.get("context") or ""
        if not caption and context_snippet:
            caption = context_snippet[:60]
        item: dict[str, Any] = {
            "url": f"/api/media/image?token={token}",
            "caption": caption,
        }
        if ref.get("page") is not None:
            item["page"] = ref["page"]
        if context_snippet:
            item["context"] = context_snippet
        out.append(item)

    if out:
        logger.info(
            "Resolved %d related image(s) for query (%d candidate path(s))",
            len(out),
            len(refs),
        )
    return out


def resolve_query_images(
    context: str,
    media_roots: list[Path],
    *,
    query: str | None = None,
    extra_context: str | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
    limit: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authoritative query-time images + debug (Web finalize and batch test)."""
    primary_text = (context or "").strip()
    debug = explain_query_images(
        primary_text,
        media_roots,
        query=query,
        extra_context=extra_context,
        retrieved_docs=retrieved_docs,
        limit=limit,
    )
    if not (debug.get("gate") or {}).get("ok"):
        return [], debug
    images = images_for_api(
        primary_text,
        media_roots,
        query=query,
        extra_context=extra_context,
        retrieved_docs=retrieved_docs,
        limit=limit,
    )
    return images, debug

# =============================================================================
# LEGACY — NOT ON MAIN PIPELINE (Route A: ingest coalesce + primary-only finalize)
#
# focus_text_for_images, _focus_qualifies_*, supplement_refs_from_content_lists,
# supplement_refs_for_listing_targets, and private helpers below.
#
# Replaced by: coalesce_text_image_segments at ingest; finalize parses inline
# [图片] from rerank-filtered primary_text only.
#
# TODO(remove): delete this entire block after production validation.
# =============================================================================
if False:  # preserved reference source — never executed
    from raganything.utils import best_image_for_text_item  # noqa: F401
    def _focus_has_query_anchored_inline_figures(query: str, focus: str) -> bool:
        """True when a ``[图片]`` block lies in the span of a high-overlap focus line."""
        text = normalize_context_for_image_parse(focus).strip()
        if not text:
            return False
        min_ov = _image_focus_min_overlap()
        ranked = _ranked_retrieval_lines(query, text, limit=12)
        if not ranked:
            return bool(extract_image_refs_from_context(text))
        tail = _image_focus_expand_tail_chars()
        for overlap, line in ranked:
            if overlap < min_ov:
                continue
            if not _line_has_specific_query_overlap(query, line, ""):
                continue
            idx = text.find(line)
            if idx < 0:
                continue
            window = text[idx : min(len(text), idx + len(line) + tail)]
            if "[图片]" not in window and not _IMAGE_PATH_RE.search(window):
                continue
            for ref in extract_image_refs_from_context(window):
                label = _ref_effective_label(ref)
                if _figure_label_matches_query(query, label) or _topic_overlaps_query(
                    label, query
                ):
                    return True
        return False

    def _focus_has_query_aligned_figure_refs(query: str, focus: str) -> bool:
        """Focus body already includes a query-aligned ``[图片]`` caption (single-topic procedure)."""
        text = normalize_context_for_image_parse(focus).strip()
        if not text or "[图片]" not in text:
            return False
        for ref in extract_image_refs_from_context(text):
            label = _ref_effective_label(ref)
            if _figure_label_matches_query(query, label) or _topic_overlaps_query(
                label, query
            ):
                return True
        return False

    def _focus_qualifies_for_image_lookup(
        query: str, full_text: str, focus: str
    ) -> bool:
        """Multi-topic listing in focus, or inline figures adjacent to query-aligned lines."""
        q = (query or "").strip()
        full = (full_text or "").strip()
        focus_norm = normalize_context_for_image_parse(focus).strip()
        if not focus_norm:
            return False
        if len(_listing_targets_with_query_line_overlap(q, full, focus_norm)) >= 2:
            return True
        if _focus_has_query_aligned_figure_refs(q, focus_norm):
            return True
        return _focus_has_query_anchored_inline_figures(q, focus_norm)

    def _image_focus_min_overlap() -> float:
        raw = (
            os.getenv("RAG_IMAGE_FOCUS_MIN_OVERLAP")
            or os.getenv("RAG_IMAGE_RETRIEVAL_FOCUS_MIN_OVERLAP")
            or "0.12"
        )
        try:
            return max(0.05, min(0.5, float(raw)))
        except ValueError:
            return 0.12

    def _image_focus_expand_tail_chars() -> int:
        raw = os.getenv("RAG_IMAGE_FOCUS_EXPAND_TAIL") or "600"
        try:
            return max(120, min(2000, int(raw)))
        except ValueError:
            return 600

    def _image_focus_max_lines() -> int:
        raw = os.getenv("RAG_IMAGE_FOCUS_MAX_LINES") or "10"
        try:
            return max(2, min(24, int(raw)))
        except ValueError:
            return 10

    def _image_focus_max_chars() -> int:
        raw = os.getenv("RAG_IMAGE_FOCUS_MAX_CHARS") or "6000"
        try:
            return max(800, min(20000, int(raw)))
        except ValueError:
            return 6000

    def _build_focus_context_from_lines(full: str, focus_lines: list[str]) -> str:
        """Merge spans around focus lines and keep adjacent ``[图片]`` blocks."""
        if not focus_lines:
            return ""
        tail = _image_focus_expand_tail_chars()
        spans: list[tuple[int, int]] = []
    
        for fl in focus_lines:
            start = 0
            while True:
                idx = full.find(fl, start)
                if idx < 0:
                    break
                span_start = full.rfind("\n\n", 0, idx)
                span_start = 0 if span_start < 0 else span_start + 2
                span_end = idx + len(fl)
                search_from = span_end
                for _ in range(4):
                    img = full.find("[图片]", search_from)
                    if img < 0 or img > span_end + tail:
                        break
                    block_end = full.find("\n\n", img)
                    if block_end < 0:
                        block_end = min(len(full), img + tail)
                    else:
                        block_end = min(len(full), block_end + 2)
                    span_end = max(span_end, block_end)
                    search_from = span_end
                next_break = full.find("\n\n", span_end)
                if next_break > 0:
                    span_end = min(len(full), max(span_end, next_break + 2))
                else:
                    span_end = min(len(full), span_end + 200)
                spans.append((span_start, span_end))
                start = idx + max(1, len(fl) // 2)
    
        if not spans:
            return "\n\n".join(focus_lines)
    
        spans.sort()
        merged: list[tuple[int, int]] = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        parts = [full[s:e].strip() for s, e in merged]
        out = "\n\n".join(p for p in parts if p)
        cap = _image_focus_max_chars()
        if len(out) > cap:
            return out[:cap]
        return out

    def focus_text_for_images(query: str, full_text: str) -> str:
        """Query-aligned retrieval lines (+ nearby ``[图片]`` blocks) used only for image lookup."""
        full = normalize_context_for_image_parse(full_text).strip()
        if not full:
            return ""
        q = (query or "").strip()
        if not q:
            return ""
    
        min_ov = _image_focus_min_overlap()
        max_lines = _image_focus_max_lines()
        ranked = _ranked_retrieval_lines(q, full, limit=max_lines * 3)
        focus_lines: list[str] = []
        seen_line: set[str] = set()
        catalog_mode = _retrieval_prefers_catalog_field(q, full)
        top_overlap = ranked[0][0] if ranked else 0.0
        sparse_overlap = top_overlap < min_ov
        if sparse_overlap and top_overlap > 0:
            # Long questions dilute term-overlap ratio; keep top relative lines with subject hits.
            rel_floor = top_overlap * 0.85
        else:
            rel_floor = max(min_ov, top_overlap * 0.62)
    
        def add_line(line: str) -> None:
            line = line.strip()
            if len(line) < _min_substantive_term_len():
                return
            if line in seen_line:
                return
            if catalog_mode and _CATALOG_MODEL_MARKER not in line:
                return
            if len(focus_lines) >= max_lines:
                return
            seen_line.add(line)
            focus_lines.append(line)
    
        if catalog_mode:
            for overlap, line in ranked:
                if overlap >= min_ov and _CATALOG_MODEL_MARKER in line:
                    add_line(line)
        else:
            for overlap, line in ranked:
                if overlap < rel_floor:
                    continue
                if sparse_overlap and not (
                    _line_has_query_subject_hit(q, line)
                    or _line_has_specific_query_overlap(q, line, "")
                ):
                    continue
                add_line(line)
    
        targets = _listing_target_phrases(q, full)
        listing_budget = max_lines * 2 if len(targets) >= 2 else max_lines
        if len(targets) >= 2 and not catalog_mode:
            for line in re.split(r"[\n\r]+", full):
                line = line.strip()
                if len(line) < 4 or len(focus_lines) >= listing_budget:
                    continue
                if any(
                    target in line or _label_matches_listing_target(line, target)
                    for target in targets
                ):
                    add_line(line)
    
        if not focus_lines:
            return ""
    
        if catalog_mode:
            return "\n\n".join(focus_lines)
    
        return _build_focus_context_from_lines(full, focus_lines)

    def supplement_refs_for_listing_targets(
        text: str,
        media_roots: list[Path],
        *,
        query: str,
        source_hints: set[str] | None = None,
        listing_source_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """Attach figures for each distinct listing target (same manual, multiple captions)."""
        focus = (text or "").strip()
        source = (listing_source_text or focus).strip()
        targets = _listing_targets_with_query_line_overlap(query, source, focus)
        if len(targets) < 2:
            return []
    
        from raganything.utils import image_label_for_item  # noqa: WPS433
    
        refs: list[dict[str, Any]] = []
        seen: set[str] = set()
        hints = source_hints or _source_hints_from_text(focus)
    
        for root in media_roots:
            try:
                content_lists = list(root.rglob("*_content_list.json"))
            except OSError:
                continue
            for cl_path in content_lists:
                items = _load_content_list_items(cl_path)
                if not items:
                    continue
                auto_dir = cl_path.parent
                doc_hint = cl_path.stem.replace("_content_list", "").replace(
                    "_content_list_v2", ""
                )
                if hints and not any(hint in doc_hint for hint in hints):
                    continue
                for image_item in items:
                    if not isinstance(image_item, dict) or image_item.get("type") != "image":
                        continue
                    label = image_label_for_item(items, image_item)
                    if not label:
                        continue
                    if not any(
                        _label_matches_listing_target(label, target) for target in targets
                    ):
                        continue
                    _append_content_list_image_ref(
                        refs,
                        seen,
                        items=items,
                        image_item=image_item,
                        auto_dir=auto_dir,
                        caption=label,
                    )
        return refs

    def _load_content_list_items(path: Path) -> list[dict[str, Any]] | None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            return raw[0]
        if isinstance(raw, list):
            return raw
        return None

    def _append_content_list_image_ref(
        refs: list[dict[str, Any]],
        seen: set[str],
        *,
        items: list[dict[str, Any]],
        image_item: dict[str, Any],
        auto_dir: Path,
        caption: str,
        context: str = "",
    ) -> None:
        from raganything.utils import image_label_for_item  # noqa: WPS433
    
        rel_path = (image_item.get("img_path") or "").strip()
        if not rel_path:
            return
        full_path = (auto_dir / rel_path).resolve()
        key = full_path.name
        if key in seen:
            return
        seen.add(key)
        page_idx = image_item.get("page_idx")
        label = (caption or image_label_for_item(items, image_item) or "").strip()
        ref = {
            "path": str(full_path),
            "page": page_idx if isinstance(page_idx, int) else None,
            "caption": label,
            "label": label,
            "context": context.strip()[:300],
        }
        if label:
            _preserve_source_figure_labels(ref)
        else:
            _enrich_ref_from_image_block(ref, context)
        refs.append(ref)

    def supplement_refs_from_content_lists(
        text: str,
        media_roots: list[Path],
        *,
        query: str | None = None,
        source_hints: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """BBox-link images only when an anchor phrase appears in rerank-filtered chunk text."""
        anchors = _extract_context_anchors(query, text)
        if not anchors:
            anchors = [t for t in _query_terms(query or "") if len(t) >= 4][:6]
    
        refs: list[dict[str, Any]] = []
        seen: set[str] = set()
        q = (query or "").strip()
        hints = source_hints or _source_hints_from_text(text)
    
        for root in media_roots:
            try:
                content_lists = list(root.rglob("*_content_list.json"))
            except OSError:
                continue
            for cl_path in content_lists:
                items = _load_content_list_items(cl_path)
                if not items:
                    continue
                auto_dir = cl_path.parent
                doc_hint = cl_path.stem.replace("_content_list", "").replace("_content_list_v2", "")
                if hints and not any(hint in doc_hint for hint in hints):
                    continue
    
                for anchor in anchors:
                    for text_idx, item in enumerate(items):
                        if item.get("type") != "text":
                            continue
                        body = item.get("text")
                        if not isinstance(body, str) or anchor not in body:
                            continue
                        image_item = best_image_for_text_item(items, text_idx)
                        if image_item is None:
                            continue
                        from raganything.utils import image_label_for_item  # noqa: WPS433
    
                        label = image_label_for_item(items, image_item) or anchor[:80]
                        _append_content_list_image_ref(
                            refs,
                            seen,
                            items=items,
                            image_item=image_item,
                            auto_dir=auto_dir,
                            caption=label,
                            context=body.strip(),
                        )
        return refs
