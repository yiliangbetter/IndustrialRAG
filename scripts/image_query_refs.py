"""Query-time image resolution for RAG Q&A (Plan B).

Ingest co-locates ``[图片]`` blocks with anchor text in vector chunks (Route A).
This module parses inline refs from LLM-input chunk bodies at finalize (Plan A).
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
    "build_inline_placements",
    "images_for_api",
    "resolve_query_images",
    "default_image_selection_limit",
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


def default_image_selection_limit() -> int:
    """Global cap for selected figures; ``0`` env means no practical cap (24)."""
    raw = os.getenv("RAG_IMAGE_MULTI_LIMIT") or "0"
    try:
        val = int(raw)
    except ValueError:
        return 24
    if val <= 0:
        return 24
    return max(1, min(24, val))


def _multi_figure_image_limit() -> int:
    return default_image_selection_limit()


def _inline_min_place_score() -> float:
    raw = os.getenv("RAG_IMAGE_INLINE_MIN_PLACE_SCORE") or "0.35"
    try:
        return float(raw)
    except ValueError:
        return 0.35


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


def _answer_text_for_listing() -> str:
    try:
        from query_progress_hooks import get_answer_text_for_images  # noqa: WPS433

        return (get_answer_text_for_images() or "").strip()
    except ImportError:
        return ""


def _listing_target_head(span: str) -> str:
    """Component name before generic qualifiers (e.g. 涂胶轴及其周围 → 涂胶轴)."""
    span = (span or "").strip().rstrip("：:")
    for sep in ("及其", "及"):
        if sep in span:
            span = span.split(sep, 1)[0].strip()
    return span


def _answer_listing_spans(answer: str) -> list[str]:
    """Component headers from answer markdown (e.g. **压带轮** / ### 压带轮)."""
    spans: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        span = _listing_target_head(raw)
        key = _normalize_label_key(span)
        if len(span) >= 2 and len(span) <= 24 and key not in seen:
            seen.add(key)
            spans.append(span)

    for match in re.finditer(r"\*\*([^*]{2,32})\*\*", answer or ""):
        add(match.group(1).strip())

    if len(spans) >= 2:
        return spans

    for line in (answer or "").splitlines():
        line = line.strip()
        m = re.match(r"^#{1,4}\s+(.+?)\s*$", line)
        if not m:
            m = re.match(r"^[-*•]\s*\*\*([^*]{2,32})\*\*", line)
        if not m:
            m = re.match(r"^[-*•]\s*(\S{2,16})[：:]", line)
        if not m:
            continue
        head = m.group(1).strip()
        if re.search(r"[。；;，,]|参考|Reference|PDF", head, re.I):
            continue
        if "[" in head or "]" in head:
            continue
        add(head)
    return spans


def _listing_spans_match_figure_labels(spans: list[str], text: str) -> list[str]:
    if not spans or not (text or "").strip():
        return []
    labels = [
        lab
        for ref in extract_image_refs_from_context(text)
        if (lab := _ref_effective_label(ref))
    ]
    matched: list[str] = []
    seen: set[str] = set()
    for span in spans:
        key = _normalize_label_key(span)
        if key in seen:
            continue
        if any(_label_matches_listing_target(lab, span) for lab in labels):
            seen.add(key)
            matched.append(span)
            continue
        if span in text:
            seen.add(key)
            matched.append(span)
    return matched


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


def _machine_spans_from_answer(answer: str) -> list[str]:
    return [
        span
        for span in _answer_listing_spans(_answer_primary_listing_body(answer))
        if "封边机" in span or "加工中心" in span
    ]


def _answer_primary_listing_body(answer: str) -> str:
    """Answer body for listing spans; drop trailing digressions (e.g. 此外…)."""
    text = (answer or "").strip()
    text = _ANSWER_REF_RE.sub("", text)
    for marker in ("此外", "另外", "同时", "除此之外"):
        match = re.search(rf"(?:^|\n)\s*{marker}", text)
        if match:
            text = text[: match.start()]
    return text.strip()


def _component_spans_from_answer(answer: str) -> list[str]:
    return [
        span
        for span in _answer_listing_spans(_answer_primary_listing_body(answer))
        if "封边机" not in span and "加工中心" not in span
    ]


def _is_component_listing_across_machines(query: str) -> bool:
    """Cross-manual questions listing parts/components (e.g. Q17), not periods per line."""
    q = (query or "").strip()
    if not q:
        return False
    return bool(
        re.search(r"部件|零件|组件", q)
        and re.search(r"哪些|有什么|有哪|各自", q)
    )


def _answer_image_span_targets(query: str, answer: str) -> list[str]:
    """One figure per answer item: component names for part listings, machine lines for Q15-style."""
    components = _component_spans_from_answer(answer)
    machines = _machine_spans_from_answer(answer)
    if _is_multi_machine_comparison_query(query):
        if _is_component_listing_across_machines(query) and len(components) >= 2:
            return components
        if len(machines) >= 2:
            return machines
    if len(components) >= 2:
        return components
    spans = _answer_listing_spans(_answer_primary_listing_body(answer))
    if len(spans) >= 2:
        return spans
    return _answer_section_topics(answer)


def _order_listing_targets_by_hints(
    targets: list[str],
    hints: list[str],
) -> list[str]:
    """Reorder pool-derived listing targets using answer item order as a hint only."""
    if not targets or not hints:
        return targets
    ordered: list[str] = []
    used: set[str] = set()
    for hint in hints:
        head = _listing_target_head(hint)
        hint_key = _normalize_label_key(head)
        for topic in targets:
            tkey = _normalize_label_key(topic)
            if tkey in used:
                continue
            if (
                hint_key in tkey
                or tkey in hint_key
                or _label_matches_listing_target(topic, head)
                or _label_matches_listing_target(head, topic)
            ):
                ordered.append(topic)
                used.add(tkey)
    for topic in targets:
        tkey = _normalize_label_key(topic)
        if tkey not in used:
            ordered.append(topic)
    return ordered


def _topic_has_figure_support_in_pool(
    topic: str,
    pool: list[dict[str, Any]],
) -> bool:
    for doc in pool:
        content = _doc_content(doc).strip()
        if not content or not extract_image_refs_from_context(content):
            continue
        labels = [
            lab
            for ref in extract_image_refs_from_context(content)
            if (lab := _ref_effective_label(ref))
        ]
        if any(_label_matches_listing_target(lab, topic) for lab in labels):
            return True
        if _chunk_matches_answer_topic(content, topic):
            return True
    return False


def _topic_supported_by_kept_chunks(
    topic: str,
    kept: list[dict[str, Any]],
) -> bool:
    for doc in kept:
        content = _doc_content(doc).strip()
        if not content:
            continue
        if topic in content or _chunk_matches_answer_topic(content, topic):
            return True
        for ref in extract_image_refs_from_context(content):
            lab = _ref_effective_label(ref)
            if lab and _label_matches_listing_target(lab, topic):
                return True
    return False


def _listing_topics_from_pool(
    query: str,
    pool: list[dict[str, Any]],
) -> list[str]:
    """``保养内容：`` topics and figure labels in pool that overlap the query."""
    q = (query or "").strip()
    if not q or not pool:
        return []
    pool_text = text_from_retrieved_docs(pool)
    topics: list[str] = []
    seen: set[str] = set()

    def add(topic: str) -> None:
        topic = topic.strip()
        if len(topic) < 4:
            return
        key = _normalize_label_key(topic)
        if key in seen:
            return
        seen.add(key)
        topics.append(topic)

    for topic in _maintenance_topics_in_text(pool_text, q):
        add(topic)
    for doc in pool:
        content = _doc_content(doc).strip()
        for ref in extract_image_refs_from_context(content):
            lab = _ref_effective_label(ref)
            if lab and len(lab) >= 4 and _topic_overlaps_query(lab, q):
                add(lab)
    return [t for t in topics if _topic_has_figure_support_in_pool(t, pool)]


def _span_keep_listing_targets(
    query: str,
    answer: str,
    pool: list[dict[str, Any]],
    *,
    kept: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Span targets for span_keep: pool/query maintenance entries first; answer bold for order."""
    q = (query or "").strip()
    kept_docs = list(kept or [])
    if _is_multi_machine_comparison_query(q) and not _is_component_listing_across_machines(
        q
    ):
        machines = _machine_spans_from_answer(answer)
        if len(machines) >= 2:
            return machines
    if _is_listing_scope_query(q) and not _is_catalog_or_model_listing_query(q):
        topics = _listing_topics_from_pool(q, pool)
        if kept_docs:
            anchored = [
                t for t in topics if _topic_supported_by_kept_chunks(t, kept_docs)
            ]
            if len(anchored) >= 2:
                topics = anchored
        if len(topics) >= 2:
            hints = _component_spans_from_answer(answer)
            if len(hints) < 2:
                hints = _answer_listing_spans(_answer_primary_listing_body(answer))
            if len(hints) >= 2:
                topics = _order_listing_targets_by_hints(topics, hints)
            return topics[:12]
    return _answer_image_span_targets(q, answer)


def _listing_mode_active(query: str, listing_targets: list[str]) -> bool:
    return _is_listing_scope_query(query) and len(listing_targets) >= 2


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

    if _is_listing_scope_query(query):
        answer = _answer_text_for_listing()
        if answer:
            hints = _component_spans_from_answer(answer)
            if len(hints) < 2:
                hints = _answer_listing_spans(_answer_primary_listing_body(answer))
            pool_topics = _maintenance_topics_in_text(text, query)
            if len(pool_topics) >= 2:
                return _order_listing_targets_by_hints(pool_topics, hints)[:12]

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

    for ref in extract_image_refs_from_context(text):
        label = _ref_effective_label(ref)
        if label and len(label) >= 4 and _topic_overlaps_query(label, query):
            add(label)

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
    focus_norm = normalize_context_for_image_parse(focus)
    answer = _answer_text_for_listing()
    if _is_listing_scope_query(query) and answer:
        bold = _component_spans_from_answer(answer)
        if len(bold) < 2:
            bold = _answer_listing_spans(_answer_primary_listing_body(answer))
        if len(bold) >= 2:
            return bold

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
    lk = _normalize_label_key(label)
    for cand in (target, _listing_target_head(target)):
        if not cand:
            continue
        if short_label_bag_aligns(cand, label):
            return True
        if _figure_label_matches_query(cand, label):
            return True
        tk = _normalize_label_key(cand)
        if len(tk) >= 3 and tk in lk:
            return True
        if len(lk) >= 4 and lk in tk:
            return True
        for term in discriminative_terms(cand, min_len=2):
            if len(term) < 2 or term not in label:
                continue
            if len(term) >= 3 or len(cand) <= 4:
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
    blob = _ref_blob(ref)
    source = (listing_source_text or retrieved_text or "").strip()
    anchor = (retrieved_text or "").strip()
    for target in _listing_targets_with_query_line_overlap(query, source, anchor):
        if label and _label_matches_listing_target(label, target):
            return True
        if blob and (
            _label_matches_listing_target(blob, target)
            or text_term_alignment_symmetric(blob, target) >= 0.18
            or _listing_target_head(target) in blob
        ):
            return True
    if len(label) < _min_substantive_term_len():
        return False
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
    blob = _ref_blob(ref)
    if _strict_object_image_gate(query):
        return _figure_matches_query_object(query, blob)
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
    listing_targets = _listing_targets_with_query_line_overlap(
        query, list_source, text
    )
    if _trust_llm_chunk_images() and not _listing_mode_active(query, listing_targets):
        if text and _ref_inline_in_retrieved_context(ref, text):
            return _ref_matches_figure_focus(query, ref)
        return False
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
    listing_label_ok = _listing_mode_active(query, listing_targets) and any(
        _label_matches_listing_target(_ref_effective_label(ref), target)
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
            if _listing_mode_active(query, listing_targets):
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
    if _listing_mode_active(query, listing_targets):
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


def _doc_basename(doc: dict[str, Any]) -> str:
    fp = str(doc.get("file_path") or "").strip()
    if not fp:
        return ""
    return Path(fp.replace("\\", "/")).name


def _answer_has_multi_section_markdown(answer: str) -> bool:
    body = _answer_text_for_placement(answer)
    titles = [
        m.group(1).strip()
        for m in re.finditer(r"\*\*([^*]+)\*\*", body)
        if len(m.group(1).strip()) >= 4
    ]
    return len(titles) >= 2


def _supplement_cross_manual_figure_chunks(
    all_docs: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    *,
    query: str | None,
    answer: str,
) -> list[dict[str, Any]]:
    """Keep one inline-figure chunk per manual when the answer compares multiple models."""
    if _unique_retrieved_doc_count(all_docs) < 2:
        return kept
    if not _answer_has_multi_section_markdown(answer):
        return kept
    q_blob = _normalize_citation_blob(query or "")
    best_per_manual: dict[str, tuple[float, dict[str, Any]]] = {}
    for doc in all_docs:
        content = _doc_content(doc).strip()
        if not content or not extract_image_refs_from_context(content):
            continue
        q_ov = _answer_chunk_term_overlap(q_blob, content) if q_blob else 0.0
        cite = _chunk_citation_score(
            _answer_body_for_citation_match(answer), content
        )
        combined = max(q_ov, cite)
        min_combined = (
            0.04 if _is_multi_machine_comparison_query(query or "") else 0.08
        )
        if combined < min_combined:
            continue
        if _is_multi_machine_comparison_query(query or ""):
            q_terms = [
                t
                for t in discriminative_terms(query or "", min_len=2)
                if len(t) >= 2
            ]
            if q_terms and not any(term in content for term in q_terms):
                continue
        manual = _doc_basename(doc)
        if not manual:
            continue
        prev = best_per_manual.get(manual)
        if prev is None or combined > prev[0]:
            best_per_manual[manual] = (combined, doc)
    if len(best_per_manual) < 2:
        return kept
    out = list(kept)
    kept_ids = {id(doc) for doc in out}
    for _, doc in best_per_manual.values():
        if id(doc) not in kept_ids:
            out.append(doc)
            kept_ids.add(id(doc))
    return out


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


def _trust_llm_chunk_images() -> bool:
    """Plan A: inline figures in LLM chunks need no query-object / caption alignment."""
    raw = (os.getenv("RAG_IMAGE_TRUST_LLM_CHUNKS") or "1").strip().lower()
    return raw not in ("", "0", "false", "no", "off", "none")


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


def query_section_anchor_enabled() -> bool:
    raw = (os.getenv("RAG_IMAGE_QUERY_SECTION_ANCHOR") or "1").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _query_section_anchor_min_score() -> float:
    raw = os.getenv("RAG_IMAGE_QUERY_SECTION_MIN_SCORE") or "0.28"
    try:
        return max(0.12, min(2.0, float(raw)))
    except ValueError:
        return 0.28


def _answer_weak_consistency_gate(answer_blob: str, content: str) -> bool:
    """Answer terms overlap chunk body (not bold-span expansion)."""
    if not answer_blob.strip() or not (content or "").strip():
        return False
    terms = [
        t
        for t in discriminative_terms(answer_blob, min_len=2)
        if len(t) >= 2
    ]
    if not terms:
        return False
    body = _normalize_citation_blob(content)
    hits = sum(1 for term in terms if term in body)
    if hits >= 1 and hits / len(terms) >= 0.12:
        return True
    return text_term_alignment_symmetric(answer_blob, content) >= 0.12


def _pool_has_query_aligned_figure_chunks(
    query: str,
    pool: list[dict[str, Any]],
    *,
    answer: str | None = None,
    require_answer_gate: bool = False,
) -> bool:
    """Pool has inline figures aligned to query; optional answer weak-consistency."""
    return bool(
        _query_aligned_figure_candidate_docs(
            query,
            pool,
            answer=answer,
            require_answer_gate=require_answer_gate,
        )
    )


def _query_aligned_figure_candidate_docs(
    query: str,
    pool: list[dict[str, Any]],
    *,
    answer: str | None = None,
    require_answer_gate: bool = False,
) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    answer_blob = (
        _answer_body_for_citation_match(answer or "") if require_answer_gate else ""
    )
    if require_answer_gate and not answer_blob.strip():
        return []
    out: list[dict[str, Any]] = []
    for doc in pool:
        content = _doc_content(doc).strip()
        if (
            not content
            or _chunk_is_toc_heavy(content)
            or _chunk_is_title_only(content)
            or not extract_image_refs_from_context(content)
        ):
            continue
        if not _chunk_figure_context_aligns_query(q, content):
            continue
        if require_answer_gate and not _answer_weak_consistency_gate(
            answer_blob, content
        ):
            continue
        out.append(doc)
    return out


def _chunk_query_section_score(
    query: str,
    content: str,
    *,
    anchor_sections: list[str],
    pool_primary: str,
) -> float:
    score = 0.0
    if anchor_sections and _chunk_belongs_to_anchor_sections(
        query, pool_primary, content, anchor_sections
    ):
        score += 1.2
    score += _chunk_subject_score(query, content) * 0.65
    for match in _MAINT_TOPIC_RE.finditer(content):
        topic = match.group(1).strip()
        if any(
            len(needle) >= 3 and needle in topic
            for needle in _query_subject_needles(query)
        ):
            score += 0.45
            break
    if _chunk_figure_context_aligns_query(query, content):
        score += 0.35
    return score


def _anchor_chunks_by_query_section(
    query: str,
    pool: list[dict[str, Any]],
    *,
    answer: str,
    kept: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Query-led section anchor when citation kept nothing (single-topic, terse answers)."""
    meta: dict[str, Any] = {
        "mode": "query_section_anchor",
        "anchor_sections": [],
        "picked": 0,
    }
    out = list(kept or [])
    q = (query or "").strip()
    if out or not q or _is_listing_scope_query(q) or not query_section_anchor_enabled():
        return out, meta

    answer_blob = _answer_body_for_citation_match(answer)
    if not answer_blob.strip():
        meta["reason"] = "no_answer_body"
        return out, meta

    pool_primary = text_from_retrieved_docs(pool)
    if _retrieval_prefers_catalog_field(q, pool_primary):
        meta["reason"] = "catalog_query"
        return out, meta
    if re.search(r"开机前", q) and not _is_listing_scope_query(q):
        meta["reason"] = "preflight_query"
        return out, meta
    if not _pool_has_query_aligned_figure_chunks(q, pool):
        meta["reason"] = "no_query_aligned_figure_in_pool"
        return out, meta
    anchor_sections = _pick_anchor_sections(q, pool_primary, pool)
    meta["anchor_sections"] = anchor_sections

    best_doc: dict[str, Any] | None = None
    best_score = 0.0
    min_score = _query_section_anchor_min_score()
    subject_needles = [n for n in _query_subject_needles(q) if len(n) >= 4]
    align_candidates = _query_aligned_figure_candidate_docs(
        q, pool, answer=answer, require_answer_gate=True
    )
    needles_required = len(align_candidates) > 1

    for doc in pool:
        content = _doc_content(doc).strip()
        if (
            not content
            or _chunk_is_toc_heavy(content)
            or _chunk_is_title_only(content)
            or not extract_image_refs_from_context(content)
        ):
            continue
        if not _answer_weak_consistency_gate(answer_blob, content):
            continue
        if (
            needles_required
            and subject_needles
            and not any(needle in content for needle in subject_needles)
        ):
            continue
        if not _chunk_figure_context_aligns_query(q, content):
            continue
        q_score = _chunk_query_section_score(
            q,
            content,
            anchor_sections=anchor_sections,
            pool_primary=pool_primary,
        )
        if anchor_sections and not _chunk_belongs_to_anchor_sections(
            q, pool_primary, content, anchor_sections
        ):
            q_score *= 0.55
        if q_score > best_score:
            best_score = q_score
            best_doc = doc

    if best_doc is None or best_score < min_score:
        fallback_doc: dict[str, Any] | None = None
        fallback_score = 0.0
        for doc in pool:
            content = _doc_content(doc).strip()
            if (
                not content
                or _chunk_is_toc_heavy(content)
                or not extract_image_refs_from_context(content)
                or not _chunk_figure_context_aligns_query(q, content)
                or not _answer_weak_consistency_gate(answer_blob, content)
            ):
                continue
            if (
                needles_required
                and subject_needles
                and not any(
                    needle in content for needle in subject_needles
                )
            ):
                continue
            score = _chunk_subject_score(q, content)
            if score > fallback_score:
                fallback_score = score
                fallback_doc = doc
        if fallback_doc is not None and fallback_score >= 0.18:
            best_doc = fallback_doc
            best_score = fallback_score
            meta["fallback"] = True

    if best_doc is None or best_score < min(min_score, 0.18):
        meta["reason"] = "no_query_section_match"
        meta["best_score"] = round(best_score, 3)
        return out, meta

    meta["picked"] = 1
    meta["best_score"] = round(best_score, 3)
    return [best_doc], meta


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
        if len(needle) > 12:
            continue
        if len(needle) >= 3 and needle in content:
            score += min(len(needle), 12) * 0.12
    for line in content.splitlines():
        line = line.strip()
        if _is_toc_or_directory_line(line) or _is_image_metadata_line(line):
            continue
        if not _SECTION_HEADING_RE.match(line):
            continue
        hits = sum(
            1
            for needle in needles
            if len(needle) <= 12 and len(needle) >= 3 and needle in line
        )
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


def _llm_subject_chunk_min_score() -> float:
    raw = os.getenv("RAG_IMAGE_LLM_CHUNK_MIN_SUBJECT") or "0.24"
    try:
        return max(0.08, min(2.0, float(raw)))
    except ValueError:
        return 0.24


def llm_rerank_figure_supplement_enabled() -> bool:
    """Off by default: do not widen the LLM batch from rerank just to hunt figures."""
    raw = (os.getenv("RAG_LLM_RERANK_FIGURE_SUPPLEMENT") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _answer_topic_min_overlap() -> float:
    raw = os.getenv("RAG_IMAGE_ANSWER_TOPIC_MIN_OVERLAP") or "0.12"
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.12


def _chunk_matches_answer_topic(content: str, topic: str) -> bool:
    threshold = _answer_topic_min_overlap()
    if _answer_chunk_term_overlap(topic, content) >= threshold:
        return True
    if text_term_alignment_symmetric(topic, content) >= max(threshold, 0.18):
        return True
    for ref in extract_image_refs_from_context(content):
        label = _ref_effective_label(ref)
        blob = " ".join(
            str(ref.get(key) or "") for key in ("context", "caption", "label")
        ).strip()
        if label and (
            _answer_chunk_term_overlap(topic, label) >= threshold
            or text_term_alignment_symmetric(topic, label) >= max(threshold, 0.18)
        ):
            return True
        if blob and text_term_alignment_symmetric(topic, blob) >= max(threshold, 0.18):
            return True
    return False


def _figure_context_from_answer_docs(
    answer: str,
    docs: list[dict[str, Any]],
    *,
    query: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Figure scan scope = cited chunks whose inline figures match answer topics only."""
    body = _answer_body_for_citation_match(answer)
    topics = _answer_image_span_targets(query or "", answer)
    if len(topics) < 2:
        topics = _answer_section_topics(answer)
    meta: dict[str, Any] = {
        "mode": "answer_topics",
        "topics": topics[:24],
        "anchor_chunks": 0,
        "anchor_chars": 0,
    }
    if not docs:
        meta["reason"] = "no_docs"
        return "", meta

    parts: list[str] = []
    min_cite = 0.12
    q = (query or "").strip()
    machines = _machine_spans_from_answer(answer)
    machine_names = set(machines)
    multi_machine = _is_multi_machine_comparison_query(q) and len(machines) >= 2
    seen_parts: set[str] = set()

    def _topic_matches_chunk(topic: str, content: str, doc: dict[str, Any]) -> bool:
        if _chunk_matches_answer_topic(content, topic):
            return True
        if multi_machine and topic in machine_names:
            manual = _doc_basename(doc)
            return bool(manual and topic in manual)
        return False

    def _add_part(content: str) -> None:
        if content and content not in seen_parts:
            seen_parts.add(content)
            parts.append(content)

    if len(topics) >= 2:
        for topic in topics:
            best_content = ""
            best_score = 0.0
            for doc in docs:
                content = _doc_content(doc).strip()
                if not content or not extract_image_refs_from_context(content):
                    continue
                if not _topic_matches_chunk(topic, content, doc):
                    continue
                score = max(
                    _answer_chunk_term_overlap(topic, content),
                    text_term_alignment_symmetric(topic, content),
                )
                if score > best_score:
                    best_score = score
                    best_content = content
            _add_part(best_content)
    else:
        for doc in docs:
            content = _doc_content(doc).strip()
            if not content or not extract_image_refs_from_context(content):
                continue
            if topics:
                if not any(
                    _topic_matches_chunk(topic, content, doc) for topic in topics
                ):
                    continue
            elif body:
                if _chunk_citation_score(body, content) < min_cite:
                    continue
            else:
                continue
            _add_part(content)

    if multi_machine:
        q_terms = [
            t for t in discriminative_terms(q, min_len=2) if len(t) >= 2
        ]
        for topic in machines:
            if any(
                topic in _doc_basename(doc)
                and _doc_content(doc).strip() in seen_parts
                for doc in docs
            ):
                continue
            best_content = ""
            best_score = 0.0
            for doc in docs:
                content = _doc_content(doc).strip()
                if not content or not extract_image_refs_from_context(content):
                    continue
                manual = _doc_basename(doc)
                if not (manual and topic in manual):
                    continue
                if q_terms and not any(term in content for term in q_terms):
                    continue
                score = _chunk_subject_score(q, content)
                if score > best_score:
                    best_score = score
                    best_content = content
            _add_part(best_content)
    query_aligned = [p for p in parts if _chunk_figure_context_aligns_query(q, p)]
    if (
        not query_aligned
        and not _is_listing_scope_query(q)
        and not multi_machine
    ):
        replacement = ""
        for doc in docs:
            content = _doc_content(doc).strip()
            if not content or not extract_image_refs_from_context(content):
                continue
            if not _chunk_figure_context_aligns_query(q, content):
                continue
            if body and not (
                _answer_weak_consistency_gate(body, content)
                or _chunk_citation_score(body, content) >= min_cite
            ):
                continue
            replacement = content
            break
        if replacement:
            parts = []
            seen_parts = set()
            _add_part(replacement)

    meta["anchor_chunks"] = len(parts)
    meta["anchor_chars"] = sum(len(p) for p in parts)
    if not parts:
        meta["reason"] = "no_answer_topic_figure_chunks"
    return "\n\n".join(parts), meta


def _action_focus_text(query: str) -> str:
    """Subject clause with leading device profile label stripped (image / chunk gate)."""
    clauses = _subject_action_clauses(query)
    text = max(clauses, key=len) if clauses else (query or "")
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    merged = "".join(cjk_runs) if cjk_runs else text
    try:
        from query_doc_steering import resolve_machine_profile  # noqa: WPS433

        profile = resolve_machine_profile(query)
        label = str((profile or {}).get("label") or "").replace(" ", "")
        if label and merged.startswith(label) and len(merged) > len(label):
            merged = merged[len(label) :]
    except Exception:
        pass
    return merged.strip() or text


def _chunk_figure_context_aligns_query(query: str, content: str) -> bool:
    """Figure caption/context must align with the query action focus, not incidental terms."""
    focus = _action_focus_text(query)
    if not focus.strip():
        return False
    strict = _strict_object_image_gate(query)
    for ref in extract_image_refs_from_context(content):
        label = _ref_effective_label(ref)
        blob = f"{label} {ref.get('context') or ''}"
        if strict:
            if _figure_matches_query_object(query, blob):
                return True
            continue
        if short_label_bag_aligns(focus, label) or short_label_bag_aligns(focus, blob[:80]):
            return True
        if any(
            len(term) >= 4 and term in blob
            for term in discriminative_terms(focus, min_len=3)
        ):
            return True
        focus_tail = focus[-6:] if len(focus) >= 6 else focus
        if len(substantive_bigrams(focus_tail) & substantive_bigrams(blob)) >= 2:
            return True
    return False


def _llm_chunks_with_inline_figures(
    retrieved_docs: list[dict[str, Any]] | None,
) -> str:
    """LLM-input chunks that carry inline ``[图片]`` markers (no query-subject filter)."""
    parts: list[str] = []
    for doc in retrieved_docs or []:
        content = _doc_content(doc).strip()
        if content and extract_image_refs_from_context(content):
            parts.append(content)
    return "\n\n".join(parts)


def _llm_chunks_with_subject_figures(
    query: str, retrieved_docs: list[dict[str, Any]] | None
) -> str:
    """LLM-input chunks with inline figures; optional legacy query-subject filter."""
    if _trust_llm_chunk_images():
        return _llm_chunks_with_inline_figures(retrieved_docs)
    q = (query or "").strip()
    min_score = _llm_subject_chunk_min_score()
    parts: list[str] = []
    for doc in retrieved_docs or []:
        content = _doc_content(doc).strip()
        if not content or not extract_image_refs_from_context(content):
            continue
        if not _chunk_figure_context_aligns_query(q, content):
            continue
        if _chunk_subject_score(q, content) < min_score:
            continue
        parts.append(content)
    return "\n\n".join(parts)


def _chunk_overlaps_ranked_lines(
    content: str, ranked: list[tuple[float, str]], *, min_align: float = 0.28
) -> bool:
    body = content.strip()
    if not body or not ranked:
        return False
    for _overlap, line in ranked:
        if line in body or body in line:
            return True
        if _text_alignment(body, line) >= min_align:
            return True
    return False


def _figure_anchor_blob(content: str, ref: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("context", "caption", "label"):
        val = str(ref.get(key) or "").strip()
        if val:
            parts.append(val)
    path = str(ref.get("path") or "")
    if path and path in content:
        pos = content.find(path)
        if pos > 0:
            preceding: list[str] = []
            for ln in reversed(content[:pos].splitlines()):
                line = ln.strip()
                if not line:
                    if preceding:
                        break
                    continue
                if line == "[图片]" or _is_image_metadata_line(line):
                    continue
                preceding.append(line)
                break
            if preceding:
                parts.append(preceding[0])
    return " ".join(parts).strip()


def _figures_anchor_to_ranked_lines(
    content: str,
    ranked: list[tuple[float, str]],
    *,
    min_align: float = 0.18,
) -> bool:
    if not content.strip() or not ranked:
        return False
    for ref in extract_image_refs_from_context(content):
        blob = _figure_anchor_blob(content, ref)
        if not blob:
            continue
        for _overlap, line in ranked:
            if line in blob or blob in line:
                return True
            if _text_alignment(blob, line) >= min_align:
                return True
    return False


def _figure_chunks_near_primary_top_lines(
    query: str,
    primary: str,
    retrieved_docs: list[dict[str, Any]] | None,
) -> str:
    """Keep inline-figure chunks only when figure anchors sit on primary top lines."""
    q = (query or "").strip()
    docs = retrieved_docs or []
    ranked = _ranked_retrieval_lines(q, primary, limit=5)
    if not ranked:
        return _llm_chunks_with_subject_figures(q, docs)

    min_line = _image_anchor_min_line_overlap()
    best_overlap = ranked[0][0]
    top_ranked = [(ov, line) for ov, line in ranked if ov >= min_line * 0.6]

    fig_near_top = any(
        _figures_anchor_to_ranked_lines(_doc_content(doc).strip(), top_ranked)
        for doc in docs
        if extract_image_refs_from_context(_doc_content(doc))
    )

    if best_overlap >= min_line and not fig_near_top:
        return ""

    parts: list[str] = []
    for doc in docs:
        content = _doc_content(doc).strip()
        if not content or not extract_image_refs_from_context(content):
            continue
        if top_ranked and not _figures_anchor_to_ranked_lines(content, top_ranked):
            continue
        if not _trust_llm_chunk_images():
            if not _chunk_figure_context_aligns_query(q, content):
                continue
            if _chunk_subject_score(q, content) < _llm_subject_chunk_min_score():
                continue
        parts.append(content)

    if parts:
        return "\n\n".join(parts)
    return _llm_chunks_with_subject_figures(q, docs)


def _answer_section_topics(answer: str) -> list[str]:
    """Bold spans in the answer body (machine headers, component names)."""
    body = _answer_text_for_placement(answer)
    topics: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\*\*([^*]+)\*\*", body):
        topic = match.group(1).strip()
        if len(topic) < 3 or topic in seen:
            continue
        seen.add(topic)
        topics.append(topic)
    return topics


def _supplement_query_topic_figure_chunks(
    pool: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    *,
    query: str,
) -> list[dict[str, Any]]:
    """When the answer is terse, keep one query-aligned inline-figure chunk from the pool."""
    if kept or _is_listing_scope_query(query):
        return kept
    q = (query or "").strip()
    if not q or not _pool_has_query_aligned_figure_chunks(q, pool):
        return kept
    best_doc: dict[str, Any] | None = None
    best_score = 0.0
    for doc in pool:
        content = _doc_content(doc).strip()
        if not content or not extract_image_refs_from_context(content):
            continue
        if not _chunk_figure_context_aligns_query(q, content):
            continue
        score = max(
            _chunk_subject_score(q, content),
            text_term_alignment_symmetric(q, content),
        )
        if score > best_score:
            best_score = score
            best_doc = doc
    if best_doc is None:
        return kept
    out = list(kept)
    if id(best_doc) not in {id(doc) for doc in out}:
        out.append(best_doc)
    return out


def _supplement_answer_topic_figure_chunks(
    pool: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    *,
    answer: str,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Add one inline-figure chunk per answer component topic from the search pool."""
    topics = _span_keep_listing_targets(
        query or "", answer, pool, kept=kept
    )
    if len(topics) < 2:
        return kept
    answer_blob = _answer_body_for_citation_match(answer)
    machine_names = set(_machine_spans_from_answer(answer))
    out = list(kept)
    kept_ids = {id(doc) for doc in out}
    for topic in topics:
        best_doc: dict[str, Any] | None = None
        best_score = 0.0
        for doc in pool:
            content = _doc_content(doc).strip()
            if not content or not extract_image_refs_from_context(content):
                continue
            topic_ov = max(
                _answer_chunk_term_overlap(topic, content),
                text_term_alignment_symmetric(topic, content),
            )
            cite = (
                _chunk_citation_score(answer_blob, content) if answer_blob else 0.0
            )
            label_hit = any(
                _label_matches_listing_target(lab, topic)
                or text_term_alignment_symmetric(topic, lab) >= 0.18
                for ref in extract_image_refs_from_context(content)
                if (lab := _ref_effective_label(ref))
            )
            topic_match = _chunk_matches_answer_topic(content, topic)
            manual = _doc_basename(doc)
            machine_match = (
                _is_multi_machine_comparison_query(query or "")
                and topic in machine_names
                and bool(manual and topic in manual)
            )
            if not topic_match and not label_hit and not machine_match:
                continue
            score = max(topic_ov, cite * 0.85)
            if label_hit:
                score += 0.12
            if machine_match:
                score = max(score, 0.2)
            if (topic_match or machine_match) and score < 0.15:
                score = 0.15
            if score < 0.06:
                continue
            if score > best_score:
                best_score = score
                best_doc = doc
        if best_doc is not None and id(best_doc) not in kept_ids:
            out.append(best_doc)
            kept_ids.add(id(best_doc))
    return out


def _doc_content_key(doc: dict[str, Any]) -> str:
    content = _doc_content(doc).strip()
    if content:
        return f"content:{content}"
    chunk_id = str(doc.get("id") or doc.get("chunk_id") or "").strip()
    if chunk_id:
        return f"id:{chunk_id}"
    return ""


def _dedupe_doc_list(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for doc in docs:
        key = _doc_content_key(doc)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def supplement_llm_docs_with_rerank_figures(
    query: str,
    llm_docs: list[dict[str, Any]],
    rerank_docs: list[dict[str, Any]] | None,
    *,
    max_add: int = 2,
) -> tuple[list[dict[str, Any]], int]:
    """When token truncation drops inline-figure chunks from LLM input, add rerank ones."""
    q = (query or "").strip()
    if not q or not rerank_docs or max_add <= 0:
        return list(llm_docs), 0

    seen = {_doc_content_key(doc) for doc in llm_docs if _doc_content_key(doc)}
    out = list(llm_docs)
    min_score = _llm_subject_chunk_min_score()
    trust = _trust_llm_chunk_images()
    added = 0
    for doc in rerank_docs:
        key = _doc_content_key(doc)
        if not key or key in seen:
            continue
        content = _doc_content(doc).strip()
        if not content or not extract_image_refs_from_context(content):
            continue
        if trust:
            out.append(doc)
            seen.add(key)
            added += 1
            if added >= max_add:
                break
            continue
        if not _chunk_figure_context_aligns_query(q, content):
            continue
        if _chunk_subject_score(q, content) < min_score:
            continue
        out.append(doc)
        seen.add(key)
        added += 1
        if added >= max_add:
            break
    return _dedupe_doc_list(out), added


def _context_for_image_scan(
    query: str,
    primary_text: str,
    retrieved_docs: list[dict[str, Any]] | None,
    *,
    answer: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Inline-image scan: answer-topic chunks only (no query-primary expansion)."""
    if (answer or "").strip() and retrieved_docs:
        return _figure_context_from_answer_docs(
            answer, list(retrieved_docs or []), query=query
        )

    primary = (primary_text or "").strip()
    meta: dict[str, Any] = {
        "mode": "off",
        "primary_chars": len(primary),
        "anchor_sections": [],
        "anchor_chars": 0,
        "anchor_chunks": 0,
        "reason": "no_answer",
    }
    return primary, meta


def _section_fidelity_enabled() -> bool:
    raw = (os.getenv("RAG_QUERY_SECTION_FIDELITY") or "1").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _sanitize_maintenance_chunk_content(content: str) -> str:
    """When a chunk already has 保养步骤, drop image 关联正文 to avoid merged neighbor text."""
    text = (content or "").strip()
    if not text or "保养步骤：" not in text or "关联正文：" not in text:
        return text
    lines: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("关联正文："):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def sanitize_retrieved_docs_content(
    docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strip polluted image 关联正文 from maintenance chunks before LLM input."""
    out: list[dict[str, Any]] = []
    for doc in docs:
        content = _doc_content(doc).strip()
        cleaned = _sanitize_maintenance_chunk_content(content)
        if cleaned != content:
            doc = dict(doc)
            for key in ("content", "text", "chunk_content", "page_content"):
                if key in doc and isinstance(doc[key], str):
                    doc[key] = cleaned
                    break
            else:
                doc["content"] = cleaned
        out.append(doc)
    return out


def narrow_retrieved_docs_to_anchor_sections(
    query: str,
    docs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep chunks in the top anchor section(s) for single-item maintenance answers."""
    meta: dict[str, Any] = {
        "narrowed": False,
        "anchor_sections": [],
        "before": len(docs or []),
        "after": len(docs or []),
    }
    q = (query or "").strip()
    if not _section_fidelity_enabled() or not q or not docs:
        return list(docs or []), meta
    if _is_listing_scope_query(q):
        return list(docs), meta
    try:
        from query_doc_steering import table_filter_needle  # noqa: WPS433

        if table_filter_needle(q):
            return list(docs), meta
    except Exception:
        pass
    if not re.search(r"保养|加注|润滑|清洁|步骤|周期|检查|更换|调整", q):
        return list(docs), meta
    if re.search(r"哪个|哪种|哪类|品牌|型号|什么产品|多少|几号", q):
        return list(docs), meta

    primary = text_from_retrieved_docs(docs)
    anchor_sections = _pick_anchor_sections(q, primary, docs)
    meta["anchor_sections"] = anchor_sections
    if not anchor_sections:
        return list(docs), meta

    kept: list[dict[str, Any]] = []
    for doc in docs:
        content = _doc_content(doc).strip()
        if content and _chunk_belongs_to_anchor_sections(
            q, primary, content, anchor_sections
        ):
            kept.append(doc)
    if not kept:
        return list(docs), meta
    meta["narrowed"] = True
    meta["after"] = len(kept)
    return sanitize_retrieved_docs_content(kept), meta


_ANSWER_REF_RE = re.compile(r"###\s*References\b.*", re.I | re.S)


_CITATION_PUNCT_RE = re.compile(
    r"[\s!！?？。.，,~、；;：:\"'（）()\[\]【】\-/／·]+"
)


def _normalize_citation_blob(text: str) -> str:
    """Strip layout/punctuation so answer paraphrases still match chunk lines."""
    text = (text or "").strip()
    text = text.replace("*", "").replace("＜", "<").replace("＞", ">")
    return _CITATION_PUNCT_RE.sub("", text)


def _chunk_is_title_only(content: str) -> bool:
    """Skip catalog/title-only chunks that echo the device name in answers."""
    text = (content or "").strip()
    if not text or "[图片]" in text:
        return False
    norm = _normalize_citation_blob(text)
    if len(norm) <= 12:
        return True
    if "保养步骤" in text or "保养内容" in text or "保养周期" in text:
        return False
    return len(norm) <= 24 and not re.search(r"[\d\.]+\s*\S", text)


def _answer_chunk_term_overlap(answer_blob: str, content: str) -> float:
    terms = [
        t
        for t in discriminative_terms(answer_blob, min_len=_min_substantive_term_len())
        if len(t) >= _min_substantive_term_len()
    ]
    if not terms:
        return 0.0
    body = _normalize_citation_blob(content)
    hits = sum(1 for term in terms if term in body)
    return hits / len(terms)


def _answer_body_for_citation_match(answer: str) -> str:
    text = (answer or "").strip()
    text = _ANSWER_REF_RE.sub("", text)
    return _normalize_citation_blob(text)


def _line_citation_overlap(answer_blob: str, line: str) -> float:
    line = (line or "").strip()
    if not line or _is_image_metadata_line(line) or _is_toc_or_directory_line(line):
        return 0.0
    if line.startswith(("[图片]", "图片路径", "页码", "关联正文", "图注", "脚注")):
        return 0.0
    norm = _normalize_citation_blob(line)
    if len(norm) < 12:
        return 0.0
    min_sub = 7 if len(norm) >= 14 else 6
    best_len = 0
    for i in range(len(norm)):
        for j in range(i + min_sub, len(norm) + 1):
            sub = norm[i:j]
            if sub in answer_blob and len(sub) > best_len:
                best_len = len(sub)
    if best_len < min_sub:
        return 0.0
    return best_len / max(len(norm), 1)


def _chunk_citation_score(answer_blob: str, content: str) -> float:
    if not answer_blob or not (content or "").strip():
        return 0.0
    best = 0.0
    for line in content.splitlines():
        best = max(best, _line_citation_overlap(answer_blob, line))
    for field in ("保养步骤", "保养内容", "保养周期"):
        for match in re.finditer(rf"{field}[：:]([^\n]+)", content):
            val = _normalize_citation_blob(match.group(1).strip())
            if len(val) >= 6:
                probe = val[: min(28, len(val))]
                if probe in answer_blob:
                    ratio = min(1.0, len(probe) / 28.0)
                    best = max(best, 0.7 + 0.3 * ratio)
                else:
                    ov = _answer_chunk_term_overlap(answer_blob, val)
                    if ov >= 0.34:
                        best = max(best, 0.55 + 0.45 * ov)
    term_ov = _answer_chunk_term_overlap(answer_blob, content)
    if term_ov >= 0.34:
        best = max(best, 0.5 + 0.5 * term_ov)
    for sid in _section_ids_in_text(content):
        if sid in answer_blob:
            best = max(best, 0.5)
            break
    return best


def filter_docs_cited_by_answer(
    answer: str,
    docs: list[dict[str, Any]],
    *,
    query: str | None = None,
    pool: list[dict[str, Any]] | None = None,
    anchor_pool: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep LLM chunks whose body lines appear in the generated answer."""
    meta: dict[str, Any] = {
        "mode": "answer_citation",
        "before": len(docs or []),
        "after": len(docs or []),
    }
    answer_blob = _answer_body_for_citation_match(answer)
    if not answer_blob.strip():
        meta["mode"] = "no_answer"
        return list(docs or []), meta

    search_pool = _dedupe_doc_list(list(pool or docs or []))
    section_pool = _dedupe_doc_list(
        list(anchor_pool or pool or docs or [])
    )
    pool_for_spans = section_pool

    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in docs or []:
        content = _doc_content(doc).strip()
        if (
            not content
            or _chunk_is_toc_heavy(content)
            or _chunk_is_title_only(content)
        ):
            continue
        score = _chunk_citation_score(answer_blob, content)
        if score > 0:
            scored.append((score, doc))

    span_keep = (
        _is_listing_scope_query(query or "")
        and not _is_catalog_or_model_listing_query(query or "")
    ) or (
        _is_multi_machine_comparison_query(query or "")
        and len(_machine_spans_from_answer(answer)) >= 2
    )

    if scored:
        scored.sort(key=lambda pair: pair[0], reverse=True)
        max_score = scored[0][0]
        min_keep = max(0.18, max_score * 0.45)
        if _is_listing_scope_query(query or ""):
            min_keep = max(0.12, max_score * 0.32)
        kept: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for score, doc in scored:
            if score >= min_keep:
                doc_id = id(doc)
                if doc_id not in seen_ids:
                    kept.append(doc)
                    seen_ids.add(doc_id)
    else:
        max_score = 0.0
        min_keep = 0.12
        kept = []
        meta["mode"] = (
            "answer_span_keep_only" if span_keep else "pending_query_section_anchor"
        )
    if span_keep:
        answer_spans = _span_keep_listing_targets(
            query or "",
            answer,
            pool_for_spans,
            kept=kept,
        )
        if len(answer_spans) >= 2:
            kept_ids = {id(doc) for doc in kept}
            for span in answer_spans:
                best_doc: dict[str, Any] | None = None
                best_score = 0.0
                for doc in pool_for_spans:
                    content = _doc_content(doc).strip()
                    if (
                        not content
                        or _chunk_is_toc_heavy(content)
                        or _chunk_is_title_only(content)
                    ):
                        continue
                    if not extract_image_refs_from_context(content):
                        continue
                    labels = [
                        lab
                        for ref in extract_image_refs_from_context(content)
                        if (lab := _ref_effective_label(ref))
                    ]
                    manual = _doc_basename(doc)
                    matched = (
                        any(_label_matches_listing_target(lab, span) for lab in labels)
                        or span in content
                        or span in manual
                        or _chunk_matches_answer_topic(content, span)
                    )
                    if not matched:
                        continue
                    if (
                        _is_multi_machine_comparison_query(query or "")
                        and not _is_component_listing_across_machines(query or "")
                    ):
                        q_terms = [
                            t
                            for t in discriminative_terms(query or "", min_len=2)
                            if len(t) >= 2
                        ]
                        if q_terms and not any(term in content for term in q_terms):
                            continue
                    score = max(_chunk_citation_score(answer_blob, content), 0.15)
                    if score > best_score:
                        best_score = score
                        best_doc = doc
                if best_doc is not None and id(best_doc) not in kept_ids:
                    kept.append(best_doc)
                    kept_ids.add(id(best_doc))
            meta["listing_span_keep"] = True
    kept = _supplement_cross_manual_figure_chunks(
        pool_for_spans, kept, query=query, answer=answer
    )
    kept = _supplement_answer_topic_figure_chunks(
        pool_for_spans, kept, answer=answer, query=query
    )
    if (
        not _is_listing_scope_query(query or "")
        and not _is_multi_machine_comparison_query(query or "")
        and kept
    ):
        query_aligned: list[dict[str, Any]] = []
        for doc in kept:
            content = _doc_content(doc).strip()
            if extract_image_refs_from_context(content):
                if _chunk_figure_context_aligns_query(query or "", content):
                    query_aligned.append(doc)
            else:
                query_aligned.append(doc)
        kept = query_aligned
    if not _is_listing_scope_query(query or ""):
        kept_has_figures = any(
            extract_image_refs_from_context(_doc_content(doc).strip())
            for doc in kept
        )
        if not kept or not kept_has_figures:
            anchored, qsec_meta = _anchor_chunks_by_query_section(
                query or "",
                section_pool,
                answer=answer,
                kept=[],
            )
            if qsec_meta.get("picked"):
                kept = anchored
                meta["mode"] = "query_section_anchor"
                meta["query_section_anchor"] = qsec_meta
            elif meta.get("mode") == "pending_query_section_anchor":
                meta["query_section_anchor"] = qsec_meta
    if re.search(r"开机前", query or "") and not span_keep:
        kept = [
            doc
            for doc in kept
            if not extract_image_refs_from_context(_doc_content(doc).strip())
        ]
    if not kept:
        meta["mode"] = "answer_no_chunk_match"
        meta["after"] = 0
        return [], meta
    meta["after"] = len(kept)
    meta["max_score"] = round(max_score, 3)
    meta["min_keep"] = round(min_keep, 3)
    meta["kept_scores"] = [
        {"score": round(score, 3), "chars": len(_doc_content(doc))}
        for score, doc in scored[:10]
    ]
    return kept, meta


def _answer_text_for_placement(answer: str) -> str:
    text = (answer or "").strip()
    parts = re.split(r"\n###\s*References\b", text, maxsplit=1, flags=re.I)
    return parts[0].strip()


def _find_anchor_in_answer(
    answer: str, anchor: str
) -> tuple[int, int, float]:
    """Return (start, end, score) in answer text for inserting a figure after anchor."""
    body = _answer_text_for_placement(answer)
    anchor = (anchor or "").strip()
    if not body or not anchor:
        return -1, -1, 0.0
    idx = body.find(anchor)
    if idx >= 0:
        end = idx + len(anchor)
        return idx, end, min(1.0, 0.85 + 0.15 * min(1.0, len(anchor) / 12.0))
    bare = anchor.replace("*", "")
    idx = body.find(bare)
    if idx >= 0:
        end = idx + len(bare)
        return idx, end, min(1.0, 0.75 + 0.15 * min(1.0, len(bare) / 12.0))
    norm_body = _normalize_citation_blob(body)
    norm_anchor = _normalize_citation_blob(anchor)
    if len(norm_anchor) >= 4 and norm_anchor in norm_body:
        terms = sorted(
            discriminative_terms(anchor, min_len=2),
            key=len,
            reverse=True,
        )
        for term in terms:
            if len(term) < 2:
                continue
            idx = body.find(term)
            if idx >= 0:
                return idx, idx + len(term), 0.55 + 0.05 * min(4, len(term))
    sem_start, sem_end, sem_score = _find_semantic_anchor_in_answer(body, anchor)
    if sem_start >= 0 and sem_score > 0:
        return sem_start, sem_end, sem_score
    return -1, -1, 0.0


def _find_semantic_anchor_in_answer(body: str, anchor: str) -> tuple[int, int, float]:
    """Match paraphrased / reordered short phrases (e.g. 机床内部清洁 ~ 清洁机床内部)."""
    anchor = (anchor or "").strip()
    if not body or not anchor or len(anchor) > 32:
        return -1, -1, 0.0
    best_start, best_end, best_score = -1, -1, 0.0
    seen_spans: set[tuple[int, int]] = set()

    def consider(start: int, end: int, snippet: str) -> None:
        nonlocal best_start, best_end, best_score
        if start < 0 or end <= start or (start, end) in seen_spans:
            return
        snippet = snippet.strip()
        if len(snippet) < 3:
            return
        seen_spans.add((start, end))
        bag = short_label_bag_aligns(snippet, anchor) or short_label_bag_aligns(
            anchor, snippet
        )
        sym = text_term_alignment_symmetric(snippet, anchor)
        if not bag and sym < 0.45:
            return
        score = min(0.88, 0.55 + sym * 0.32 + (0.14 if bag else 0.0))
        if score > best_score:
            best_start, best_end, best_score = start, end, score

    for line in re.split(r"[\n\r]+", body):
        line = line.strip()
        if len(line) < 3:
            continue
        idx = body.find(line)
        if idx >= 0:
            consider(idx, idx + len(line), line)
        for run in re.findall(r"[\u4e00-\u9fff]{3,28}", line):
            run_idx = body.find(run, max(0, idx))
            if run_idx >= 0:
                consider(run_idx, run_idx + len(run), run)

    for match in re.finditer(r"[\u4e00-\u9fff]{3,28}", body):
        consider(match.start(), match.end(), match.group(0))

    return best_start, best_end, best_score


def _fallback_end_placements(
    answer: str, images: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """When no inline anchor matches, attach figure(s) after the last answer paragraph."""
    body = _answer_text_for_placement(answer)
    if not body or not images:
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]
    target = blocks[-1] if blocks else body.strip()
    idx = body.rfind(target)
    if idx < 0:
        idx = 0
        target = body.strip()
    end = idx + len(target)
    anchor_text = target if len(target) <= 160 else target[-120:]
    placements: list[dict[str, Any]] = []
    for image_index, _img in enumerate(images):
        placements.append(
            {
                "anchor_text": anchor_text,
                "match_start": idx,
                "match_end": end,
                "image_index": image_index,
                "score": 0.4,
                "fallback": "answer_end",
            }
        )
    return placements


def _apply_placement_reindex(
    images: list[dict[str, Any]],
    placements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only images referenced by placements; compact image_index values."""
    reindexed: list[dict[str, Any]] = []
    kept_images: list[dict[str, Any]] = []
    for pl in placements:
        idx = int(pl["image_index"])
        if idx < 0 or idx >= len(images):
            continue
        new_idx = len(kept_images)
        kept_images.append(images[idx])
        reindexed.append({**pl, "image_index": new_idx})
    if reindexed:
        images.clear()
        images.extend(kept_images)
    return reindexed


def _build_listing_inline_placements(
    answer: str,
    images: list[dict[str, Any]],
    spans: list[str],
    min_score: float,
) -> list[dict[str, Any]]:
    """One figure per answer listing span, paired by caption/label alignment."""
    placements: list[dict[str, Any]] = []
    used_indices: set[int] = set()

    for span in spans:
        if not span:
            continue
        start, end, span_score = _find_anchor_in_answer(answer, span)
        anchor_text = span
        if start < 0:
            head = _listing_target_head(span)
            if head and head != span:
                start, end, span_score = _find_anchor_in_answer(answer, head)
                if start >= 0:
                    anchor_text = head
        if start < 0 or span_score < min_score:
            continue

        best_idx = -1
        best_align = -1.0
        for idx, img in enumerate(images):
            if idx in used_indices:
                continue
            caption = str(img.get("caption") or "").strip()
            if _label_matches_listing_target(caption, span):
                align = 1.0
            else:
                align = text_term_alignment_symmetric(span, caption)
            if align < 0.35:
                continue
            if align > best_align:
                best_align = align
                best_idx = idx
        if best_idx < 0:
            continue
        used_indices.add(best_idx)
        body = _answer_text_for_placement(answer)
        display_anchor = anchor_text
        if 0 <= start < end <= len(body):
            snippet = body[start:end].strip()
            if snippet:
                display_anchor = snippet
        placements.append(
            {
                "anchor_text": display_anchor,
                "match_start": start,
                "match_end": end,
                "image_index": best_idx,
                "score": round(span_score, 3),
            }
        )

    placements.sort(key=lambda item: item["match_start"])
    return placements


def _source_key_from_image(img: dict[str, Any]) -> str:
    sk = str(img.get("source_key") or "").strip()
    if sk:
        return sk
    ctx = str(img.get("context") or img.get("caption") or "")
    return _source_key_from_path(ctx) if ctx else ""


def _answer_section_anchors(answer: str) -> list[tuple[int, int, str]]:
    body = _answer_text_for_placement(answer)
    sections: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\*\*([^*]+)\*\*", body):
        title = match.group(1).strip()
        if len(title) < 4:
            continue
        sections.append((match.start(), match.end(), title))
    return sections


def _section_matches_source(title: str, source_key: str) -> bool:
    title_n = _normalize_citation_blob(title)
    key_n = _normalize_citation_blob(source_key)
    if not title_n or not key_n:
        return False
    if title_n in key_n or key_n in title_n:
        return True
    terms = [
        t
        for t in discriminative_terms(source_key, min_len=3)
        if len(t) >= 3
    ]
    if any(len(t) >= 4 and t in title_n for t in terms):
        return True
    return text_term_alignment_symmetric(title, source_key) >= 0.28


def _build_cross_manual_inline_placements(
    answer: str,
    images: list[dict[str, Any]],
    min_score: float,
) -> list[dict[str, Any]]:
    sections = _answer_section_anchors(answer)
    if len(sections) < 2:
        return []
    placements: list[dict[str, Any]] = []
    used_sections: set[int] = set()
    for image_index, img in enumerate(images):
        source_key = _source_key_from_image(img)
        if not source_key:
            continue
        best: tuple[int, int, str, float, int] | None = None
        for sec_idx, (start, end, title) in enumerate(sections):
            if sec_idx in used_sections:
                continue
            if not _section_matches_source(title, source_key):
                continue
            score = 0.85
            if best is None or score > best[3]:
                best = (start, end, title, score, sec_idx)
        if best is None:
            continue
        start, end, title, score, sec_idx = best
        if score < min_score:
            continue
        used_sections.add(sec_idx)
        placements.append(
            {
                "anchor_text": title,
                "match_start": start,
                "match_end": end,
                "image_index": image_index,
                "score": round(score, 3),
            }
        )
    placed_indices = {pl["image_index"] for pl in placements}
    unused_sections = [
        idx for idx in range(len(sections)) if idx not in used_sections
    ]
    for image_index, img in enumerate(images):
        if image_index in placed_indices or not unused_sections:
            continue
        sec_idx = unused_sections.pop(0)
        start, end, title = sections[sec_idx]
        placements.append(
            {
                "anchor_text": title,
                "match_start": start,
                "match_end": end,
                "image_index": image_index,
                "score": 0.5,
                "fallback": "cross_manual_order",
            }
        )
    placements.sort(key=lambda item: item["match_start"])
    return placements


def build_inline_placements(
    answer: str,
    images: list[dict[str, Any]],
    *,
    retrieved_docs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Map each selected image to an answer span for inline Web rendering."""
    del retrieved_docs  # reserved for future chunk-id anchoring
    if not (answer or "").strip() or not images:
        return []
    min_score = _inline_min_place_score()
    answer_body = _answer_text_for_placement(answer)
    listing_spans = _answer_listing_spans(answer)
    source_keys = {
        sk
        for sk in (_source_key_from_image(img) for img in images)
        if sk
    }
    if len(source_keys) >= 2 and _answer_has_multi_section_markdown(answer):
        cross_placements = _build_cross_manual_inline_placements(
            answer, images, min_score
        )
        if len(cross_placements) >= 2:
            return _apply_placement_reindex(images, cross_placements)
    if len(listing_spans) >= 2:
        listing_placements = _build_listing_inline_placements(
            answer, images, listing_spans, min_score
        )
        if listing_placements:
            return _apply_placement_reindex(images, listing_placements)

    placements: list[dict[str, Any]] = []
    used_ranges: list[tuple[int, int]] = []

    for image_index, img in enumerate(images):
        caption = str(img.get("caption") or "").strip()
        anchors: list[str] = []
        if caption:
            anchors.append(caption)
        for span in listing_spans:
            if not span:
                continue
            head = _listing_target_head(span)
            if head and head not in anchors:
                anchors.append(head)
            if caption and _label_matches_listing_target(caption, span):
                anchors.insert(0, span)
            elif caption and (
                span in caption
                or caption in span
                or text_term_alignment_symmetric(span, caption) >= 0.35
            ):
                anchors.append(span)
        if not anchors and listing_spans:
            anchors.extend(listing_spans)

        best_start, best_end, best_score, best_anchor = -1, -1, 0.0, ""
        seen_anchor: set[str] = set()
        for anchor in anchors:
            key = _normalize_label_key(anchor)
            if not key or key in seen_anchor:
                continue
            seen_anchor.add(key)
            start, end, score = _find_anchor_in_answer(answer, anchor)
            label_match = bool(
                caption
                and (
                    _label_matches_listing_target(caption, anchor)
                    or _label_matches_listing_target(
                        caption, _listing_target_head(anchor)
                    )
                )
            )
            effective = score + (0.5 if label_match else 0.0)
            if effective > best_score:
                best_start, best_end, best_score, best_anchor = (
                    start,
                    end,
                    score,
                    anchor,
                )

        if best_start < 0 or best_score < min_score:
            continue
        overlap = any(not (best_end <= u0 or best_start >= u1) for u0, u1 in used_ranges)
        if overlap:
            continue
        used_ranges.append((best_start, best_end))
        display_anchor = best_anchor
        if 0 <= best_start < best_end <= len(answer_body):
            snippet = answer_body[best_start:best_end].strip()
            if snippet:
                display_anchor = snippet
        placements.append(
            {
                "anchor_text": display_anchor,
                "match_start": best_start,
                "match_end": best_end,
                "image_index": image_index,
                "score": round(best_score, 3),
            }
        )

    if not placements:
        return _apply_placement_reindex(
            images, _fallback_end_placements(answer, images)
        )

    placements.sort(key=lambda item: item["match_start"])
    return _apply_placement_reindex(images, placements)


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


def _action_object_cjk(query: str) -> str:
    """Maintenance object phrase from the action clause (question frame stripped)."""
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", _action_focus_text(query)))
    if not cjk:
        return ""
    cjk = re.sub(r"^[对向]", "", cjk)
    try:
        from query_doc_steering import resolve_machine_profile  # noqa: WPS433

        profile = resolve_machine_profile(query)
        label = str((profile or {}).get("label") or "").replace(" ", "")
        if label and label in cjk:
            cjk = cjk.replace(label, "", 1)
    except Exception:
        pass
    cjk = re.sub(r"^[的]", "", cjk)
    cjk = re.sub(r"(，|,).*$", "", cjk)
    cjk = re.sub(r"时(?:如果|若|当|在).*$", "", cjk)
    cjk = re.sub(
        r"(应该|需要|要我|我要|还须|须)?"
        r"(?:使用|用|加注|注入|添加|加入|加|做|选|读|量)?"
        r"(?:什么|哪些|哪种|哪个|多少|几).*$",
        "",
        cjk,
    )
    cjk = re.sub(r"(?:需要|须要|应)?(?:注入|添加|加入|加注).*$", "", cjk)
    cjk = re.sub(r"(要多长|多久|多长时间|做一次).*$", "", cjk)
    cjk = re.sub(r"(进行保养|进行清洁|保养时|保养)$", "", cjk)
    how = re.search(
        r"(?:如何|怎样|怎么|要如何)(?:检查|清洁|更换|调整|清理)?(.+)$",
        cjk,
    )
    if how:
        tail = how.group(1).strip()
        if len(tail) >= _min_substantive_term_len():
            cjk = tail
    for prefix in ("清理", "检查", "更换", "调整", "清洁"):
        if cjk.startswith(prefix) and len(cjk) > len(prefix) + 2:
            cjk = cjk[len(prefix) :]
            break
    return cjk.strip()


def _query_object_terms(query: str) -> list[str]:
    """Longest-first terms from the action object phrase (not shared inspection verbs)."""
    tail = _action_object_cjk(query)
    if len(tail) < _min_substantive_term_len():
        return []
    seen: set[str] = set()
    terms: list[str] = []

    def add(term: str) -> None:
        term = term.strip()
        if len(term) < _min_substantive_term_len() or term in seen:
            return
        if re.search(r"什么|哪些|多少|如何|怎样|怎么", term):
            return
        seen.add(term)
        terms.append(term)

    add(tail)
    for term in discriminative_terms(tail, min_len=_min_substantive_term_len()):
        add(term)
    return sorted(terms, key=len, reverse=True)


def _query_primary_object_term(query: str) -> str:
    """Concrete object span (e.g. 压带轮残胶), not interrogative tails like 什么工具."""
    obj = _action_object_cjk(query)
    if re.search(r"什么|哪些|多少|如何|怎样|怎么|哪个", obj):
        return ""
    if len(obj) >= 4:
        return obj
    terms = _query_object_terms(query)
    return terms[0] if terms else ""


def _strict_object_image_gate(query: str) -> bool:
    """Single-item maintenance questions get strict object matching; listings stay permissive."""
    if _trust_llm_chunk_images():
        return False
    if _is_listing_scope_query(query):
        return False
    primary = _query_primary_object_term(query)
    if not primary or re.search(r"什么|哪些|多少|如何|怎样|怎么|哪个", primary):
        return False
    return len(primary) >= 4


def _figure_matches_query_object(query: str, text: str) -> bool:
    """Query object must appear in figure text; no 开关-in-保护开关 substring hits."""
    blob = (text or "").strip()
    if not blob:
        return False
    if not _strict_object_image_gate(query):
        focus = _action_focus_text(query)
        if short_label_bag_aligns(focus, blob):
            return True
        if any(
            len(term) >= 4 and term in blob
            for term in discriminative_terms(focus, min_len=3)
        ):
            return True
        focus_tail = focus[-6:] if len(focus) >= 6 else focus
        return len(substantive_bigrams(focus_tail) & substantive_bigrams(blob)) >= 2
    primary = _query_primary_object_term(query)
    if primary in blob:
        return True
    if short_label_bag_aligns(primary, blob):
        return True
    shared = substantive_bigrams(primary) & substantive_bigrams(blob)
    if len(shared) >= 2:
        return True
    for term in _query_object_terms(query):
        if term in blob or short_label_bag_aligns(term, blob):
            return True
    return False


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
    if _strict_object_image_gate(query):
        return _figure_matches_query_object(query, blob)
    for term in _query_terms(query):
        if len(term) >= 4 and term in blob:
            return True
        if len(term) == 3 and term in label:
            return True
    focus = _action_focus_bigrams(query, retrieved_text)
    if not focus:
        return True
    return bool(focus & substantive_bigrams(blob))


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
    if _strict_object_image_gate(query):
        return _figure_matches_query_object(query, core)
    if short_label_bag_aligns(query, core):
        return True
    qb = substantive_bigrams(query)
    lb = substantive_bigrams(core)
    if len(qb & lb) >= 2:
        return True
    for term in _query_terms(query):
        if len(term) >= 3 and term in core:
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
        hinted = [ref for ref in refs if _ref_matches_source_hints(ref, hints)]
        if hinted:
            refs = hinted
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
    answer: str | None = None,
) -> bool:
    """True when answer-cited chunks have inline figures aligned to answer topics."""
    del focus_text  # legacy kwarg; primary-only pipeline
    if not query_wants_kb_images(query):
        logger.info("Skip related images: non-KB / chitchat query")
        return False
    primary = (context_text or "").strip()
    q = (query or "").strip()
    if primary and detect_table_filter_signal(q, primary):
        logger.info("Skip related images: confirmed table filter listing in context")
        return False

    if not primary and not (answer or "").strip():
        logger.info("Skip related images: empty retrieval context")
        return False

    scan_text, anchor_meta = _context_for_image_scan(
        q, primary, retrieved_docs, answer=answer
    )
    figure_context = scan_text
    if anchor_meta.get("mode") == "off":
        figure_context = primary
    if anchor_meta.get("mode") == "answer_topics" and not figure_context.strip():
        logger.info(
            "Skip related images: answer topics have no matching inline figures (%s)",
            anchor_meta.get("reason") or "empty",
        )
        return False
    if anchor_meta.get("mode") not in ("off", "answer_topics") and not figure_context.strip():
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
    listing_ok = _listing_mode_active(
        q, _listing_targets_with_query_line_overlap(q, primary, primary)
    )
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


def _maintenance_content_label_for_ref(
    ref: dict[str, Any], retrieved_docs: list[dict[str, Any]] | None
) -> str:
    """``保养内容：`` line from the chunk that cites this inline figure."""
    path_name = Path(str(ref.get("path") or "")).name
    if not path_name:
        return ""
    for doc in retrieved_docs or []:
        content = _doc_content(doc)
        if path_name not in content:
            continue
        match = re.search(r"保养内容[：:]\s*([^\n]+)", content)
        if match:
            return match.group(1).strip()
    return ""


def _chunk_rerank_for_ref(
    ref: dict[str, Any], retrieved_docs: list[dict[str, Any]] | None
) -> float:
    """Rerank score of the chunk that cites this inline figure."""
    path_name = Path(str(ref.get("path") or "")).name
    if not path_name:
        return 0.0
    best = 0.0
    for doc in retrieved_docs or []:
        content = _doc_content(doc)
        if path_name not in content:
            continue
        try:
            best = max(best, float(doc.get("rerank_score") or doc.get("score") or 0))
        except (TypeError, ValueError):
            continue
    return best


def _score_ref_for_query(
    ref: dict[str, Any],
    query: str | None,
    *,
    anchor_phrases: list[str] | None = None,
    source_hints: set[str] | None = None,
    retrieved_text: str | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
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
    label = _ref_effective_label(ref)

    if _trust_llm_chunk_images() and retrieved_docs:
        score = int(_chunk_rerank_for_ref(ref, retrieved_docs) * 300)
        path_name = Path(str(ref.get("path") or "")).name
        top_doc = max(
            retrieved_docs,
            key=lambda d: float(d.get("rerank_score") or d.get("score") or 0),
        )
        if path_name and path_name in _doc_content(top_doc):
            score += 500
        eff_label = _ref_effective_label(ref) or _maintenance_content_label_for_ref(
            ref, retrieved_docs
        )
        if eff_label:
            score += 80
        elif not str(ref.get("context") or "").strip():
            score -= 100
        if source_hints and _ref_matches_source_hints(ref, source_hints):
            score += 10
        if _ref_aligns_with_query_label(
            q,
            ref,
            threshold=_image_min_ref_align(),
            retrieved_text=retrieved_text or ctx,
        ):
            score += 650
        elif _ref_matches_figure_focus(q, ref):
            score += 320
        ranked = (
            _ranked_retrieval_lines(q, retrieved_text or "", limit=1)
            if retrieved_text
            else []
        )
        if ranked and ranked[0][0] >= _image_anchor_min_line_overlap():
            top_line = ranked[0][1]
            ref_blob = " ".join(
                str(ref.get(key) or "") for key in ("caption", "context", "label")
            ).strip()
            if top_line and ref_blob and top_line not in ref_blob and ref_blob not in top_line:
                if (
                    _text_alignment(ref_blob, top_line) < 0.18
                    and _text_alignment(top_line, ref_blob) < 0.18
                ):
                    score -= 520
        if _is_generic_cycle_only_label(label) and not _ref_matches_figure_focus(q, ref):
            score -= 120
        return max(0, score)

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
    targets: list[str] | None = None,
) -> list[dict[str, Any]]:
    """One figure per listing target phrase (e.g. three glue-cleaning procedures)."""
    targets = targets or _listing_target_phrases(query, retrieved_text)
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


def _ref_topic_overlaps_query(query: str, ref: dict[str, Any]) -> bool:
    blob = " ".join(
        str(ref.get(key) or "") for key in ("caption", "context", "label")
    )
    if not blob.strip():
        return False
    terms = [t for t in _query_terms(query) if len(t) >= 3]
    if not terms:
        return True
    long_terms = sorted({t for t in terms if len(t) >= 4}, key=len, reverse=True)
    if long_terms and any(term in blob for term in long_terms):
        return True
    return any(term in blob for term in terms)


def _select_scored_refs_cross_manual(
    scored: list[tuple[int, dict[str, Any]]],
    *,
    limit: int,
    min_absolute: int = 12,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """One on-topic figure per ingested manual for multi-handbook comparison answers."""
    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    for score, ref in sorted(scored, key=lambda pair: pair[0], reverse=True):
        if score < min_absolute:
            continue
        if query and not _ref_topic_overlaps_query(query, ref):
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
    answer: str | None = None,
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

    component_spans = _component_spans_from_answer(answer or "")
    machine_spans = _machine_spans_from_answer(answer or "")
    targets = _listing_target_phrases(query or "", retrieved_text or "")
    if _is_listing_scope_query(query or "") and len(component_spans) >= 2:
        cap = _multi_figure_image_limit()
        if retrieved_text:
            return _select_scored_refs_for_listing(
                scored,
                query=query or "",
                retrieved_text=retrieved_text,
                limit=cap,
                min_absolute=min(12, min_absolute),
                targets=component_spans,
            )
        return _select_scored_refs_by_label(
            scored,
            limit=cap,
            min_absolute=min(12, min_absolute),
        )
    if _is_listing_scope_query(query or "") and len(targets) >= 2:
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

    if len(distinct_sources) >= 2 and (
        machine_spans
        or _is_multi_machine_comparison_query(query or "")
        or _answer_has_multi_section_markdown(answer or "")
    ):
        cap = min(limit, _multi_figure_image_limit())
        want = max(cap, len(machine_spans), 4 if _is_multi_machine_comparison_query(query or "") else 0)
        return _select_scored_refs_cross_manual(
            scored,
            limit=want or cap,
            min_absolute=min(12, min_absolute),
            query=query,
        )

    if len(distinct_sources) >= 2:
        cap = min(limit, _multi_figure_image_limit())
        return _select_scored_refs_cross_manual(
            scored, limit=cap, min_absolute=min(12, min_absolute), query=query
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
    answer: str | None = None,
) -> dict[str, Any]:
    """Explain image gate decisions for debug dumps."""
    del focus_text  # legacy kwarg; primary-only pipeline
    q = (query or "").strip()
    if not query_wants_kb_images(query):
        return {"ok": False, "reason": "non_kb_query"}

    primary = (context_text or "").strip()
    if not primary and not (answer or "").strip():
        return {"ok": False, "reason": "empty_retrieval_context"}

    scan_for_gate = primary
    if (answer or "").strip() and retrieved_docs:
        scan_for_gate, _topic_meta = _figure_context_from_answer_docs(
            answer, list(retrieved_docs or [])
        )
    if not _eligible_figure_refs(scan_for_gate):
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
        answer=answer,
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
    answer: str | None = None,
) -> dict[str, Any]:
    """Full image pipeline trace for debug dumps (no side effects)."""
    primary_text = (context or "").strip()
    scan_text, anchor_scan = _context_for_image_scan(
        query or "", primary_text, retrieved_docs, answer=answer
    )
    figure_context = scan_text if anchor_scan.get("mode") != "off" else primary_text
    component_spans = _component_spans_from_answer(answer or "")
    machine_spans = _machine_spans_from_answer(answer or "")
    ref_scan_text = (
        primary_text
        if len(component_spans) >= 2 or len(machine_spans) >= 2
        else figure_context
    )
    gate = explain_retrieval_supports_images(
        query,
        retrieved_docs=retrieved_docs,
        context_text=primary_text,
        media_roots=media_roots,
        answer=answer,
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

    from_context = _refs_from_retrieved_docs_text(
        ref_scan_text,
        media_roots,
        query=query,
        retrieved_docs=retrieved_docs,
        full_context=primary_text,
    )
    debug["refs_from_context"] = len(from_context)
    debug["listing_targets"] = (
        component_spans
        if len(component_spans) >= 2
        else machine_spans
        if len(machine_spans) >= 2
        else _listing_target_phrases(query or "", primary_text)
    )

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
                retrieved_text=ref_scan_text,
                retrieved_docs=retrieved_docs,
            ),
            ref,
        )
        for ref in aligned_refs
    ]
    debug["scored"] = [
        {"score": score, **_summarize_ref(ref)} for score, ref in scored_pairs
    ]
    selected = _select_scored_refs(
        scored_pairs,
        limit=limit,
        query=query,
        retrieved_text=ref_scan_text,
        answer=answer,
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
    answer: str | None = None,
) -> list[dict[str, Any]]:
    """Turn answer-cited chunks into Web-safe image descriptors."""
    primary_text = (context or "").strip()
    if not primary_text and not (answer or "").strip():
        return []

    if not retrieval_supports_images(
        query,
        retrieved_docs=retrieved_docs,
        context_text=primary_text,
        media_roots=media_roots,
        answer=answer,
    ):
        return []

    scan_text, anchor_scan = _context_for_image_scan(
        query or "", primary_text, retrieved_docs, answer=answer
    )
    figure_context = scan_text if anchor_scan.get("mode") != "off" else primary_text
    if anchor_scan.get("mode") == "answer_topics" and not figure_context.strip():
        logger.info("Skip related images: no answer-topic inline figures")
        return []

    component_spans = _component_spans_from_answer(answer or "")
    machine_spans = _machine_spans_from_answer(answer or "")
    ref_scan_text = (
        primary_text
        if len(component_spans) >= 2 or len(machine_spans) >= 2
        else figure_context
    )

    anchor_phrases = _extract_context_anchors(query, primary_text)
    source_hints = _merged_source_hints(primary_text, retrieved_docs)

    refs = _refs_from_retrieved_docs_text(
        ref_scan_text,
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
                retrieved_docs=retrieved_docs,
            ),
            ref,
        )
        for ref in refs
    ]
    selected = _select_scored_refs(
        scored,
        limit=limit,
        query=query,
        retrieved_text=ref_scan_text,
        answer=answer,
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
        caption = (
            _ref_effective_label(ref)
            or _maintenance_content_label_for_ref(ref, retrieved_docs)
            or ref.get("caption")
            or ""
        )
        context_snippet = ref.get("context") or ""
        if not caption and context_snippet:
            caption = context_snippet[:60]
        item: dict[str, Any] = {
            "url": f"/api/media/image?token={token}",
            "caption": caption,
            "source_key": _source_key_from_path(path_str),
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
    answer: str | None = None,
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
        answer=answer,
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
        answer=answer,
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
