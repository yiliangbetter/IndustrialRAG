"""Query-time image resolution for RAG Q&A (Plan B).

Ingest writes ``[图片]`` metadata blocks via ``raganything.utils`` / ``processor.py``.
This module is used when answering questions (``query_progress_hooks``, media API).
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
    best_image_for_text_item,
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
    "images_for_api",
    "query_wants_kb_images",
    "retrieval_supports_images",
    "encode_media_token",
    "decode_media_token",
    "is_safe_media_path",
    "normalize_context_for_image_parse",
    "supplement_refs_from_content_lists",
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


def _is_metadata_or_inspection_query(query: str) -> bool:
    """Catalog / spec / electrical-check questions should not emit procedure figures."""
    q = (query or "").strip()
    if not q:
        return True
    if re.search(r"型号|适用于哪些|哪些产品|适用范围|产品说明", q):
        return True
    if re.search(r"电源|开关", q) and re.search(r"检查|查电", q):
        if not re.search(r"清洁|清理|保养|润滑|加注|残胶|内部|外部|床身", q):
            return True
    return False


def _query_requests_multiple_figures(query: str) -> bool:
    """Listing questions (e.g. which parts need maintenance) may need several figures."""
    q = (query or "").strip()
    if not q:
        return False
    if re.search(r"哪些|有哪几种|包括哪些|哪几项|列举|分别有哪些", q):
        return bool(
            re.search(r"部件|零件|项目|保养|清洁|清理|润滑|加注|检查", q)
        )
    return False


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


def _listing_target_phrases(query: str, retrieved_text: str) -> list[str]:
    """Distinct maintenance phrases in retrieval that anchor separate figures."""
    if not _query_requests_multiple_figures(query):
        return []
    text = (retrieved_text or "").strip()
    if not text:
        return []
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

    q_has_glue = "残胶" in query or re.search(r"清理.*胶|胶.*清理", query)
    for line in re.split(r"[\n\r]+", text):
        line = line.strip()
        if len(line) < 4 or _is_image_metadata_line(line) or _is_toc_or_directory_line(line):
            continue
        if q_has_glue:
            if "残胶" not in line and not (
                "清理" in line
                and re.search(r"靠|压带|涂胶|胶轴|仿形", line)
            ):
                continue
        elif not any(term in line for term in _query_terms(query)):
            continue
        topic = _maintenance_topic_from_text(line)
        if topic:
            if not q_has_glue or (
                "残胶" in topic
                or re.search(r"压带轮|仿形|涂胶|胶轴", topic)
            ):
                add(topic)
        for match in re.finditer(
            r"[\u4e00-\u9fff]{2,24}(?:残胶清理|残胶|老化胶水)", line
        ):
            add(match.group(0))
        if not q_has_glue:
            for term in _query_terms(query):
                if len(term) >= 4 and term in line:
                    add(term)

    return phrases[:12]


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
) -> bool:
    label = _ref_effective_label(ref)
    if len(label) < _min_substantive_term_len():
        return False
    for target in _listing_target_phrases(query, retrieved_text or ""):
        if _label_matches_listing_target(label, target):
            return True
    if "残胶" in query and "残胶" in label:
        return True
    return _ref_aligns_with_query_label(
        query, ref, threshold=threshold, retrieved_text=retrieved_text
    )


def _ref_passes_image_align_gate(
    query: str,
    ref: dict[str, Any],
    *,
    retrieved_text: str | None,
) -> bool:
    threshold = _image_min_ref_align()
    if _query_requests_multiple_figures(query):
        return _ref_aligns_for_multi_figure_listing(
            query, ref, threshold=threshold, retrieved_text=retrieved_text
        )
    return _ref_aligns_with_query_label(
        query, ref, threshold=threshold, retrieved_text=retrieved_text
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
    """True when a substantive query term hits the answer line but not figure metadata."""
    blob = ref_text or ""
    for term in _query_terms(query):
        if len(term) < 4:
            continue
        if term in line and term not in blob:
            return True
    return False


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
        overlap = _term_overlap_ratio(query, line)
        if overlap <= 0:
            continue
        scored.append((overlap, line))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:limit]


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
    if label and _figure_label_matches_query(query, label):
        return True
    if not label:
        return False
    blob = label + str(ref.get("context") or "")
    label_bgs = substantive_bigrams(blob)
    for term in _query_terms(query):
        if len(term) >= 3 and term in blob:
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
) -> bool:
    """Parser caption/footnote/section heading aligns with the query."""
    label = _ref_effective_label(ref)
    if len(label) < _min_substantive_term_len():
        return False

    core_label = _strip_section_prefix(label) if _is_section_number_heading(label) else label
    label_ok = _figure_label_matches_query(query, core_label)
    if not label_ok and _text_alignment(query, core_label) >= threshold and any(
        len(term) >= 4 and term in core_label for term in _query_terms(query)
    ):
        label_ok = True

    if not label_ok:
        return False
    if not _ref_passes_focus_bigram_gate(query, ref, retrieved_text):
        return False
    if _figure_label_matches_query(query, core_label):
        return True
    return _ref_aligns_with_retrieval_focus(query, ref, retrieved_text)


def supplement_refs_for_listing_targets(
    text: str,
    media_roots: list[Path],
    *,
    query: str,
) -> list[dict[str, Any]]:
    """Attach figures for each distinct listing target (same manual, multiple captions)."""
    if not _query_requests_multiple_figures(query):
        return []
    targets = _listing_target_phrases(query, text)
    if not targets:
        return []

    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_hints = _source_hints_from_text(text)

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
            if source_hints and not any(hint in doc_hint for hint in source_hints):
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


def _refs_from_retrieved_docs_text(
    context: str,
    media_roots: list[Path],
    *,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Image refs parsed only from rerank-filtered chunk bodies (no global context bleed)."""
    q = (query or "").strip()
    return _merge_refs(
        extract_image_refs_from_context(context),
        supplement_refs_from_content_lists(context, media_roots, query=query),
        supplement_refs_for_listing_targets(context, media_roots, query=q) if q else [],
    )


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
) -> list[dict[str, Any]]:
    """Inline chunk figures plus MinerU bbox supplements for text-only rerank slices."""
    refs = _eligible_figure_refs(context)
    if media_roots:
        refs = _merge_refs(
            refs,
            supplement_refs_from_content_lists(context, media_roots, query=query),
        )
    return [ref for ref in refs if not _is_cover_page_ref(ref)]


def retrieval_supports_images(
    query: str | None,
    *,
    retrieved_docs: list[dict[str, Any]] | None = None,
    context_text: str | None = None,
    media_roots: list[Path] | None = None,
) -> bool:
    """True only when retrieval looks substantively relevant to the query."""
    if _is_metadata_or_inspection_query(query or ""):
        logger.info("Skip related images: metadata or inspection-only query")
        return False
    if not query_wants_kb_images(query):
        logger.info("Skip related images: non-KB / chitchat query")
        return False
    text = (context_text or "").strip()
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
        if not text:
            logger.info("Skip related images: empty retrieval context")
            return False
        eligible = _collect_figure_refs(text, media_roots, query=query)
        if not eligible:
            logger.info("Skip related images: no non-cover figures in retrieval")
            return False
        if not any(
            _ref_passes_image_align_gate(
                query or "", ref, retrieved_text=text
            )
            for ref in eligible
        ):
            logger.info(
                "Skip related images: no figure label/heading matches query focus"
            )
            return False
        return True

    text = (context_text or "").strip()
    if not text:
        logger.info("Skip related images: empty retrieval context")
        return False

    q = (query or "").strip()
    terms = _query_terms(q)
    if not terms:
        logger.info("Skip related images: no substantive query terms")
        return False

    overlap = _term_overlap_ratio(q, text)
    min_overlap = _image_min_term_overlap()
    eligible = _collect_figure_refs(text, media_roots, query=query)
    listing_ok = (
        _query_requests_multiple_figures(q) and bool(_listing_target_phrases(q, text))
    )
    near_miss = overlap + 0.051 >= min_overlap
    if overlap < min_overlap and not listing_ok and not (
        near_miss
        and eligible
        and any(
            _figure_label_matches_query(q, _ref_effective_label(ref))
            for ref in eligible
        )
    ):
        logger.info(
            "Skip related images: term overlap %.2f < %.2f",
            overlap,
            min_overlap,
        )
        return False
    if not eligible:
        logger.info("Skip related images: no non-cover figures in retrieval")
        return False
    if not any(
        _ref_passes_image_align_gate(q, ref, retrieved_text=text) for ref in eligible
    ):
        logger.info("Skip related images: no figure label/heading matches query focus")
        return False
    return True


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
        scored.append((overlap, line[:120]))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    for overlap, line in scored:
        if overlap < scored[0][0] * 0.45:
            break
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
        ranked = _ranked_retrieval_lines(q, retrieved_text, limit=1)
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

    label = _ref_effective_label(ref)
    if _is_section_number_heading(str(ref.get("caption") or "")):
        score -= 45
    if label and _figure_label_matches_query(q, label):
        score += 40
    if retrieved_text and _query_requests_multiple_figures(q):
        for target in _listing_target_phrases(q, retrieved_text):
            if _label_matches_listing_target(label, target):
                score += 85
                break
    ctx = str(ref.get("context") or "")
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
) -> list[dict[str, Any]]:
    """Match query-linked phrases in retrieved text to same-page images via MinerU bbox."""
    anchors = _extract_context_anchors(query, text)
    if not anchors:
        anchors = [
            t for t in _query_terms(query or "") if len(t) >= 4
        ][:6]

    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    q = (query or "").strip()
    source_hints = _source_hints_from_text(text)

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
            if source_hints and not any(hint in doc_hint for hint in source_hints):
                continue

            if q and not _is_metadata_or_inspection_query(q):
                from raganything.utils import image_label_for_item  # noqa: WPS433

                for image_item in items:
                    if not isinstance(image_item, dict) or image_item.get("type") != "image":
                        continue
                    label = image_label_for_item(items, image_item)
                    topic = _maintenance_topic_from_text(label) or label
                    if not topic or not _figure_label_matches_query(q, topic):
                        continue
                    _append_content_list_image_ref(
                        refs,
                        seen,
                        items=items,
                        image_item=image_item,
                        auto_dir=auto_dir,
                        caption=topic,
                    )

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

    if _query_requests_multiple_figures(query or ""):
        cap = _multi_figure_image_limit()
        if retrieved_text and _listing_target_phrases(query or "", retrieved_text):
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
) -> dict[str, Any]:
    """Explain image gate decisions for debug dumps."""
    q = (query or "").strip()
    if not query_wants_kb_images(query):
        return {"ok": False, "reason": "non_kb_query"}

    text = (context_text or "").strip()
    max_score = _max_rerank_score(retrieved_docs)
    if max_score is not None:
        threshold = _image_min_rerank_score()
        if max_score < threshold:
            return {
                "ok": False,
                "reason": "low_rerank_score",
                "max_rerank_score": max_score,
                "threshold": threshold,
            }
        if not text:
            return {"ok": False, "reason": "empty_retrieval_context"}
        eligible = _collect_figure_refs(text, media_roots, query=query)
        if not eligible:
            return {"ok": False, "reason": "no_non_cover_figures"}
        if not any(
            _ref_passes_image_align_gate(q, ref, retrieved_text=text)
            for ref in eligible
        ):
            return {"ok": False, "reason": "no_query_label_match"}
        return {"ok": True, "reason": "rerank_score_ok", "max_rerank_score": max_score}

    text = (context_text or "").strip()
    if not text:
        return {"ok": False, "reason": "empty_retrieval_context"}

    terms = _query_terms(q)
    if not terms:
        return {"ok": False, "reason": "no_query_terms"}

    overlap = _term_overlap_ratio(q, text)
    min_overlap = _image_min_term_overlap()
    eligible = _collect_figure_refs(text, media_roots, query=query)
    listing_ok = (
        _query_requests_multiple_figures(q) and bool(_listing_target_phrases(q, text))
    )
    near_miss = overlap + 0.051 >= min_overlap
    if overlap < min_overlap and not listing_ok and not (
        near_miss
        and eligible
        and any(_figure_label_matches_query(q, _ref_effective_label(ref)) for ref in eligible)
    ):
        return {
            "ok": False,
            "reason": "low_term_overlap",
            "overlap": overlap,
            "threshold": min_overlap,
        }
    if not eligible:
        return {"ok": False, "reason": "no_non_cover_figures"}
    if not any(
        _ref_passes_image_align_gate(q, ref, retrieved_text=text) for ref in eligible
    ):
        return {"ok": False, "reason": "no_query_label_match"}
    return {"ok": True, "reason": "term_overlap_ok", "overlap": overlap}


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
    gate = explain_retrieval_supports_images(
        query,
        retrieved_docs=retrieved_docs,
        context_text=primary_text,
        media_roots=media_roots,
    )
    debug: dict[str, Any] = {
        "gate": gate,
        "anchor_phrases": [],
        "top_retrieval_lines": [],
        "refs_from_context": 0,
        "refs_from_supplement": 0,
        "refs_merged": 0,
        "refs_after_align": [],
        "refs_dropped_align": [],
        "scored": [],
        "selected_paths": [],
    }
    if not gate.get("ok"):
        return debug

    merged_context = primary_text
    anchor_phrases = _extract_context_anchors(query, merged_context)
    debug["anchor_phrases"] = anchor_phrases[:12]
    debug["top_retrieval_lines"] = [
        {"overlap": overlap, "line": line[:200]}
        for overlap, line in _ranked_retrieval_lines(query or "", merged_context, limit=6)
    ]

    from_context = extract_image_refs_from_context(primary_text)
    from_supplement = supplement_refs_from_content_lists(
        primary_text, media_roots, query=query
    )
    from_listing = supplement_refs_for_listing_targets(
        primary_text, media_roots, query=query or ""
    )
    debug["refs_from_context"] = len(from_context)
    debug["refs_from_supplement"] = len(from_supplement)
    debug["refs_from_listing"] = len(from_listing)
    debug["listing_targets"] = _listing_target_phrases(query or "", primary_text)

    refs = _merge_refs(from_context, from_supplement, from_listing)
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
            query or "", ref, retrieved_text=merged_context
        ):
            aligned_refs.append(ref)
        else:
            dropped.append({**summary, "drop_reason": "query_label_mismatch"})
    debug["refs_after_align"] = [_summarize_ref(ref) for ref in aligned_refs]
    debug["refs_dropped_align"] = dropped

    if not aligned_refs:
        debug["gate"] = {"ok": False, "reason": "no_query_label_match"}
        return debug

    source_hints = _source_hints_from_text(merged_context)
    scored_pairs = [
        (
            _score_ref_for_query(
                ref,
                query,
                anchor_phrases=anchor_phrases,
                source_hints=source_hints,
                retrieved_text=merged_context,
            ),
            ref,
        )
        for ref in aligned_refs
    ]
    debug["scored"] = [
        {"score": score, **_summarize_ref(ref)} for score, ref in scored_pairs
    ]
    selected = _select_scored_refs(
        scored_pairs, limit=limit, query=query, retrieved_text=merged_context
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
    """Turn retrieved context into Web-safe image descriptors."""
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

    anchor_phrases = _extract_context_anchors(query, primary_text)
    source_hints = _source_hints_from_text(primary_text)

    refs = _refs_from_retrieved_docs_text(
        primary_text, media_roots, query=query
    )
    refs = [
        ref
        for ref in refs
        if not _is_cover_page_ref(ref)
        and _ref_passes_image_align_gate(
            query or "", ref, retrieved_text=primary_text
        )
    ]
    if not refs:
        logger.info("Skip related images: no figure aligns with retrieved lines")
        return []

    scored = [
        (
            _score_ref_for_query(
                ref,
                query,
                anchor_phrases=anchor_phrases,
                source_hints=source_hints,
                retrieved_text=primary_text,
            ),
            ref,
        )
        for ref in refs
    ]
    selected = _select_scored_refs(
        scored, limit=limit, query=query, retrieved_text=primary_text
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
