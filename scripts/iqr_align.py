"""image_query_refs submodule ``iqr_align``.

Reference/chunk alignment scoring: figure-ref align gates, subject/citation
chunk scores, and answer-to-chunk citation overlap.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from raganything.utils import (
    discriminative_terms,
    image_label_text,
    short_label_bag_aligns,
    substantive_bigrams,
    text_term_alignment_symmetric,
)
from iqr_protocol import (
    _is_image_metadata_line,
    extract_image_refs_from_context,
)
from iqr_config import (
    _image_min_ref_align,
    _min_substantive_term_len,
)
from iqr_domain_schema import schema as _domain_schema
from iqr_terms import (
    _action_focus_bigrams,
    _action_focus_text,
    _figure_matches_query_object,
    _query_subject_needles,
    _query_terms,
    _ref_passes_focus_bigram_gate,
    _subject_object_bigrams,
)
from iqr_store import (
    _collect_figure_refs,
    _doc_content,
    _is_multi_source_retrieval,
    _merged_source_hints,
    _ranked_retrieval_lines,
    _ref_matches_source_hints,
)


_SECTION_HEADING_RE = re.compile(r"^[\d\.]+\s*\S")


_SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)")


_PDF_NAME_RE = re.compile(r"[^\s\\n/]+\.pdf", re.IGNORECASE)


_REF_LINE_RE = re.compile(r"^\s*(?:[-*•]\s*)?\[(\d+)\]\s*(.+?)\s*$", re.MULTILINE)


def _ref_structure_align_text(ref: dict[str, Any]) -> str:
    """Structure-driven align text: parser context/步骤 over caption when both exist."""
    ctx = str(ref.get("context") or "").strip()
    if ctx:
        return ctx
    label = _ref_effective_label(ref)
    if label:
        return label
    return str(ref.get("caption") or "").strip()


def _ref_inline_context_text(ref: dict[str, Any]) -> str:
    return str(ref.get("context") or "").strip()


def _figure_ref_passes_align_gate(
    anchor_text: str,
    query: str,
    ref: dict[str, Any],
    *,
    doc_content: str = "",
) -> bool:
    from iqr_anchor import _anchor_align_blob

    blob = _anchor_align_blob(anchor_text, query)
    if not blob.strip():
        return True
    anchor_only = (anchor_text or "").strip()
    label = _ref_effective_label(ref) or ""
    ctx = str(ref.get("context") or ref.get("inline_context") or "").strip()
    chunk = (doc_content or "").strip()
    struct = _ref_structure_align_text(ref)
    score = 0.0
    if anchor_only and struct:
        score = max(score, text_term_alignment_symmetric(anchor_only, struct))
    if struct:
        score = max(score, text_term_alignment_symmetric(blob, struct))
    if anchor_only:
        if label:
            score = max(score, text_term_alignment_symmetric(anchor_only, label))
        if ctx:
            score = max(score, text_term_alignment_symmetric(anchor_only, ctx))
        if chunk:
            score = max(score, text_term_alignment_symmetric(anchor_only, chunk))
    if label:
        score = max(score, text_term_alignment_symmetric(blob, label))
    if ctx:
        score = max(score, text_term_alignment_symmetric(blob, ctx))
    if label and ctx:
        score = max(score, text_term_alignment_symmetric(blob, f"{label}\n{ctx}"))
    if chunk:
        score = max(score, text_term_alignment_symmetric(blob, chunk))
        chunk_label = image_label_text(chunk)
        if chunk_label:
            score = max(score, text_term_alignment_symmetric(blob, chunk_label))
    for needle in _query_subject_needles(query):
        if len(needle) < 2:
            continue
        haystacks = (label, ctx, chunk)
        if anchor_only:
            haystacks = haystacks + (anchor_only,)
        if any(needle in h for h in haystacks if h):
            score = max(score, _image_min_ref_align())
    return score >= _image_min_ref_align()


def _apply_unified_figure_debug(
    debug: dict[str, Any],
    refs: list[dict[str, Any]],
    meta: dict[str, Any],
    *,
    answer: str,
) -> None:
    from iqr_figure_target import _machine_targets_from_answer

    debug["image_selection"] = "unified_figure_targets"
    debug["unified_figure_targets"] = meta
    debug["chunk_locality"] = meta
    debug["refs_from_context"] = len(refs)
    debug["refs_merged"] = len(refs)
    debug["refs_after_align"] = [_summarize_ref(ref) for ref in refs]
    debug["refs_dropped_align"] = []
    debug["scored"] = [
        {"score": 1000 - idx, **_summarize_ref(ref)} for idx, ref in enumerate(refs)
    ]
    debug["selected_paths"] = [str(ref.get("path") or "") for ref in refs]
    debug["listing_targets"] = _machine_targets_from_answer(answer or "")


def _ref_blob(ref: dict[str, Any]) -> str:
    return " ".join(
        str(ref.get(key) or "") for key in ("label", "caption", "context")
    ).strip()


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


def _ref_passes_image_align_gate(
    query: str,
    ref: dict[str, Any],
    *,
    retrieved_text: str | None,
    source_hints: set[str] | None = None,
    listing_source_text: str | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
) -> bool:
    from iqr_anchor import (
        _image_anchor_mode,
        _pick_anchor_sections,
        _ref_anchored_in_retrieved_text,
        _ref_conflicts_anchor_sections,
        _ref_context_subject_aligns,
    )
    from iqr_figure_target import (
        _anchor_maintenance_spans,
        _answer_text_for_listing,
        _label_matches_listing_target,
        _listing_mode_active,
        _listing_targets_with_query_line_overlap,
        _maintenance_content_spans,
        _ref_aligns_answer_bullets_via_inline_context,
        _ref_aligns_for_multi_figure_listing,
        _ref_matches_figure_focus,
    )

    if source_hints and not _ref_matches_source_hints(ref, source_hints):
        return False
    text = (retrieved_text or "").strip()
    list_source = (listing_source_text or text).strip()
    listing_targets = _listing_targets_with_query_line_overlap(query, list_source, text)
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
        if (
            ref_maint
            and anchor_maint
            and not any(
                text_term_alignment_symmetric(rm, am) >= 0.42
                for rm in ref_maint
                for am in anchor_maint
            )
        ):
            return False
    answer_for_images = _answer_text_for_listing()
    inline_bullet_ok = bool(
        answer_for_images
        and _ref_aligns_answer_bullets_via_inline_context(ref, answer_for_images)
    )
    listing_label_ok = _listing_mode_active(query, listing_targets) and (
        inline_bullet_ok
        or any(
            _label_matches_listing_target(_ref_effective_label(ref), target)
            for target in listing_targets
        )
    )
    if (
        retrieved_text
        and not listing_label_ok
        and not _ref_anchored_in_retrieved_text(ref, retrieved_text, query)
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
            elif _ref_inline_in_retrieved_context(ref, text):
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


def _text_alignment(left: str, right: str) -> float:
    """Symmetric overlap of discriminative terms between two text snippets."""
    return text_term_alignment_symmetric(
        left, right, min_len=_min_substantive_term_len()
    )


def _ref_label_aligns_with_line(ref: dict[str, Any], line: str) -> bool:
    label = str(ref.get("label") or ref.get("caption") or "").strip()
    if len(label) < _min_substantive_term_len():
        return False
    if label in line:
        return True
    return _text_alignment(label, line) >= _image_min_ref_align()


def _trust_llm_chunk_images() -> bool:
    """Plan A: inline figures in LLM chunks need no query-object / caption alignment."""
    raw = (os.getenv("RAG_IMAGE_TRUST_LLM_CHUNKS") or "1").strip().lower()
    return raw not in ("", "0", "false", "no", "off", "none")


def _query_aligned_figure_candidate_docs(
    query: str,
    pool: list[dict[str, Any]],
    *,
    answer: str | None = None,
    require_answer_gate: bool = False,
) -> list[dict[str, Any]]:
    from iqr_figure_target import (
        _answer_body_for_citation_match,
        _answer_weak_consistency_gate,
        _chunk_is_title_only,
        _chunk_is_toc_heavy,
    )

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


def _chunk_subject_score(query: str, content: str) -> float:
    from iqr_figure_target import _is_toc_or_directory_line

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


def _chunk_figure_context_aligns_query(query: str, content: str) -> bool:
    """Figure caption/context must align with the query action focus, not incidental terms."""
    from iqr_figure_target import _strict_object_image_gate

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
        if short_label_bag_aligns(focus, label) or short_label_bag_aligns(
            focus, blob[:80]
        ):
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


_ANSWER_REF_RE = re.compile(r"###\s*References\b.*", re.I | re.S)


_CITATION_PUNCT_RE = re.compile(r"[\s!！?？。.，,~、；;：:" r"''（）()\[\]【】\-/／·]+")


def _normalize_citation_blob(text: str) -> str:
    """Strip layout/punctuation so answer paraphrases still match chunk lines."""
    text = (text or "").strip()
    text = text.replace("*", "").replace("＜", "<").replace("＞", ">")
    return _CITATION_PUNCT_RE.sub("", text)


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


@lru_cache(maxsize=16384)
def _line_citation_overlap(answer_blob: str, line: str) -> float:
    from iqr_figure_target import _is_toc_or_directory_line

    line = (line or "").strip()
    if not line or _is_image_metadata_line(line) or _is_toc_or_directory_line(line):
        return 0.0
    if line.startswith(tuple(["[图片]", *_domain_schema.image_block_fields])):
        return 0.0
    norm = _normalize_citation_blob(line)
    if len(norm) < 12:
        return 0.0
    if norm in answer_blob:
        return 1.0
    min_sub = 7 if len(norm) >= 14 else 6
    best_len = 0
    if len(norm) > 320:
        ov = _answer_chunk_term_overlap(answer_blob, norm)
        if ov >= 0.34:
            return min(1.0, 0.55 + 0.45 * ov)
        return 0.0
    for length in range(len(norm), min_sub - 1, -1):
        for i in range(0, len(norm) - length + 1):
            if norm[i : i + length] in answer_blob:
                best_len = length
                break
        if best_len:
            break
    if best_len < min_sub:
        return 0.0
    return best_len / max(len(norm), 1)


@lru_cache(maxsize=4096)
def _chunk_citation_score(answer_blob: str, content: str) -> float:
    from iqr_anchor import _section_ids_in_text

    if not answer_blob or not (content or "").strip():
        return 0.0
    best = 0.0
    for line in content.splitlines():
        best = max(best, _line_citation_overlap(answer_blob, line))
    for field in _domain_schema.section_markers:
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


def _image_retrieval_focus_min_overlap() -> float:
    raw = os.getenv("RAG_IMAGE_RETRIEVAL_FOCUS_MIN_OVERLAP") or "0.1"
    try:
        return float(raw)
    except ValueError:
        return 0.1


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


def _ref_effective_label(ref: dict[str, Any]) -> str:
    from iqr_anchor import _is_section_number_heading, _strip_section_prefix
    from iqr_figure_target import _source_figure_label

    raw = _source_figure_label(ref) or str(ref.get("label") or "").strip()
    if _is_section_number_heading(raw):
        stripped = _strip_section_prefix(raw)
        if stripped:
            return stripped
    return raw


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
    from iqr_anchor import (
        _is_section_number_heading,
        _ref_context_subject_aligns,
        _strip_section_prefix,
    )
    from iqr_figure_target import (
        _figure_label_matches_query,
        _is_generic_cycle_only_label,
        _ref_matches_figure_focus,
    )

    label = _ref_effective_label(ref)
    if len(label) < _min_substantive_term_len():
        return False

    core_label = (
        _strip_section_prefix(label) if _is_section_number_heading(label) else label
    )
    if _is_generic_cycle_only_label(core_label) and not _ref_matches_figure_focus(
        query, ref
    ):
        return False
    label_ok = _figure_label_matches_query(query, core_label)
    if (
        not label_ok
        and _text_alignment(query, core_label) >= threshold
        and any(len(term) >= 4 and term in core_label for term in _query_terms(query))
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
