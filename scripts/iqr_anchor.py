"""image_query_refs submodule ``iqr_anchor``.

Section anchoring and chunk locality: anchor-section selection, chunk
section scoring, and order-neighbor expansion for LLM chunks.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from raganything.utils import (
    discriminative_terms,
    image_label_for_item,
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
    _env_bool_image,
    _env_int_image,
    _image_min_ref_align,
    _min_substantive_term_len,
)
from iqr_terms import (
    _listing_target_head,
    _normalize_label_key,
    _query_subject_needles,
    _query_terms,
    _subject_action_clauses,
    _term_overlap_ratio,
)
from iqr_store import (
    _dedupe_doc_list_by_chunk_identity,
    _doc_basename,
    _doc_chunk_order_index,
    _doc_content,
    _doc_content_key,
    _doc_matches_cited_hints,
    _doc_storage_chunk_id,
    _load_content_list_items,
    _load_manual_chunks_for_locality,
    _pipeline_content_list_entries,
    _ranked_retrieval_lines,
    _source_hint_matches_doc,
    _unique_retrieved_doc_count,
    text_from_retrieved_docs,
)
from iqr_align import (
    _SECTION_HEADING_RE,
    _SECTION_NUM_RE,
    _answer_chunk_term_overlap,
    _chunk_citation_score,
    _chunk_figure_context_aligns_query,
    _chunk_subject_score,
    _figure_ref_passes_align_gate,
    _line_citation_overlap,
    _normalize_citation_blob,
    _query_aligned_figure_candidate_docs,
    _ref_aligns_with_retrieval_focus,
    _ref_effective_label,
    _ref_inline_context_text,
    _ref_inline_in_retrieved_context,
)


def _ref_section_subject(ref: dict[str, Any]) -> str:
    """Section subject before ``保养`` from parser heading in ref context."""
    ctx = str(ref.get("context") or "")
    match = re.search(r"\d+(?:\.\d+)+\s*(\S+?)保养", ctx)
    return match.group(1).strip() if match else ""


def _should_expand_cited_manual_kv_pool(query: str, answer: str) -> bool:
    """True when cited manuals need KV figure chunks beyond LLM retrieval."""
    from iqr_figure_target import (
        _answer_has_multi_section_markdown,
        _cited_manual_pdf_stems,
        _component_spans_from_answer,
        _is_component_listing_across_machines,
        _is_listing_scope_query,
        _is_multi_machine_comparison_query,
        _machine_from_section_title,
        _machine_spans_from_answer,
    )

    stems = _cited_manual_pdf_stems(answer)
    q = (query or "").strip()
    if len(stems) >= 2:
        if len(_machine_spans_from_answer(answer)) >= 2:
            return True
        if _is_component_listing_across_machines(q):
            return True
        if _is_multi_machine_comparison_query(q):
            return True
        if _answer_has_multi_section_markdown(answer):
            return True
        return False
    if len(stems) == 1 and _is_listing_scope_query(q):
        components = [
            c
            for c in _component_spans_from_answer(answer)
            if c and not _machine_from_section_title(c)
        ]
        if len(components) >= 2 and not _machine_spans_from_answer(answer):
            return True
    return False


def _expand_pool_with_same_section_neighbors(
    pool: list[dict[str, Any]],
    kept: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Citation pool ∪ same-manual same-section chunks (phase 3.1)."""
    if not pool or not kept:
        return list(pool or [])
    sections_by_manual: dict[str, set[str]] = {}
    for doc in kept:
        manual = _doc_basename(doc)
        section = _primary_section_id(_doc_content(doc))
        if manual and section:
            sections_by_manual.setdefault(manual, set()).add(section)
    if not sections_by_manual:
        return list(pool)
    out = list(pool)
    seen = {_doc_content_key(doc) for doc in out if _doc_content_key(doc)}
    for doc in pool:
        manual = _doc_basename(doc)
        section = _primary_section_id(_doc_content(doc))
        key = _doc_content_key(doc)
        if not manual or not section or not key or key in seen:
            continue
        if section in sections_by_manual.get(manual, set()):
            out.append(doc)
            seen.add(key)
    return out


def _chunk_locality_image_enabled() -> bool:
    return _env_bool_image("RAG_IMAGE_CHUNK_LOCALITY", True)


def _chunk_locality_window() -> int:
    return _env_int_image(
        "RAG_CHUNK_ORDER_WINDOW",
        "RAG_IMAGE_CHUNK_LOCALITY_WINDOW",
        default=8,
    )


def _chunk_is_short_section_heading(content: str) -> bool:
    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    if not lines:
        return False
    first = lines[0]
    if not _SECTION_NUM_RE.match(first):
        return False
    non_meta = [
        line
        for line in lines
        if not _is_image_metadata_line(line) and "[图片]" not in line
    ]
    return len("\n".join(non_meta)) <= 220


def _chunk_anchor_structural_adjustment(content: str) -> float:
    """Boost short section headings; penalize table mega-chunks and TOC (structural only)."""
    from iqr_figure_target import _chunk_is_table_heavy, _chunk_is_toc_heavy

    adj = 0.0
    if _chunk_is_table_heavy(content):
        adj -= 2.0
    if _chunk_is_toc_heavy(content):
        adj -= 1.5
    if _chunk_is_short_section_heading(content):
        adj += 1.25
    elif len((content or "").strip()) > 1200:
        adj -= 0.75
    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    if lines and re.match(r"^附表", lines[0]):
        adj -= 0.85
    return adj


def _anchor_cite_bonus(
    anchor_text: str,
    query: str,
    content: str,
) -> float:
    """Cite-pool boost only when chunk text aligns with bullet/query (not rerank noise)."""
    align = max(
        text_term_alignment_symmetric(anchor_text, content),
        text_term_alignment_symmetric(query, content) if query else 0.0,
    )
    if align >= 0.38:
        return 0.35
    if align >= 0.28:
        return 0.12
    return 0.0


def _chunk_section_title_line(content: str) -> str:
    """First numbered section heading in a chunk (e.g. ``3.2.1 预铣刀检查``)."""
    for line in (content or "").splitlines():
        line = line.strip()
        if not line or _is_image_metadata_line(line):
            continue
        if _SECTION_NUM_RE.match(line):
            return line[:120]
    lines = [ln.strip() for ln in (content or "").splitlines() if ln.strip()]
    return lines[0][:120] if lines else ""


def _chunk_discriminative_heading_bonus(query: str, content: str) -> float:
    """Boost anchor when section title overlaps query ``discriminative_terms``."""
    title = _chunk_section_title_line(content)
    if not title or not (query or "").strip():
        return 0.0
    bonus = 0.0
    for term in discriminative_terms(query, min_len=3):
        if len(term) < 3 or term not in title:
            continue
        bonus += min(len(term), 12) * 0.07
    return bonus


def _anchor_section_carries_subject(
    anchor_content: str,
    subjects: list[str],
) -> bool:
    from iqr_figure_target import _chunk_heading_blob

    blob = _chunk_heading_blob(anchor_content)
    if not blob:
        return False
    for subject in subjects:
        subject = (subject or "").strip()
        if len(subject) < 2:
            continue
        compact = _listing_target_head(subject)
        if subject in blob or (compact and compact in blob):
            return True
    return False


def _best_anchor_chunk(
    manual_chunks: list[dict[str, Any]],
    anchor_text: str,
    *,
    cite_pool: list[dict[str, Any]] | None = None,
    query: str = "",
    manual_hint: str = "",
) -> dict[str, Any] | None:
    """Pick anchor chunk from full manual order + manual-scoped cite pool."""
    from iqr_figure_target import (
        _cite_pool_chunk_ids,
        _doc_matches_manual_hint,
        _manual_scoped_cite_pool,
        _resolve_doc_with_order_index,
    )

    scoped_cite = (
        _manual_scoped_cite_pool(cite_pool, manual_hint)
        if manual_hint
        else list(cite_pool or [])
    )
    cite_ids = _cite_pool_chunk_ids(scoped_cite)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add_candidate(doc: dict[str, Any]) -> None:
        resolved = _resolve_doc_with_order_index(doc)
        if manual_hint and not _doc_matches_manual_hint(resolved, manual_hint):
            return
        cid = _doc_storage_chunk_id(resolved)
        key = cid or str(id(resolved))
        if key in seen:
            return
        seen.add(key)
        candidates.append(resolved)

    for doc in manual_chunks or []:
        _add_candidate(doc)
    for doc in scoped_cite:
        _add_candidate(doc)

    if not candidates:
        return None

    answer_blob = _normalize_citation_blob(anchor_text)
    scored: list[tuple[float, float, float, dict[str, Any]]] = []
    for doc in candidates:
        content = _doc_content(doc).strip()
        if not content:
            continue
        anchor_align = text_term_alignment_symmetric(anchor_text, content)
        query_align = text_term_alignment_symmetric(query, content) if query else 0.0
        score = max(
            _line_citation_overlap(answer_blob, content),
            _chunk_citation_score(answer_blob, content),
        )
        score += anchor_align * 0.45
        if query:
            score += query_align * 0.25
            score += _chunk_subject_score(query, content) * 0.15
            score += _chunk_discriminative_heading_bonus(query, content)
        score += _chunk_anchor_structural_adjustment(content)
        doc_id = _doc_storage_chunk_id(doc)
        if doc_id and doc_id in cite_ids:
            score += _anchor_cite_bonus(anchor_text, query, content)
        scored.append((score, anchor_align, query_align, doc))

    if not scored:
        return None
    scored.sort(
        key=lambda row: (
            -row[0],
            -row[1],
            -row[2],
            _doc_chunk_order_index(row[3]) or 0,
        )
    )
    return _resolve_doc_with_order_index(scored[0][3])


def _anchor_align_blob(anchor_text: str, query: str) -> str:
    parts = [(anchor_text or "").strip(), (query or "").strip()]
    return "\n".join(p for p in parts if p)


def _best_inline_figure_from_pool(
    anchor_text: str,
    pool: list[dict[str, Any]],
    *,
    query: str,
    manual_hint: str = "",
    kind: str = "",
    component: str = "",
    strict_topic: bool = False,
    anchor_content: str = "",
    exclude_paths: set[str] | None = None,
    exclude_chunk_ids: set[str] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    """Best inline-figure chunk in pool aligned to anchor (cite / manual evidence)."""
    from iqr_figure_target import (
        _answer_bullet_component_head,
        _doc_matches_manual_hint,
        _figure_ref_matches_listing_target,
        _figure_ref_matches_target_topic,
        _is_cover_page_ref,
        _machine_bullet_subject,
    )

    align_blob = _anchor_align_blob(anchor_text, query)
    head = (
        _answer_bullet_component_head(anchor_text)
        if kind == "answer_bullet"
        else _listing_target_head(anchor_text)
    )
    machine_subject = _machine_bullet_subject(anchor_text)
    excluded_paths = set(exclude_paths or [])
    excluded_chunks = set(exclude_chunk_ids or [])
    ranked: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    section_anchor = (anchor_content or "").strip()
    for doc in pool:
        if manual_hint and not _doc_matches_manual_hint(doc, manual_hint):
            continue
        doc_id = _doc_storage_chunk_id(doc)
        if doc_id and doc_id in excluded_chunks:
            continue
        content = _doc_content(doc).strip()
        if not content or not extract_image_refs_from_context(content):
            continue
        refs = extract_image_refs_from_context(content)
        if not refs:
            continue
        for ref in refs:
            if _is_cover_page_ref(ref):
                continue
            path_key = Path(str(ref.get("path") or "")).name
            if path_key and path_key in excluded_paths:
                continue
            if not _figure_ref_passes_align_gate(
                anchor_text,
                query,
                ref,
                doc_content=content,
            ):
                continue
            if strict_topic and not _figure_ref_matches_target_topic(
                anchor_text,
                query,
                ref,
                kind=kind,
                component=component,
                anchor_content=section_anchor,
            ):
                continue
            hay = "\n".join(
                p
                for p in (
                    _ref_effective_label(ref),
                    _ref_inline_context_text(ref),
                    content,
                )
                if p
            )
            score = text_term_alignment_symmetric(align_blob, hay)
            if head and _figure_ref_matches_listing_target(ref, head):
                score += 0.25
            if machine_subject and _figure_ref_matches_listing_target(
                ref, machine_subject
            ):
                score += 0.25
            ranked.append((score, ref, doc))
    if not ranked:
        return None, None, ""
    ranked.sort(key=lambda row: (-row[0], _doc_chunk_order_index(row[2]) or 0))
    best_ref, best_doc = ranked[0][1], ranked[0][2]
    return best_ref, best_doc, "cite_pool"


def _figure_ref_from_content_list_for_target(
    anchor_text: str,
    manual_hint: str,
    *,
    query: str,
    kind: str = "",
    component: str = "",
    anchor_content: str = "",
) -> tuple[dict[str, Any] | None, str]:
    """content_list + ``best_image_for_text_item`` fallback (same priority as ingest)."""
    from iqr_figure_target import (
        _doc_matches_manual_hint,
        _figure_ref_matches_target_topic,
        _is_cover_page_ref,
    )
    from raganything.utils import best_image_for_text_item, context_text_for_image

    align_blob = _anchor_align_blob(anchor_text, query)
    head = _listing_target_head(anchor_text)
    min_align = _image_min_ref_align()
    best_ref: dict[str, Any] | None = None
    best_score = 0.0

    for cl_path, doc_hint, auto_dir in _pipeline_content_list_entries():
        if manual_hint and not (
            _doc_matches_manual_hint({"file_path": doc_hint + ".pdf"}, manual_hint)
            or _source_hint_matches_doc(manual_hint, doc_hint)
        ):
            continue
        items = _load_content_list_items(cl_path)
        if not items:
            continue
        for ti, item in enumerate(items):
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = str(item.get("text") or "").strip()
            if len(text) < 4:
                continue
            align = text_term_alignment_symmetric(align_blob, text)
            if head and head in text:
                align = max(align, min_align)
            if align < min_align * 0.7:
                continue
            img_item = best_image_for_text_item(items, ti)
            if img_item is None:
                continue
            try:
                img_idx = items.index(img_item)
            except ValueError:
                img_idx = -1
            ctx = context_text_for_image(items, img_idx) if img_idx >= 0 else ""
            label = image_label_for_item(items, img_item) if img_idx >= 0 else ""
            rel_path = (img_item.get("img_path") or "").strip()
            if not rel_path:
                continue
            full_path = (auto_dir / rel_path).resolve()
            ref = {
                "path": str(full_path),
                "page": img_item.get("page_idx")
                if isinstance(img_item.get("page_idx"), int)
                else None,
                "caption": label,
                "label": label,
                "context": "\n".join(p for p in (text[:300], ctx.strip()) if p)[:400],
            }
            if _is_cover_page_ref(ref):
                continue
            if not _figure_ref_passes_align_gate(
                anchor_text,
                query,
                ref,
                doc_content=ctx,
            ):
                continue
            section_anchor = (anchor_content or text).strip()
            if not _figure_ref_matches_target_topic(
                anchor_text,
                query,
                ref,
                kind=kind,
                component=component,
                anchor_content=section_anchor,
            ):
                continue
            score = align + text_term_alignment_symmetric(
                align_blob,
                " ".join(part for part in (label, ctx) if part),
            )
            if score > best_score:
                best_score = score
                best_ref = ref
    if best_ref is None:
        return None, ""
    return best_ref, "content_list"


def _figure_doc_for_anchor_neighbor(
    anchor: dict[str, Any],
    manual_chunks: list[dict[str, Any]],
    *,
    window: int,
    anchor_text: str = "",
    query: str = "",
    manual_hint: str = "",
    kind: str = "",
    component: str = "",
) -> tuple[dict[str, Any] | None, str]:
    """Return (doc_with_figure, source) where source is ``anchor`` or ``neighbor``."""
    from iqr_figure_target import (
        _doc_matches_manual_hint,
        _figure_ref_matches_target_topic,
        _is_cover_page_ref,
        _pick_figure_ref_for_target_kind,
        _resolve_doc_with_order_index,
    )

    anchor = _resolve_doc_with_order_index(anchor)
    if manual_hint and not _doc_matches_manual_hint(anchor, manual_hint):
        return None, ""

    align_blob = _anchor_align_blob(anchor_text, query)
    min_align = _image_min_ref_align()
    anchor_content = _doc_content(anchor).strip()
    if anchor_content and extract_image_refs_from_context(anchor_content):
        ref = _pick_figure_ref_for_target_kind(
            anchor,
            kind=kind,
            anchor_text=anchor_text,
            query=query,
            component=component,
            anchor_content=anchor_content,
        )
        if ref is not None and not _is_cover_page_ref(ref):
            return anchor, "anchor"

    anchor_idx = _doc_chunk_order_index(anchor)
    if anchor_idx is None or not manual_chunks:
        return None, ""

    anchor_id = str(anchor.get("id") or anchor.get("chunk_id") or "").strip()
    anchor_section = _primary_section_id(anchor_content)
    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in manual_chunks:
        doc = _resolve_doc_with_order_index(doc)
        if manual_hint and not _doc_matches_manual_hint(doc, manual_hint):
            continue
        doc_id = str(doc.get("id") or doc.get("chunk_id") or "").strip()
        if anchor_id and doc_id and doc_id == anchor_id:
            continue
        idx = _doc_chunk_order_index(doc)
        if idx is None:
            continue
        dist = abs(idx - anchor_idx)
        if dist <= 0 or dist > window:
            continue
        content = _doc_content(doc).strip()
        if not content or not extract_image_refs_from_context(content):
            continue
        score = 100.0 - float(dist)
        sec = _primary_section_id(content)
        if anchor_section and sec and _sections_compatible(anchor_section, sec):
            score += 50.0
        elif anchor_section and anchor_section in content:
            score += 25.0
        if align_blob:
            label = image_label_text(content) if content else ""
            align = max(
                text_term_alignment_symmetric(align_blob, content),
                text_term_alignment_symmetric(align_blob, label) if label else 0.0,
            )
            score += align * 45.0
            if align < min_align * 0.85:
                score -= 25.0
        for needle in _query_subject_needles(query):
            if len(needle) >= 3 and needle in content:
                score += 22.0
                break
        scored.append((score, doc))
    if not scored:
        return None, ""
    scored.sort(key=lambda pair: (-pair[0], _doc_chunk_order_index(pair[1]) or 0))
    for _score, doc in scored:
        ref = _pick_figure_ref_for_target_kind(
            doc,
            kind=kind,
            anchor_text=anchor_text,
            query=query,
            component=component,
            anchor_content=anchor_content,
        )
        if ref is None:
            continue
        align_content = "\n".join(
            p
            for p in (
                anchor_content,
                _doc_content(doc).strip(),
            )
            if p
        )
        if not _figure_ref_passes_align_gate(
            anchor_text,
            query,
            ref,
            doc_content=align_content,
        ):
            continue
        if _is_cover_page_ref(ref):
            continue
        if not _figure_ref_matches_target_topic(
            anchor_text,
            query,
            ref,
            kind=kind,
            component=component,
            anchor_content=anchor_content,
        ):
            continue
        return doc, "neighbor"
    return None, ""


def _best_figure_ref_for_anchor_align(
    doc: dict[str, Any],
    *,
    anchor_text: str = "",
    query: str = "",
    anchor_content: str = "",
) -> dict[str, Any] | None:
    """Best inline figure in ``doc`` for anchor/query alignment (not always first)."""
    doc_content = _doc_content(doc).strip()
    refs = extract_image_refs_from_context(doc_content)
    if not refs:
        return None
    align_blob = _anchor_align_blob(anchor_text, query)
    anchor_blob = (anchor_content or "").strip()
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, ref in enumerate(refs):
        hay = "\n".join(
            p
            for p in (
                _ref_effective_label(ref),
                _ref_inline_context_text(ref),
            )
            if p
        )
        score = 0.0
        if align_blob:
            score += text_term_alignment_symmetric(align_blob, hay) * 2.0
        if anchor_blob:
            score += text_term_alignment_symmetric(anchor_blob, hay) * 3.0
        scored.append((score, idx, ref))
    if not scored:
        return None
    scored.sort(key=lambda row: (row[0], row[1]))
    best_score = scored[-1][0]
    tied = [ref for score, _idx, ref in scored if score >= best_score - 1e-6]
    return tied[-1]


def _load_figure_chunks_for_manual_paths(
    allowed_paths: set[str],
    query: str,
    *,
    max_per_manual: int = 14,
) -> list[dict[str, Any]]:
    """Load inline-figure chunks from kv store for cited / pool manuals (plan C)."""
    if not _env_bool_image("RAG_IMAGE_CITED_MANUAL_KV_POOL", True):
        return []
    if not allowed_paths:
        return []
    try:
        from query_doc_steering import (  # noqa: WPS433
            _load_manual_chunks_for_paths,
            active_deny_substrings,
        )
    except ImportError:
        return []
    deny, _ = active_deny_substrings(query or "")
    loaded = _load_manual_chunks_for_paths(allowed_paths, deny)
    if not loaded:
        return []
    q_terms = [t for t in discriminative_terms(query or "", min_len=2) if len(t) >= 2]
    focus_terms = sorted(
        {t for t in q_terms if 2 <= len(t) <= 8},
        key=len,
        reverse=True,
    )
    by_manual: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for doc in loaded:
        content = _doc_content(doc).strip()
        if not content or not extract_image_refs_from_context(content):
            continue
        term_hit = focus_terms and any(term in content for term in focus_terms)
        if (
            q_terms
            and not term_hit
            and not _chunk_figure_context_aligns_query(query or "", content)
        ):
            continue
        manual = _doc_basename(doc)
        if not manual:
            continue
        score = max(
            _term_overlap_ratio(query or "", content),
            _chunk_subject_score(query or "", content),
        )
        if term_hit:
            score = max(score, 0.2)
        if _chunk_figure_context_aligns_query(query or "", content):
            score += 0.22
        if score < 0.08 and not term_hit:
            continue
        by_manual.setdefault(manual, []).append((score, doc))
    out: list[dict[str, Any]] = []
    for items in by_manual.values():
        items.sort(key=lambda pair: pair[0], reverse=True)
        out.extend(doc for _, doc in items[:max_per_manual])
    return out


def _supplement_component_listing_figure_chunks(
    pool: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    *,
    query: str | None,
    answer: str,
) -> list[dict[str, Any]]:
    """One inline-figure chunk per (machine, component) pair, scoped to cited manuals."""
    from iqr_figure_target import (
        _answer_body_for_citation_match,
        _best_figure_doc_for_component,
        _cited_manual_hints_from_answer,
        _machine_component_listing_pair_targets,
    )

    pairs = _machine_component_listing_pair_targets(
        query or "", answer, pool, kept=kept
    )
    if len(pairs) < 2:
        return kept
    cited_hints = _cited_manual_hints_from_answer(answer)
    answer_blob = _answer_body_for_citation_match(answer)
    expanded = _expand_pool_with_same_section_neighbors(pool, kept)
    out = list(kept)
    kept_ids = {id(doc) for doc in out}
    seen_pairs: set[tuple[str, str]] = set()
    for machine_hint, component in pairs:
        pair_key = (
            _normalize_label_key(machine_hint),
            _normalize_label_key(component),
        )
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        best_doc = _best_figure_doc_for_component(
            component,
            expanded,
            answer_blob=answer_blob,
            manual_hint=machine_hint,
            cited_hints=cited_hints,
            query=query or "",
        )
        if best_doc is not None and id(best_doc) not in kept_ids:
            out.append(best_doc)
            kept_ids.add(id(best_doc))
    return out


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


def _supplement_cross_manual_figure_chunks(
    all_docs: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    *,
    query: str | None,
    answer: str,
) -> list[dict[str, Any]]:
    """Keep one inline-figure chunk per manual when the answer compares multiple models."""
    from iqr_figure_target import (
        _answer_body_for_citation_match,
        _answer_has_multi_section_markdown,
        _is_component_listing_across_machines,
        _is_multi_machine_comparison_query,
    )

    if _is_component_listing_across_machines(query or ""):
        return _supplement_component_listing_figure_chunks(
            all_docs, kept, query=query, answer=answer
        )
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
        cite = _chunk_citation_score(_answer_body_for_citation_match(answer), content)
        combined = max(q_ov, cite)
        min_combined = 0.04 if _is_multi_machine_comparison_query(query or "") else 0.08
        if combined < min_combined:
            continue
        if _is_multi_machine_comparison_query(query or ""):
            q_terms = [
                t for t in discriminative_terms(query or "", min_len=2) if len(t) >= 2
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


def _ref_conflicts_anchor_sections(
    anchor_sections: list[str],
    ref: dict[str, Any],
    *,
    query: str | None = None,
    primary_text: str | None = None,
) -> bool:
    """True when a figure cites an explicit section incompatible with answer anchors."""
    from iqr_figure_target import _ref_maint_section_id

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
    from iqr_figure_target import _anchor_maintenance_spans, _maintenance_content_spans

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
        if (
            anchor_sections
            and sid
            and not any(_sections_compatible(anchor, sid) for anchor in anchor_sections)
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


def query_section_anchor_enabled() -> bool:
    raw = (os.getenv("RAG_IMAGE_QUERY_SECTION_ANCHOR") or "1").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _query_section_anchor_min_score() -> float:
    raw = os.getenv("RAG_IMAGE_QUERY_SECTION_MIN_SCORE") or "0.28"
    try:
        return max(0.12, min(2.0, float(raw)))
    except ValueError:
        return 0.28


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


def _chunk_query_section_score(
    query: str,
    content: str,
    *,
    anchor_sections: list[str],
    pool_primary: str,
) -> float:
    from iqr_figure_target import _MAINT_TOPIC_RE

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
    from iqr_figure_target import (
        _answer_body_for_citation_match,
        _answer_weak_consistency_gate,
        _chunk_is_title_only,
        _chunk_is_toc_heavy,
        _is_listing_scope_query,
        _retrieval_prefers_catalog_field,
    )

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
                and not any(needle in content for needle in subject_needles)
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


def _best_section_id_from_content(query: str, content: str) -> str | None:
    """Prefer a section heading that contains query subject terms (e.g. 2.1.1 机床床身清洁)."""
    from iqr_figure_target import _chunk_is_toc_heavy, _is_toc_or_directory_line

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
    from iqr_figure_target import _chunk_is_toc_heavy, _is_toc_or_directory_line

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
        section_id = _best_section_id_from_content(
            query, content
        ) or _primary_section_id(content)
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


def _anchor_inline_same_chunk(
    anchor_doc: dict[str, Any] | None,
    fig_doc: dict[str, Any] | None,
) -> bool:
    if anchor_doc is None or fig_doc is None:
        return False
    aid = _doc_storage_chunk_id(anchor_doc)
    fid = _doc_storage_chunk_id(fig_doc)
    return bool(aid and fid and aid == fid)


def _supplement_answer_topic_figure_chunks(
    pool: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    *,
    answer: str,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Add one inline-figure chunk per answer component topic from the search pool."""
    from iqr_figure_target import (
        _answer_body_for_citation_match,
        _chunk_matches_answer_topic,
        _cited_manual_hints_from_answer,
        _doc_matches_manual_hint,
        _is_component_listing_across_machines,
        _is_multi_machine_comparison_query,
        _label_matches_listing_target,
        _machine_spans_from_answer,
        _manual_hint_for_component,
        _span_keep_listing_targets,
    )

    topics = _span_keep_listing_targets(query or "", answer, pool, kept=kept)
    if len(topics) < 2:
        return kept
    answer_blob = _answer_body_for_citation_match(answer)
    machine_names = set(_machine_spans_from_answer(answer))
    cited_hints = (
        _cited_manual_hints_from_answer(answer)
        if _is_component_listing_across_machines(query or "")
        else set()
    )
    expanded = (
        _expand_pool_with_same_section_neighbors(pool, kept)
        if _is_component_listing_across_machines(query or "")
        else pool
    )
    out = list(kept)
    kept_ids = {id(doc) for doc in out}
    for topic in topics:
        manual_hint = (
            _manual_hint_for_component(topic, answer)
            if _is_component_listing_across_machines(query or "")
            else ""
        )
        best_doc: dict[str, Any] | None = None
        best_score = 0.0
        for doc in expanded:
            if cited_hints and not _doc_matches_cited_hints(doc, cited_hints):
                continue
            if manual_hint and not _doc_matches_manual_hint(doc, manual_hint):
                continue
            content = _doc_content(doc).strip()
            if not content or not extract_image_refs_from_context(content):
                continue
            topic_ov = max(
                _answer_chunk_term_overlap(topic, content),
                text_term_alignment_symmetric(topic, content),
            )
            cite = _chunk_citation_score(answer_blob, content) if answer_blob else 0.0
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


def llm_chunk_locality_enabled() -> bool:
    raw = (os.getenv("RAG_QUERY_CHUNK_LOCALITY") or "1").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _llm_chunk_locality_window() -> int:
    return _env_int_image(
        "RAG_CHUNK_ORDER_WINDOW",
        "RAG_QUERY_CHUNK_LOCALITY_WINDOW",
        "RAG_IMAGE_CHUNK_LOCALITY_WINDOW",
        default=2,
    )


def _llm_chunk_locality_max_add() -> int:
    return _env_int_image(
        "RAG_CHUNK_ORDER_MAX_ADD",
        "RAG_QUERY_CHUNK_LOCALITY_MAX_ADD",
        default=3,
    )


def _llm_chunk_locality_skipped(query: str) -> str | None:
    from iqr_figure_target import (
        _is_catalog_or_model_listing_query,
        _is_listing_scope_query,
    )

    q = (query or "").strip()
    if not q:
        return "empty_query"
    if _is_listing_scope_query(q) or _is_catalog_or_model_listing_query(q):
        return "listing_or_catalog"
    return None


def _order_neighbor_candidates_for_anchor(
    anchor: dict[str, Any],
    manual_chunks: list[dict[str, Any]],
    *,
    window: int,
    seen: set[str],
) -> list[tuple[int, dict[str, Any]]]:
    from iqr_figure_target import _chunk_is_toc_heavy, _resolve_doc_with_order_index

    anchor = _resolve_doc_with_order_index(anchor)
    anchor_idx = _doc_chunk_order_index(anchor)
    if anchor_idx is None or not manual_chunks:
        return []
    anchor_id = _doc_storage_chunk_id(anchor)
    out: list[tuple[int, dict[str, Any]]] = []
    for doc in manual_chunks:
        doc_id = _doc_storage_chunk_id(doc)
        if anchor_id and doc_id and doc_id == anchor_id:
            continue
        idx = _doc_chunk_order_index(doc)
        if idx is None:
            continue
        dist = abs(idx - anchor_idx)
        if dist <= 0 or dist > window:
            continue
        ident = _doc_storage_chunk_id(doc) or _doc_content_key(doc)
        if not ident or ident in seen:
            continue
        content = _doc_content(doc).strip()
        if not content or _chunk_is_toc_heavy(content):
            continue
        out.append((dist, doc))
    anchor_idx = _doc_chunk_order_index(anchor) or 0
    out.sort(
        key=lambda pair: (
            pair[0],
            0 if (_doc_chunk_order_index(pair[1]) or 0) > anchor_idx else 1,
            _doc_chunk_order_index(pair[1]) or 0,
        )
    )
    return out


def supplement_unique_chunks_with_order_neighbors(
    query: str,
    docs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pre-rerank: expand retrieval pool with ±order_index neighbors (same manual)."""
    from iqr_figure_target import _resolve_doc_with_order_index

    meta: dict[str, Any] = {"mode": "off", "added": 0, "phase": "pre_rerank"}
    if not llm_chunk_locality_enabled():
        meta["reason"] = "disabled"
        return list(docs or []), meta
    skip = _llm_chunk_locality_skipped(query)
    if skip:
        meta["reason"] = skip
        return list(docs or []), meta
    if not docs:
        return [], meta

    window = _llm_chunk_locality_window()
    max_add = _llm_chunk_locality_max_add()
    seen: set[str] = set()
    for doc in docs:
        ident = _doc_storage_chunk_id(doc) or _doc_content_key(doc)
        if ident:
            seen.add(ident)

    manual_cache: dict[str, list[dict[str, Any]]] = {}
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for doc in docs:
        resolved = _resolve_doc_with_order_index(doc)
        manual = _doc_basename(resolved)
        idx = _doc_chunk_order_index(resolved)
        if not manual or idx is None:
            continue
        if manual not in manual_cache:
            manual_cache[manual] = _load_manual_chunks_for_locality(manual)
        for dist, neighbor in _order_neighbor_candidates_for_anchor(
            resolved, manual_cache[manual], window=window, seen=seen
        ):
            n_idx = _doc_chunk_order_index(neighbor) or 0
            candidates.append((dist, n_idx, neighbor))

    candidates.sort(key=lambda row: (row[0], row[1]))
    out = list(docs)
    added = 0
    for _dist, _idx, neighbor in candidates:
        ident = _doc_storage_chunk_id(neighbor) or _doc_content_key(neighbor)
        if not ident or ident in seen:
            continue
        out.append(neighbor)
        seen.add(ident)
        added += 1
        if added >= max_add:
            break

    meta["mode"] = "order_neighbors" if added else "off"
    meta["added"] = added
    meta["window"] = window
    return _dedupe_doc_list_by_chunk_identity(out), meta


def merge_order_neighbors_into_llm_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    global_config: dict[str, Any] | None = None,
    query_param: Any | None = None,
) -> list[dict[str, Any]]:
    """Post top-k: insert missing ±order_index neighbors, then re-apply token truncation."""
    from iqr_figure_target import _relabel_dc_chunks, _resolve_doc_with_order_index

    if not llm_chunk_locality_enabled() or not chunks:
        return list(chunks)
    if _llm_chunk_locality_skipped(query):
        return list(chunks)

    window = _llm_chunk_locality_window()
    max_add = _llm_chunk_locality_max_add()
    present: set[str] = set()
    for doc in chunks:
        ident = _doc_storage_chunk_id(doc) or _doc_content_key(doc)
        if ident:
            present.add(ident)

    manual_cache: dict[str, list[dict[str, Any]]] = {}
    out: list[dict[str, Any]] = []
    added = 0
    for doc in chunks:
        out.append(doc)
        if added >= max_add:
            continue
        resolved = _resolve_doc_with_order_index(doc)
        manual = _doc_basename(resolved)
        if not manual or _doc_chunk_order_index(resolved) is None:
            continue
        if manual not in manual_cache:
            manual_cache[manual] = _load_manual_chunks_for_locality(manual)
        for _dist, neighbor in _order_neighbor_candidates_for_anchor(
            resolved, manual_cache[manual], window=window, seen=present
        ):
            ident = _doc_storage_chunk_id(neighbor) or _doc_content_key(neighbor)
            if not ident or ident in present:
                continue
            out.append(neighbor)
            present.add(ident)
            added += 1
            if added >= max_add:
                break

    if not added:
        return list(chunks)

    gconf = global_config or {}
    tokenizer = gconf.get("tokenizer")
    if tokenizer and query_param is not None:
        try:
            from lightrag.utils import (  # noqa: WPS433
                DEFAULT_MAX_TOTAL_TOKENS,
                truncate_list_by_token_size,
            )
        except ImportError:
            return _relabel_dc_chunks(out)

        chunk_token_limit = getattr(query_param, "max_total_tokens", None) or gconf.get(
            "MAX_TOTAL_TOKENS", DEFAULT_MAX_TOTAL_TOKENS
        )
        out = truncate_list_by_token_size(
            out,
            key=lambda x: "\n".join(
                json.dumps(item, ensure_ascii=False) for item in [x]
            ),
            max_token_size=chunk_token_limit,
            tokenizer=tokenizer,
        )
    return _relabel_dc_chunks(out)


def _find_anchor_in_answer(answer: str, anchor: str) -> tuple[int, int, float]:
    """Return (start, end, score) in answer text for inserting a figure after anchor."""
    from iqr_figure_target import _answer_text_for_placement

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


def _strip_section_prefix(label: str) -> str:
    return re.sub(r"^[\d\.\s]+", "", (label or "").strip()).strip()


def _is_section_number_heading(label: str) -> bool:
    stripped = (label or "").strip()
    if not stripped:
        return False
    return bool(_SECTION_HEADING_RE.match(stripped))
