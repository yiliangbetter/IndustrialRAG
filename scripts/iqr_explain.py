"""image_query_refs submodule ``iqr_explain``.

Diagnostics and explanations for image-selection decisions (query intent,
retrieval support, human-readable explain output).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from iqr_config import (
    _image_min_rerank_score,
    _max_rerank_score,
    _min_query_chars_for_images,
)
from iqr_terms import (
    _image_min_term_overlap,
    _listing_target_head,
    _normalize_label_key,
    _normalize_query_for_match,
    _query_terms,
    _term_overlap_ratio,
)
from iqr_store import (
    _collect_figure_refs,
    _context_for_image_scan,
    _eligible_figure_refs,
    _is_multi_source_retrieval,
    _merged_source_hints,
    logger,
)
from iqr_align import (
    _apply_unified_figure_debug,
    _cross_manual_has_aligned_figures,
    _ref_effective_label,
    _ref_passes_image_align_gate,
    _truncate_ref_text,
)
from iqr_anchor import _chunk_locality_image_enabled
from iqr_figure_target import (
    _figure_context_from_answer_docs,
    _figure_label_matches_query,
    _is_answer_structural_label,
    _is_component_listing_across_machines,
    _listing_mode_active,
    _listing_targets_with_query_line_overlap,
    _machine_component_targets_from_answer,
    _machine_spans_from_answer,
    _pair_component_ref_align,
    _pair_component_spans_from_answer,
    _ref_matches_manual_hint,
    _refs_from_unified_figure_targets,
    _should_use_unified_figure_targets,
    detect_table_filter_signal,
    extract_figure_targets,
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
    if (
        anchor_meta.get("mode") not in ("off", "answer_topics")
        and not figure_context.strip()
    ):
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

    if _is_component_listing_across_machines(q) and (answer or "").strip():
        pairs = _machine_component_targets_from_answer(answer or "", query=q)
        machines = {
            m.strip()
            for m, c in pairs
            if m.strip() and c.strip() and not _is_answer_structural_label(c)
        }
        if len(machines) < 2:
            machines = set(_machine_spans_from_answer(answer or ""))
        if len(machines) >= 2:
            for scan_blob in (figure_context, primary):
                scan_blob = (scan_blob or "").strip()
                if not scan_blob:
                    continue
                eligible = _collect_figure_refs(
                    scan_blob,
                    media_roots,
                    query=query,
                    retrieved_docs=retrieved_docs,
                    full_context=primary,
                )
                if not eligible:
                    continue
                for machine in machines:
                    comp = next(
                        (
                            _listing_target_head(c)
                            for m, c in pairs
                            if _normalize_label_key(m) == _normalize_label_key(machine)
                            and c.strip()
                            and not _is_answer_structural_label(c)
                        ),
                        "",
                    )
                    for ref in eligible:
                        if not _ref_matches_manual_hint(ref, machine):
                            continue
                        if comp and _pair_component_ref_align(comp, ref) >= 0.38:
                            return True
                        for shared in _pair_component_spans_from_answer(
                            answer or "", query=q
                        ):
                            if _pair_component_ref_align(shared, ref) >= 0.38:
                                return True
                if eligible:
                    return True
            max_score = _max_rerank_score(retrieved_docs)
            if max_score is not None and max_score >= _image_min_rerank_score():
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
    if (
        overlap < min_overlap
        and not listing_ok
        and not cross_ok
        and not (
            near_miss
            and any(
                _figure_label_matches_query(q, _ref_effective_label(ref))
                for ref in eligible
            )
        )
    ):
        logger.info(
            "Skip related images: primary term overlap %.2f < %.2f",
            overlap,
            min_overlap,
        )
        return False
    logger.info("Skip related images: no figure label/heading matches query")
    if (
        _is_component_listing_across_machines(q)
        and (answer or "").strip()
        and len(_machine_spans_from_answer(answer)) >= 2
    ):
        return True
    return False


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
            answer, list(retrieved_docs or []), query=q
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
        reason = (
            "rerank_score_ok" if max_score is not None else "aligned_figures_in_primary"
        )
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
    if (
        not ok
        and _is_component_listing_across_machines(q)
        and (answer or "").strip()
        and len(_machine_spans_from_answer(answer)) >= 2
    ):
        return {
            "ok": True,
            "reason": "cross_listing_answer_machines",
            "primary_chars": len(primary),
            "overlap": overlap,
        }
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
    figure_pool: list[dict[str, Any]] | None = None,
    limit: int = 4,
    answer: str | None = None,
) -> dict[str, Any]:
    """Full image pipeline trace for debug dumps (no side effects)."""
    primary_text = (context or "").strip()
    scan_text, anchor_scan = _context_for_image_scan(
        query or "", primary_text, retrieved_docs, answer=answer
    )
    figure_context = scan_text if anchor_scan.get("mode") != "off" else primary_text
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

    if _should_use_unified_figure_targets(query or "", answer or ""):
        refs, uni_meta = _refs_from_unified_figure_targets(
            query or "",
            answer or "",
            retrieved_docs=retrieved_docs,
            cite_pool=list(figure_pool or retrieved_docs or []),
        )
        _apply_unified_figure_debug(debug, refs, uni_meta, answer=answer or "")
        return debug

    debug["image_selection"] = "legacy_retired"
    debug["unified_skip"] = {
        "chunk_locality": _chunk_locality_image_enabled(),
        "target_count": len(extract_figure_targets(query or "", answer or "")),
    }
    return debug


