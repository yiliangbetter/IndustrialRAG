"""image_query_refs submodule ``iqr_placement``.

Inline image placement: semantic/caption placements, end fallback, and
API image assembly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from raganything.utils import text_term_alignment_symmetric
from iqr_protocol import _image_score_for_logic_line
from iqr_media import (
    encode_media_token,
    resolve_media_path,
)
from iqr_config import (
    _inline_min_place_score,
    _multi_figure_image_limit,
)
from iqr_terms import (
    _is_procedure_steps_query,
    _listing_target_head,
    _normalize_label_key,
    _procedure_step_spans,
    _query_subject_needles,
)
from iqr_store import (
    _source_key_from_path,
    logger,
)
from iqr_align import _ref_effective_label
from iqr_anchor import _find_anchor_in_answer
from iqr_figure_target import (
    _answer_listing_spans,
    _answer_logic_lines,
    _answer_paragraph_blocks,
    _answer_text_for_placement,
    _build_machine_bullet_manual_placements,
    _is_maintenance_cycle_query,
    _is_maintenance_cycle_value,
    _label_matches_listing_target,
    _machine_bullet_lines_from_answer,
    _refs_from_unified_figure_targets,
    _should_use_unified_figure_targets,
)
from iqr_explain import (
    explain_query_images,
    retrieval_supports_images,
)


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
    *,
    keep_unplaced: bool = False,
) -> list[dict[str, Any]]:
    """Compact image_index values; optionally drop images without placements."""
    if keep_unplaced:
        reindexed: list[dict[str, Any]] = []
        for pl in placements:
            idx = int(pl["image_index"])
            if idx < 0 or idx >= len(images):
                continue
            reindexed.append(dict(pl))
        return reindexed
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


def _cycle_caption_block_placement(
    answer: str, caption: str
) -> tuple[int, int, str] | None:
    """Cycle answers: anchor figure after a full paragraph, not a parenthetical aside."""
    caption = (caption or "").strip()
    if not caption:
        return None
    body = _answer_text_for_placement(answer)
    blocks = _answer_paragraph_blocks(answer)
    if not blocks:
        return None

    def chunk_matches_caption(chunk: str) -> bool:
        return caption in chunk or text_term_alignment_symmetric(caption, chunk) >= 0.4

    for chunk, bstart, bend in reversed(blocks):
        for line in chunk.splitlines():
            stripped = line.strip()
            bullet = re.match(r"^[-*•]\s*\*\*([^*]+)\*\*", stripped)
            if not bullet:
                continue
            title = _listing_target_head(bullet.group(1).strip())
            if not title or not _label_matches_listing_target(caption, title):
                continue
            line_start = body.find(stripped, bstart, bend)
            if line_start < 0:
                line_start = body.find(stripped)
            if line_start < 0:
                continue
            line_end = line_start + len(stripped)
            return line_start, line_end, stripped

    for chunk, bstart, bend in reversed(blocks):
        if not chunk_matches_caption(chunk):
            continue
        if len(blocks) > 1 and chunk == blocks[0][0]:
            lead = chunk.splitlines()[0].strip()
            if re.search(r"保养一次|每[天周月季年]", lead) and caption in lead:
                tail = chunk[chunk.find(caption) + len(caption) :]
                if (
                    not tail.strip()
                    or tail.strip().startswith("）")
                    or tail.strip().startswith(")")
                ):
                    continue
        return bstart, bend, chunk

    chunk, bstart, bend = blocks[-1]
    return bstart, bend, chunk


def _build_caption_fallback_placements(
    answer: str,
    images: list[dict[str, Any]],
    listing_spans: list[str],
    min_score: float,
    *,
    query: str = "",
) -> list[dict[str, Any]]:
    """Single-entry answers: pair each image with best caption/listing anchor."""
    answer_body = _answer_text_for_placement(answer)
    placements: list[dict[str, Any]] = []
    used_ranges: list[tuple[int, int]] = []
    q = (query or "").strip()
    procedure_steps = (
        _procedure_step_spans(answer) if _is_procedure_steps_query(q) else []
    )
    cycle_query = _is_maintenance_cycle_query(q)

    for image_index, img in enumerate(images):
        caption = str(img.get("caption") or "").strip()
        if cycle_query and caption:
            block_hit = _cycle_caption_block_placement(answer, caption)
            if block_hit:
                bstart, bend, display_anchor = block_hit
                overlap = any(
                    not (bend <= u0 or bstart >= u1) for u0, u1 in used_ranges
                )
                if not overlap:
                    used_ranges.append((bstart, bend))
                    placements.append(
                        {
                            "anchor_text": display_anchor,
                            "match_start": bstart,
                            "match_end": bend,
                            "image_index": image_index,
                            "score": 0.88,
                        }
                    )
                    continue
        anchors: list[str] = []
        if caption:
            anchors.append(caption)
        if procedure_steps:
            for step in procedure_steps:
                if step not in anchors:
                    anchors.append(step)
        elif cycle_query:
            for needle in _query_subject_needles(q):
                if needle not in anchors:
                    anchors.append(needle)
            for span in listing_spans:
                if (
                    span
                    and not _is_maintenance_cycle_value(span)
                    and span not in anchors
                ):
                    anchors.append(span)
        else:
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

        best_start, best_end, best_effective, best_anchor = -1, -1, -1.0, ""
        best_label_match = False
        seen_anchor: set[str] = set()
        for anchor in anchors:
            key = _normalize_label_key(anchor)
            if not key or key in seen_anchor:
                continue
            seen_anchor.add(key)
            start, end, score = _find_anchor_in_answer(answer, anchor)
            if start < 0:
                continue
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
            if caption:
                head = _listing_target_head(anchor)
                if text_term_alignment_symmetric(head or anchor, caption) >= 0.45:
                    effective += 0.08
            if cycle_query and _is_maintenance_cycle_value(anchor):
                effective -= 0.6
            elif cycle_query and caption:
                for needle in _query_subject_needles(q):
                    if (
                        needle in anchor
                        and text_term_alignment_symmetric(needle, caption) >= 0.35
                    ):
                        effective += 0.15
                        break
            if procedure_steps and anchor not in procedure_steps:
                first_step = procedure_steps[0]
                step_pos = answer_body.find(f"**{first_step}**")
                if step_pos > 0 and 0 <= start < step_pos:
                    continue
            if effective > best_effective or (
                abs(effective - best_effective) < 1e-6
                and label_match
                and not best_label_match
            ):
                best_start, best_end, best_effective, best_anchor = (
                    start,
                    end,
                    effective,
                    anchor,
                )
                best_label_match = label_match

        if best_start < 0 or best_effective < min_score:
            continue
        overlap = any(
            not (best_end <= u0 or best_start >= u1) for u0, u1 in used_ranges
        )
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
                "score": round(best_effective, 3),
            }
        )

    placements.sort(key=lambda item: item["match_start"])
    return placements


def _build_semantic_inline_placements(
    answer: str,
    images: list[dict[str, Any]],
    *,
    query: str | None = None,
    min_score: float,
) -> list[dict[str, Any]]:
    """Format-agnostic placement: one figure per semantic answer line when possible."""
    logic_lines = _answer_logic_lines(answer, query=query or "")
    placements: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    used_ranges: list[tuple[int, int]] = []

    for line in logic_lines:
        span_score = min(1.0, 0.85 + 0.15 * min(1.0, len(line.text) / 40.0))
        if span_score < min_score:
            continue
        best_idx = -1
        best_align = -1.0
        for idx, img in enumerate(images):
            if idx in used_indices:
                continue
            align = _image_score_for_logic_line(line, img, query=query or "")
            if align > best_align:
                best_align = align
                best_idx = idx
        if best_idx < 0:
            continue
        overlap = any(
            not (line.end <= u0 or line.start >= u1) for u0, u1 in used_ranges
        )
        if overlap:
            continue
        used_indices.add(best_idx)
        used_ranges.append((line.start, line.end))
        placements.append(
            {
                "anchor_text": line.text,
                "match_start": line.start,
                "match_end": line.end,
                "image_index": best_idx,
                "score": round(span_score, 3),
            }
        )

    if placements:
        placements.sort(key=lambda item: item["match_start"])
        return placements

    listing_spans = _answer_listing_spans(answer)
    return _build_caption_fallback_placements(
        answer, images, listing_spans, min_score, query=query or ""
    )


def build_inline_placements(
    answer: str,
    images: list[dict[str, Any]],
    *,
    query: str | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Map each selected image to an answer span for inline Web rendering."""
    del retrieved_docs
    if not (answer or "").strip() or not images:
        return []
    min_score = _inline_min_place_score()
    machine_bullets = _machine_bullet_lines_from_answer(answer)
    multi_machine = len(machine_bullets) >= 2

    if multi_machine:
        bullet_placements = _build_machine_bullet_manual_placements(
            answer,
            images,
            min_score=min_score,
        )
        if bullet_placements:
            return _apply_placement_reindex(
                images,
                bullet_placements,
                keep_unplaced=True,
            )

    placements = _build_semantic_inline_placements(
        answer,
        images,
        query=query,
        min_score=min_score,
    )
    if not placements:
        return _apply_placement_reindex(
            images, _fallback_end_placements(answer, images)
        )
    keep_unplaced = (
        len(placements) >= 2
        or multi_machine
        or bool(query and _should_use_unified_figure_targets(query, answer))
    )
    return _apply_placement_reindex(images, placements, keep_unplaced=keep_unplaced)


def images_for_api(
    context: str,
    media_roots: list[Path],
    *,
    query: str | None = None,
    extra_context: str | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
    figure_pool: list[dict[str, Any]] | None = None,
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

    if _should_use_unified_figure_targets(query or "", answer or ""):
        refs, _uni_meta = _refs_from_unified_figure_targets(
            query or "",
            answer or "",
            retrieved_docs=retrieved_docs,
            cite_pool=list(figure_pool or retrieved_docs or []),
        )
        return _api_images_from_figure_refs(
            refs[: max(limit, _multi_figure_image_limit())],
            media_roots,
        )

    logger.info(
        "Skip related images: unified figure targets not applicable (legacy path retired)"
    )
    return []


def _api_images_from_figure_refs(
    refs: list[dict[str, Any]],
    media_roots: list[Path],
) -> list[dict[str, Any]]:
    """Resolve figure refs to Web API image descriptors."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in refs:
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
        caption = _ref_effective_label(ref) or str(ref.get("caption") or "").strip()
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
            "Resolved %d chunk-locality image(s) (%d candidate ref(s))",
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
    figure_pool: list[dict[str, Any]] | None = None,
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
        figure_pool=figure_pool,
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
        figure_pool=figure_pool,
        limit=limit,
        answer=answer,
    )
    return images, debug
