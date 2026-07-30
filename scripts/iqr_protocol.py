"""image_query_refs submodule ``iqr_protocol``.

Shared data types (FigureTarget, logic lines), image-ref regexes, and
context parsing/normalization.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple
from raganything.utils import text_term_alignment_symmetric
from iqr_domain_schema import schema as _domain_schema


_SECTION_MARKER_SPLIT_RE = re.compile(
    "|".join(re.escape(m) for m in _domain_schema.section_markers)
)


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


_IMAGE_EXTS = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
)


class FigureTarget(NamedTuple):
    manual_hint: str
    anchor_text: str
    kind: str
    component: str = ""


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


class _LogicLine(NamedTuple):
    start: int
    end: int
    text: str
    machine: str = ""
    subject: str = ""


def _append_logic_line(
    out: list[_LogicLine],
    seen: set[tuple[str, str, int, int]],
    *,
    start: int,
    end: int,
    text: str,
    machine: str = "",
    subject: str = "",
) -> None:
    from iqr_terms import _normalize_label_key
    if not machine and not subject:
        return
    key = (
        _normalize_label_key(machine),
        _normalize_label_key(subject),
        start,
        end,
    )
    if key in seen:
        return
    seen.add(key)
    out.append(_LogicLine(start, end, text, machine, subject))


def _image_score_for_logic_line(
    line: _LogicLine, img: dict[str, Any], *, query: str = ""
) -> float:
    from iqr_terms import _is_procedure_steps_query, _line_has_query_subject_hit, _listing_target_head, _normalize_label_key, _query_subject_needles, _subject_from_machine_field_line
    from iqr_figure_target import _label_matches_listing_target, _pair_component_ref_align, _ref_matches_manual_hint
    q = (query or "").strip()
    if _is_procedure_steps_query(q) and (line.subject or line.machine):
        caption = str(img.get("caption") or "").strip()
        label = str(img.get("label") or caption).strip()
        best = 0.0
        for needle in _query_subject_needles(q):
            best = max(best, text_term_alignment_symmetric(needle, caption))
            if label:
                best = max(best, text_term_alignment_symmetric(needle, label))
        if caption and _line_has_query_subject_hit(q, line.text):
            best = max(best, 0.45)
        return best if best >= 0.35 else -1.0
    if line.machine and not _ref_matches_manual_hint(img, line.machine):
        return -1.0
    caption = str(img.get("caption") or "").strip()
    if line.machine and line.subject:
        heads: list[str] = []
        seen: set[str] = set()
        for cand in (line.subject, _listing_target_head(line.subject)):
            key = _normalize_label_key(cand)
            if cand and key not in seen:
                seen.add(key)
                heads.append(cand)
        align = max(_pair_component_ref_align(h, img) for h in heads)
        min_align = 0.38 if align >= 0.99 else 0.45
        return align if align >= min_align else -1.0
    if line.machine:
        align = 0.35
        subj = line.subject or _subject_from_machine_field_line(line.text)
        if subj and caption:
            align = max(align, text_term_alignment_symmetric(subj, caption))
            if _label_matches_listing_target(caption, subj):
                align = max(align, 1.0)
        return align
    if line.subject:
        if caption and _label_matches_listing_target(caption, line.subject):
            return 1.0
        sym = text_term_alignment_symmetric(line.subject, caption)
        return sym if sym >= 0.35 else -1.0
    return -1.0


def _maintenance_topic_from_text(text: str) -> str:
    from iqr_figure_target import _MAINT_TOPIC_RE
    if not text:
        return ""
    match = _MAINT_TOPIC_RE.search(text)
    if not match:
        return ""
    topic = match.group(1).strip()
    topic = re.split(r"\s*\d+\.\d+", topic, maxsplit=1)[0].strip()
    topic = _SECTION_MARKER_SPLIT_RE.split(topic, maxsplit=1)[0].strip()
    topic = re.split(r"[。\n]", topic, maxsplit=1)[0].strip()
    return topic[:32]


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
    from iqr_figure_target import _is_usable_source_figure_label
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


def extract_image_refs_from_context(context: str) -> list[dict[str, Any]]:
    """Parse image metadata from retrieved LightRAG context text."""
    from iqr_figure_target import _enrich_ref_from_image_block, _heading_before_image_block
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


