"""image_query_refs submodule ``iqr_figure_target``.

Figure-target extraction from answers: machine/component/listing targets,
cited-manual hints, and citation-based doc filtering.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from raganything.utils import (
    discriminative_terms,
    short_label_bag_aligns,
    substantive_bigrams,
    text_term_alignment_symmetric,
)
from iqr_protocol import (
    FigureTarget,
    _LogicLine,
    _append_logic_line,
    _is_image_metadata_line,
    _maintenance_topic_from_text,
    extract_image_refs_from_context,
    normalize_context_for_image_parse,
)
from iqr_config import (
    _image_min_ref_align,
    _min_substantive_term_len,
)
from iqr_terms import (
    _figure_matches_query_object,
    _is_procedure_steps_query,
    _is_query_subject_echo,
    _line_has_specific_query_overlap,
    _line_spans_in_body,
    _listing_target_head,
    _normalize_label_key,
    _query_subject_needles,
    _query_terms,
    _subject_from_machine_field_line,
    _subject_object_bigrams,
)
from iqr_store import (
    _dedupe_doc_list,
    _dedupe_doc_list_by_chunk_identity,
    _doc_basename,
    _doc_chunk_order_index,
    _doc_content,
    _doc_matches_cited_hints,
    _doc_storage_chunk_id,
    _first_figure_ref_from_doc,
    _kv_chunk_row,
    _load_manual_chunks_for_locality,
    _ranked_retrieval_lines,
    _source_key_from_path,
    text_from_retrieved_docs,
)
from iqr_align import (
    _ANSWER_REF_RE,
    _PDF_NAME_RE,
    _REF_LINE_RE,
    _answer_chunk_term_overlap,
    _chunk_citation_score,
    _chunk_figure_context_aligns_query,
    _chunk_subject_score,
    _figure_ref_passes_align_gate,
    _normalize_citation_blob,
    _ref_aligns_with_query_label,
    _ref_blob,
    _ref_effective_label,
    _ref_inline_context_text,
    _ref_structure_align_text,
)
from iqr_anchor import (
    _anchor_chunks_by_query_section,
    _anchor_inline_same_chunk,
    _anchor_section_carries_subject,
    _best_anchor_chunk,
    _best_figure_ref_for_anchor_align,
    _best_inline_figure_from_pool,
    _chunk_locality_image_enabled,
    _chunk_locality_window,
    _expand_pool_with_same_section_neighbors,
    _figure_doc_for_anchor_neighbor,
    _figure_ref_from_content_list_for_target,
    _is_section_number_heading,
    _load_figure_chunks_for_manual_paths,
    _ref_section_subject,
    _section_id_from_line_or_context,
    _should_expand_cited_manual_kv_pool,
    _strip_section_prefix,
    _supplement_answer_topic_figure_chunks,
    _supplement_cross_manual_figure_chunks,
)
from iqr_machine import known_machine_names, resolve_machine_name
from iqr_domain_schema import schema as _domain_schema
from iqr_query_intent import (  # noqa: F401 re-exports for backward compat
    _BOLD_RE,
    _BULLET_PREFIX_RE,
    _CATALOG_MODEL_MARKER,
    _CONJUNCTION_RE,
    _GENERIC_CYCLE_LABEL_RE,
    _MAINT_CYCLE_VALUE_RE,
    _MAINT_TOPIC_RE,
    _NEWLINES_RE,
    _PAREN_SPLIT_RE,
    _PERIOD_NL_SPLIT_RE,
    _REFERENCES_SPLIT_RE,
    _SECTION_MARKER_CONTENT_RE,
    _SECTION_MARKER_SPLIT_RE,
    _STRUCTURAL_LABEL_SUFFIXES,
    _WS_RE,
    _is_catalog_or_model_listing_query,
    _is_component_listing_across_machines,
    _is_listing_scope_query,
    _is_maintenance_cycle_query,
    _is_multi_machine_comparison_query,
    _retrieval_prefers_catalog_field,
    _strict_object_image_gate,
    _topic_overlaps_query,
    detect_table_filter_signal,
)


def _answer_text_for_listing() -> str:
    try:
        from query_progress_hooks import get_answer_text_for_images  # noqa: WPS433

        return (get_answer_text_for_images() or "").strip()
    except ImportError:
        return ""


def _answer_bullet_component_head(line: str) -> str:
    """Component span from markdown answer bullet (``**部件**``), not a domain lexicon."""
    payload = _BULLET_PREFIX_RE.sub("", (line or "").strip())
    bold = re.match(r"^\*\*([^*]+)\*\*", payload)
    if bold:
        return _listing_target_head(bold.group(1).strip())
    return _listing_target_head(payload)


def _figure_target_topic_text(target: FigureTarget) -> str:
    """Answer-structure topic for figure gates (selection), not placement."""
    if target.kind == "answer_bullet":
        return _answer_bullet_component_head(target.anchor_text)
    if target.kind == "machine_component_pair":
        return (target.component or target.anchor_text).strip()
    return target.anchor_text


def _ref_maintenance_topic(ref: dict[str, Any]) -> str:
    """Primary ``保养内容`` topic from ref context (ingest template field)."""
    blob = " ".join(str(ref.get(key) or "") for key in ("context", "caption", "label"))
    match = _MAINT_TOPIC_RE.search(blob)
    if not match:
        return ""
    topic = match.group(1).strip()
    topic = _SECTION_MARKER_SPLIT_RE.split(topic, maxsplit=1)[0].strip()
    topic = _PERIOD_NL_SPLIT_RE.split(topic, maxsplit=1)[0].strip()[:32]
    return topic


def _listing_component_from_maint_chunk(content: str, topic: str) -> str:
    """Map a ``保养内容`` line to its listing component via nearest section heading."""
    head = _listing_target_head(topic)
    topic_pos = content.find(topic)
    if topic_pos < 0:
        return head
    best_subj = ""
    best_dist = 10**9
    for match in re.finditer(r"\d+(?:\.\d+)+\s*(\S+?)保养", content):
        dist = abs(match.start() - topic_pos)
        if dist < best_dist:
            best_dist = dist
            best_subj = match.group(1).strip()
    if best_subj:
        subj_head = _listing_target_head(best_subj)
        if subj_head:
            return subj_head
    return head


def _bullet_head_in_inline_context(head: str, ref: dict[str, Any]) -> bool:
    """True when ingest ``关联正文`` already ties the figure to an answer bullet head."""
    head = (head or "").strip()
    ctx = _ref_inline_context_text(ref)
    if not head or not ctx:
        return False
    if head in ctx:
        return True
    compact_head = _listing_target_head(head)
    if compact_head and compact_head in ctx:
        return True
    return text_term_alignment_symmetric(head, ctx) >= 0.18


def _topic_in_inline_figure_context(topic: str, content: str) -> bool:
    """Topic appears in any inline figure ``关联正文`` inside ``content``."""
    topic = (topic or "").strip()
    if not topic or not (content or "").strip():
        return False
    for ref in extract_image_refs_from_context(content):
        if _bullet_head_in_inline_context(topic, ref):
            return True
    return False


def _answer_bullets_for_inline_figure_match(answer: str) -> list[str]:
    bullets = _component_spans_from_answer(answer)
    if len(bullets) >= 2:
        return bullets
    return _answer_listing_spans(_answer_primary_listing_body(answer))


def _ref_aligns_answer_bullets_via_inline_context(
    ref: dict[str, Any], answer: str
) -> bool:
    for head in _answer_bullets_for_inline_figure_match(answer):
        if _bullet_head_in_inline_context(head, ref):
            return True
    return False


def _chunk_aligns_answer_bullet_via_inline_figure(
    answer: str,
    content: str,
    *,
    bullets: list[str] | None = None,
) -> bool:
    """Cited chunk carries inline figures whose ingest context matches an answer item."""
    content = (content or "").strip()
    if not content or not extract_image_refs_from_context(content):
        return False
    heads = bullets or _answer_bullets_for_inline_figure_match(answer)
    for head in heads:
        if _topic_in_inline_figure_context(head, content):
            return True
    return False


def _pair_component_ref_align(component: str, ref: dict[str, Any]) -> float:
    """Score (machine, component) ↔ figure using parser ``保养内容`` / section fields."""
    head = _listing_target_head(component)
    if not head:
        return 0.0
    topic = _ref_maintenance_topic(ref)
    caption = str(ref.get("caption") or "").strip()
    section_subj = _ref_section_subject(ref)
    ctx = str(ref.get("context") or "").strip()

    if section_subj and caption:
        cap_sec = text_term_alignment_symmetric(section_subj, caption)
        cap_comp = text_term_alignment_symmetric(head, caption)
        sec_comp = text_term_alignment_symmetric(head, section_subj)
        if cap_sec >= 0.4 and cap_comp < 0.35 and sec_comp < 0.35:
            return 0.0

    if topic and _label_matches_listing_target(topic, head):
        return 1.0
    if section_subj:
        subj_head = _listing_target_head(section_subj)
        if subj_head == head:
            return 0.98
        if len(head) >= 4 and head in section_subj:
            return 0.98
        if len(subj_head) >= 4 and subj_head in head:
            return 0.98
    if topic:
        sym = text_term_alignment_symmetric(head, topic)
        if sym >= 0.5:
            return 0.8 + sym * 0.15
        if head in ctx and not _label_matches_listing_target(topic, head):
            body_sym = max(
                text_term_alignment_symmetric(head, caption),
                sym,
            )
            return max(0.0, body_sym - 0.4)
    if caption and _label_matches_listing_target(caption, head):
        return 0.88
    ctx_sym = text_term_alignment_symmetric(head, ctx) if ctx else 0.0
    in_ctx = 0.92 if head in ctx else 0.0
    return max(
        text_term_alignment_symmetric(head, caption),
        text_term_alignment_symmetric(head, topic) if topic else 0.0,
        ctx_sym,
        in_ctx,
    )


def _answer_listing_spans(answer: str) -> list[str]:
    """Component headers from answer markdown (e.g. **压带轮** / ### 压带轮)."""
    spans: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        span = _listing_target_head(raw)
        key = _normalize_label_key(span)
        if _is_answer_structural_label(span):
            return
        if len(span) >= 2 and len(span) <= 24 and key not in seen:
            seen.add(key)
            spans.append(span)

    for line in (answer or "").splitlines():
        field = re.match(
            r"^\s*[*\-•]\s*(?:\*\*([^*]+)\*\*|([^*：:\n]+))[：:]\s*(.+)$",
            line,
        )
        if not field:
            continue
        fname = (field.group(1) or field.group(2) or "").strip()
        if _is_component_field_name(fname):
            value = field.group(3).strip().strip("*").rstrip("。")
            value = _PAREN_SPLIT_RE.split(value, maxsplit=1)[0].strip()
            if value:
                add(value)
            continue
        machine = _machine_from_section_title(fname)
        if machine:
            for extra in _BOLD_RE.findall(field.group(3)):
                add(extra.strip())

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


def _maintenance_topics_in_text(text: str, query: str) -> list[str]:
    """All ``保养内容：`` topics in retrieval bodies (ingest template, not phrase lists)."""
    topics: list[str] = []
    seen: set[str] = set()
    for match in _MAINT_TOPIC_RE.finditer(text or ""):
        topic = match.group(1).strip()
        topic = _SECTION_MARKER_SPLIT_RE.split(topic, maxsplit=1)[0].strip()
        topic = _PERIOD_NL_SPLIT_RE.split(topic, maxsplit=1)[0].strip()[:32]
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


def _machine_spans_from_answer(answer: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for span in _answer_listing_spans(_answer_primary_listing_body(answer)):
        machine = _machine_from_section_title(span)
        if not machine:
            # Data-driven fallback: dynamic KB machine vocab (whitespace-tolerant)
            # replaces the former hard-coded 封边机/加工中心 substring literals.
            machine = resolve_machine_name(span)
        if not machine:
            continue
        key = _normalize_label_key(machine)
        if key in seen:
            continue
        seen.add(key)
        out.append(machine)
    return out


def _answer_primary_listing_body(answer: str) -> str:
    """Answer body for listing spans; drop trailing digressions (e.g. 此外…)."""
    text = (answer or "").strip()
    text = _ANSWER_REF_RE.sub("", text)
    for marker in _domain_schema.paragraph_connectors:
        match = re.search(rf"(?:^|\n)\s*{marker}", text)
        if match:
            text = text[: match.start()]
    return text.strip()


def _component_spans_from_answer(answer: str) -> list[str]:
    comps = _pair_component_spans_from_answer(answer)
    if comps:
        return comps
    return [
        span
        for span in _answer_listing_spans(_answer_primary_listing_body(answer))
        if not resolve_machine_name(span) and not _is_answer_structural_label(span)
    ]


def _cited_manual_hints_from_answer(answer: str) -> set[str]:
    """PDF stems from the answer References block (phase 3 cross-manual scope)."""
    text = (answer or "").strip()
    if not text:
        return set()
    parts = _REFERENCES_SPLIT_RE.split(text, maxsplit=1)
    ref_blob = parts[1] if len(parts) > 1 else text
    hints: set[str] = set()
    for match in _REF_LINE_RE.finditer(ref_blob):
        title = match.group(2).strip()
        if title:
            stem = title.split(".pdf")[0].split(".PDF")[0].strip()
            if stem:
                hints.add(stem[:80])
                compact = _WS_RE.sub("", stem)
                if compact:
                    hints.add(compact[:80])
                machine = _machine_from_section_title(stem)
                if machine:
                    hints.add(machine)
                    hints.add(_WS_RE.sub("", machine))
    for match in _PDF_NAME_RE.finditer(ref_blob):
        stem = Path(match.group(0)).stem[:80]
        if stem:
            hints.add(stem)
            compact = _WS_RE.sub("", stem)
            if compact:
                hints.add(compact[:80])
    return {h for h in hints if h and len(h) >= 4}


def _cited_manual_pdf_stems(answer: str) -> list[str]:
    """Distinct PDF stems from answer References (one entry per cited manual)."""
    text = (answer or "").strip()
    if not text:
        return []
    parts = _REFERENCES_SPLIT_RE.split(text, maxsplit=1)
    ref_blob = parts[1] if len(parts) > 1 else ""
    stems: list[str] = []
    seen: set[str] = set()
    for match in _REF_LINE_RE.finditer(ref_blob):
        title = match.group(2).strip()
        if not title:
            continue
        stem = title.split(".pdf")[0].split(".PDF")[0].strip()
        key = _WS_RE.sub("", stem)
        if key and key not in seen:
            seen.add(key)
            stems.append(stem)
    return stems


# Generic structural suffixes that answer field labels end with (保养周期 /
# 操作步骤 / 加注工具 / 表针读数标准 / 润滑部位 / 保养部件 …). Language-level
# presentation structure (same class as ``DOC_TYPE_KEYWORDS``), NOT business
# data: a span ending in one of these is a field label, never a figure subject.
# Replaces the former 14-entry ``_ANSWER_STRUCTURAL_LABEL_KEYS`` enumeration —
# every one of those labels ends in a suffix here, and the suffix form also
# generalizes to unseen labels (LLM answer wording is low-randomness, but new
# field labels still follow these endings). Real components (压带轮 / 涂胶轴 /
# 仿形靠模 / 电机散热风扇) end in none of them.


def _is_answer_structural_label(span: str) -> bool:
    """Answer field label (ends in a structural suffix), not a figure subject."""
    key = _normalize_label_key(span)
    return bool(key) and key.endswith(_STRUCTURAL_LABEL_SUFFIXES)


def _is_ordinal_enumeration_bullet_head(head: str) -> bool:
    """LLM ordinal splits (第一把刀 / 第二把预铣刀) — one figure subject, not N targets."""
    return bool(re.match(r"^第[一二三四五六七八九十0-9]+把", (head or "").strip()))


def _is_answer_footnote_line(line: str) -> bool:
    """Skip ``*(注：...)`` digressions when parsing machine/component pairs."""
    s = (line or "").strip()
    return bool(s and re.match(r"^\*?\(注[：:]", s))


def _is_maintenance_cycle_value(span: str) -> bool:
    """Answer cadence phrases (每季度一次), not maintainable components."""
    span = (span or "").strip()
    if not span:
        return False
    compact = _WS_RE.sub("", span)
    return bool(
        _MAINT_CYCLE_VALUE_RE.match(span) or _MAINT_CYCLE_VALUE_RE.match(compact)
    )


def _pair_component_spans_from_answer(answer: str, *, query: str = "") -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for _, comp in _machine_component_targets_from_answer(answer, query=query):
        comp = _listing_target_head(comp)
        if not comp or _is_answer_structural_label(comp):
            continue
        key = _normalize_label_key(comp)
        if key in seen:
            continue
        seen.add(key)
        out.append(comp)
    return out


def _cross_listing_figure_topics(query: str, answer: str) -> list[str]:
    """Topics for figure scan on cross-manual component listings (e.g. Q16/Q17)."""
    if not _is_component_listing_across_machines(query):
        return []
    machines: list[str] = []
    components: list[str] = []
    seen_m: set[str] = set()
    seen_c: set[str] = set()
    for machine, comp in _machine_component_targets_from_answer(answer, query=query):
        machine = machine.strip()
        comp = _listing_target_head(comp)
        if not machine or not comp or _is_answer_structural_label(comp):
            continue
        mk, ck = _normalize_label_key(machine), _normalize_label_key(comp)
        if mk not in seen_m:
            seen_m.add(mk)
            machines.append(machine)
        if ck not in seen_c:
            seen_c.add(ck)
            components.append(comp)
    if len(machines) >= 2 and len(components) == 1:
        return machines
    if len(machines) >= 2 and re.search(r"哪些机型|用来保养哪个部件", query or ""):
        return machines
    if len(components) >= 2:
        return components
    if components:
        return components
    return []


def _is_component_field_name(field_name: str) -> bool:
    """Field label whose *value* lists components (保养部件 / 润滑部位 [+ 序号])."""
    name = (field_name or "").strip()
    return bool(re.fullmatch(r"(?:保养部件|润滑部位)\d*", name))


_MACHINE_LINE_COMPONENT_PATTERNS = (
    r"用于保养\s*\*\*([^*]+)\*\*",
    r"对\s*\*\*([^*]+)\*\*\s*进行",
    r"保养\s*\*\*([^*]+)\*\*",
)


def _components_from_machine_line_value(value: str, query: str) -> list[str]:
    """Maintenance components from a machine bullet value (structural + bold)."""
    out: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        comp = _listing_target_head(raw.strip())
        if (
            not comp
            or _machine_from_section_title(comp)
            or _is_answer_structural_label(comp)
            or _is_maintenance_cycle_value(comp)
            or _is_query_subject_echo(comp, query)
        ):
            return
        key = _normalize_label_key(comp)
        if key in seen:
            return
        seen.add(key)
        out.append(comp)

    for pat in _MACHINE_LINE_COMPONENT_PATTERNS:
        for match in re.finditer(pat, value):
            add(match.group(1))
    if not out and not any(m in value for m in _domain_schema.section_markers):
        for extra in _BOLD_RE.findall(value):
            add(extra)
    return out


def _is_machine_class_span(text: str) -> bool:
    """Generic machine-*class* suffix (封边机 / 钻 / 中心).

    Language-level structural cue (same class as ``DOC_TYPE_KEYWORDS``), NOT
    business data: specific machine names come from the dynamic KB vocab
    (``known_machine_names`` / ``resolve_machine_name``); this only catches a
    not-yet-ingested machine whose name still ends with a known class suffix.
    """
    blob = (text or "").strip()
    return any(s in blob for s in _domain_schema.machine_class_suffixes)


def _machine_from_section_title(title: str) -> str:
    title = re.sub(r"^\d+\.\s*", "", title).strip()
    title = re.sub(r"^[一二三四五六七八九十]+、", "", title).strip()
    title = re.sub(r"^机型[：:]\s*", "", title).strip()
    title = title.strip("《》").strip()
    if _is_answer_structural_label(title):
        return ""
    for name in known_machine_names():
        if name in title:
            return name
    if _is_machine_class_span(title):
        return _domain_schema.truncate_filename(title)
    return ""


def _machine_component_targets_from_answer(
    answer: str, *, query: str = ""
) -> list[tuple[str, str]]:
    """(machine line, component) pairs from numbered markdown sections in the answer."""
    body = _answer_primary_listing_body(answer)
    pairs: list[tuple[str, str]] = []
    current_machine = ""
    q = (query or "").strip()

    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if _is_answer_footnote_line(line):
            continue
        if line.startswith("#"):
            machine = _machine_from_section_title(line.lstrip("#").strip())
            if machine:
                current_machine = machine
            continue
        standalone_machine = re.match(r"^\*\*(?:\d+\.\s*)?([^*]+)\*\*\s*$", line)
        if standalone_machine:
            machine = _machine_from_section_title(standalone_machine.group(1).strip())
            if machine:
                current_machine = machine
            continue
        bare_field_match = re.match(
            r"^\s*\*\*([^*]+)\*\*[：:]\s*(.+)$",
            line,
        )
        if bare_field_match:
            field_name = bare_field_match.group(1).strip()
            value = bare_field_match.group(2).strip()
            machine = _machine_from_section_title(field_name)
            if machine:
                inline_added = False
                for comp in _components_from_machine_line_value(value, q):
                    pairs.append((machine, comp))
                    inline_added = True
                if not inline_added and "**" not in value:
                    plain = _BOLD_RE.sub(r"\1", value)
                    plain = plain.strip().rstrip("。")
                    plain = _PAREN_SPLIT_RE.split(plain, maxsplit=1)[0].strip()
                    if (
                        plain
                        and len(plain) <= 32
                        and not _is_query_subject_echo(plain, q)
                    ):
                        comp = _listing_target_head(plain)
                        if (
                            comp
                            and not _machine_from_section_title(comp)
                            and not _is_answer_structural_label(comp)
                            and not _is_maintenance_cycle_value(comp)
                            and not _is_query_subject_echo(comp, q)
                        ):
                            pairs.append((machine, comp))
                continue
        prefix_match = re.match(r"^\*\*(?:\d+\.\s*)?([^*]+)\*\*", line)
        if prefix_match:
            machine = _machine_from_section_title(prefix_match.group(1).strip())
            if machine:
                current_machine = machine
            continue
        num_bullet_machine = re.match(
            r"^\s*[*\-•]\s*\*\*(?:\d+\.\s*)?([^*]+)\*\*\s*$", line
        )
        if num_bullet_machine:
            machine = _machine_from_section_title(num_bullet_machine.group(1).strip())
            if machine:
                current_machine = machine
            continue
        field_match = re.match(
            r"^\s*[*\-•]\s*(?:\*\*([^*]+)\*\*|([^*：:\n]+))[：:]\s*(.+)$",
            line,
        )
        if field_match:
            field_name = (field_match.group(1) or field_match.group(2) or "").strip()
            value = field_match.group(3).strip()
            if field_name in _domain_schema.footnote_labels or re.fullmatch(
                r"注\d*", field_name
            ):
                continue
            machine = _machine_from_section_title(field_name)
            if machine:
                current_machine = machine
                inline_added = False
                for comp in _components_from_machine_line_value(value, q):
                    pairs.append((machine, comp))
                    inline_added = True
                if not inline_added and "**" not in value:
                    plain = _BOLD_RE.sub(r"\1", value)
                    plain = plain.strip().rstrip("。")
                    plain = _PAREN_SPLIT_RE.split(plain, maxsplit=1)[0].strip()
                    if (
                        plain
                        and len(plain) <= 32
                        and not _is_query_subject_echo(plain, q)
                    ):
                        comp = _listing_target_head(plain)
                        if (
                            comp
                            and not _machine_from_section_title(comp)
                            and not _is_answer_structural_label(comp)
                            and not _is_query_subject_echo(comp, q)
                        ):
                            pairs.append((machine, comp))
                continue
            if current_machine and _is_component_field_name(field_name):
                bold_parts = _BOLD_RE.findall(value)
                if bold_parts:
                    for part in bold_parts:
                        comp = _listing_target_head(part.strip())
                        if comp and not _is_answer_structural_label(comp):
                            pairs.append((current_machine, comp))
                else:
                    value = value.strip().strip("*").rstrip("。")
                    value = _PAREN_SPLIT_RE.split(value, maxsplit=1)[0].strip()
                    parts = re.split(r"[/／、与和及]", value)
                    for part in parts:
                        part = part.strip()
                        if not part or _machine_from_section_title(part):
                            continue
                        comp = _listing_target_head(part)
                        if comp and not _is_answer_structural_label(comp):
                            pairs.append((current_machine, comp))
            elif current_machine and not _is_answer_structural_label(field_name):
                comp = _listing_target_head(field_name)
                if comp and not _machine_from_section_title(comp):
                    pairs.append((current_machine, comp))
            continue
        bullet_match = re.match(r"^\s*[*\-•]\s*\*\*([^*]+)\*\*", line)
        if bullet_match:
            bullet_title = bullet_match.group(1).strip()
            machine = _machine_from_section_title(bullet_title)
            if machine:
                current_machine = machine
                inline_added = False
                for comp in _components_from_machine_line_value(line, q):
                    parts = _CONJUNCTION_RE.split(comp)
                    for part in parts:
                        part = part.strip()
                        if part:
                            pairs.append((machine, part))
                            inline_added = True
                if inline_added:
                    continue
                continue
            if current_machine:
                comp_raw = _listing_target_head(bullet_title)
                parts = _CONJUNCTION_RE.split(comp_raw)
                for part in parts:
                    part = part.strip()
                    if part:
                        pairs.append((current_machine, part))
                continue
        maint_match = re.match(r"^\s*[*\-•]\s*保养部件[：:]\s*(.+)$", line)
        if maint_match and current_machine:
            comp_blob = maint_match.group(1).strip().strip("*").strip()
            comp_blob = comp_blob.split("/")[0].strip()
            comp_raw = _listing_target_head(comp_blob)
            parts = _CONJUNCTION_RE.split(comp_raw)
            for part in parts:
                part = part.strip()
                if part:
                    pairs.append((current_machine, part))
            continue
        comp_match = re.match(r"^\s*\d+\.\s*\*\*([^*]+)\*\*", line)
        if comp_match and current_machine:
            comp_raw = _listing_target_head(comp_match.group(1))
            parts = _CONJUNCTION_RE.split(comp_raw)
            for part in parts:
                part = part.strip()
                if part:
                    pairs.append((current_machine, part))
    return pairs


def _infer_listing_pairs_from_cited_chunks(
    query: str,
    answer: str,
    pool: list[dict[str, Any]],
    existing: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Add (machine, component) pairs from cited chunks when answer omits distinct 保养内容 items."""
    if not _is_component_listing_across_machines(query) or not pool:
        return existing
    cited = _cited_manual_hints_from_answer(answer)
    seen = {
        (_normalize_label_key(machine), _normalize_label_key(comp))
        for machine, comp in existing
    }
    out = list(existing)
    for doc in pool:
        if cited and not _doc_matches_cited_hints(doc, cited):
            continue
        content = _doc_content(doc).strip()
        if not content or not extract_image_refs_from_context(content):
            continue
        # Query-topic relevance is enforced per-component below via
        # ``_pair_component_ref_align`` (保养内容/图注/小节主语 alignment ≥0.75);
        # the former hard-coded 残胶/老化胶水 doc pre-filter was redundant.
        machine = ""
        fp = _doc_basename(doc)
        for name in known_machine_names():
            if name in fp:
                machine = name
                break
        if not machine:
            continue
        refs = extract_image_refs_from_context(content)
        for topic in _maintenance_content_spans(content):
            comp = _listing_component_from_maint_chunk(content, topic)
            if len(comp) < 2:
                continue
            key = (_normalize_label_key(machine), _normalize_label_key(comp))
            if key in seen:
                continue
            if not any(_pair_component_ref_align(comp, ref) >= 0.75 for ref in refs):
                continue
            seen.add(key)
            out.append((machine, comp))
    return out


def _machine_component_listing_pair_targets(
    query: str,
    answer: str,
    pool: list[dict[str, Any]],
    *,
    kept: list[dict[str, Any]] | None = None,
) -> list[tuple[str, str]]:
    """Ordered (machine, component) pairs for cross-manual listings (Q17)."""
    del kept
    pairs = _machine_component_targets_from_answer(answer, query=query)
    if len(pairs) < 2:
        return []
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for machine, comp in pairs:
        machine = machine.strip()
        comp = _listing_target_head(comp)
        if not machine or not comp:
            continue
        key = (_normalize_label_key(machine), _normalize_label_key(comp))
        if key in seen:
            continue
        seen.add(key)
        out.append((machine, comp))
    if pool:
        out = _infer_listing_pairs_from_cited_chunks(query, answer, pool, out)
    out = _filter_cross_listing_pair_targets(query, answer, out)
    return out[:16]


def _filter_cross_listing_pair_targets(
    query: str,
    answer: str,
    pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Drop non-cited manuals, query echoes, and oil-query footnote extras."""
    if not pairs:
        return pairs
    cited = _cited_manual_hints_from_answer(answer)
    filtered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for machine, comp in pairs:
        machine = machine.strip()
        comp = _listing_target_head(comp)
        if not machine or not comp or _is_answer_structural_label(comp):
            continue
        if _is_maintenance_cycle_value(comp):
            continue
        if _is_query_subject_echo(comp, query):
            continue
        if cited and not any(
            _doc_matches_manual_hint({"file_path": h, "source_key": h}, machine)
            for h in cited
        ):
            continue
        key = (_normalize_label_key(machine), _normalize_label_key(comp))
        if key in seen:
            continue
        seen.add(key)
        filtered.append((machine, comp))
    return filtered if len(filtered) >= 2 else pairs


def _manual_hint_for_component(
    component: str,
    answer: str,
    *,
    pool: list[dict[str, Any]] | None = None,
    cited_hints: set[str] | None = None,
) -> str:
    head = _listing_target_head(component)
    for machine, comp in _machine_component_targets_from_answer(answer):
        if comp == head or _label_matches_listing_target(comp, head):
            return machine
    for doc in pool or []:
        if cited_hints and not _doc_matches_cited_hints(doc, cited_hints):
            continue
        content = _doc_content(doc).strip()
        if not content or not extract_image_refs_from_context(content):
            continue
        labels = [
            lab
            for ref in extract_image_refs_from_context(content)
            if (lab := _ref_effective_label(ref))
        ]
        if not (
            any(_label_matches_listing_target(lab, head) for lab in labels)
            or _chunk_matches_answer_topic(content, head)
        ):
            continue
        fp = _doc_basename(doc)
        for name in known_machine_names():
            if name in fp:
                return name
    return ""


def _resolve_known_machine_name(text: str) -> str:
    """Longest KB machine name in *text* (avoids 自动封边机 ⊂ 高速自动封边机 false positives)."""
    return resolve_machine_name(text)


def _doc_matches_manual_hint(doc: dict[str, Any], manual_hint: str) -> bool:
    hint = (manual_hint or "").strip()
    if not hint:
        return True
    fp = str(doc.get("file_path") or doc.get("path") or "").replace("\\", "/")
    if not fp:
        fp = _doc_basename(doc)
    fp_compact = _WS_RE.sub("", fp)
    hint_compact = _WS_RE.sub("", hint)
    if not fp_compact:
        return False
    # Data-driven anti-bleed: resolve both sides to their canonical KB machine
    # and require agreement (replaces the former hard-coded 高速智能/高速自动/
    # 自动封边机 substring special cases).
    hint_machine = _resolve_known_machine_name(hint)
    fp_machine = _resolve_known_machine_name(fp)
    if hint_machine and fp_machine:
        return hint_machine == fp_machine
    if hint_machine:
        nc = _WS_RE.sub("", hint_machine)
        if nc not in fp_compact and hint_machine not in fp:
            return False
        return True
    if hint_compact in fp_compact:
        return True
    if hint_compact and len(hint_compact) >= 8 and hint_compact[:8] in fp_compact:
        return True
    for term in discriminative_terms(hint, min_len=4):
        if len(term) >= 4 and term in fp:
            return True
    return False


def _machine_targets_from_answer(answer: str) -> list[str]:
    """Machines for multi-manual listing: prefer ``**机型**`` bullets over span heuristics."""
    bullets = _machine_bullet_lines_from_answer(answer)
    if len(bullets) >= 2:
        out: list[str] = []
        seen: set[str] = set()
        for machine, _line in bullets:
            key = _normalize_label_key(machine)
            if key in seen:
                continue
            seen.add(key)
            out.append(machine)
        return out
    return _machine_spans_from_answer(answer)


def _machine_bullet_lines_from_answer(answer: str) -> list[tuple[str, str]]:
    """``(machine, bullet_line)`` from markdown machine bullets."""
    body = _answer_text_for_placement(answer)
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_line in body.splitlines():
        line = raw_line.strip()
        bullet = re.match(r"^[-*•]\s*\*\*([^*]+)\*\*[：:].+$", line)
        if not bullet:
            continue
        machine = _machine_from_section_title(bullet.group(1).strip())
        if not machine:
            continue
        key = _normalize_label_key(machine)
        if key in seen:
            continue
        seen.add(key)
        out.append((machine, line))
    return out


def _answer_bullet_lines_for_figure_targets(answer: str) -> list[str]:
    """Full non-machine listing bullet lines for ``answer_bullet`` FigureTargets."""
    if len(_machine_bullet_lines_from_answer(answer)) >= 2:
        return []
    body = _answer_text_for_placement(answer)
    out: list[str] = []
    seen: set[str] = set()
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not _BULLET_PREFIX_RE.match(line):
            continue
        payload = _BULLET_PREFIX_RE.sub("", line).strip()
        if len(payload) < 4:
            continue
        machine_bullet = re.match(r"^\*\*([^*]+)\*\*[：:]", payload)
        if machine_bullet and _machine_from_section_title(
            machine_bullet.group(1).strip()
        ):
            continue
        head = _answer_bullet_component_head(line)
        if _is_answer_structural_label(head) or _is_ordinal_enumeration_bullet_head(
            head
        ):
            continue
        key = _normalize_label_key(head)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def _resolve_manual_hint_for_figure_targets(answer: str, query: str) -> str:
    """Single manual scope for ``single`` / ``answer_bullet`` targets."""
    machines = _machine_targets_from_answer(answer)
    if len(machines) == 1:
        return machines[0]
    cited = _cited_manual_hints_from_answer(answer)
    if len(cited) == 1:
        return next(iter(cited))
    if machines:
        return machines[0]
    q = (query or "").strip()
    if q:
        machine = _machine_from_section_title(q)
        if machine:
            return machine
    return ""


def _machine_bullet_subject_variants(line: str) -> list[str]:
    """Subject spans from machine bullet body (primary + parenthetical qualifiers)."""
    variants: list[str] = []
    seen: set[str] = set()

    def add(span: str) -> None:
        span = (span or "").strip()
        if len(span) < 2:
            return
        key = _normalize_label_key(span)
        if not key or key in seen:
            return
        seen.add(key)
        variants.append(span)

    add(_machine_bullet_subject(line))
    m = re.match(
        r"^(?:\d+\.\s*)?[-*•]?\s*\*\*[^*]+\*\*[：:]\s*(.+)$",
        (line or "").strip(),
    )
    if not m:
        return variants
    body = _BOLD_RE.sub(r"\1", m.group(1).strip())
    body = _SECTION_MARKER_SPLIT_RE.split(body, maxsplit=1)[0].strip().rstrip("。")
    for inner in re.findall(r"[（(]([^）)]+)[）)]", body):
        add(_listing_target_head(inner))
    for part in _PAREN_SPLIT_RE.split(body, maxsplit=1):
        add(_listing_target_head(part))
    return variants


def _chunk_is_table_heavy(content: str) -> bool:
    """Structural signal: HTML table blocks dominate the chunk (not domain keywords)."""
    stripped = (content or "").strip()
    if not stripped or "[Table]" not in stripped:
        return False
    table_markers = stripped.count("<tr>") + stripped.count("<td")
    if table_markers >= 3:
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines and re.match(r"^附表", lines[0]) and len(stripped) > 120:
        return True
    return len(stripped) > 900 and table_markers >= 1


def _cite_pool_chunk_ids(pool: list[dict[str, Any]] | None) -> set[str]:
    out: set[str] = set()
    for doc in pool or []:
        cid = _doc_storage_chunk_id(doc)
        if cid:
            out.add(cid)
    return out


def _ref_matches_query_theme_in_evidence(
    query: str,
    ref_hay: str,
) -> bool:
    """Query theme in ref evidence (substring or symmetric term align)."""
    terms: list[str] = []
    seen: set[str] = set()

    def add(span: str) -> None:
        span = span.strip()
        if len(span) < _min_substantive_term_len() or span in seen:
            return
        seen.add(span)
        terms.append(span)

    for needle in _query_subject_needles(query):
        add(needle)
    for term in discriminative_terms(query, min_len=_min_substantive_term_len()):
        add(term)

    if not terms:
        return True
    if any(t in ref_hay for t in terms):
        return True
    compact = [t for t in terms if len(t) <= 8]
    theme_blob = " ".join(compact[:10] if compact else terms[:8])
    return text_term_alignment_symmetric(theme_blob, ref_hay) >= (
        _image_min_ref_align() * 0.85
    )


def _chunk_heading_blob(content: str, *, limit: int = 200) -> str:
    """First lines of a chunk (section heading), not full mega-chunk body."""
    lines = [ln.strip() for ln in (content or "").splitlines() if ln.strip()]
    return "\n".join(lines[:5])[:limit]


def _ref_matches_machine_bullet_theme(
    anchor_text: str,
    ref: dict[str, Any],
    *,
    anchor_content: str = "",
) -> bool:
    """Machine-line bullets: ref label/context or anchor-section align to bullet subject."""
    variants = _machine_bullet_subject_variants(anchor_text)
    for subject in variants:
        if _figure_ref_matches_listing_target(ref, subject):
            return True
    anchor = (anchor_content or "").strip()
    if not anchor or not _anchor_section_carries_subject(anchor, variants):
        return False
    struct = _ref_structure_align_text(ref)
    if not struct:
        return False
    return text_term_alignment_symmetric(anchor, struct) >= _image_min_ref_align()


def _figure_ref_matches_target_topic(
    anchor_text: str,
    query: str,
    ref: dict[str, Any],
    *,
    kind: str = "",
    component: str = "",
    doc_content: str = "",
    anchor_content: str = "",
) -> bool:
    """Ref-level topic gate dispatched by ``FigureTarget.kind`` (§2.7c)."""
    target_kind = (kind or "").strip()
    section_anchor = (anchor_content or doc_content or "").strip()

    if target_kind == "machine_component_pair":
        comp = (component or "").strip()
        if not comp:
            return True
        return _figure_ref_matches_listing_target(ref, comp)

    if target_kind == "machine_bullet":
        return _ref_matches_machine_bullet_theme(
            anchor_text,
            ref,
            anchor_content=section_anchor,
        )

    if target_kind == "answer_bullet":
        head = _answer_bullet_component_head(anchor_text)
        if not head or len(head) < 2:
            return False
        return _figure_ref_matches_listing_target(ref, head)

    if target_kind == "single":
        struct = _ref_structure_align_text(ref)
        chunk = (doc_content or "").strip()
        ref_hay = "\n".join(p for p in (struct, chunk) if p.strip())
        if not ref_hay:
            return False
        topics = _single_figure_supplement_topics(anchor_text, query)
        if topics:
            return any(_chunk_matches_answer_topic(ref_hay, t) for t in topics)
        return _ref_matches_query_theme_in_evidence(query, ref_hay)

    return True


def _pick_figure_ref_for_target_kind(
    doc: dict[str, Any],
    *,
    kind: str,
    anchor_text: str,
    query: str,
    component: str = "",
    anchor_content: str = "",
) -> dict[str, Any] | None:
    """Pick inline figure in ``doc`` using kind-aware listing/subject priority."""
    content = (anchor_content or _doc_content(doc)).strip()
    target_kind = (kind or "").strip()

    if target_kind == "machine_bullet":
        subject = _machine_bullet_subject(anchor_text)
        if subject:
            ref = _best_figure_ref_for_listing_target(doc, subject)
            if ref is not None and _figure_ref_matches_listing_target(ref, subject):
                return ref
    elif target_kind == "answer_bullet":
        head = _answer_bullet_component_head(anchor_text)
        if head:
            ref = _best_figure_ref_for_listing_target(doc, head)
            if ref is not None and _figure_ref_matches_listing_target(ref, head):
                return ref
    elif target_kind == "machine_component_pair":
        comp = (component or "").strip()
        if comp:
            ref = _best_figure_ref_for_listing_target(doc, comp)
            if ref is not None and _figure_ref_matches_listing_target(ref, comp):
                return ref

    return _best_figure_ref_for_anchor_align(
        doc,
        anchor_text=anchor_text,
        query=query,
        anchor_content=content,
    )


def _best_anchor_doc_for_machine(
    machine: str,
    bullet_line: str,
    pool: list[dict[str, Any]],
    *,
    query: str = "",
    manual_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if manual_chunks is None:
        manual_chunks = _load_manual_chunks_for_locality(machine)
    anchor = _best_anchor_chunk(
        manual_chunks,
        bullet_line,
        cite_pool=pool,
        query=query,
        manual_hint=machine,
    )
    if anchor is not None:
        return anchor
    candidates = [doc for doc in pool if _doc_matches_manual_hint(doc, machine)]
    if not candidates:
        return None
    return _resolve_doc_with_order_index(candidates[0])


def _manual_scoped_cite_pool(
    cite_pool: list[dict[str, Any]] | None,
    manual_hint: str,
) -> list[dict[str, Any]]:
    hint = (manual_hint or "").strip()
    if not hint:
        return list(cite_pool or [])
    return [doc for doc in cite_pool or [] if _doc_matches_manual_hint(doc, hint)]


def extract_figure_targets(
    query: str,
    answer: str,
    *,
    cite_pool: list[dict[str, Any]] | None = None,
    kept_docs: list[dict[str, Any]] | None = None,
) -> list[FigureTarget]:
    """Answer-structure targets for unified figure selection (not query-type routing)."""
    q = (query or "").strip()
    ans = (answer or "").strip()
    if not ans:
        return []

    if _is_component_listing_across_machines(q):
        pairs = _machine_component_listing_pair_targets(
            q,
            ans,
            list(cite_pool or kept_docs or []),
            kept=list(kept_docs or []),
        )
        if len(pairs) >= 2:
            return [
                FigureTarget(
                    manual_hint=machine,
                    anchor_text=f"{machine} {comp}".strip(),
                    kind="machine_component_pair",
                    component=comp,
                )
                for machine, comp in pairs
            ]

    bullets = _machine_bullet_lines_from_answer(ans)
    if len(bullets) >= 2:
        return [
            FigureTarget(
                manual_hint=machine,
                anchor_text=line,
                kind="machine_bullet",
            )
            for machine, line in bullets
        ]

    machines = _machine_targets_from_answer(ans)
    if len(machines) >= 2:
        return [
            FigureTarget(
                manual_hint=machine,
                anchor_text=machine,
                kind="machine_bullet",
            )
            for machine in machines
        ]

    answer_lines = _answer_bullet_lines_for_figure_targets(ans)
    if len(answer_lines) >= 2:
        manual = _resolve_manual_hint_for_figure_targets(ans, q)
        return [
            FigureTarget(
                manual_hint=manual,
                anchor_text=line,
                kind="answer_bullet",
            )
            for line in answer_lines
        ]

    anchor_text = ans
    if q:
        anchor_text = f"{q}\n{ans}"
    manual = _resolve_manual_hint_for_figure_targets(ans, q)
    return [
        FigureTarget(
            manual_hint=manual,
            anchor_text=anchor_text,
            kind="single",
        )
    ]


def _should_use_unified_figure_targets(query: str, answer: str) -> bool:
    """True when answer maps to unified ``extract_figure_targets`` pipeline."""
    if not _chunk_locality_image_enabled():
        return False
    if not (answer or "").strip():
        return False
    targets = extract_figure_targets(query, answer)
    if not targets:
        return False
    if len(targets) == 1 and targets[0].kind == "single":
        return True
    if len(targets) < 2:
        return False
    kind = targets[0].kind
    if kind == "machine_component_pair":
        return True
    if kind == "answer_bullet":
        return True
    if kind == "machine_bullet":
        return True
    return False


def _finalize_figure_target_row(
    row: dict[str, Any],
    ref: dict[str, Any],
    *,
    source: str,
    anchor: dict[str, Any] | None,
    fig_doc: dict[str, Any] | None,
) -> None:
    row["status"] = "ok"
    row["figure_source"] = source
    if anchor is not None:
        row["anchor_id"] = str(anchor.get("id") or anchor.get("chunk_id") or "")
        row["anchor_idx"] = _doc_chunk_order_index(anchor)
    if fig_doc is not None:
        row["figure_idx"] = _doc_chunk_order_index(fig_doc)
    row["path"] = Path(str(ref.get("path") or "")).name


def _figure_for_target(
    target: FigureTarget,
    *,
    query: str,
    cite_pool: list[dict[str, Any]] | None,
    manual_cache: dict[str, list[dict[str, Any]]],
    exclude_paths: set[str] | None = None,
    exclude_chunk_ids: set[str] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """One figure ref for a single ``FigureTarget`` (anchor/neighbor → cite → pool → content_list)."""
    row: dict[str, Any] = {
        "manual_hint": target.manual_hint,
        "kind": target.kind,
        "status": "pending",
    }
    excluded_paths = set(exclude_paths or [])
    excluded_chunks = set(exclude_chunk_ids or [])
    machine = (target.manual_hint or "").strip()
    if not machine:
        row["status"] = "no_manual_hint"
        return None, row

    hint_key = _normalize_label_key(machine)
    if hint_key not in manual_cache:
        manual_cache[hint_key] = _load_manual_chunks_for_locality(machine)
    manual_chunks = manual_cache[hint_key]
    row["manual_chunk_n"] = len(manual_chunks)

    scoped_cite = _manual_scoped_cite_pool(cite_pool, machine)
    row["scoped_cite_n"] = len(scoped_cite)
    scoped_pool = _dedupe_doc_list_by_chunk_identity(
        list(manual_chunks) + list(scoped_cite)
    )

    if target.kind == "machine_component_pair" and target.component:
        answer_blob = _normalize_citation_blob(target.anchor_text)
        fig_doc = _best_figure_doc_for_component(
            target.component,
            scoped_pool,
            answer_blob=answer_blob,
            manual_hint=machine,
            query=query,
        )
        if fig_doc is None:
            row["status"] = "no_component_figure"
            return None, row
        ref = _best_figure_ref_for_listing_target(fig_doc, target.component)
        if ref is None:
            row["status"] = "no_figure_extracted"
            return None, row
        if not _figure_ref_passes_align_gate(
            target.anchor_text,
            query,
            ref,
            doc_content=_doc_content(fig_doc),
        ):
            row["status"] = "align_gate"
            return None, row
        row["status"] = "ok"
        row["figure_source"] = "component_pool"
        row["path"] = Path(str(ref.get("path") or "")).name
        return ref, row

    anchor_search_text = target.anchor_text
    if target.kind == "single" and (query or "").strip():
        anchor_search_text = (query or "").strip()

    anchor = _best_anchor_chunk(
        manual_chunks,
        anchor_search_text,
        cite_pool=scoped_cite,
        query=query,
        manual_hint=machine,
    )
    manual_anchor = _best_anchor_chunk(
        manual_chunks,
        anchor_search_text,
        cite_pool=[],
        query=query,
        manual_hint=machine,
    )
    anchor_candidates: list[dict[str, Any]] = []
    for candidate in (anchor, manual_anchor):
        if candidate is None:
            continue
        candidate = _resolve_doc_with_order_index(candidate)
        cid = _doc_storage_chunk_id(candidate)
        if cid and any(_doc_storage_chunk_id(a) == cid for a in anchor_candidates):
            continue
        anchor_candidates.append(candidate)

    if not anchor_candidates:
        row["status"] = "no_anchor"
        return None, row

    primary_anchor = anchor_candidates[0]
    row["anchor_id"] = str(
        primary_anchor.get("id") or primary_anchor.get("chunk_id") or ""
    )
    row["anchor_idx"] = _doc_chunk_order_index(primary_anchor)

    anchor_text = target.anchor_text
    topic_text = _figure_target_topic_text(target)
    window = _chunk_locality_window()
    anchor_section = _doc_content(primary_anchor)

    def _accept(
        ref: dict[str, Any] | None,
        *,
        source: str,
        anchor_doc: dict[str, Any] | None,
        fig_doc: dict[str, Any] | None,
        align_extra: str = "",
        align_anchor: str | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if ref is None:
            return None, row
        path_key = Path(str(ref.get("path") or "")).name
        if path_key and path_key in excluded_paths:
            return None, row
        fig_cid = _doc_storage_chunk_id(fig_doc) if fig_doc else ""
        if fig_cid and fig_cid in excluded_chunks:
            return None, row
        if (
            source == "anchor"
            and _anchor_inline_same_chunk(anchor_doc, fig_doc)
            and not _is_cover_page_ref(ref)
        ):
            topic_ok = target.kind != "single" or _figure_ref_matches_target_topic(
                anchor_text,
                query,
                ref,
                kind=target.kind,
                doc_content=_doc_content(fig_doc) if fig_doc else "",
                anchor_content=_doc_content(anchor_doc) if anchor_doc else "",
            )
            if topic_ok:
                _finalize_figure_target_row(
                    row,
                    ref,
                    source=source,
                    anchor=anchor_doc,
                    fig_doc=fig_doc,
                )
                return ref, row
            row["status"] = "topic_gate"
            row["figure_source"] = source
            return None, row
        align_anchor_text = (align_anchor or topic_text or anchor_text).strip()
        align_content = "\n".join(
            p
            for p in (
                _doc_content(anchor_doc) if anchor_doc else "",
                _doc_content(fig_doc) if fig_doc else "",
                align_extra,
            )
            if p.strip()
        )
        if not _figure_ref_passes_align_gate(
            align_anchor_text,
            query,
            ref,
            doc_content=align_content or None,
        ):
            row["status"] = "align_gate"
            row["figure_source"] = source
            return None, row
        _finalize_figure_target_row(
            row,
            ref,
            source=source,
            anchor=anchor_doc,
            fig_doc=fig_doc,
        )
        return ref, row

    # 1. Anchor inline figure + order_index neighbors (structure-first)
    for candidate in anchor_candidates:
        if not _doc_matches_manual_hint(candidate, machine):
            continue
        cand_fig, cand_src = _figure_doc_for_anchor_neighbor(
            candidate,
            manual_chunks,
            window=window,
            anchor_text=topic_text,
            query=query,
            manual_hint=machine,
            kind=target.kind,
            component=target.component,
        )
        if cand_fig is None:
            continue
        ref = _pick_figure_ref_for_target_kind(
            cand_fig,
            kind=target.kind,
            anchor_text=topic_text,
            query=query,
            component=target.component,
            anchor_content=_doc_content(candidate),
        )
        accepted, out_row = _accept(
            ref,
            source=cand_src,
            anchor_doc=candidate,
            fig_doc=cand_fig,
            align_anchor=topic_text,
        )
        if accepted is not None:
            return accepted, out_row

    # 2. Scoped cite pool (evidence supplement, not global query-subject pick)
    cite_ref, cite_doc, cite_src = _best_inline_figure_from_pool(
        topic_text,
        scoped_cite,
        query=query,
        manual_hint=machine,
        kind=target.kind,
        component=target.component,
        strict_topic=True,
        anchor_content=anchor_section,
        exclude_paths=excluded_paths,
        exclude_chunk_ids=excluded_chunks,
    )
    accepted, out_row = _accept(
        cite_ref,
        source=cite_src,
        anchor_doc=primary_anchor if cite_doc is None else cite_doc,
        fig_doc=cite_doc,
        align_anchor=topic_text,
    )
    if accepted is not None:
        return accepted, out_row

    # 3. Full manual + cite pool (strict answer topic)
    pool_ref, pool_doc, pool_src = _best_inline_figure_from_pool(
        topic_text,
        scoped_pool,
        query=query,
        manual_hint=machine,
        kind=target.kind,
        component=target.component,
        strict_topic=True,
        anchor_content=anchor_section,
        exclude_paths=excluded_paths,
        exclude_chunk_ids=excluded_chunks,
    )
    accepted, out_row = _accept(
        pool_ref,
        source=pool_src,
        anchor_doc=pool_doc or primary_anchor,
        fig_doc=pool_doc,
        align_anchor=topic_text,
    )
    if accepted is not None:
        return accepted, out_row

    # 4. content_list / utils ingest pairing
    cl_ref, cl_src = _figure_ref_from_content_list_for_target(
        topic_text,
        machine,
        query=query,
        kind=target.kind,
        component=target.component,
        anchor_content=anchor_section,
    )
    accepted, out_row = _accept(
        cl_ref,
        source=cl_src,
        anchor_doc=primary_anchor,
        fig_doc=None,
        align_extra=topic_text,
        align_anchor=topic_text,
    )
    if accepted is not None:
        return accepted, out_row

    row["status"] = "no_figure_near_anchor"
    return None, row


def _refs_from_unified_figure_targets(
    query: str,
    answer: str,
    *,
    retrieved_docs: list[dict[str, Any]] | None,
    cite_pool: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Unified: one figure per ``FigureTarget`` via ``_figure_for_target``."""
    pool = _dedupe_doc_list_by_chunk_identity(
        list(cite_pool or []) + list(retrieved_docs or [])
    )
    pool = [_resolve_doc_with_order_index(doc) for doc in pool]
    targets = extract_figure_targets(
        query,
        answer,
        cite_pool=pool,
        kept_docs=list(retrieved_docs or []),
    )
    meta: dict[str, Any] = {
        "mode": "unified_figure_targets",
        "window": _chunk_locality_window(),
        "pool_size": len(pool),
        "target_count": len(targets),
        "targets": [],
    }
    refs: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    manual_cache: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        machine = (target.manual_hint or "").strip()
        if not machine:
            continue
        hint_key = _normalize_label_key(machine)
        if hint_key not in manual_cache:
            manual_cache[hint_key] = _load_manual_chunks_for_locality(machine)

    for target in targets:
        ref, row = _figure_for_target(
            target,
            query=query,
            cite_pool=pool,
            manual_cache=manual_cache,
            exclude_paths=seen_paths,
        )
        meta["targets"].append(row)
        if ref is None:
            continue
        path_key = Path(str(ref.get("path") or "")).name
        if not path_key or path_key in seen_paths:
            row["status"] = "duplicate_path"
            continue
        if _is_cover_page_ref(ref):
            row["status"] = "cover_page"
            continue
        seen_paths.add(path_key)
        ref = dict(ref)
        ref["figure_target_kind"] = target.kind
        ref["locality_machine"] = target.manual_hint
        ref["locality_source"] = row.get("figure_source")
        ref["source_key"] = _source_key_from_path(str(ref.get("path") or ""))
        refs.append(ref)

    meta["picked"] = len(refs)
    return refs, meta


def _component_listing_span_targets(
    query: str,
    answer: str,
    pool: list[dict[str, Any]],
    *,
    kept: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Ordered component targets for cross-manual part listings (Q17)."""
    pairs = _machine_component_targets_from_answer(answer, query=query)
    components: list[str] = []
    seen: set[str] = set()
    for _, comp in pairs:
        key = _normalize_label_key(comp)
        if key and key not in seen:
            seen.add(key)
            components.append(comp)
    hints = list(components)
    kept_docs = list(kept or [])
    pool_topics = _listing_topics_from_pool(query, pool)
    if kept_docs and pool_topics:
        anchored = [
            t for t in pool_topics if _topic_supported_by_kept_chunks(t, kept_docs)
        ]
        if anchored:
            pool_topics = anchored
    for topic in pool_topics:
        key = _normalize_label_key(topic)
        if key and key not in seen:
            seen.add(key)
            components.append(topic)
    if len(components) < 2:
        components = _component_spans_from_answer(answer)
    if len(components) < 2:
        return []
    if len(hints) >= 2:
        components = _order_listing_targets_by_hints(components, hints)
    return components[:12]


def _figure_ref_matches_listing_target(ref: dict[str, Any], head: str) -> bool:
    """True when figure label or inline 关联正文 ties to a listing/component head."""
    head = (head or "").strip()
    if not head:
        return False
    label = _ref_effective_label(ref)
    if label and _label_matches_listing_target(label, head):
        return True
    if _bullet_head_in_inline_context(head, ref):
        return True
    ctx = _ref_inline_context_text(ref)
    compact = _listing_target_head(head)
    return bool(
        (compact and compact in ctx)
        or (head in ctx)
        or text_term_alignment_symmetric(head, ctx) >= 0.22
    )


def _best_figure_ref_for_listing_target(
    doc: dict[str, Any],
    head: str,
) -> dict[str, Any] | None:
    """Best inline figure in ``doc`` for a component/listing head (not always first)."""
    head = (head or "").strip()
    if not head:
        return _first_figure_ref_from_doc(doc)
    refs = extract_image_refs_from_context(_doc_content(doc))
    if not refs:
        return None
    best_ref: dict[str, Any] | None = None
    best_score = 0.0
    for ref in refs:
        if not _figure_ref_matches_listing_target(ref, head):
            continue
        label = _ref_effective_label(ref)
        ctx = _ref_inline_context_text(ref)
        score = max(
            text_term_alignment_symmetric(head, label) if label else 0.0,
            text_term_alignment_symmetric(head, ctx),
            _answer_chunk_term_overlap(head, ctx),
        )
        if score > best_score:
            best_score = score
            best_ref = ref
    return best_ref or refs[0]


def _best_figure_doc_for_component(
    component: str,
    pool: list[dict[str, Any]],
    *,
    answer_blob: str,
    manual_hint: str = "",
    cited_hints: set[str] | None = None,
    query: str = "",
) -> dict[str, Any] | None:
    head = _listing_target_head(component)
    best_doc: dict[str, Any] | None = None
    best_score = 0.0
    for doc in pool:
        if cited_hints and not _doc_matches_cited_hints(doc, cited_hints):
            continue
        if manual_hint and not _doc_matches_manual_hint(doc, manual_hint):
            continue
        content = _doc_content(doc).strip()
        if not content or not extract_image_refs_from_context(content):
            continue
        refs = extract_image_refs_from_context(content)
        label_hit = any(
            _label_matches_listing_target(lab, head)
            for ref in refs
            if (lab := _ref_effective_label(ref))
        )
        ref_topic_match = any(
            _figure_ref_matches_listing_target(ref, head) for ref in refs
        )
        if not label_hit and not ref_topic_match:
            continue
        if not _answer_weak_consistency_gate(answer_blob, content):
            if (
                head not in answer_blob
                and _listing_target_head(head) not in answer_blob
            ):
                continue
        score = max(
            _answer_chunk_term_overlap(head, content),
            text_term_alignment_symmetric(head, content),
            _chunk_citation_score(answer_blob, content) if answer_blob else 0.0,
        )
        if label_hit:
            score += 0.15
        if query and _chunk_figure_context_aligns_query(query, content):
            score += 0.08
        if score < 0.08:
            continue
        if score > best_score:
            best_score = score
            best_doc = doc
    return best_doc


def _answer_image_span_targets(query: str, answer: str) -> list[str]:
    """One figure per answer item: component names for part listings, machine lines for Q15-style."""
    cross = _cross_listing_figure_topics(query, answer)
    if cross:
        return cross
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
    if _is_component_listing_across_machines(q):
        targets = _component_listing_span_targets(q, answer, pool, kept=kept_docs)
        if len(targets) >= 2:
            return targets
    if _is_multi_machine_comparison_query(
        q
    ) and not _is_component_listing_across_machines(q):
        machines = _machine_spans_from_answer(answer)
        if len(machines) >= 2:
            return machines
    if _is_listing_scope_query(q) and not _is_catalog_or_model_listing_query(q):
        answer_bullets = [
            span
            for span in _answer_bullets_for_inline_figure_match(answer)
            if span and not resolve_machine_name(span)
        ]
        if len(answer_bullets) >= 2:
            return answer_bullets[:12]
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

    for line in _NEWLINES_RE.split(text):
        line = line.strip()
        if (
            len(line) < 4
            or _is_image_metadata_line(line)
            or _is_toc_or_directory_line(line)
        ):
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
    lines = [line.strip() for line in _NEWLINES_RE.split(anchor) if line.strip()]
    return [
        t
        for t in targets
        if any(t in line or _label_matches_listing_target(line, t) for line in lines)
    ]


def _listing_targets_with_query_line_overlap(
    query: str, full_text: str, focus: str
) -> list[str]:
    """Listing targets whose focus lines also share substantive query terms."""
    focus_norm = normalize_context_for_image_parse(focus)
    answer = _answer_text_for_listing()
    if answer:
        cross = _cross_listing_figure_topics(query, answer)
        if len(cross) >= 2:
            return cross
    if _is_listing_scope_query(query) and answer:
        bold = _component_spans_from_answer(answer)
        if len(bold) < 2:
            bold = _answer_listing_spans(_answer_primary_listing_body(answer))
        if len(bold) >= 2:
            return bold

    targets = _listing_targets_anchored_in_text(query, full_text, anchor_text=focus)
    if not targets:
        return []
    lines = [line.strip() for line in _NEWLINES_RE.split(focus_norm) if line.strip()]
    return [
        t
        for t in targets
        if any(
            (t in line or _label_matches_listing_target(line, t))
            and _line_has_specific_query_overlap(query, line, "")
            for line in lines
        )
    ]


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
    answer = _answer_text_for_listing()
    if answer and _ref_aligns_answer_bullets_via_inline_context(ref, answer):
        return True
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


def _is_cover_page_ref(ref: dict[str, Any]) -> bool:
    """Decorative cover figures (page 0) are never shown as query illustrations."""
    page = ref.get("page")
    if page is None:
        return False
    try:
        return int(page) == 0
    except (TypeError, ValueError):
        return False


def _answer_has_multi_section_markdown(answer: str) -> bool:
    body = _answer_text_for_placement(answer)
    titles = [
        m.group(1).strip()
        for m in _BOLD_RE.finditer(body)
        if len(m.group(1).strip()) >= 4
    ]
    return len(titles) >= 2


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


def _maintenance_content_spans(text: str) -> list[str]:
    spans: list[str] = []
    for match in _SECTION_MARKER_CONTENT_RE.finditer(text or ""):
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


def _answer_weak_consistency_gate(answer_blob: str, content: str) -> bool:
    """Answer terms overlap chunk body (not bold-span expansion)."""
    if not answer_blob.strip() or not (content or "").strip():
        return False
    terms = [t for t in discriminative_terms(answer_blob, min_len=2) if len(t) >= 2]
    if not terms:
        return False
    body = _normalize_citation_blob(content)
    hits = sum(1 for term in terms if term in body)
    if hits >= 1 and hits / len(terms) >= 0.12:
        return True
    return text_term_alignment_symmetric(answer_blob, content) >= 0.12


def _chunk_is_toc_heavy(content: str) -> bool:
    lines = [line.strip() for line in content.splitlines() if len(line.strip()) >= 4]
    if len(lines) < 4:
        return False
    toc_count = sum(1 for line in lines if _is_toc_or_directory_line(line))
    return toc_count / len(lines) >= 0.35


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
    cross_listing = _is_component_listing_across_machines(q)
    pair_components = _pair_component_spans_from_answer(answer)
    shared_component = pair_components[0] if len(pair_components) == 1 else ""
    multi_machine = (_is_multi_machine_comparison_query(q) and len(machines) >= 2) or (
        cross_listing and len(machines) >= 2 and topics and topics[0] in machine_names
    )
    seen_parts: set[str] = set()

    def _topic_matches_chunk(topic: str, content: str, doc: dict[str, Any]) -> bool:
        if _topic_in_inline_figure_context(topic, content):
            return True
        if _chunk_matches_answer_topic(content, topic):
            if shared_component and topic in machine_names:
                return _chunk_matches_answer_topic(content, shared_component)
            return True
        if multi_machine and topic in machine_names:
            manual = _doc_basename(doc)
            if not (manual and topic in manual):
                return False
            if shared_component:
                return _chunk_matches_answer_topic(content, shared_component)
            return True
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
        q_terms = [t for t in discriminative_terms(q, min_len=2) if len(t) >= 2]
        for topic in machines:
            if any(
                topic in _doc_basename(doc) and _doc_content(doc).strip() in seen_parts
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
    if not query_aligned and not _is_listing_scope_query(q) and not multi_machine:
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


def _answer_section_topics(answer: str) -> list[str]:
    """Bold spans in the answer body (machine headers, component names).

    Note: LLM markdown bold is volatile; do not use alone to gate anchor inline figures.
    """
    body = _answer_text_for_placement(answer)
    topics: list[str] = []
    seen: set[str] = set()
    for match in _BOLD_RE.finditer(body):
        topic = match.group(1).strip()
        if len(topic) < 3 or topic in seen:
            continue
        seen.add(topic)
        topics.append(topic)
    return topics


def _single_figure_supplement_topics(anchor_text: str, query: str) -> list[str]:
    """Topic spans for cite/neighbor supplement gates on ``single`` targets.

    Combines query terms, answer listing spans, and optional LLM-bold — bold alone
    must not be the only signal (formatting varies per generation).
    """
    spans: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        raw = (raw or "").strip()
        if len(raw) < 2:
            return
        if (
            _is_answer_structural_label(raw)
            or _is_maintenance_cycle_value(raw)
            or _is_ordinal_enumeration_bullet_head(raw)
        ):
            return
        key = _normalize_label_key(raw)
        if not key or key in seen:
            return
        seen.add(key)
        spans.append(raw)

    ans_blob = (anchor_text or "").strip()
    for topic in _answer_image_span_targets(query, ans_blob):
        add(topic)
    for topic in _answer_section_topics(ans_blob):
        add(topic)
    body = _answer_text_for_placement(ans_blob)
    hay = f"{body}\n{ans_blob}"
    for term in discriminative_terms(query, min_len=3):
        if len(term) < 3:
            continue
        if term in hay or term in (query or ""):
            add(term)
    return spans


def _resolve_doc_with_order_index(doc: dict[str, Any]) -> dict[str, Any]:
    """Fill ``chunk_order_index`` / ``file_path`` from KV when rerank docs omit them."""
    out = dict(doc)
    if _doc_chunk_order_index(out) is not None:
        return out
    cid = _doc_storage_chunk_id(out)
    if not cid:
        return out
    row = _kv_chunk_row(cid)
    if not row:
        return out
    raw_idx = row.get("chunk_order_index")
    if raw_idx is not None:
        try:
            out["chunk_order_index"] = int(raw_idx)
        except (TypeError, ValueError):
            pass
    if not out.get("file_path") and row.get("file_path"):
        out["file_path"] = row["file_path"]
    if not out.get("content") and row.get("content"):
        out["content"] = row["content"]
    return out


def _relabel_dc_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relabeled: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        row = dict(chunk)
        row["id"] = f"DC{i + 1}"
        relabeled.append(row)
    return relabeled


def _chunk_is_title_only(content: str) -> bool:
    """Skip catalog/title-only chunks that echo the device name in answers."""
    text = (content or "").strip()
    if not text or "[图片]" in text:
        return False
    norm = _normalize_citation_blob(text)
    if len(norm) <= 12:
        return True
    if any(m in text for m in _domain_schema.section_markers):
        return False
    return len(norm) <= 24 and not re.search(r"[\d\.]+\s*\S", text)


def _answer_body_for_citation_match(answer: str) -> str:
    text = (answer or "").strip()
    text = _ANSWER_REF_RE.sub("", text)
    return _normalize_citation_blob(text)


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

    section_pool = _dedupe_doc_list(list(anchor_pool or pool or docs or []))
    pool_for_spans = section_pool
    if _should_expand_cited_manual_kv_pool(query or "", answer):
        cited_early = _cited_manual_hints_from_answer(answer)
        allowed: set[str] = set(cited_early)
        for doc in section_pool:
            fp = _doc_basename(doc)
            if fp:
                allowed.add(fp)
        if allowed:
            kv_extra = _load_figure_chunks_for_manual_paths(allowed, query or "")
            if kv_extra:
                pool_for_spans = _dedupe_doc_list(section_pool + kv_extra)
                meta["cited_manual_kv_pool"] = len(kv_extra)

    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in docs or []:
        content = _doc_content(doc).strip()
        if not content or _chunk_is_toc_heavy(content) or _chunk_is_title_only(content):
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
        if _is_listing_scope_query(query or ""):
            bullets = _answer_bullets_for_inline_figure_match(answer)
            inline_kept = 0
            for score, doc in scored:
                if id(doc) in seen_ids:
                    continue
                content = _doc_content(doc).strip()
                if _chunk_aligns_answer_bullet_via_inline_figure(
                    answer, content, bullets=bullets
                ):
                    kept.append(doc)
                    seen_ids.add(id(doc))
                    inline_kept += 1
            if inline_kept:
                meta["inline_figure_citation_keep"] = inline_kept
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
        cited_hints = (
            _cited_manual_hints_from_answer(answer)
            if _is_component_listing_across_machines(query or "")
            else set()
        )
        expanded_spans_pool = (
            _expand_pool_with_same_section_neighbors(pool_for_spans, kept)
            if _is_component_listing_across_machines(query or "")
            else pool_for_spans
        )
        if len(answer_spans) >= 2 or (
            _is_component_listing_across_machines(query or "")
            and len(
                _machine_component_listing_pair_targets(
                    query or "", answer, pool_for_spans, kept=kept
                )
            )
            >= 2
        ):
            kept_ids = {id(doc) for doc in kept}
            pair_targets = (
                _machine_component_listing_pair_targets(
                    query or "", answer, pool_for_spans, kept=kept
                )
                if _is_component_listing_across_machines(query or "")
                else []
            )
            span_items: list[tuple[str, str]] = (
                pair_targets if pair_targets else [("", span) for span in answer_spans]
            )
            for manual_hint, span in span_items:
                if not manual_hint and _is_component_listing_across_machines(
                    query or ""
                ):
                    manual_hint = _manual_hint_for_component(
                        span,
                        answer,
                        pool=expanded_spans_pool,
                        cited_hints=cited_hints,
                    )
                best_doc: dict[str, Any] | None = None
                best_score = 0.0
                for doc in expanded_spans_pool:
                    if cited_hints and not _doc_matches_cited_hints(doc, cited_hints):
                        continue
                    if manual_hint and not _doc_matches_manual_hint(doc, manual_hint):
                        continue
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
                    figure_refs = extract_image_refs_from_context(content)
                    manual = _doc_basename(doc)
                    matched = (
                        any(_label_matches_listing_target(lab, span) for lab in labels)
                        or any(
                            _pair_component_ref_align(span, ref) >= 0.45
                            for ref in figure_refs
                        )
                        or _topic_in_inline_figure_context(span, content)
                        or span in content
                        or span in manual
                        or _chunk_matches_answer_topic(content, span)
                    )
                    if not matched:
                        continue
                    if _is_multi_machine_comparison_query(
                        query or ""
                    ) and not _is_component_listing_across_machines(query or ""):
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
            extract_image_refs_from_context(_doc_content(doc).strip()) for doc in kept
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
    if _domain_schema.matches_special_pattern(query or "") and not span_keep:
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
    parts = _REFERENCES_SPLIT_RE.split(text, maxsplit=1)
    return parts[0].strip()


def _machine_bullet_subject(line: str) -> str:
    """Topic phrase from a ``[-*•] **机型**：…`` answer line (bullet optional)."""
    return _subject_from_machine_field_line(line)


_BULLET_ALIAS_SUFFIX = r"(?:[（(][^）)]*[）)])?"


_MACHINE_FIELD_LINE_RE = re.compile(
    rf"^(?:\d+\.\s*)?[-*•]?\s*\*\*([^*]+)\*\*{_BULLET_ALIAS_SUFFIX}[：:]\s*(.+)$"
)


_COMPONENT_BULLET_RE = re.compile(
    rf"^[-*•]\s*\*\*([^*]+)\*\*{_BULLET_ALIAS_SUFFIX}(?:[：:]\s*(.*))?$"
)


def _subjects_from_machine_field_value(value: str, query: str) -> list[str]:
    comps = _components_from_machine_line_value(value, query)
    if comps:
        return comps
    tail = re.split(
        r"|".join(re.escape(m) for m in _domain_schema.section_markers),
        value,
        maxsplit=1,
    )[0].strip()
    tail = _BOLD_RE.sub(r"\1", tail).strip().rstrip("。")
    tail = _PAREN_SPLIT_RE.split(tail, maxsplit=1)[0].strip()
    subj = _listing_target_head(tail)
    if (
        subj
        and not _machine_from_section_title(subj)
        and not _is_answer_structural_label(subj)
        and not _is_maintenance_cycle_value(subj)
        and not _is_query_subject_echo(subj, query)
    ):
        return [subj]
    return []


def _infer_listing_machine_context(query: str, answer: str) -> str:
    """Infer machine for single-manual component listings when answer omits ### headers."""
    q = (query or "").strip()
    if not q or not (answer or "").strip():
        return ""
    if _is_procedure_steps_query(q) or _is_maintenance_cycle_query(q):
        return ""
    if len(_cited_manual_pdf_stems(answer)) != 1:
        return ""
    components = [
        c
        for c in _answer_listing_spans(_answer_primary_listing_body(answer))
        if c and not _machine_from_section_title(c)
    ]
    if len(components) < 2:
        return ""
    if _machine_spans_from_answer(answer):
        return ""
    cited = _cited_manual_hints_from_answer(answer)
    for name in known_machine_names():
        if name not in q:
            continue
        for hint in cited:
            compact_hint = _WS_RE.sub("", hint)
            compact_name = _WS_RE.sub("", name)
            if compact_name in compact_hint or name in hint:
                return name
        machine = _machine_from_section_title(name)
        if machine:
            return machine
    machine = _machine_from_section_title(q)
    if machine:
        return machine
    return ""


def _answer_logic_lines(answer: str, *, query: str = "") -> list[_LogicLine]:
    """Semantic answer lines for placement (format-agnostic machine / subject rows)."""
    body = _answer_text_for_placement(answer)
    q = (query or "").strip()
    out: list[_LogicLine] = []
    seen: set[tuple[str, str, int, int]] = set()
    current_machine = _infer_listing_machine_context(q, answer)

    for start, end, line in _line_spans_in_body(body):
        if line.startswith("#"):
            machine = _machine_from_section_title(line.lstrip("#").strip())
            if machine:
                current_machine = machine
            continue

        standalone = re.match(r"^\*\*(?:\d+\.\s*)?([^*]+)\*\*\s*$", line)
        if standalone:
            machine = _machine_from_section_title(standalone.group(1).strip())
            if machine:
                current_machine = machine
            continue

        num_bullet_machine = re.match(
            r"^[-*•]\s*\*\*(?:\d+\.\s*)?([^*]+)\*\*\s*$",
            line,
        )
        if num_bullet_machine:
            machine = _machine_from_section_title(num_bullet_machine.group(1).strip())
            if machine:
                current_machine = machine
            continue

        mfield = _MACHINE_FIELD_LINE_RE.match(line)
        if mfield:
            machine = _machine_from_section_title(mfield.group(1).strip())
            if machine:
                current_machine = machine
                value = mfield.group(2).strip()
                subjects = _subjects_from_machine_field_value(value, q)
                if subjects:
                    for subj in subjects:
                        if _is_maintenance_cycle_value(subj):
                            continue
                        _append_logic_line(
                            out,
                            seen,
                            start=start,
                            end=end,
                            text=line,
                            machine=machine,
                            subject=subj,
                        )
                else:
                    subj = _subject_from_machine_field_line(line)
                    _append_logic_line(
                        out,
                        seen,
                        start=start,
                        end=end,
                        text=line,
                        machine=machine,
                        subject=subj,
                    )
                continue

        bullet = _COMPONENT_BULLET_RE.match(line)
        if bullet:
            title = bullet.group(1).strip()
            rest = (bullet.group(2) or "").strip()
            machine = _machine_from_section_title(title)
            if machine:
                current_machine = machine
                if rest:
                    for subj in _subjects_from_machine_field_value(rest, q):
                        if not _is_maintenance_cycle_value(subj):
                            _append_logic_line(
                                out,
                                seen,
                                start=start,
                                end=end,
                                text=line,
                                machine=machine,
                                subject=subj,
                            )
                continue
            comp = _listing_target_head(title)
            if (
                current_machine
                and comp
                and not _machine_from_section_title(comp)
                and not _is_answer_structural_label(comp)
                and not _is_maintenance_cycle_value(comp)
                and not _is_query_subject_echo(comp, q)
            ):
                _append_logic_line(
                    out,
                    seen,
                    start=start,
                    end=end,
                    text=line,
                    machine=current_machine,
                    subject=comp,
                )
            continue

        num_comp = re.match(r"^\d+\.\s*\*\*([^*]+)\*\*", line)
        if num_comp:
            comp = _listing_target_head(num_comp.group(1).strip())
            if not comp or _is_answer_structural_label(comp):
                continue
            if _is_procedure_steps_query(q):
                _append_logic_line(
                    out,
                    seen,
                    start=start,
                    end=end,
                    text=line,
                    subject=comp,
                )
            elif current_machine:
                _append_logic_line(
                    out,
                    seen,
                    start=start,
                    end=end,
                    text=line,
                    machine=current_machine,
                    subject=comp,
                )

    out.sort(key=lambda row: row.start)
    return out


def _answer_paragraph_blocks(answer: str) -> list[tuple[str, int, int]]:
    """Non-empty paragraphs in placement body as (text, start, end) char spans."""
    body = _answer_text_for_placement(answer)
    if not body:
        return []
    blocks: list[tuple[str, int, int]] = []
    pos = 0
    for raw in re.split(r"\n\s*\n", body):
        chunk = raw.strip()
        if not chunk:
            continue
        idx = body.find(chunk, pos)
        if idx < 0:
            idx = body.find(chunk)
        if idx < 0:
            continue
        blocks.append((chunk, idx, idx + len(chunk)))
        pos = idx + len(chunk)
    return blocks


def _build_machine_bullet_manual_placements(
    answer: str,
    images: list[dict[str, Any]],
    min_score: float,
) -> list[dict[str, Any]]:
    """Pair each machine-named bullet with a figure from the cited manual (Q15-style)."""
    body = _answer_text_for_placement(answer)
    placements: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    used_ranges: list[tuple[int, int]] = []

    for raw_line in body.splitlines():
        line = raw_line.strip()
        bullet = re.match(r"^[-*•]\s*\*\*([^*]+)\*\*[：:].+$", line)
        if not bullet:
            continue
        machine = _machine_from_section_title(bullet.group(1).strip())
        if not machine:
            continue
        start = body.find(line)
        if start < 0:
            continue
        end = start + len(line)
        span_score = min(1.0, 0.85 + 0.15 * min(1.0, len(machine) / 12.0))
        if span_score < min_score:
            continue
        subject = _machine_bullet_subject(line)
        best_idx = -1
        best_align = -1.0
        for idx, img in enumerate(images):
            if idx in used_indices:
                continue
            if not _ref_matches_manual_hint(img, machine):
                continue
            caption = str(img.get("caption") or "").strip()
            align = 0.35
            if subject and caption:
                align = max(
                    align,
                    text_term_alignment_symmetric(subject, caption),
                )
                if _label_matches_listing_target(caption, subject):
                    align = max(align, 1.0)
            if align > best_align:
                best_align = align
                best_idx = idx
        if best_idx < 0:
            continue
        overlap = any(not (end <= u0 or start >= u1) for u0, u1 in used_ranges)
        if overlap:
            continue
        used_indices.add(best_idx)
        used_ranges.append((start, end))
        placements.append(
            {
                "anchor_text": line,
                "match_start": start,
                "match_end": end,
                "image_index": best_idx,
                "score": round(span_score, 3),
            }
        )

    placements.sort(key=lambda item: item["match_start"])
    return placements


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
    """Section title immediately preceding an inline ``[图片]`` block (same chunk only)."""
    chunk_start = context.rfind("\n\n", 0, path_match_start)
    if chunk_start < 0:
        chunk_start = 0
    else:
        chunk_start += 2
    window_start = max(chunk_start, path_match_start - 400)
    window = context[window_start:path_match_start]
    lines = [ln.strip() for ln in window.splitlines() if ln.strip()]
    for line in reversed(lines):
        if _is_image_metadata_line(line) or _is_toc_or_directory_line(line):
            continue
        if re.match(r"^\d+\.\d+(?:\.\d+)?\s+\S", line) and len(line) <= 80:
            return line
        if (
            any(line.startswith(m) for m in _domain_schema.section_markers)
            and "：" in line
        ):
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
        match = _SECTION_MARKER_CONTENT_RE.search(content)
        if match:
            return match.group(1).strip()
    return ""


def _ref_matches_manual_hint(ref: dict[str, Any], manual_hint: str) -> bool:
    hint = (manual_hint or "").strip()
    if not hint:
        return True
    for key in ("path", "source_key", "context", "caption", "label"):
        val = str(ref.get(key) or "").strip()
        if not val:
            continue
        if _doc_matches_manual_hint({"file_path": val}, hint):
            return True
    return False
