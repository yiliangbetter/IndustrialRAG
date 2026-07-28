"""image_query_refs submodule ``iqr_terms`` (stage-0 relocation, auto-generated).

Behavior-preserving split of scripts/image_query_refs.py. Do not hand-edit;
regenerate via scripts/_tmp_iqr_gen.py.
"""
from __future__ import annotations

import os
import re
from typing import Any
from raganything.utils import (
    discriminative_terms,
    short_label_bag_aligns,
    substantive_bigrams,
    text_term_alignment_symmetric,
)
from iqr_config import _min_substantive_term_len


def _normalize_query_for_match(query: str) -> str:
    q = (query or "").strip()
    q = re.sub(r"[\s!！?？。.，,~、；;：:" "''\"']+", "", q, flags=re.I)
    return q


def _query_terms(query: str) -> list[str]:
    """Extract discriminative terms from the query (length-based, no phrase lists)."""
    return discriminative_terms(query, min_len=_min_substantive_term_len())


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


def _normalize_label_key(label: str) -> str:
    return re.sub(r"\s+", "", (label or "").strip())


def _listing_target_head(span: str) -> str:
    """Component name before generic qualifiers (e.g. 涂胶轴及其周围 → 涂胶轴)."""
    span = (span or "").strip().rstrip("：:")
    span = re.sub(r"[（(][^）)]*[）)]", "", span).strip()
    for sep in ("及其", "及"):
        if sep in span:
            span = span.split(sep, 1)[0].strip()
    return span


def _is_procedure_steps_query(query: str) -> bool:
    """How-to / procedure steps, not multi-item component listings."""
    q = (query or "").strip()
    if not q:
        return False
    return bool(re.search(r"步骤|怎么做|如何操作|操作方法|操作流程|怎样|如何进行", q))


def _is_query_subject_echo(span: str, query: str) -> bool:
    """True when *span* mostly restates the query subject rather than a distinct part."""
    s = (span or "").strip()
    q = (query or "").strip()
    if not s or not q:
        return False
    if text_term_alignment_symmetric(s, q, min_len=2) < 0.5:
        return False
    s_terms = discriminative_terms(s, min_len=2)
    if not s_terms:
        return False
    q_compact = re.sub(r"\s+", "", q)
    hit = sum(
        1
        for t in s_terms
        if t in q or t in q_compact or re.sub(r"\s+", "", t) in q_compact
    )
    return hit / len(s_terms) >= 0.5


def _term_overlap_ratio(query: str, text: str) -> float:
    terms = _query_terms(query)
    if not terms or not text.strip():
        return 0.0
    hits = sum(1 for term in terms if term in text)
    return hits / len(terms)


def _image_min_term_overlap() -> float:
    raw = os.getenv("RAG_IMAGE_MIN_TERM_OVERLAP") or "0.28"
    try:
        return float(raw)
    except ValueError:
        return 0.34


def _line_has_specific_query_overlap(query: str, line: str, ref_text: str) -> bool:
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


def _subject_from_machine_field_line(line: str) -> str:
    m = re.match(
        r"^(?:\d+\.\s*)?[-*•]?\s*\*\*[^*]+\*\*[：:]\s*(.+)$",
        (line or "").strip(),
    )
    if not m:
        return ""
    tail = re.split(r"保养周期", m.group(1), maxsplit=1)[0].strip()
    tail = re.sub(r"\*\*([^*]+)\*\*", r"\1", tail).strip().rstrip("。")
    tail = re.split(r"[（(]", tail, maxsplit=1)[0].strip()
    return _listing_target_head(tail)


def _line_spans_in_body(body: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    offset = 0
    for part in body.splitlines(keepends=True):
        stripped = part.strip()
        if not stripped:
            offset += len(part)
            continue
        start = body.find(stripped, offset)
        if start < 0:
            start = offset
        end = start + len(stripped)
        spans.append((start, end, stripped))
        offset = end
    return spans


def _procedure_step_spans(answer: str) -> list[str]:
    """Bold titles from numbered procedure steps in the answer body."""
    from iqr_figure_target import _answer_text_for_placement
    body = _answer_text_for_placement(answer)
    spans: list[str] = []
    seen: set[str] = set()
    for raw_line in body.splitlines():
        line = raw_line.strip()
        match = re.match(r"^\d+\.\s*\*\*([^*]+)\*\*", line)
        if not match:
            continue
        head = _listing_target_head(match.group(1).strip())
        if not head:
            continue
        key = _normalize_label_key(head)
        if key in seen:
            continue
        seen.add(key)
        spans.append(head)
    return spans


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


def _figure_matches_query_object(query: str, text: str) -> bool:
    """Query object must appear in figure text; no 开关-in-保护开关 substring hits."""
    from iqr_figure_target import _strict_object_image_gate
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


def _action_focus_bigrams(query: str, retrieved_text: str | None = None) -> set[str]:
    """Bigrams for the query's concrete subject/action (not the device name echo)."""
    from iqr_store import _ranked_retrieval_lines
    from iqr_figure_target import _is_toc_or_directory_line
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
            if len(
                action
            ) >= _min_substantive_term_len() and not _is_toc_or_directory_line(line):
                candidate = _suffix_focus_bigrams(action)
                if candidate:
                    return candidate
    return _suffix_focus_bigrams(_query_tail_text(query))


def _ref_passes_focus_bigram_gate(
    query: str,
    ref: dict[str, Any],
    retrieved_text: str | None = None,
) -> bool:
    """Reject figures whose label only shares generic inspection bigrams with the query."""
    from iqr_align import _ref_effective_label
    from iqr_figure_target import _strict_object_image_gate
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


