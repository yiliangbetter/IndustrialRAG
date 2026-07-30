"""Term-alignment primitives shared by ingest and query paths.

Split from utils.py (stage-1 refactor, see
docs/utils_refactor_schema_induction.md). Pure text heuristics with no
image / ingest dependencies.
"""

import re
from typing import Any, Dict, List


def _join_caption_field(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(x) for x in value if x).strip()
    return str(value or "").strip()


def image_label_text(item: Dict[str, Any]) -> str:
    """MinerU image caption or footnote label (either may be empty)."""
    if not isinstance(item, dict) or item.get("type") != "image":
        return ""
    caption = _join_caption_field(
        item.get("image_caption", item.get("img_caption", ""))
    )
    footnote = _join_caption_field(
        item.get("image_footnote", item.get("img_footnote", ""))
    )
    return " ".join(part for part in (caption, footnote) if part).strip()


def discriminative_terms(text: str, *, min_len: int = 2) -> List[str]:
    """Length-based terms for any snippet (no domain phrase lists)."""
    terms: List[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        term = term.strip()
        if len(term) < min_len or term in seen:
            return
        seen.add(term)
        terms.append(term)

    for run in re.findall(r"[\u4e00-\u9fff]+", text or ""):
        if min_len <= len(run) <= 24:
            add(run)
        for size in (min_len, min_len + 1):
            if size > len(run):
                continue
            for i in range(len(run) - size + 1):
                add(run[i : i + size])

    for term in re.findall(r"[a-zA-Z0-9]{4,}", (text or "").lower()):
        add(term)
    return terms


def text_term_alignment(left: str, right: str, *, min_len: int = 2) -> float:
    """Share of discriminative terms from ``left`` found in ``right``."""
    terms = discriminative_terms(left, min_len=min_len)
    if not terms or not right.strip():
        return 0.0
    hits = sum(1 for term in terms if term in right)
    return hits / len(terms)


def text_term_alignment_symmetric(left: str, right: str, *, min_len: int = 2) -> float:
    if not left.strip() or not right.strip():
        return 0.0
    if left in right or right in left:
        return 1.0
    return max(
        text_term_alignment(left, right, min_len=min_len),
        text_term_alignment(right, left, min_len=min_len),
    )


def substantive_bigrams(text: str) -> set[str]:
    """Unique 2-character CJK runs (length-based, no domain phrase lists)."""
    bigrams: set[str] = set()
    for run in re.findall(r"[\u4e00-\u9fff]+", text or ""):
        for i in range(len(run) - 1):
            bigrams.add(run[i : i + 2])
    return bigrams


_SHORT_LABEL_MAX_LEN = 20


_SHORT_LABEL_MIN_SHARED_BIGRAMS = 2


_SHORT_LABEL_ANCHOR_RUN_LEN = 4


def _longest_cjk_run(text: str) -> str:
    runs = re.findall(r"[\u4e00-\u9fff]+", text or "")
    return max(runs, key=len, default="")


def _best_overlap_cjk_run(query: str, label: str, *, min_len: int = 4) -> str:
    """CJK span in the query whose bigrams best match the figure label."""
    label_bigrams = substantive_bigrams(label)
    best_run = ""
    best_score = 0
    for run in re.findall(r"[\u4e00-\u9fff]+", query or ""):
        if len(run) < min_len:
            continue
        score = len(substantive_bigrams(run) & label_bigrams)
        if score > best_score or (score == best_score and len(run) > len(best_run)):
            best_score = score
            best_run = run
    return best_run


def _focus_run_for_bag(query: str, label: str, *, min_len: int = 4) -> str:
    """Shortest query CJK span with strong bigram overlap to the label."""
    label_bigrams = substantive_bigrams(label)
    best_run = ""
    best_key: tuple[int, int] = (0, 0)
    for run in re.findall(r"[\u4e00-\u9fff]+", query or ""):
        if len(run) < min_len:
            continue
        overlap = len(substantive_bigrams(run) & label_bigrams)
        if overlap < _SHORT_LABEL_MIN_SHARED_BIGRAMS:
            continue
        key = (overlap, -len(run))
        if key > best_key:
            best_key = key
            best_run = run
    return best_run


def _best_focus_subspan(
    focus: str, label: str, *, min_len: int = _SHORT_LABEL_ANCHOR_RUN_LEN
) -> str:
    """Shortest sub-span with label overlap; prefer tight match at minimum length."""
    label_bigrams = substantive_bigrams(label)
    min_sub_len = max(min_len, 6)
    matches: list[tuple[int, str]] = []
    for start in range(len(focus)):
        for end in range(start + min_sub_len, len(focus) + 1):
            sub = focus[start:end]
            overlap = len(substantive_bigrams(sub) & label_bigrams)
            if overlap < _SHORT_LABEL_MIN_SHARED_BIGRAMS:
                continue
            matches.append((len(sub), sub))
    if not matches:
        return ""
    min_length = min(length for length, _ in matches)
    for length, sub in sorted(matches, key=lambda item: item[0]):
        if length != min_length:
            continue
        if _focus_midsection_bigram_hits(sub, label):
            return sub
    return ""


def _focus_midsection_bigram_hits(focus: str, label: str) -> set[str]:
    """Shared bigrams from the interior of the focus span (excludes edge-only matches)."""
    if len(focus) < 4:
        return set()
    mid = focus[1:-1]
    if len(mid) < 2:
        return set()
    return substantive_bigrams(mid) & substantive_bigrams(label)


def label_bigram_coverage(query: str, label: str) -> float:
    """Share of figure-label bigrams also present in the query."""
    label_bgs = substantive_bigrams(label)
    if not label_bgs:
        return 0.0
    return len(label_bgs & substantive_bigrams(query)) / len(label_bgs)


def short_label_bag_aligns(
    query: str,
    label: str,
    *,
    max_label_len: int = _SHORT_LABEL_MAX_LEN,
    min_shared: int = _SHORT_LABEL_MIN_SHARED_BIGRAMS,
) -> bool:
    """Align short figure labels when word order differs (e.g. 清洁机器床身 vs 机床床身清洁)."""
    label = (label or "").strip()
    query = (query or "").strip()
    if not label or not query or len(label) > max_label_len:
        return False
    if re.search(r"[。；;，,：:]", label):
        return False

    shared = substantive_bigrams(query) & substantive_bigrams(label)
    if len(shared) < min_shared:
        return False

    focus = _focus_run_for_bag(query, label)
    if len(focus) >= _SHORT_LABEL_ANCHOR_RUN_LEN:
        subspan = _best_focus_subspan(focus, label)
        if not subspan:
            return False
        focus = subspan
        if not _focus_midsection_bigram_hits(focus, label):
            return False
        label_bgs = substantive_bigrams(label)
        sub_hits = len(label_bgs & substantive_bigrams(focus))
        min_hits = max(min_shared, int(len(label_bgs) * 0.34))
        if len(label) >= 6:
            min_hits = max(min_hits, int(len(label_bgs) * 0.5))
        if sub_hits < min_hits:
            return False
    else:
        anchor = _best_overlap_cjk_run(query, label)
        if len(anchor) >= _SHORT_LABEL_ANCHOR_RUN_LEN:
            anchor_bigrams = {anchor[i : i + 2] for i in range(len(anchor) - 1)}
            if not any(bigram in label for bigram in anchor_bigrams):
                return False

    return True


_SECTION_HEADING_LINE_RE = re.compile(r"^(?:\d+\.){1,3}\d+\s+\S")


_COALESCE_HEADING_MAX_CHARS = 120


def _is_section_heading_line(line: str) -> bool:
    line = (line or "").strip()
    if not line or len(line) > _COALESCE_HEADING_MAX_CHARS:
        return False
    return bool(_SECTION_HEADING_LINE_RE.match(line))
