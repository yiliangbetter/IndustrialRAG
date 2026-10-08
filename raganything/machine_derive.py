"""Derive the machine-model name from a manual document name.

Industrial maintenance manuals are named ``<machine><doc-type><noise>`` —
e.g. ``高速智能封边机维护保养手册``, ``数控六面钻产品维护保养说明-20240427``,
``双端封边机维护保养手册 新版8-24最终版``.  The machine-model segment is the
prefix that precedes a *document-type* keyword.  Document-type keywords are
structural vocabulary (they describe the kind of manual, not a machine), so
this module carries **no machine-model literals**: it works for any manual
whose name follows the ``<machine><doc-type>…`` convention.

Used at ingest to stamp every chunk with a ``machine`` field (see
``processor.py`` / ``ingest_insert.py``), so the query side can read the
machine from stored metadata instead of reverse-engineering it from
hard-coded machine lists.

Hard-coded vs derived — a deliberate split:
  * ``DOC_TYPE_KEYWORDS`` and the date/version noise pattern below are
    hard-coded *structural* vocabulary: they name the manual genre
    (maintenance / alarm-clearing / operation manuals), not any specific
    machine.  They are the stable cutting rule, not business data, and are
    NOT the kind of hardcode the de-hardcode effort targets.
  * The machine-model names themselves are derived at ingest by cutting the
    document name at the earliest doc-type keyword — they appear in no
    hard-coded list.  Adding a new manual needs no code change so long as its
    name follows ``<machine><doc-type>…``; only a brand-new manual *genre*
    would require extending ``DOC_TYPE_KEYWORDS`` (rare, and structural).
"""

from __future__ import annotations

import re

__all__ = ["derive_machine_from_docname", "DOC_TYPE_KEYWORDS"]

# Document-type keywords: the machine-model name is the prefix before the
# earliest of these in the file stem.  These describe the *kind* of manual
# (maintenance / alarm-clearing / operation), never a specific machine.
DOC_TYPE_KEYWORDS: tuple[str, ...] = (
    "维护保养",
    "维修保养",
    "电气报警",
    "报警排除",
    "故障排除",
    "操作说明",
    "使用说明",
    "安装说明",
    "产品说明",
    "产品",  # 「产品维护保养说明」中的前缀噪声，须早于「维护保养」切分
)

# Trailing date / version / edition noise that may follow the machine name
# when no document-type keyword is present (``…8-25``, ``…-20240510``,
# ``… 新版8-24最终版``).
_TRAILING_NOISE_RE = re.compile(
    r"[\s_\-]*("
    r"新版|最终版|定稿|终稿|修订版?|版"
    r"|\d{4}[-./年]\d{1,2}[-./月]\d{1,2}日?"
    r"|\d{6,8}"
    r"|\d{1,2}[-./月]\d{1,2}日?"
    r")$"
)


def _strip_trailing_noise(name: str) -> str:
    """Iteratively drop trailing date/version/edition markers."""
    prev = None
    cur = name.strip()
    while cur and cur != prev:
        prev = cur
        cur = _TRAILING_NOISE_RE.sub("", cur).strip(" _-·．.")
    return cur


def derive_machine_from_docname(name: str) -> str:
    """Return the machine-model segment of a manual document name.

    Cuts the stem at the earliest document-type keyword; falls back to
    stripping trailing date/version noise when no keyword is present.
    Returns ``""`` only when nothing usable remains.
    """
    stem = (name or "").strip()
    if not stem:
        return ""
    # Drop path and file extension (``…/双端封边机维护保养手册.pdf``).
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", stem.replace("\\", "/").split("/")[-1])
    stem = stem.strip()

    cut = -1
    for kw in DOC_TYPE_KEYWORDS:
        idx = stem.find(kw)
        if idx > 0 and (cut < 0 or idx < cut):
            cut = idx
    if cut > 0:
        return stem[:cut].strip(" _-·．.")

    return _strip_trailing_noise(stem)
