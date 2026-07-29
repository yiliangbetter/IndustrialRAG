"""Dynamic machine-model vocabulary derived from the knowledge base.

Replaces the former hard-coded ``_KNOWN_MACHINE_NAMES`` tuple.  The vocabulary
is induced from the KB's stored chunks: the ``machine`` field stamped at ingest
(see ``raganything/machine_derive.py``), with a fallback that derives the
machine from each chunk's ``file_path`` so it also works on a KB ingested
before the ``machine`` field existed.  Adding a new manual to the KB extends
the vocabulary automatically — no code change required.
"""

from __future__ import annotations

import re

from raganything.machine_derive import derive_machine_from_docname

__all__ = ["known_machine_names", "resolve_machine_name"]

_vocab_cache: tuple[str, ...] | None = None
_vocab_mtime: float | None = None


def known_machine_names() -> tuple[str, ...]:
    """Distinct machine-model names present in the KB (longest first).

    Cached against the text-chunks KV store mtime, so it rebuilds only when
    the KB changes.  Returns ``()`` when the store is unavailable.
    """
    global _vocab_cache, _vocab_mtime
    try:
        from iqr_store import _kv_text_chunks_store  # noqa: WPS433  (lazy: avoid cycle)
    except ImportError:
        return _vocab_cache or ()
    store = _kv_text_chunks_store()
    try:
        import iqr_store as _s  # noqa: WPS433

        mtime = getattr(_s, "_kv_store_mtime", None)
    except ImportError:  # pragma: no cover
        mtime = None
    if _vocab_cache is not None and mtime == _vocab_mtime:
        return _vocab_cache
    names: set[str] = set()
    for row in store.values():
        if not isinstance(row, dict):
            continue
        machine = str(row.get("machine") or "").strip()
        if not machine:
            machine = derive_machine_from_docname(str(row.get("file_path") or ""))
        machine = machine.strip()
        if 2 <= len(machine) <= 24:
            names.add(machine)
    _vocab_cache = tuple(sorted(names, key=len, reverse=True))
    _vocab_mtime = mtime
    return _vocab_cache


def resolve_machine_name(text: str) -> str:
    """Longest KB machine name occurring in *text* (canonical, bleed-safe).

    Longest-first matching keeps ``高速自动封边机`` from collapsing to
    ``自动封边机``.  Returns ``""`` when no vocabulary machine occurs.
    """
    blob = (text or "").strip()
    if not blob:
        return ""
    compact = re.sub(r"\s+", "", blob)
    for name in known_machine_names():
        if re.sub(r"\s+", "", name) in compact or name in blob:
            return name
    return ""
