"""image_query_refs submodule ``iqr_store``.

KV-store access, doc/content helpers, source hints, and content-list loading.
"""
from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from iqr_protocol import (
    _is_image_metadata_line,
    extract_image_refs_from_context,
    normalize_image_path,
)
from iqr_config import _min_substantive_term_len
from iqr_terms import (
    _query_subject_needles,
    _term_overlap_ratio,
)


logger = logging.getLogger(__name__)


_kv_store_cache: dict[str, Any] | None = None


_kv_store_mtime: float = -1.0


_cl_path_index: list[tuple[Path, str, Path]] | None = None


def _unique_retrieved_doc_count(retrieved_docs: list[dict[str, Any]] | None) -> int:
    paths: set[str] = set()
    for doc in retrieved_docs or []:
        if not isinstance(doc, dict):
            continue
        fp = str(doc.get("file_path") or "").strip()
        if fp:
            paths.add(fp)
    return len(paths)


def _is_multi_source_retrieval(retrieved_docs: list[dict[str, Any]] | None) -> bool:
    return _unique_retrieved_doc_count(retrieved_docs) >= 2


def _source_hint_matches_doc(hint: str, doc_hint: str) -> bool:
    """Machine-aware doc_hint filter for content_list supplement (no substring bleed)."""
    from iqr_figure_target import _resolve_known_machine_name
    hint = (hint or "").strip()
    doc_hint = (doc_hint or "").strip()
    if not hint or not doc_hint:
        return False
    hint_compact = re.sub(r"\s+", "", hint)
    doc_compact = re.sub(r"\s+", "", doc_hint)
    # Data-driven anti-bleed: resolve both sides to their canonical KB machine
    # and require agreement (replaces the former hard-coded 高速智能/高速自动/
    # 自动封边机 substring special cases).
    hint_machine = _resolve_known_machine_name(hint)
    doc_machine = _resolve_known_machine_name(doc_hint)
    if hint_machine and doc_machine:
        return hint_machine == doc_machine
    if hint_machine:
        return (
            hint_machine in doc_hint or re.sub(r"\s+", "", hint_machine) in doc_compact
        )
    if hint_compact in doc_compact or doc_compact in hint_compact:
        return True
    if len(hint_compact) >= 8 and hint_compact[:8] in doc_compact:
        return True
    return False


def _doc_matches_cited_hints(doc: dict[str, Any], cited_hints: set[str]) -> bool:
    if not cited_hints:
        return True
    fp = _doc_basename(doc)
    if not fp:
        return False
    fp_compact = re.sub(r"\s+", "", fp)
    for hint in cited_hints:
        compact = re.sub(r"\s+", "", hint)
        if hint in fp or (compact and compact[:6] in fp_compact):
            return True
    return False


def _doc_chunk_order_index(doc: dict[str, Any]) -> int | None:
    raw = doc.get("chunk_order_index")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _load_manual_chunks_for_locality(manual_hint: str) -> list[dict[str, Any]]:
    """All KB chunks for one manual, sorted by ingest order (includes short heading chunks)."""
    from iqr_figure_target import _doc_matches_manual_hint
    hint = (manual_hint or "").strip()
    if not hint:
        return []
    try:
        from query_doc_steering import (  # noqa: WPS433
            _path_hits_deny,
            active_deny_substrings,
        )
    except ImportError:
        return []
    deny, _ = active_deny_substrings("")
    raw = _kv_text_chunks_store()
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for chunk_id, row in raw.items():
        if not isinstance(row, dict):
            continue
        fp = str(row.get("file_path") or "")
        if not fp or _path_hits_deny(fp, deny):
            continue
        doc = dict(row)
        doc.setdefault("content", str(row.get("content") or ""))
        doc.setdefault("id", chunk_id)
        if not _doc_matches_manual_hint(doc, hint):
            continue
        out.append(doc)
    out.sort(
        key=lambda doc: (
            _doc_chunk_order_index(doc)
            if _doc_chunk_order_index(doc) is not None
            else 10**9,
            str(doc.get("id") or ""),
        )
    )
    return out


def _pipeline_content_list_entries() -> list[tuple[Path, str, Path]]:
    """Cached ``(content_list_path, doc_hint, auto_dir)`` under parse roots."""
    global _cl_path_index
    if _cl_path_index is not None:
        return _cl_path_index
    entries: list[tuple[Path, str, Path]] = []
    for root in _pipeline_parse_roots():
        if not root.is_dir():
            continue
        try:
            cl_paths = list(root.rglob("*_content_list.json"))
        except OSError:
            continue
        for cl_path in cl_paths:
            doc_hint = cl_path.stem.replace("_content_list", "").replace(
                "_content_list_v2", ""
            )
            entries.append((cl_path, doc_hint, cl_path.parent))
    _cl_path_index = entries
    return entries


def _pipeline_parse_roots() -> list[Path]:
    """Parse output dirs for content_list fallback (§4.2 step 3)."""
    roots: list[Path] = []
    for key in ("RAG_WEB_PARSER_OUTPUT_DIR", "RAG_PARSER_OUTPUT_DIR"):
        raw = (os.getenv(key) or "").strip()
        if raw:
            roots.append(Path(raw))
    repo = Path(__file__).resolve().parents[1]
    for rel in ("data/pipeline_parse", "output/pipeline_parse"):
        roots.append(repo / rel)
    seen: set[str] = set()
    out: list[Path] = []
    for path in roots:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _first_figure_ref_from_doc(doc: dict[str, Any]) -> dict[str, Any] | None:
    refs = extract_image_refs_from_context(_doc_content(doc))
    return refs[0] if refs else None


def _source_hints_from_retrieved_docs(
    retrieved_docs: list[dict[str, Any]] | None,
) -> set[str]:
    hints: set[str] = set()
    for doc in retrieved_docs or []:
        if not isinstance(doc, dict):
            continue
        fp = str(doc.get("file_path") or "").strip()
        if not fp:
            continue
        stem = Path(fp).stem[:80]
        hints.add(stem)
        compact = re.sub(r"\s+", "", stem)
        if compact:
            hints.add(compact[:80])
    return hints


def _merged_source_hints(
    text: str,
    retrieved_docs: list[dict[str, Any]] | None = None,
) -> set[str]:
    hints = _source_hints_from_text(text)
    hints |= _source_hints_from_retrieved_docs(retrieved_docs)
    return {h for h in hints if h and len(h) >= 4}


def _ref_matches_source_hints(ref: dict[str, Any], hints: set[str]) -> bool:
    if not hints:
        return True
    path = str(ref.get("path") or "")
    return any(hint in path for hint in hints)


def _ranked_retrieval_lines(
    query: str, text: str, *, limit: int = 8
) -> list[tuple[float, str]]:
    """Lines from retrieved context ranked by query term overlap."""
    from iqr_align import _PDF_NAME_RE, _SECTION_NUM_RE
    from iqr_figure_target import _is_toc_or_directory_line
    if not text.strip():
        return []
    scored: list[tuple[float, str]] = []
    for line in re.split(r"[\n\r]+", text):
        line = line.strip()
        if len(line) < _min_substantive_term_len():
            continue
        if _is_image_metadata_line(line) or _is_toc_or_directory_line(line):
            continue
        if (
            _PDF_NAME_RE.search(line)
            or ".jpg" in line.lower()
            or ".png" in line.lower()
        ):
            continue
        cjk = "".join(re.findall(r"[\u4e00-\u9fff]", line))
        if len(cjk) < _min_substantive_term_len():
            continue
        overlap = _term_overlap_ratio(query, line)
        if overlap <= 0:
            continue
        cjk_only = "".join(re.findall(r"[\u4e00-\u9fff]", line))
        is_short_title = not _SECTION_NUM_RE.match(line) and len(cjk_only) <= 10
        if not is_short_title:
            for needle in _query_subject_needles(query):
                if len(needle) >= 4 and needle in line:
                    overlap += 0.22
            if _SECTION_NUM_RE.match(line):
                overlap += 0.08
        if is_short_title and overlap < 0.35:
            overlap *= 0.45
        scored.append((overlap, line))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:limit]


def _doc_content(doc: dict[str, Any]) -> str:
    for key in ("content", "text", "chunk_content", "page_content"):
        val = doc.get(key)
        if isinstance(val, str):
            return val
    return ""


def _doc_basename(doc: dict[str, Any]) -> str:
    fp = str(doc.get("file_path") or "").strip()
    if not fp:
        return ""
    return Path(fp.replace("\\", "/")).name


def _doc_content_key(doc: dict[str, Any]) -> str:
    content = _doc_content(doc).strip()
    if content:
        return f"content:{content}"
    chunk_id = str(doc.get("id") or doc.get("chunk_id") or "").strip()
    if chunk_id:
        return f"id:{chunk_id}"
    return ""


def _dedupe_doc_list(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for doc in docs:
        key = _doc_content_key(doc)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def _dedupe_doc_list_by_chunk_identity(
    docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Dedupe retrieval docs by chunk id / manual+content (not content alone)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for doc in docs:
        chunk_id = str(doc.get("id") or doc.get("chunk_id") or "").strip()
        if chunk_id:
            key = f"id:{chunk_id}"
        else:
            key = f"{_doc_basename(doc)}:{_doc_content(doc).strip()[:120]}"
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def _doc_storage_chunk_id(doc: dict[str, Any]) -> str:
    for key in ("chunk_id", "id"):
        val = str(doc.get(key) or "").strip()
        if val.startswith("chunk-"):
            return val
    return ""


def _kv_text_chunks_store() -> dict[str, Any]:
    global _kv_store_cache, _kv_store_mtime
    try:
        from query_doc_steering import _text_chunks_store_path  # noqa: WPS433
    except ImportError:
        return {}
    path = _text_chunks_store_path()
    if not path.is_file():
        _kv_store_cache = {}
        _kv_store_mtime = -1.0
        return {}
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if _kv_store_cache is not None and mtime == _kv_store_mtime:
        return _kv_store_cache
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    _kv_store_cache = raw if isinstance(raw, dict) else {}
    _kv_store_mtime = mtime
    return _kv_store_cache


def _kv_chunk_row(chunk_id: str) -> dict[str, Any] | None:
    cid = (chunk_id or "").strip()
    if not cid:
        return None
    row = _kv_text_chunks_store().get(cid)
    return row if isinstance(row, dict) else None


def _context_for_image_scan(
    query: str,
    primary_text: str,
    retrieved_docs: list[dict[str, Any]] | None,
    *,
    answer: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Inline-image scan: answer-topic chunks only (no query-primary expansion)."""
    from iqr_figure_target import _figure_context_from_answer_docs
    if (answer or "").strip() and retrieved_docs:
        return _figure_context_from_answer_docs(
            answer, list(retrieved_docs or []), query=query
        )

    primary = (primary_text or "").strip()
    meta: dict[str, Any] = {
        "mode": "off",
        "primary_chars": len(primary),
        "anchor_sections": [],
        "anchor_chars": 0,
        "anchor_chunks": 0,
        "reason": "no_answer",
    }
    return primary, meta


def _refs_from_retrieved_docs_text(
    context: str,
    media_roots: list[Path],
    *,
    query: str | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
    full_context: str | None = None,
) -> list[dict[str, Any]]:
    """Inline ``[图片]`` refs from rerank-filtered primary text only."""
    ctx = (context or "").strip()
    hint_source = (full_context or ctx).strip()
    hints = _merged_source_hints(hint_source, retrieved_docs)
    refs = extract_image_refs_from_context(ctx)
    if hints:
        hinted = [ref for ref in refs if _ref_matches_source_hints(ref, hints)]
        if hinted:
            refs = hinted
    return refs


def _eligible_figure_refs(context: str) -> list[dict[str, Any]]:
    from iqr_figure_target import _is_cover_page_ref
    return [
        ref
        for ref in extract_image_refs_from_context(context)
        if not _is_cover_page_ref(ref)
    ]


def _collect_figure_refs(
    context: str,
    media_roots: list[Path] | None = None,
    *,
    query: str | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
    full_context: str | None = None,
) -> list[dict[str, Any]]:
    """Inline chunk figures from rerank-filtered primary context."""
    if media_roots:
        return _refs_from_retrieved_docs_text(
            context,
            media_roots,
            query=query,
            retrieved_docs=retrieved_docs,
            full_context=full_context,
        )
    return _eligible_figure_refs(context)


def _source_key_from_path(path_str: str) -> str:
    """Group images by parser output folder (one key per ingested document tree)."""
    path = Path(normalize_image_path(path_str))
    parts = [p for p in path.parts if p.lower() not in {"images", "auto", "figures"}]
    if len(parts) >= 2:
        return parts[-2]
    return path.stem


def _source_hints_from_text(text: str) -> set[str]:
    """Document titles/paths mentioned in retrieved context."""
    from iqr_align import _PDF_NAME_RE, _REF_LINE_RE
    hints: set[str] = set()
    for match in _REF_LINE_RE.finditer(text):
        title = match.group(2).strip()
        if title:
            hints.add(title.split(".pdf")[0].split(".PDF")[0][:80])
    for match in _PDF_NAME_RE.finditer(text):
        hints.add(Path(match.group(0)).stem[:80])
    return hints


def text_from_retrieved_docs(docs: list[dict[str, Any]] | None) -> str:
    """Concatenate chunk bodies returned by LightRAG rerank / retrieval."""
    parts: list[str] = []
    for doc in docs or []:
        if not isinstance(doc, dict):
            continue
        for key in ("content", "text", "chunk_content", "page_content"):
            val = doc.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
                break
    return "\n\n".join(parts)


def merge_context_for_images(*sources: str | None) -> str:
    parts = [s.strip() for s in sources if isinstance(s, str) and s.strip()]
    return "\n\n".join(parts)


@lru_cache(maxsize=64)
def _load_content_list_items_cached(
    path_str: str, mtime_ns: int
) -> tuple[Any, ...] | None:
    del mtime_ns
    path = Path(path_str)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        return tuple(raw[0])
    if isinstance(raw, list):
        return tuple(raw)
    return None


def _load_content_list_items(path: Path) -> list[dict[str, Any]] | None:
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return None
    cached = _load_content_list_items_cached(str(path.resolve()), mtime_ns)
    if cached is None:
        return None
    return list(cached)


