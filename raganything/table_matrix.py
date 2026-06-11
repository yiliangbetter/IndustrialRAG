"""Row-matrix index for MinerU HTML tables (ingest + query supplement).

Tables are parsed into ``{table_id, headers, rows, chunk_ids, ...}`` and stored
in ``kv_store_table_matrix.json`` under the LightRAG working directory.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lightrag.utils import compute_mdhash_id, logger

from raganything.utils import (
    _TABLE_INGEST_MARKER,
    discriminative_terms,
    text_term_alignment_symmetric,
)

_TABLE_LEGACY_FLAT_MARKER = "[TableFlat]"

_MATRIX_STORE_NAME = "kv_store_table_matrix.json"
_TABLE_ROW_RE = re.compile(r"<tr>.*?</tr>", re.IGNORECASE | re.DOTALL)
_TABLE_CELL_RE = re.compile(r"<td[^>]*>([^<]*)</td>", re.IGNORECASE)


def strip_table_flat_from_content(content: str) -> str:
    """Remove legacy ``[TableFlat]`` block; keep optional caption + ``[Table]`` HTML."""
    t = (content or "").strip()
    if _TABLE_LEGACY_FLAT_MARKER not in t:
        return content
    before, _, after = t.partition(_TABLE_LEGACY_FLAT_MARKER)
    caption = before.strip()
    table_idx = after.find(_TABLE_INGEST_MARKER)
    if table_idx < 0:
        return caption if caption else content
    table_part = after[table_idx:].lstrip()
    if caption:
        return f"{caption}\n\n{table_part}".strip()
    return table_part


def table_matrix_ingest_enabled() -> bool:
    raw = (os.getenv("RAG_TABLE_MATRIX_INGEST") or "true").strip().lower()
    return raw in ("1", "true", "yes", "on")


def table_matrix_query_boost_enabled() -> bool:
    raw = (os.getenv("RAG_TABLE_MATRIX_QUERY_BOOST") or "true").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _matrix_min_align() -> float:
    raw = os.getenv("RAG_TABLE_MATRIX_MIN_ALIGN") or "0.08"
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.08


def _matrix_min_matching_rows() -> int:
    raw = os.getenv("RAG_TABLE_MATRIX_MIN_ROWS") or "2"
    try:
        return max(1, int(raw))
    except ValueError:
        return 2


def matrix_store_path(working_dir: str | Path) -> Path:
    return Path(working_dir) / _MATRIX_STORE_NAME


def _table_row_cells(row_html: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", cell.strip())
        for cell in _TABLE_CELL_RE.findall(row_html or "")
    ]


def parse_table_html(html: str) -> tuple[list[str], list[list[str]]]:
    """Return (header cells, data rows). Header is first ``<tr>`` when present."""
    body = (html or "").strip()
    if not body:
        return [], []
    rows = _TABLE_ROW_RE.findall(body)
    if not rows:
        return [], []
    parsed = [_table_row_cells(row) for row in rows]
    parsed = [cells for cells in parsed if cells]
    if not parsed:
        return [], []
    if len(parsed) == 1:
        return parsed[0], []
    return parsed[0], parsed[1:]


def extract_table_segment_parts(segment: str) -> tuple[str, str]:
    """Parse ingest segment → (caption, html_body)."""
    piece = strip_table_flat_from_content((segment or "").strip())
    if not piece:
        return "", ""
    marker_idx = piece.find(_TABLE_INGEST_MARKER)
    if marker_idx < 0:
        return "", ""
    caption = piece[:marker_idx].strip()
    html_body = piece[marker_idx + len(_TABLE_INGEST_MARKER) :].lstrip("\n")
    return caption, html_body


def extract_table_from_chunk_content(content: str) -> tuple[str, str]:
    """Parse stored chunk text → (caption, html)."""
    return extract_table_segment_parts(content)


@dataclass
class TableMatrixRecord:
    table_id: str
    full_doc_id: str
    file_path: str
    caption: str = ""
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    html_signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableMatrixRecord:
        return cls(
            table_id=str(data.get("table_id") or ""),
            full_doc_id=str(data.get("full_doc_id") or ""),
            file_path=str(data.get("file_path") or ""),
            caption=str(data.get("caption") or ""),
            headers=list(data.get("headers") or []),
            rows=[list(r) for r in (data.get("rows") or [])],
            chunk_ids=list(data.get("chunk_ids") or []),
            html_signature=str(data.get("html_signature") or ""),
        )


def row_query_text(headers: list[str], cells: list[str]) -> str:
    width = len(headers)
    padded = (cells + [""] * width)[:width] if width else cells
    if headers and width:
        return " ".join(
            f"{headers[i]} {padded[i]}"
            for i in range(width)
            if padded[i]
        )
    return " ".join(c for c in padded if c)


def html_signature(html: str) -> str:
    body = re.sub(r"\s+", "", (html or "")[:500])
    return body[:240]


def build_records_from_segments(
    segments: list[str],
    *,
    full_doc_id: str,
    file_path: str,
) -> list[TableMatrixRecord]:
    records: list[TableMatrixRecord] = []
    for idx, segment in enumerate(segments):
        caption, html = extract_table_segment_parts(segment)
        if not html:
            continue
        headers, rows = parse_table_html(html)
        if not headers and not rows:
            continue
        sig = html_signature(html)
        content_sig = f"{full_doc_id}:{idx}:{sig}:{len(rows)}"
        table_id = compute_mdhash_id(content_sig, prefix="table-")
        records.append(
            TableMatrixRecord(
                table_id=table_id,
                full_doc_id=full_doc_id,
                file_path=file_path,
                caption=caption,
                headers=headers,
                rows=rows,
                html_signature=sig,
            )
        )
    return records


def build_records_from_document_parts(
    document_parts: list[str],
    *,
    full_doc_id: str,
    file_path: str,
) -> list[TableMatrixRecord]:
    """Build matrix rows from pre-split ``document_parts`` (one record per table block)."""
    records: list[TableMatrixRecord] = []
    table_idx = 0
    for part in document_parts:
        piece = (part or "").strip()
        if not piece:
            continue
        caption, html = extract_table_segment_parts(piece)
        if not html and _TABLE_INGEST_MARKER not in piece:
            continue
        if not html:
            marker_idx = piece.find(_TABLE_INGEST_MARKER)
            if marker_idx < 0:
                continue
            caption = piece[:marker_idx].strip()
            html = piece[marker_idx + len(_TABLE_INGEST_MARKER) :].lstrip("\n")
        headers, rows = parse_table_html(html)
        if not headers and not rows:
            continue
        sig = html_signature(html)
        content_sig = f"{full_doc_id}:{table_idx}:{sig}:{len(rows)}"
        table_id = compute_mdhash_id(content_sig, prefix="table-")
        records.append(
            TableMatrixRecord(
                table_id=table_id,
                full_doc_id=full_doc_id,
                file_path=file_path,
                caption=caption,
                headers=headers,
                rows=rows,
                html_signature=sig,
            )
        )
        table_idx += 1
    return records


def link_records_to_chunks(
    records: list[TableMatrixRecord],
    chunks: dict[str, dict[str, Any]],
    *,
    full_doc_id: str,
) -> None:
    """Attach ``chunk_ids`` by matching ``[Table]`` HTML signatures on the same doc."""
    doc_chunks: list[tuple[str, str]] = []
    for cid, row in chunks.items():
        if not isinstance(row, dict):
            continue
        if str(row.get("full_doc_id") or "") != full_doc_id:
            continue
        content = str(row.get("content") or "")
        if _TABLE_INGEST_MARKER not in content:
            continue
        _, html = extract_table_from_chunk_content(content)
        if html:
            doc_chunks.append((cid, html))

    for record in records:
        record.chunk_ids = []
        sig = record.html_signature
        if not sig:
            continue
        for cid, html in doc_chunks:
            if html_signature(html) == sig or sig in re.sub(r"\s+", "", html):
                record.chunk_ids.append(cid)
        if not record.chunk_ids and record.rows:
            first_row = [c for c in record.rows[0] if c]
            for cid, html in doc_chunks:
                if first_row and all(c in html for c in first_row[: min(3, len(first_row))]):
                    record.chunk_ids.append(cid)


def score_table_for_query(query: str, record: TableMatrixRecord) -> tuple[float, int]:
    """Return (best row align, count of rows above min align)."""
    min_align = _matrix_min_align()
    best = 0.0
    matches = 0
    blob_parts = [record.caption] if record.caption else []
    for row in record.rows:
        text = row_query_text(record.headers, row)
        if not text:
            continue
        align = text_term_alignment_symmetric(query, text)
        best = max(best, align)
        if align >= min_align:
            matches += 1
    if blob_parts:
        best = max(best, text_term_alignment_symmetric(query, " ".join(blob_parts)))
    return best, matches


def table_matrix_matches_query(query: str, record: TableMatrixRecord) -> bool:
    best, matches = score_table_for_query(query, record)
    if matches >= _matrix_min_matching_rows():
        return True
    terms = discriminative_terms(query, min_len=3)
    if not terms:
        return False
    row_blob = " ".join(
        row_query_text(record.headers, row) for row in record.rows[:40]
    )
    if record.caption:
        row_blob = f"{record.caption} {row_blob}"
    hit = sum(1 for t in terms if t in row_blob)
    if hit >= 2 and matches >= 1:
        return True
    return best >= min(_matrix_min_align() * 3, 0.35) and matches >= 1


def load_matrix_store(working_dir: str | Path) -> dict[str, TableMatrixRecord]:
    path = matrix_store_path(working_dir)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("table_matrix: read failed: %s", exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, TableMatrixRecord] = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            out[str(key)] = TableMatrixRecord.from_dict(val)
    return out


def save_matrix_store(working_dir: str | Path, store: dict[str, TableMatrixRecord]) -> None:
    path = matrix_store_path(working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.to_dict() for k, v in store.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_matrix_for_doc(working_dir: str | Path, full_doc_id: str) -> int:
    if not full_doc_id:
        return 0
    store = load_matrix_store(working_dir)
    removed = 0
    for tid in list(store):
        if store[tid].full_doc_id == full_doc_id:
            del store[tid]
            removed += 1
    if removed:
        save_matrix_store(working_dir, store)
    return removed


def upsert_matrix_for_doc(
    working_dir: str | Path,
    records: list[TableMatrixRecord],
    *,
    full_doc_id: str,
) -> int:
    delete_matrix_for_doc(working_dir, full_doc_id)
    if not records:
        return 0
    store = load_matrix_store(working_dir)
    for record in records:
        if record.table_id:
            store[record.table_id] = record
    save_matrix_store(working_dir, store)
    return len(records)


async def persist_table_matrix_after_ingest(
    lightrag,
    *,
    full_doc_id: str,
    file_path: str,
    ingest_segments: list[str] | None = None,
    document_parts: list[str] | None = None,
) -> int:
    """Link table matrix rows to LightRAG chunks and persist."""
    if not table_matrix_ingest_enabled():
        return 0
    wd = getattr(lightrag, "working_dir", None) or ""
    if not wd:
        return 0

    if ingest_segments:
        records = build_records_from_segments(
            ingest_segments, full_doc_id=full_doc_id, file_path=file_path
        )
    elif document_parts:
        records = build_records_from_document_parts(
            document_parts, full_doc_id=full_doc_id, file_path=file_path
        )
    else:
        return 0

    if not records:
        return 0

    chunks_storage = getattr(lightrag, "text_chunks", None)
    all_chunks: dict[str, dict[str, Any]] = {}
    if chunks_storage is not None:
        doc_status = getattr(lightrag, "doc_status", None)
        chunk_ids: list[str] = []
        if doc_status is not None:
            meta = await doc_status.get_by_id(full_doc_id)
            if isinstance(meta, dict):
                chunk_ids = list(meta.get("chunks_list") or [])
        if chunk_ids:
            rows = await chunks_storage.get_by_ids(chunk_ids)
            for cid, row in zip(chunk_ids, rows or []):
                if isinstance(row, dict):
                    all_chunks[cid] = row
        if not all_chunks:
            path = Path(wd) / "kv_store_text_chunks.json"
            if path.is_file():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        for cid, row in raw.items():
                            if isinstance(row, dict) and row.get("full_doc_id") == full_doc_id:
                                all_chunks[str(cid)] = row
                except (OSError, json.JSONDecodeError):
                    pass

    link_records_to_chunks(records, all_chunks, full_doc_id=full_doc_id)
    count = upsert_matrix_for_doc(wd, records, full_doc_id=full_doc_id)
    logger.info(
        "table_matrix: persisted %d table(s) for %s (%d linked chunk refs)",
        count,
        file_path,
        sum(len(r.chunk_ids) for r in records),
    )
    return count


def build_matrix_from_text_chunks(
    chunks: dict[str, dict[str, Any]],
) -> dict[str, TableMatrixRecord]:
    """Rebuild matrix store entries from ``kv_store_text_chunks`` (backfill)."""
    by_doc: dict[str, list[tuple[str, str, str]]] = {}
    for cid, row in chunks.items():
        if not isinstance(row, dict):
            continue
        content = str(row.get("content") or "")
        if _TABLE_INGEST_MARKER not in content:
            continue
        doc_id = str(row.get("full_doc_id") or "")
        fp = str(row.get("file_path") or "")
        caption, html = extract_table_from_chunk_content(
            strip_table_flat_from_content(content)
        )
        if not html:
            continue
        by_doc.setdefault(doc_id, []).append((cid, caption, html))

    store: dict[str, TableMatrixRecord] = {}
    for doc_id, items in by_doc.items():
        file_path = ""
        for row in chunks.values():
            if isinstance(row, dict) and str(row.get("full_doc_id") or "") == doc_id:
                file_path = str(row.get("file_path") or file_path)
                if file_path:
                    break
        for idx, (cid, caption, html) in enumerate(items):
            headers, rows = parse_table_html(html)
            if not headers and not rows:
                continue
            sig = html_signature(html)
            table_id = compute_mdhash_id(
                f"{doc_id}:{idx}:{sig}:{len(rows)}", prefix="table-"
            )
            store[table_id] = TableMatrixRecord(
                table_id=table_id,
                full_doc_id=doc_id,
                file_path=file_path,
                caption=caption,
                headers=headers,
                rows=rows,
                chunk_ids=[cid],
                html_signature=sig,
            )
    return store
