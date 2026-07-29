#!/usr/bin/env python3
"""Dump LightRAG retrieval context for any query (``aquery_data`` path, no LLM).

Prints metadata, chunk list with previews, and optional substring hit checks.
Examples::

  uv run python scripts/dump_query_context.py -w ./rag_storage_run \\
    "你的测试问题"

  uv run python scripts/dump_query_context.py -w ./rag_storage_run \\
    --markers "关键词1,关键词2" \\
    "你的测试问题"

If QUERY is omitted, uses env ``DUMP_QUERY_DEFAULT`` when set; otherwise the script exits with an error.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env", override=False)
if (os.getenv("HF_EMBED_OFFLINE") or "").strip().lower() in ("1", "true", "yes"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

spec = importlib.util.spec_from_file_location(
    "rag_pipeline_parse_graph_chat", _ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
)
rpc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rpc)

from lightrag import QueryParam  # noqa: E402


def _split_markers(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


async def _async_main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-w",
        "--working-dir",
        type=Path,
        required=True,
        help="LightRAG working dir (same as rag_pipeline_parse_graph_chat -w).",
    )
    p.add_argument(
        "--parser-output-dir",
        type=Path,
        default=_ROOT / "output" / "pipeline_parse",
        help="Parser output dir (same default as pipeline script).",
    )
    p.add_argument(
        "--query-mode",
        type=str,
        default=os.getenv("RAG_QUERY_MODE", "mix"),
        help="LightRAG mode: mix, naive, hybrid, local, global, ...",
    )
    p.add_argument(
        "--markers",
        type=str,
        default="",
        help=(
            "Optional comma-separated substrings to search in merged chunk text "
            "(e.g. '3.14.5,每季度,电控板'). Omit to skip marker sections."
        ),
    )
    p.add_argument(
        "--chunk-preview",
        type=int,
        default=2500,
        help="Max chars printed per chunk (default 2500).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write full report as UTF-8 (recommended on Windows). Default: docs/query_context_dump.txt.",
    )
    p.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Question text. If omitted, set DUMP_QUERY_DEFAULT in .env or export it.",
    )
    args = p.parse_args()

    query = (args.query or "").strip() or (os.getenv("DUMP_QUERY_DEFAULT") or "").strip()
    if not query:
        p.error(
            "Missing QUERY: pass it as the last argument, e.g. "
            'dump_query_context.py -w ./storage "你的问题"'
            " — or set DUMP_QUERY_DEFAULT in .env."
        )

    markers = _split_markers(args.markers)

    wd = args.working_dir.expanduser().resolve()
    pod = args.parser_output_dir.expanduser().resolve()
    pod.mkdir(parents=True, exist_ok=True)

    rag, _, _ = await rpc._build_rag(wd, pod)
    param = QueryParam(mode=args.query_mode.strip())
    data = await rag.lightrag.aquery_data(query, param)

    out_path = args.out
    if out_path is None:
        out_path = _ROOT / "docs" / "query_context_dump.txt"
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def out(s: str = "") -> None:
        lines.append(s)

    out("query: " + query)
    out("status: " + str(data.get("status")))
    out("message: " + str(data.get("message", "")))
    meta = data.get("metadata") or {}
    out("\n=== metadata ===")
    for k, v in meta.items():
        out(f"  {k}: {v}")

    inner = data.get("data") or {}
    entities = inner.get("entities") or []
    rels = inner.get("relationships") or []
    chunks = inner.get("chunks") or []
    out(f"\n=== counts: entities={len(entities)} relations={len(rels)} chunks={len(chunks)} ===")

    all_chunk_text = "\n".join((c.get("content") or "") for c in chunks)
    if markers:
        out("\n=== marker hits in merged chunk text (any chunk) ===")
        for m in markers:
            out(f"  {m!r}: {m in all_chunk_text}")
    else:
        out("\n=== marker hits (skipped; pass --markers 'a,b,c' to scan substrings) ===")

    out(f"\n=== chunk previews (query_mode={args.query_mode!r}) ===\n")
    for i, c in enumerate(chunks):
        content = c.get("content") or ""
        fp = c.get("file_path", "")
        hits = [m for m in markers if m in content] if markers else []
        marker_note = f"markers_in_chunk={hits} | " if markers else ""
        out(
            f"--- chunk {i + 1}/{len(chunks)} | file_path={fp!r} | "
            f"chunk_id={c.get('chunk_id', '')!r} | {marker_note}len={len(content)} ---"
        )
        limit = max(0, args.chunk_preview)
        if len(content) <= limit:
            out(content)
        else:
            out(content[:limit])
            out(f"... [{len(content) - limit} more chars]")
        out("")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    safe = out_path.as_posix()
    print(f"Wrote UTF-8 report to: {safe}", flush=True)

    await rag.finalize_storages()


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
