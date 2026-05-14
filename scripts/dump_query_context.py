#!/usr/bin/env python3
"""Print structured retrieval (entities / relations / chunks) for a query — same path as aquery, no LLM.

Uses LightRAG ``aquery_data`` (see lightrag docs). Example:

  uv run python scripts/dump_query_context.py -w ./rag_storage_run
  uv run python scripts/dump_query_context.py -w D:/data/rag_storage_run --query-mode mix \\
    "高速智能封边机的保养中，哪些部件需要使用美孚长效液压油？"
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
        "query",
        nargs="?",
        default="高速智能封边机的保养中，哪些部件需要使用美孚长效液压油？",
        help="Question text (default: Mobil oil parts question).",
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
        help="Write full report as UTF-8 to this file (recommended on Windows). Default: docs/query_context_dump.txt under repo root.",
    )
    args = p.parse_args()
    wd = args.working_dir.expanduser().resolve()
    pod = args.parser_output_dir.expanduser().resolve()
    pod.mkdir(parents=True, exist_ok=True)

    rag, _, _ = await rpc._build_rag(wd, pod)
    param = QueryParam(mode=args.query_mode.strip())
    data = await rag.lightrag.aquery_data(args.query, param)

    out_path = args.out
    if out_path is None:
        out_path = _ROOT / "docs" / "query_context_dump.txt"
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def out(s: str = "") -> None:
        lines.append(s)

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

    markers = [
        "美孚长效液压油",
        "自动注油泵",
        "导轨",
        "滑块",
        "集中润滑",
        "润滑注油",
        "注油泵",
        "辅助进料",
        "预铣",
        "平切",
        "精修",
        "仿形",
        "开槽",
        "刮边",
    ]
    all_chunk_text = "\n".join((c.get("content") or "") for c in chunks)
    out("\n=== marker hits in merged chunk text (any chunk) ===")
    for m in markers:
        out(f"  {m!r}: {m in all_chunk_text}")

    out(f"\n=== chunk previews (query_mode={args.query_mode!r}) ===\n")
    for i, c in enumerate(chunks):
        content = c.get("content") or ""
        fp = c.get("file_path", "")
        hits = [m for m in markers if m in content]
        out(
            f"--- chunk {i + 1}/{len(chunks)} | file_path={fp!r} | "
            f"chunk_id={c.get('chunk_id', '')!r} | markers_in_chunk={hits} | len={len(content)} ---"
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
