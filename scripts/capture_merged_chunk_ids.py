#!/usr/bin/env python3
"""Monkeypatch LightRAG ``_merge_all_chunks`` to write merged chunk_ids (pre-rerank) to a file.

Run after ``load_dotenv`` and before ``aquery_data`` so the patch applies.

Writes: docs/merged_pre_rerank_chunk_ids.txt (one id per line).
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


def _install_merge_hook(out: Path) -> None:
    import lightrag.operate as op

    _orig = op._merge_all_chunks

    async def _wrapped(*args, **kwargs):
        merged = await _orig(*args, **kwargs)
        lines = []
        for c in merged:
            cid = c.get("chunk_id") or c.get("id") or ""
            lines.append(str(cid))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out.with_suffix(".meta.txt").write_text(
            f"count={len(merged)}\n", encoding="utf-8"
        )
        return merged

    op._merge_all_chunks = _wrapped  # type: ignore[method-assign]


async def _async_main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-w", "--working-dir", type=Path, required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "docs" / "merged_pre_rerank_chunk_ids.txt",
    )
    p.add_argument("query", nargs="?", default=None)
    args = p.parse_args()

    query = (args.query or os.getenv("DUMP_QUERY_DEFAULT") or "").strip()
    if not query:
        p.error("QUERY is required (or set env DUMP_QUERY_DEFAULT)")

    _install_merge_hook(args.out.expanduser().resolve())

    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_parse_graph_chat", _ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
    )
    rpc = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(rpc)

    from lightrag import QueryParam  # noqa: E402

    wd = args.working_dir.expanduser().resolve()
    pod = (_ROOT / "output" / "pipeline_parse").resolve()
    pod.mkdir(parents=True, exist_ok=True)

    rag, _, _ = await rpc._build_rag(wd, pod)
    await rag.lightrag.aquery_data(query, QueryParam(mode="mix"))
    await rag.finalize_storages()

    print(f"Wrote merged pre-rerank chunk ids to: {args.out.as_posix()}", flush=True)


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
