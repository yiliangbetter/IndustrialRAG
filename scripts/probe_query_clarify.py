#!/usr/bin/env python3
"""Probe query clarification gate (assess + optional rewrite), no full answer LLM."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env", override=False)

spec = importlib.util.spec_from_file_location(
    "rag_pipeline_parse_graph_chat", _ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
)
rpc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rpc)

from client_paths import get_parser_output_dir, get_rag_storage_dir  # noqa: E402
from query_clarification import (  # noqa: E402
    assess_query_fit,
    load_clarify_config,
    run_clarification_gate,
)


async def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("query", help="Question to assess")
    p.add_argument("--mode", default=os.getenv("RAG_QUERY_MODE", "mix"))
    p.add_argument("--assess-only", action="store_true", help="Only run assess_query_fit")
    p.add_argument("-w", "--working-dir", type=Path, default=None)
    p.add_argument("--parser-output-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = load_clarify_config()
    print(f"config: enabled={cfg.enabled} lower={cfg.threshold_lower} upper={cfg.threshold_upper} k={cfg.k}")

    wd = (args.working_dir or get_rag_storage_dir()).expanduser().resolve()
    pod = (args.parser_output_dir or get_parser_output_dir()).expanduser().resolve()
    rag, _, _ = await rpc._build_rag(wd, pod)

    q = args.query.strip()
    if args.assess_only:
        assess = await assess_query_fit(rag, q, mode=args.mode.strip())
        print(f"band={assess.band} score={assess.score:.4f} top1={assess.score_top1:.4f} chunks={assess.chunk_count}")
        if assess.error:
            print(f"error: {assess.error}")
        for i, snip in enumerate(assess.preview_snippets[:3], 1):
            print(f"preview[{i}]: {snip[:120]}...")
        await rag.finalize_storages()
        return

    outcome = await run_clarification_gate(rag, q, mode=args.mode.strip())
    print(f"action={outcome.action} band={outcome.band} score={outcome.original_score:.4f}")
    if outcome.message:
        print(f"message: {outcome.message}")
    if outcome.options:
        for i, o in enumerate(outcome.options, 1):
            print(f"  [{i}] score={o.score:.4f} {o.query}")
    if outcome.session_id:
        print(f"session_id={outcome.session_id}")
    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(_main())
