#!/usr/bin/env python3
"""Query an existing local Ollama + LightRAG index (built by ingest_documents_local_ollama.py)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import importlib.util

_ingest_path = Path(__file__).resolve().parent / "ingest_documents_local_ollama.py"
_spec = importlib.util.spec_from_file_location(
    "ingest_documents_local_ollama", _ingest_path
)
_ingest = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ingest)

DEFAULT_WORKING = _ingest.DEFAULT_WORKING
_build_rag = _ingest._build_rag
_interactive_loop = _ingest._interactive_loop


async def async_main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-w",
        "--working-dir",
        type=Path,
        default=DEFAULT_WORKING,
        help=f"LightRAG storage directory (default: {DEFAULT_WORKING})",
    )
    p.add_argument(
        "--query-mode",
        default="mix",
        help="LightRAG query mode: local, global, hybrid, mix, …",
    )
    p.add_argument("--query", type=str, default="", help="Single question (non-interactive)")
    args = p.parse_args()

    if not args.working_dir.is_dir():
        raise SystemExit(f"Storage not found: {args.working_dir}. Run ingest first.")

    rag = await _build_rag(args.working_dir, args.working_dir / "_noop_parser")

    if args.query.strip():
        ans = await rag.aquery(args.query.strip(), mode=args.query_mode, vlm_enhanced=False)
        print(ans or "", flush=True)
        return

    await _interactive_loop(rag, args.query_mode)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
