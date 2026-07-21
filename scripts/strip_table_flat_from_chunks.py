#!/usr/bin/env python3
"""Strip legacy ``[TableFlat]`` blocks from indexed chunks and re-embed.

Examples::

  uv run python scripts/strip_table_flat_from_chunks.py -w data/rag_storage
  uv run python scripts/strip_table_flat_from_chunks.py -w data/rag_storage --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env", override=False)

from raganything.table_matrix import strip_table_flat_from_content  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "rag_pipeline_parse_graph_chat",
    _ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py",
)
if spec is None or spec.loader is None:
    raise SystemExit("Failed to load scripts/rag_pipeline_parse_graph_chat.py")
rpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rpc)


async def _async_main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-w",
        "--working-dir",
        type=Path,
        default=_ROOT / "data" / "rag_storage",
    )
    p.add_argument(
        "--parser-output-dir",
        type=Path,
        default=_ROOT / "data" / "pipeline_parse",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    wd = args.working_dir.expanduser().resolve()
    pod = args.parser_output_dir.expanduser().resolve()
    pod.mkdir(parents=True, exist_ok=True)

    path = wd / "kv_store_text_chunks.json"
    if not path.is_file():
        print(f"Missing {path}")
        sys.exit(1)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"Failed to read {path}: {exc}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Corrupt or incomplete JSON in {path}: {exc}")
        sys.exit(1)
    if not isinstance(raw, dict):
        print(f"Unexpected payload in {path}: expected a JSON object")
        sys.exit(1)

    updates: dict[str, dict] = {}
    for cid, item in raw.items():
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "")
        if "[TableFlat]" not in content:
            continue
        cleaned = strip_table_flat_from_content(content)
        if cleaned == content:
            continue
        row = dict(item)
        row["content"] = cleaned
        updates[cid] = row

    print(f"Chunks to clean: {len(updates)}")
    if not updates or args.dry_run:
        if args.dry_run and updates:
            for cid in list(updates)[:5]:
                print(f"  - {cid}")
        return

    rag, _, _ = await rpc._build_rag(wd, pod)
    await rag.lightrag.text_chunks.upsert(updates)
    await rag.lightrag.chunks_vdb.upsert(updates)
    await rag.lightrag.text_chunks.index_done_callback()
    await rag.lightrag.chunks_vdb.index_done_callback()
    await rag.finalize_storages()
    print(f"Cleaned and re-embedded {len(updates)} chunk(s).")


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
