#!/usr/bin/env python3
"""Build ``kv_store_table_matrix.json`` from existing ``kv_store_text_chunks.json``.

No flatten required; parses ``[Table]`` HTML in each chunk.

Examples::

  uv run python scripts/build_table_matrix_from_storage.py -w data/rag_storage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from raganything.table_matrix import (  # noqa: E402
    build_matrix_from_text_chunks,
    save_matrix_store,
    table_matrix_ingest_enabled,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-w",
        "--working-dir",
        type=Path,
        default=_ROOT / "data" / "rag_storage",
    )
    args = p.parse_args()

    if not table_matrix_ingest_enabled():
        print("RAG_TABLE_MATRIX_INGEST is off.")
        sys.exit(1)

    wd = args.working_dir.expanduser().resolve()
    chunk_path = wd / "kv_store_text_chunks.json"
    if not chunk_path.is_file():
        print(f"Missing {chunk_path}")
        sys.exit(1)

    chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
    if not isinstance(chunks, dict):
        print("Unexpected kv_store_text_chunks.json")
        sys.exit(1)

    store = build_matrix_from_text_chunks(chunks)
    save_matrix_store(wd, store)
    linked = sum(len(r.chunk_ids) for r in store.values())
    print(f"Wrote {len(store)} table matrix record(s), {linked} chunk link(s) → {wd / 'kv_store_table_matrix.json'}")


if __name__ == "__main__":
    main()
