#!/usr/bin/env python3
"""Replay inline placement from query dumps (no LLM / RAG)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from iqr_figure_target import _answer_logic_lines  # noqa: E402
from image_query_refs import build_inline_placements  # noqa: E402


def _load_dump(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_images(data: dict) -> list[dict]:
    raw = data.get("images")
    if isinstance(raw, dict):
        raw = raw.get("selected") or []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _replay_one(path: Path, *, verbose: bool) -> dict:
    data = _load_dump(path)
    answer = data.get("answer") or ""
    query = data.get("query") or ""
    images = _normalize_images(data)
    imgs = [dict(item) for item in images]
    logic = _answer_logic_lines(answer, query=query)
    placements = build_inline_placements(answer, imgs, query=query)
    row = {
        "file": path.name,
        "query": query[:60],
        "logic_lines": len(logic),
        "placements": len(placements),
        "images_in": len(images),
        "images_out": len(imgs),
    }
    if verbose:
        print(f"\n=== {path.name} ===")
        print(f"query: {query}")
        print(f"logic_lines: {len(logic)}")
        for ln in logic:
            print(
                f"  [{ln.start}:{ln.end}] machine={ln.machine!r} "
                f"subject={ln.subject!r} text={ln.text[:70]!r}"
            )
        print(f"placements: {len(placements)} (images {len(images)} -> {len(imgs)})")
        for pl in placements:
            print(
                f"  img={pl['image_index']} @{pl['match_start']} "
                f"{pl['anchor_text'][:72]!r}"
            )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        default="*",
        help="Glob under logs/query_dumps (default: all)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-dump details",
    )
    args = parser.parse_args()
    dump_dir = ROOT / "logs" / "query_dumps"
    paths = sorted(dump_dir.glob(args.glob))
    if not paths:
        print(f"No dumps matching {args.glob!r} in {dump_dir}")
        return 1
    rows = [_replay_one(p, verbose=args.verbose) for p in paths]
    print(f"\nReplayed {len(rows)} dump(s) from {dump_dir}")
    for row in rows:
        print(
            f"  {row['file']}: logic={row['logic_lines']} "
            f"pl={row['placements']} imgs {row['images_in']}->{row['images_out']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
