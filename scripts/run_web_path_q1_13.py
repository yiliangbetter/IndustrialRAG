#!/usr/bin/env python3
"""Run the legacy Q1–Q13 web-path regression suite.

The maintained implementation lives in :mod:`run_web_path_q1_17`.  This
module keeps the original command and public helpers available for existing
automation while selecting only its first thirteen cases.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_ROOT))

import run_web_path_q1_17 as _suite

_CASE_IDS = tuple(range(1, 14))

# Preserve the import surface used by ad-hoc test tooling without duplicating
# either the case data or the grading implementation.
REF = {case_id: _suite.REF[case_id] for case_id in _CASE_IDS}
grade_text = _suite.grade_text
grade_images = _suite.grade_images


def _runtime_paths() -> tuple[Path, Path]:
    working_dir = Path(
        os.getenv("RAG_WEB_WORKING_DIR") or (_ROOT / "data" / "rag_storage")
    ).resolve()
    parser_output_dir = Path(
        os.getenv("RAG_WEB_PARSER_OUTPUT_DIR") or (_ROOT / "data" / "pipeline_parse")
    ).resolve()
    return working_dir, parser_output_dir


async def run_cases(ids: list[int], *, mode: str, wd: Path, pod: Path) -> list[dict]:
    """Run cases with the pre-gate, no-dump behavior of the legacy command."""
    return await _suite.run_cases(
        ids,
        mode=mode,
        wd=wd,
        pod=pod,
        write_dumps=False,
        skip_gate=True,
        web_sim="none",
    )


def write_report(path: Path, rows: list[dict]) -> None:
    """Retain the old two-argument helper while using the shared reporter."""
    working_dir, parser_output_dir = _runtime_paths()
    _suite.write_report(
        path,
        rows,
        mode=os.getenv("RAG_QUERY_MODE", "mix"),
        wd=working_dir,
        media_root=parser_output_dir,
    )


def main() -> None:
    working_dir, parser_output_dir = _runtime_paths()
    mode = os.getenv("RAG_QUERY_MODE", "mix")
    print(f"Running Q1–Q13 web path, wd={working_dir}", flush=True)
    rows = asyncio.run(
        run_cases(
            list(_CASE_IDS),
            mode=mode,
            wd=working_dir,
            pod=parser_output_dir,
        )
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_dir = _ROOT / "logs" / "web_path_q1_13"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{stamp}.json"
    md_path = report_dir / f"{stamp}.md"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(md_path, rows)

    passed = sum(1 for row in rows if row["grade"]["ok"])
    print(f"\nReport: {md_path}", flush=True)
    print(f"Total: {passed}/{len(rows)} passed", flush=True)
    if passed < len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
