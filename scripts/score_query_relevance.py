#!/usr/bin/env python3
"""Score question / keyword relevance via naive chunk-vector retrieval.

Examples::

  uv run python scripts/score_query_relevance.py -w data/rag_storage \\
    "高速智能封边机的涂胶轴怎么保养"

  uv run python scripts/score_query_relevance.py -w data/rag_storage \\
    --source docs/测试例.txt --limit 17

  uv run python scripts/score_query_relevance.py -w data/rag_storage \\
    --source data/voice_script_tests.json --out logs/relevance_voice.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

_SOURCE_PRESETS: dict[str, Path] = {
    "shili": _ROOT / "docs" / "测试例.txt",
    "shili17": _ROOT / "docs" / "测试例.txt",
    "voice": _ROOT / "data" / "voice_script_tests.json",
    "voice29": _ROOT / "data" / "voice_script_tests.json",
    "voice_green8": _ROOT / "data" / "voice_script_green8.json",
    "green8": _ROOT / "data" / "voice_script_green8.json",
}

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

from raganything.naive_relevance import format_relevance_report  # noqa: E402


def _parse_shili_questions(path: Path, *, limit: int = 0) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("—"):
            break
        m = re.match(r"^(\d+)\.(.+)$", line)
        if not m:
            continue
        num, rest = m.group(1), m.group(2).strip()
        if not rest or rest.startswith("是否配图"):
            continue
        rows.append({"id": num, "query": rest})
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def _parse_voice_script_cases(
    path: Path, *, limit: int = 0, group: str | None = None
) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    group_filter = (group or "").strip()
    rows: list[dict[str, str]] = []
    for case in data.get("cases") or []:
        if case.get("skip"):
            continue
        case_group = str(case.get("group") or "").strip()
        if group_filter and case_group != group_filter:
            continue
        q = (case.get("standard_question") or "").strip()
        if not q:
            utterances = case.get("utterances") or []
            q = (utterances[0] if utterances else "").strip()
        if not q:
            continue
        rows.append(
            {
                "id": str(case.get("id", len(rows) + 1)),
                "query": q,
                "group": str(case.get("group") or ""),
                "category": str(case.get("category") or ""),
            }
        )
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def _load_cases(source: Path, *, limit: int, group: str | None = None) -> list[dict[str, str]]:
    if source.suffix.lower() == ".json":
        return _parse_voice_script_cases(source, limit=limit, group=group)
    return _parse_shili_questions(source, limit=limit)


async def _score_one(rag: Any, query: str) -> dict:
    payload = await rag.score_naive_relevance(query)
    payload["scored_at"] = datetime.now(timezone.utc).isoformat()
    return payload


async def _async_main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-w", "--working-dir", type=Path, required=True)
    p.add_argument(
        "--parser-output-dir",
        type=Path,
        default=None,
        help="Parser output dir (default: RAG_WEB_PARSER_OUTPUT_DIR or data/pipeline_parse).",
    )
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="Batch source path, or preset: shili / shili17 / voice / voice29",
    )
    p.add_argument("--limit", type=int, default=0, help="Max questions from --source (0=all).")
    p.add_argument(
        "--group",
        type=str,
        default="",
        help="For voice_script_tests.json: only cases with this group (e.g. 技术).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSONL batch results (one score object per line).",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write human-readable UTF-8 report for batch or single query.",
    )
    p.add_argument("query", nargs="?", default=None, help="Single question text.")
    args = p.parse_args()

    wd = args.working_dir.expanduser().resolve()
    pod = args.parser_output_dir
    if pod is None:
        raw = (os.getenv("RAG_WEB_PARSER_OUTPUT_DIR") or "data/pipeline_parse").strip()
        pod = (_ROOT / raw).resolve()
    else:
        pod = pod.expanduser().resolve()

    rag, _, _ = await rpc._build_rag(wd, pod)

    if args.source:
        preset = _SOURCE_PRESETS.get(args.source.strip().lower())
        source_path = preset if preset is not None else Path(args.source).expanduser()
        cases = _load_cases(
            source_path.resolve(),
            limit=args.limit,
            group=(args.group or "").strip() or None,
        )
        if not cases:
            raise SystemExit(f"No questions parsed from {args.source}")
        results: list[dict] = []
        report_parts: list[str] = []
        for idx, case in enumerate(cases, start=1):
            q = case["query"]
            print(f"[{idx}/{len(cases)}] {q}", flush=True)
            payload = await _score_one(rag, q)
            payload["case"] = case
            results.append(payload)
            q_score = (payload.get("scores") or {}).get("query", {}).get("max_cosine_similarity")
            h_score = (payload.get("scores") or {}).get("high_level", {}).get("max_cosine_similarity")
            l_score = (payload.get("scores") or {}).get("low_level", {}).get("max_cosine_similarity")
            print(
                f"  query={q_score} high_level={h_score} low_level={l_score}",
                flush=True,
            )
            report_parts.append(f"--- case {case.get('id')} ---\n{format_relevance_report(payload)}")

        if args.out:
            out_path = args.out.expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as fh:
                for row in results:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"Wrote JSONL: {out_path.as_posix()}", flush=True)

        report_path = args.report
        if report_path is None and not args.out:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = _ROOT / "logs" / f"relevance_batch_{stamp}.txt"
        if report_path:
            report_path = report_path.expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("\n".join(report_parts), encoding="utf-8")
            print(f"Wrote report: {report_path.as_posix()}", flush=True)
    else:
        query = (args.query or "").strip() or (os.getenv("DUMP_QUERY_DEFAULT") or "").strip()
        if not query:
            p.error("Provide QUERY or --source for batch scoring.")
        payload = await _score_one(rag, query)
        text = format_relevance_report(payload)
        print(text, flush=True)
        if args.out:
            path = args.out.expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote JSON: {path.as_posix()}", flush=True)
        if args.report:
            path = args.report.expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    await rag.finalize_storages()


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
