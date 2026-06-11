#!/usr/bin/env python3
"""Run questions from docs/测试例.txt and write an LLM answer report (no image/text grading)."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env", override=False)

_DEFAULT_CASES = _ROOT / "docs" / "测试例.txt"
_REPORT_DIR = _ROOT / "logs" / "test_cases_report"


def _load_rpc():
    spec = importlib.util.spec_from_file_location(
        "rpc", _SCRIPTS / "rag_pipeline_parse_graph_chat.py"
    )
    rpc = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(rpc)
    return rpc


def parse_cases(path: Path, *, max_id: int = 15) -> list[dict[str, object]]:
    """Parse numbered questions; skip 是否配图 / 文字答案 lines."""
    cases: list[dict[str, object]] = []
    current_id: int | None = None
    current_q: str | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\.\s*(.+)$", line)
        if m:
            if current_id is not None and current_q and current_id <= max_id:
                cases.append({"id": current_id, "query": current_q.strip()})
            current_id = int(m.group(1))
            current_q = m.group(2).strip()
            if current_id > max_id:
                break
            continue
        if line.startswith("是否配图") or line.startswith("文字答案"):
            continue

    if current_id is not None and current_q and current_id <= max_id:
        cases.append({"id": current_id, "query": current_q.strip()})
    return cases


def _resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    wd = args.working_dir or Path(
        os.getenv("RAG_WEB_WORKING_DIR") or (_ROOT / "data" / "rag_storage")
    )
    pod = args.parser_output_dir or Path(
        os.getenv("RAG_WEB_PARSER_OUTPUT_DIR") or (_ROOT / "data" / "pipeline_parse")
    )
    return wd.resolve(), pod.resolve()


async def _run_cases(
    cases: list[dict[str, object]],
    *,
    mode: str,
    wd: Path,
    pod: Path,
) -> list[dict[str, object]]:
    from query_progress_hooks import query_progress_hooks
    from stream_cot_parser import parse_complete_cot

    rpc = _load_rpc()
    rag, _, _ = await rpc._build_rag(wd, pod)
    results: list[dict[str, object]] = []

    try:
        for item in cases:
            case_id = int(item["id"])
            query = str(item["query"])
            started = time.perf_counter()
            err: str | None = None
            raw = ""
            thinking = ""
            answer = ""
            try:
                async with query_progress_hooks():
                    raw = await rag.aquery(
                        query,
                        mode=mode,
                        vlm_enhanced=False,
                        **rpc._query_extras_from_env(query),
                    )
                thinking, answer = parse_complete_cot(raw or "")
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            results.append(
                {
                    "id": case_id,
                    "query": query,
                    "mode": mode,
                    "duration_ms": elapsed_ms,
                    "error": err,
                    "answer": answer.strip(),
                    "thinking_preview": (thinking.strip()[:500] + "…")
                    if len(thinking.strip()) > 500
                    else thinking.strip(),
                    "answer_chars": len(answer.strip()),
                }
            )
            status = "ERR" if err else "OK"
            preview = (answer.strip() or err or "")[:80].replace("\n", " ")
            print(f"  [{case_id:02d}] {status} {elapsed_ms}ms | {preview}")
    finally:
        await rag.finalize_storages()

    return results


def _write_markdown(path: Path, meta: dict[str, object], results: list[dict]) -> None:
    lines = [
        "# 测试例 LLM 回答报告",
        "",
        f"- 生成时间：{meta['generated_at']}",
        f"- 用例文件：`{meta['cases_file']}`",
        f"- 查询模式：`{meta['mode']}`",
        f"- 工作目录：`{meta['working_dir']}`",
        f"- 题目数：{len(results)}",
        "",
    ]
    for row in results:
        lines.append(f"## Q{row['id']} {row['query']}")
        lines.append("")
        if row.get("error"):
            lines.append(f"**错误**：{row['error']}")
            lines.append("")
            continue
        lines.append(f"*耗时 {row['duration_ms']} ms · 答案 {row['answer_chars']} 字*")
        lines.append("")
        lines.append(row.get("answer") or "（空）")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cases-file",
        type=Path,
        default=_DEFAULT_CASES,
        help="Question bank file (default: docs/测试例.txt)",
    )
    p.add_argument("--max-id", type=int, default=15, help="Last question id to run")
    p.add_argument("--ids", type=str, default="", help="Comma-separated ids, e.g. 1,3,11")
    p.add_argument("-w", "--working-dir", type=Path, default=None)
    p.add_argument("--parser-output-dir", type=Path, default=None)
    p.add_argument("--mode", default=os.getenv("RAG_QUERY_MODE", "mix"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPORT_DIR,
        help="Report output directory",
    )
    args = p.parse_args()

    cases_path = args.cases_file.resolve()
    if not cases_path.is_file():
        raise SystemExit(f"Cases file not found: {cases_path}")

    cases = parse_cases(cases_path, max_id=args.max_id)
    if args.ids.strip():
        wanted = {int(x.strip()) for x in args.ids.split(",") if x.strip()}
        cases = [c for c in cases if int(c["id"]) in wanted]
    if not cases:
        raise SystemExit("No test cases to run.")

    wd, pod = _resolve_dirs(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stamp}.json"
    md_path = out_dir / f"{stamp}.md"

    print(f"Running {len(cases)} case(s) mode={args.mode}")
    print(f"  working_dir={wd}")
    results = asyncio.run(
        _run_cases(cases, mode=args.mode.strip(), wd=wd, pod=pod)
    )

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_file": str(cases_path),
        "mode": args.mode.strip(),
        "working_dir": str(wd),
        "parser_output_dir": str(pod),
        "case_count": len(results),
        "ok_count": sum(1 for r in results if not r.get("error")),
        "error_count": sum(1 for r in results if r.get("error")),
    }
    payload = {"meta": meta, "results": results}
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(md_path, meta, results)
    print(f"\nWrote JSON: {json_path}")
    print(f"Wrote Markdown: {md_path}")


if __name__ == "__main__":
    main()
