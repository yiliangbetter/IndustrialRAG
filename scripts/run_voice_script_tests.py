#!/usr/bin/env python3
"""Batch-test AI 智能客服话术 against data/voice_script_tests.json."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env", override=False)

_DEFAULT_CASES = _ROOT / "data" / "voice_script_tests.json"
_REPORT_DIR = _ROOT / "logs" / "voice_script_tests"
_FALLBACK_ANY = ("人工客服", "转人工", "专线", "400-888-2033")


def _load_rpc():
    path = _SCRIPTS / "rag_pipeline_parse_graph_chat.py"
    spec = importlib.util.spec_from_file_location("rag_pipeline_parse_graph_chat", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


def _contains_any(text: str, needles: list[str]) -> list[str]:
    blob = _norm(text)
    miss = []
    for n in needles:
        n = (n or "").strip()
        if not n:
            continue
        if _norm(n) not in blob and n not in (text or ""):
            miss.append(n)
    return miss


def grade_response(answer: str, case: dict) -> tuple[bool, list[str], str]:
    """Return (ok, misses, mode_used)."""
    grade = case.get("grade") or {}
    expect = case.get("expect") or "script"
    answer = (answer or "").strip()
    if not answer:
        return False, ["（空应答）"], "empty"

    fallback_hits = [t for t in _FALLBACK_ANY if t in answer]
    if grade.get("accept_fallback") and fallback_hits:
        return True, [], "fallback"

    text_all = grade.get("text_all") or []
    text_any = grade.get("text_any") or []
    miss_all = _contains_any(answer, text_all)
    if miss_all:
        if grade.get("accept_fallback") and fallback_hits:
            return True, [], "fallback"
        return False, [f"缺：{m}" for m in miss_all], "strict"

    if text_any:
        hits = [t for t in text_any if _norm(t) in _norm(answer) or t in answer]
        min_hits = 1 if expect == "technical" else 1
        if len(hits) >= min_hits:
            return True, [], "script"
        if grade.get("accept_fallback") and fallback_hits:
            return True, [], "fallback"
        return False, [f"未命中要点：{', '.join(text_any[:3])}"], "strict"

    if grade.get("accept_fallback") and fallback_hits:
        return True, [], "fallback"
    return False, ["无评分要点"], "none"


async def run_cases(
    cases: list[dict],
    *,
    mode: str,
    wd: Path,
    query_mode: str,
    include_skip: bool,
) -> list[dict]:
    rpc = _load_rpc()
    rag, _, pod = await rpc._build_rag(wd, wd)
    parser_root = pod.resolve()
    from query_doc_steering import strip_manual_circled_step_markers
    from query_progress_hooks import query_progress_hooks, set_query_text_for_images
    from stream_cot_parser import parse_complete_cot

    rows: list[dict] = []
    try:
        for case in cases:
            if case.get("skip") and not include_skip:
                continue
            utterance = (case.get("utterances") or [case.get("standard_question")])[0]
            t0 = time.perf_counter()
            async with query_progress_hooks():
                set_query_text_for_images(utterance.strip())
                raw = await rag.aquery(
                    utterance,
                    mode=query_mode,
                    vlm_enhanced=False,
                    **rpc._query_extras_from_env(utterance),
                )
                _thinking, answer = parse_complete_cot(raw or "")
                answer = strip_manual_circled_step_markers((answer or "").strip())
            elapsed = int((time.perf_counter() - t0) * 1000)
            ok, miss, grade_mode = grade_response(answer, case)
            row = {
                "id": case["id"],
                "group": case.get("group"),
                "standard_question": case.get("standard_question"),
                "utterance": utterance,
                "expect": case.get("expect"),
                "answer": answer,
                "duration_ms": elapsed,
                "grade": {
                    "ok": ok,
                    "mode": grade_mode,
                    "miss": miss,
                },
            }
            rows.append(row)
            status = "PASS" if ok else "FAIL"
            print(
                f"VS{case['id']:02d} [{case.get('group')}] {status} "
                f"{grade_mode} {elapsed}ms | {utterance[:40]}",
                flush=True,
            )
            if miss:
                print(f"  miss: {miss}", flush=True)
    finally:
        await rag.finalize_storages()
    return rows


def grade_local(answer: str, case: dict) -> dict:
    ok, miss, grade_mode = grade_response(answer, case)
    return {"ok": ok, "mode": grade_mode, "miss": miss}


def write_report(path: Path, rows: list[dict], *, query_mode: str, wd: Path) -> None:
    passed = sum(1 for r in rows if r["grade"]["ok"])
    lines = [
        "# AI 智能客服话术批测",
        "",
        f"- 时间：{datetime.now().astimezone().isoformat()}",
        f"- RAG 模式：{query_mode}",
        f"- 工作目录：`{wd}`",
        f"- 参考答案：`docs/AI智能客服话术测试参考答案.md`",
        f"- 通过：**{passed}/{len(rows)}**",
        "",
        "## 汇总",
        "",
        "| ID | 分组 | 结果 | 评分模式 | 耗时(ms) | 测试问法 |",
        "|----|------|------|----------|----------|----------|",
    ]
    for r in rows:
        g = r["grade"]
        q = r["utterance"]
        if len(q) > 36:
            q = q[:33] + "…"
        lines.append(
            f"| VS{r['id']} | {r.get('group') or ''} | "
            f"{'✓' if g['ok'] else '✗'} | {g.get('mode')} | "
            f"{r['duration_ms']} | {q} |"
        )
    lines.append("")
    for r in rows:
        g = r["grade"]
        lines.append(f"## VS{r['id']} {'✓' if g['ok'] else '✗'}")
        lines.append("")
        lines.append(f"**标准问题**：{r['standard_question']}")
        lines.append(f"**测试问法**：{r['utterance']}")
        lines.append(f"**期望**：`{r['expect']}` · 评分 `{g.get('mode')}`")
        if g.get("miss"):
            lines.append(f"**未命中**：{', '.join(g['miss'])}")
        lines.append("")
        lines.append("**系统应答**")
        lines.append("")
        lines.append(r.get("answer") or "（空）")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Voice script test batch runner")
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES)
    parser.add_argument("--ids", type=str, default="", help="Comma IDs e.g. 1,3,5")
    parser.add_argument("--group", choices=["商务", "技术", "综合", ""], default="技术")
    parser.add_argument("--expect", choices=["script", "transfer", "technical", ""], default="")
    parser.add_argument("--include-skip", action="store_true")
    parser.add_argument("--mode", choices=["rag", "local"], default="rag")
    parser.add_argument("--response", type=str, default="", help="local mode: answer text")
    parser.add_argument("--case-id", type=int, default=0, help="local mode: case id")
    parser.add_argument("--query-mode", default=os.getenv("RAG_QUERY_MODE", "mix"))
    parser.add_argument(
        "--wd",
        type=Path,
        default=Path(os.getenv("RAG_WORKING_DIR", _ROOT / "data" / "rag_storage")),
    )
    args = parser.parse_args()

    if not args.cases.is_file():
        print(f"Cases file missing. Run: python scripts/build_voice_script_tests.py", file=sys.stderr)
        return 1

    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases: list[dict] = payload.get("cases") or []

    if args.ids:
        wanted = {int(x.strip()) for x in args.ids.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    if args.group:
        cases = [c for c in cases if c.get("group") == args.group]
    if args.expect:
        cases = [c for c in cases if c.get("expect") == args.expect]

    if args.mode == "local":
        if not args.case_id or not args.response:
            print("local mode requires --case-id and --response", file=sys.stderr)
            return 1
        case = next((c for c in cases if c["id"] == args.case_id), None)
        if not case:
            print(f"Case VS{args.case_id} not found", file=sys.stderr)
            return 1
        g = grade_local(args.response, case)
        print(json.dumps(g, ensure_ascii=False, indent=2))
        return 0 if g["ok"] else 1

    if not cases:
        print("No cases to run", file=sys.stderr)
        return 1

    wd = args.wd.resolve()
    print(f"Running {len(cases)} voice script cases, wd={wd}", flush=True)
    rows = asyncio.run(
        run_cases(
            cases,
            mode=args.mode,
            wd=wd,
            query_mode=args.query_mode,
            include_skip=args.include_skip,
        )
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.group or "all"
    report = _REPORT_DIR / f"{stamp}_{tag}.md"
    write_report(report, rows, query_mode=args.query_mode, wd=wd)
    passed = sum(1 for r in rows if r["grade"]["ok"])
    print(f"\nReport: {report}")
    print(f"Total: {passed}/{len(rows)} passed")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
