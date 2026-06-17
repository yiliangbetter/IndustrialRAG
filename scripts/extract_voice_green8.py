#!/usr/bin/env python3
"""Extract Excel A-column green technical questions → test JSON + relevance sub-report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# Order matches Excel「技术问题」sheet green A rows (B column standard_question).
GREEN_QUESTIONS = (
    "漏胶",
    "仿形效果不好",
    "气压报警？",
    "靠板上限？",
    "三相电异常",
    "变频器异常报警",
    "未检测到工作（板材）",
    "未检测到工件",
)


def _extract_report_blocks(full_report: str, case_ids: set[str]) -> list[str]:
    parts = re.split(r"(?=--- case \d+ ---)", full_report)
    blocks: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"--- case (\d+) ---", part)
        if m and m.group(1) in case_ids:
            blocks.append(part)
    return blocks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--voice-json",
        type=Path,
        default=_ROOT / "data" / "voice_script_tests.json",
    )
    p.add_argument(
        "--jsonl",
        type=Path,
        default=_ROOT / "logs" / "relevance_voice_tech29.jsonl",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=_ROOT / "logs" / "relevance_voice_tech29_report.txt",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=_ROOT / "data" / "voice_script_green8.json",
    )
    p.add_argument(
        "--out-txt",
        type=Path,
        default=_ROOT / "docs" / "voice_script_green8_tests.txt",
    )
    p.add_argument(
        "--out-jsonl",
        type=Path,
        default=_ROOT / "logs" / "relevance_voice_green8.jsonl",
    )
    p.add_argument(
        "--out-report",
        type=Path,
        default=_ROOT / "logs" / "relevance_voice_green8_report.txt",
    )
    args = p.parse_args()

    src = json.loads(args.voice_json.read_text(encoding="utf-8"))
    by_q = {c["standard_question"]: c for c in src.get("cases") or []}
    cases: list[dict] = []
    for q in GREEN_QUESTIONS:
        if q not in by_q:
            raise SystemExit(f"Question not in {args.voice_json}: {q!r}")
        cases.append(by_q[q])

    green8 = {
        "schema": "voice_script_green8_v1",
        "source_xlsx": r"d:\dev\docs\AI智能语音话术 解析.xlsx",
        "source_sheet": "技术问题",
        "marker": "A列浅绿底色（Excel theme=9）",
        "count": len(cases),
        "cases": cases,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(green8, ensure_ascii=False, indent=2), encoding="utf-8")

    txt_lines = [
        "# Excel A列标绿 — 技术话术 8 题",
        "# 来源: d:\\dev\\docs\\AI智能语音话术 解析.xlsx / 技术问题",
        "",
    ]
    for i, c in enumerate(cases, 1):
        txt_lines.append(f"G{i}. [{c['category']}] {c['standard_question']}")
    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_txt.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    id_set = {str(c["id"]) for c in cases}
    order = [str(c["id"]) for c in cases]
    jsonl_rows: list[dict] = []
    for line in args.jsonl.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if str(row.get("case", {}).get("id")) in id_set:
            jsonl_rows.append(row)
    jsonl_rows.sort(key=lambda r: order.index(str(r["case"]["id"])))
    args.out_jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in jsonl_rows) + "\n",
        encoding="utf-8",
    )

    full_report = args.report.read_text(encoding="utf-8")
    blocks = _extract_report_blocks(full_report, id_set)
    block_by_id = {}
    for block in blocks:
        m = re.search(r"--- case (\d+) ---", block)
        if m:
            block_by_id[m.group(1)] = block

    qs = [r["scores"]["query"]["max_cosine_similarity"] for r in jsonl_rows]
    hs = [r["scores"]["high_level"]["max_cosine_similarity"] for r in jsonl_rows]
    ls = [r["scores"]["low_level"]["max_cosine_similarity"] for r in jsonl_rows]

    rp: list[str] = [
        "=" * 72,
        "Naive Relevance — Excel A列标绿 技术话术 8 题（子报告）",
        "=" * 72,
        "",
        "来源：",
        r"  Excel: d:\dev\docs\AI智能语音话术 解析.xlsx → sheet「技术问题」A列浅绿",
        f"  母报告: {args.report.as_posix()}",
        f"  母数据: {args.jsonl.as_posix()}",
        "",
        "题单（与 Excel 绿标顺序一致）：",
    ]
    for i, c in enumerate(cases, 1):
        rp.append(f"  G{i}  #{c['id']:>2} [{c['category']}] {c['standard_question']}")
    rp.extend(["", "-" * 72, "一、汇总（8 题）", "-" * 72, ""])
    rp.append(f"{'G#':<6} {'voice#':<8} {'query':<7} {'high':<7} {'low':<7} 题目")
    for i, (c, row) in enumerate(zip(cases, jsonl_rows, strict=True), 1):
        s = row["scores"]
        qtxt = c["standard_question"][:40]
        rp.append(
            f"G{i:<5} #{c['id']:<7} "
            f"{s['query']['max_cosine_similarity']:.3f}  "
            f"{s['high_level']['max_cosine_similarity']:.3f}  "
            f"{s['low_level']['max_cosine_similarity']:.3f}  {qtxt}"
        )
    rp.extend(
        [
            "",
            f"query  avg={sum(qs)/len(qs):.3f}  min={min(qs):.3f}  max={max(qs):.3f}",
            f"high   avg={sum(hs)/len(hs):.3f}  min={min(hs):.3f}  max={max(hs):.3f}",
            f"low    avg={sum(ls)/len(ls):.3f}  min={min(ls):.3f}  max={max(ls):.3f}",
            "",
            "对比：母集 29 题 query 平均约 0.669；绿标 8 题见上表 avg。",
            "绿标题为业务在 Excel 中单独标记的重点/可答项，非随机子集。",
            "",
            "-" * 72,
            "二、逐条明细（摘自母报告）",
            "-" * 72,
            "",
        ]
    )
    for i, c in enumerate(cases, 1):
        cid = str(c["id"])
        rp.append(f"[G{i} / voice #{cid}]")
        rp.append(block_by_id.get(cid, f"(missing block for case {cid})"))
        rp.append("")

    args.out_report.write_text("\n".join(rp), encoding="utf-8")

    print(f"Wrote test JSON: {args.out_json.as_posix()}")
    print(f"Wrote test list: {args.out_txt.as_posix()}")
    print(f"Wrote JSONL: {args.out_jsonl.as_posix()}")
    print(f"Wrote sub-report: {args.out_report.as_posix()}")


if __name__ == "__main__":
    main()
