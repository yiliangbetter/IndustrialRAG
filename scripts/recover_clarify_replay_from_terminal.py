#!/usr/bin/env python3
"""Recover a partial clarify-gate replay report from a crashed run's terminal log.

The replay script only writes JSONL/report at the end; if the run aborts mid-way,
gate stats and picked queries can still be parsed from stdout. LLM answers are **not**
in the terminal log and cannot be recovered unless query cache / debug dumps exist.

Example::

  uv run python scripts/recover_clarify_replay_from_terminal.py \\
    --terminal C:/Users/.../terminals/413407.txt \\
    --gate-jsonl logs/clarify_gate_voice29_replay.jsonl \\
    --out logs/clarify_gate_voice29_partial_recovery.txt
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent

_CASE_HDR = re.compile(r"^\[(\d+)/(\d+)\] (\w+)#(\d+) (.+)$")
_SUMMARY = re.compile(
    r"^\s+chunks=(\d+|None) rerank=(\S+) gate=(\S+) cands=(\d+) design=(\S+)$"
)
_AQUERY = re.compile(r"^INFO: Executing text query: (.+)\.\.\.$")


def _parse_terminal(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    cases: list[dict[str, Any]] = []
    picked_queries: list[str] = []
    current: dict[str, Any] | None = None

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        m = _CASE_HDR.match(line)
        if m:
            current = {
                "run_index": int(m.group(1)),
                "total": int(m.group(2)),
                "source_set": m.group(3),
                "id": int(m.group(4)),
                "original_query": m.group(5),
            }
            continue
        m = _SUMMARY.match(raw)
        if m and current is not None:
            chunks_raw, rerank_raw, gate, cands, design = m.groups()
            current["original_llm_chunks"] = int(chunks_raw) if chunks_raw != "None" else 0
            current["original_max_rerank"] = (
                None if rerank_raw == "None" else float(rerank_raw)
            )
            current["gate_outcome"] = gate
            current["candidate_count"] = int(cands)
            current["design_ok"] = design == "OK"
            cases.append(current)
            current = None
            continue
        m = _AQUERY.match(raw)
        if m:
            picked_queries.append(m.group(1).strip())

    return cases, picked_queries


def _load_gate_jsonl(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    by_id: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_id[int(row["id"])] = row
    return by_id


def _match_pick_id(candidates: list[dict[str, Any]], pick_text: str) -> str | None:
    pick = (pick_text or "").strip()
    for c in candidates:
        if (c.get("text") or "").strip() == pick:
            return str(c.get("id") or "")
    for c in candidates:
        text = (c.get("text") or "").strip()
        if pick in text or text in pick:
            return str(c.get("id") or "")
    return None


def _format_report(
    cases: list[dict[str, Any]],
    *,
    terminal: Path,
    gate_jsonl: Path | None,
    pick_seed: int | None,
) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("voice29 澄清门控批测 — 中断恢复报告（部分）")
    lines.append("=" * 72)
    lines.append(f"time: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"source_terminal: {terminal}")
    if gate_jsonl:
        lines.append(f"gate_candidates_ref: {gate_jsonl}")
    if pick_seed is not None:
        lines.append(f"pick_strategy: random seed={pick_seed}")
    lines.append(f"cases_recovered: {len(cases)}")
    lines.append("")
    lines.append("说明:")
    lines.append("  - 门控统计与随机点选问句来自中断前的终端日志。")
    lines.append("  - 推荐问列表来自 gate-jsonl（同日 gate-only 重跑，文本可能与中断跑略有差异）。")
    lines.append("  - LLM 答案 / References 未写入日志，无法恢复；标记为 aquery_done_no_text。")
    lines.append("")

    n_offer = 0
    n_picked = 0
    for row in cases:
        gid = int(row["id"])
        gate_row = row.get("gate_row") or {}
        candidates = list(gate_row.get("candidates") or [])
        pick_text = row.get("pick_text")
        pick_id = row.get("pick_candidate_id")

        lines.append("-" * 72)
        lines.append(
            f"#{gid} [{row.get('source_set')}/{gate_row.get('category', '')}] "
            f"{row['original_query']}"
        )
        lines.append(
            f"  gate={row.get('gate_outcome')} "
            f"qualifying={row.get('original_llm_chunks')} "
            f"max_rerank={row.get('original_max_rerank')}"
        )

        if row.get("gate_outcome") == "reject":
            lines.append(
                f"  拒答: {gate_row.get('pick_text') or gate_row.get('gate_reason') or 'no_final_chunks'}"
            )
            continue

        n_offer += 1
        if candidates:
            lines.append("  推荐问（参考 gate-jsonl）:")
            for c in candidates:
                lines.append(
                    f"    [{c.get('id')}] chunks={c.get('chunk_count')} "
                    f"max_rerank={c.get('max_rerank_score')} | {c.get('text')}"
                )
        else:
            lines.append("  推荐问: (gate-jsonl 无数据)")

        if pick_text:
            n_picked += 1
            lines.append(f"  随机选用: [{pick_id or '?'}] {pick_text}")
            lines.append("  LLM: aquery_done_no_text (答案未落盘，需账户恢复后重跑该题)")
            lines.append("  References 证据: UNKNOWN (答案未落盘)")
        else:
            lines.append("  随机选用: (无 — 本题为 reject 或中断前未点选)")

    lines.append("")
    lines.append("=" * 72)
    lines.append(
        f"汇总: 恢复 {len(cases)} 题 | offer {n_offer} | 已点选并完成 aquery(无文本) {n_picked}"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--terminal", type=Path, required=True, help="Crashed run terminal log")
    p.add_argument(
        "--gate-jsonl",
        type=Path,
        default=_ROOT / "logs" / "clarify_gate_voice29_replay.jsonl",
        help="Gate-only JSONL for candidate reference (optional)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random pick seed used in crashed run")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cases, picked = _parse_terminal(args.terminal.resolve())
    gate_by_id = _load_gate_jsonl(args.gate_jsonl if args.gate_jsonl else None)

    pick_i = 0
    for row in cases:
        gid = int(row["id"])
        gate_row = gate_by_id.get(gid, {})
        row["gate_row"] = gate_row
        if row.get("gate_outcome") != "offer":
            continue
        if pick_i >= len(picked):
            continue
        pick_text = picked[pick_i]
        pick_i += 1
        row["pick_text"] = pick_text
        row["pick_candidate_id"] = _match_pick_id(
            list(gate_row.get("candidates") or []), pick_text
        )

    out = args.out
    if out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = _ROOT / "logs" / f"clarify_gate_voice29_partial_recovery_{ts}.txt"

    report = _format_report(
        cases,
        terminal=args.terminal.resolve(),
        gate_jsonl=args.gate_jsonl if args.gate_jsonl else None,
        pick_seed=args.seed,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
