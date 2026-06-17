#!/usr/bin/env python3
"""Replay question sets through clarify gate: random pick if clarified, then aquery.

Reports per-question scores, candidate scores, pick, and whether an answer was produced.

Examples::

  uv run python scripts/replay_clarify_gate_green8.py --source green8
  uv run python scripts/replay_clarify_gate_green8.py --source shili17
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

load_dotenv(_ROOT / ".env", override=False)

spec = importlib.util.spec_from_file_location(
    "rag_pipeline_parse_graph_chat", _ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
)
rpc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rpc)

from score_query_relevance import _load_cases  # noqa: E402
from stream_cot_parser import parse_complete_cot  # noqa: E402
from raganything.clarify_gate import (  # noqa: E402
    ClarifyBypass,
    ClarifyRequired,
    classify_query_relevance,
    clarify_threshold_lower,
    clarify_threshold_upper,
    evaluate_clarify_gate,
    probe_query_score,
)


def _fscore(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _pick_clarify_option(data: dict[str, Any], rng: random.Random) -> tuple[str, dict[str, Any]]:
    """Return (kind, payload) where kind is candidate|keep_original."""
    candidates = list(data.get("candidates") or [])
    keep = data.get("keep_original")
    options: list[tuple[str, dict[str, Any]]] = [("candidate", c) for c in candidates]
    if isinstance(keep, dict) and (keep.get("text") or "").strip():
        options.append(("keep_original", keep))
    if not options:
        return "none", {}
    kind, payload = rng.choice(options)
    return kind, payload


async def _run_aquery(rag: Any, query: str, *, mode: str) -> tuple[str, str, str | None]:
    started = time.perf_counter()
    error: str | None = None
    raw = ""
    try:
        raw = await rag.aquery(
            query,
            mode=mode,
            vlm_enhanced=False,
            **rpc._query_extras_from_env(query),
        )
        if not isinstance(raw, str):
            parts: list[str] = []
            async for chunk in raw:
                if chunk:
                    parts.append(chunk if isinstance(chunk, str) else str(chunk))
            raw = "".join(parts)
    except Exception as exc:
        error = str(exc)
    thinking, answer = parse_complete_cot(raw or "")
    ms = int((time.perf_counter() - started) * 1000)
    return thinking, answer, error if error else (None if not (answer or "").strip() else None)


async def _replay_case(
    rag: Any,
    case: dict[str, str],
    *,
    mode: str,
    lo: float,
    hi: float,
    rng: random.Random,
) -> dict[str, Any]:
    orig_q = case["query"]
    case_id = case.get("id", "?")

    orig_probe = await probe_query_score(rag.lightrag, orig_q)
    orig_score = _fscore(orig_probe.get("max_cosine_similarity"))
    band = classify_query_relevance(orig_score, lower=lo, upper=hi).value

    gate = await evaluate_clarify_gate(rag.lightrag, orig_q, mode=mode)
    clarify_triggered = isinstance(gate, ClarifyRequired)
    row: dict[str, Any] = {
        "id": case_id,
        "category": case.get("category", ""),
        "original_query": orig_q,
        "original_score": round(orig_score, 4) if orig_score is not None else None,
        "band": band,
        "clarify_triggered": clarify_triggered,
        "candidates": [],
        "pick_kind": None,
        "pick_text": None,
        "pick_score": None,
        "final_query": orig_q,
        "final_score": None,
        "has_answer": False,
        "answer_chars": 0,
        "aquery_error": None,
        "duration_ms": None,
    }

    if clarify_triggered:
        data = gate.data
        row["clarification_id"] = data.get("clarification_id")
        row["unanswerable"] = bool(data.get("unanswerable"))
        row["candidates"] = [
            {
                "id": c.get("id"),
                "text": c.get("text"),
                "chunk_count": c.get("chunk_count"),
                "max_rerank_score": c.get("max_rerank_score"),
                "score": c.get("score"),
            }
            for c in (data.get("candidates") or [])
        ]
        if row["unanswerable"]:
            row["pick_kind"] = "unanswerable"
            row["pick_text"] = data.get("message")
            return row

        pick_kind, pick_payload = _pick_clarify_option(data, rng)
        row["pick_kind"] = pick_kind

        if pick_kind == "candidate":
            final_q = (pick_payload.get("text") or "").strip()
            row["pick_text"] = final_q
            row["pick_score"] = pick_payload.get("score")
            bypass = await evaluate_clarify_gate(
                rag.lightrag,
                final_q,
                mode=mode,
                clarify_choice="use_candidate",
                clarification_id=data.get("clarification_id"),
                candidate_id=pick_payload.get("id"),
            )
        elif pick_kind == "keep_original":
            final_q = (pick_payload.get("text") or orig_q).strip()
            row["pick_text"] = final_q
            row["pick_score"] = pick_payload.get("relevance_score")
            bypass = await evaluate_clarify_gate(
                rag.lightrag,
                final_q,
                mode=mode,
                clarify_choice="keep_original",
            )
        else:
            final_q = orig_q
            bypass = ClarifyBypass("pass")
        row["final_query"] = final_q
        if not isinstance(bypass, ClarifyBypass):
            row["aquery_error"] = "bypass_validation_failed"
            return row
    else:
        row["gate_bypass"] = gate.reason if isinstance(gate, ClarifyBypass) else None
        final_q = orig_q

    final_probe = await probe_query_score(rag.lightrag, final_q)
    final_score = _fscore(final_probe.get("max_cosine_similarity"))
    row["final_score"] = round(final_score, 4) if final_score is not None else None

    started = time.perf_counter()
    _thinking, answer, err = await _run_aquery(rag, final_q, mode=mode)
    row["duration_ms"] = int((time.perf_counter() - started) * 1000)
    row["aquery_error"] = err
    ans = (answer or "").strip()
    row["has_answer"] = bool(ans)
    row["answer_chars"] = len(ans)
    return row


_SOURCE_PRESETS: dict[str, Path] = {
    "green8": _ROOT / "data" / "voice_script_green8.json",
    "shili17": _ROOT / "docs" / "测试例.txt",
}


def _format_report(
    rows: list[dict[str, Any]], *, source: str, lo: float, hi: float, seed: int
) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"{source} 澄清门控回放批测")
    lines.append("=" * 72)
    lines.append(f"time: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"thresholds: lower={lo} upper={hi}")
    lines.append(f"random_seed: {seed}")
    lines.append(f"cases: {len(rows)}")
    lines.append("")

    for r in rows:
        lines.append("-" * 72)
        lines.append(
            f"#{r['id']} [{r.get('category','')}] {r['original_query']}"
        )
        lines.append(
            f"  原问 score={r['original_score']} band={r['band']} "
            f"clarify={'YES' if r['clarify_triggered'] else 'NO'}"
        )
        if r["clarify_triggered"]:
            if r.get("unanswerable"):
                lines.append(f"  无法回答: {r.get('pick_text') or ''}")
            else:
                lines.append("  推荐问法:")
                for c in r.get("candidates") or []:
                    rs = c.get("max_rerank_score")
                    cc = c.get("chunk_count")
                    meta = f"chunks={cc} rerank={rs}" if rs is not None else f"score={c.get('score')}"
                    lines.append(f"    - [{c.get('id')}] {meta} | {c.get('text')}")
                lines.append(
                    f"  随机选择: {r['pick_kind']} | {r['pick_text']} "
                    f"(pick_score={r['pick_score']})"
                )
        else:
            lines.append(f"  gate_bypass={r.get('gate_bypass')}")
        lines.append(
            f"  最终问: {r['final_query']}"
        )
        lines.append(
            f"  最终 score={r['final_score']} | has_answer={r['has_answer']} "
            f"answer_chars={r['answer_chars']} duration_ms={r['duration_ms']}"
        )
        if r.get("aquery_error"):
            lines.append(f"  error: {r['aquery_error']}")
        lines.append("")

    n_clarify = sum(1 for r in rows if r["clarify_triggered"])
    n_answer = sum(1 for r in rows if r["has_answer"])
    lines.append("=" * 72)
    lines.append(f"汇总: clarify触发 {n_clarify}/{len(rows)} | 出答案 {n_answer}/{len(rows)}")
    lines.append("")
    lines.append("id   orig_q   final_q   clarify  has_ans")
    for r in rows:
        lines.append(
            f"{str(r['id']):>3}  "
            f"{r['original_score']!s:>7}  "
            f"{r['final_score']!s:>7}  "
            f"{'Y' if r['clarify_triggered'] else 'N':>7}  "
            f"{'Y' if r['has_answer'] else 'N':>7}  "
            f"{r['original_query']}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


async def _main(args: argparse.Namespace) -> None:
    os.environ.setdefault("RAG_CLARIFY_ENABLED", "1")
    rng = random.Random(args.seed)
    lo, hi = clarify_threshold_lower(), clarify_threshold_upper()
    wd = (_ROOT / "data" / "rag_storage").resolve()
    pod = (_ROOT / "data" / "pipeline_parse").resolve()
    rag, _, _ = await rpc._build_rag(wd, pod)
    src = _SOURCE_PRESETS.get(args.source, Path(args.source))
    cases = _load_cases(src.resolve(), limit=args.limit)

    rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.get('id')} {case['query']}", flush=True)
        row = await _replay_case(
            rag, case, mode=args.mode, lo=lo, hi=hi, rng=rng
        )
        rows.append(row)
        print(
            f"  orig={row['original_score']} clarify={row['clarify_triggered']} "
            f"final={row['final_score']} answer={row['has_answer']}",
            flush=True,
        )

    report = _format_report(rows, source=args.source, lo=lo, hi=hi, seed=args.seed)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.out_report.write_text(report, encoding="utf-8")
    print(report)
    print(f"JSONL: {args.out_jsonl}")
    print(f"Report: {args.out_report}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        default="green8",
        help="Preset (green8, shili17) or path to question file",
    )
    p.add_argument("--mode", default="mix")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42, help="Random seed for clarify pick")
    p.add_argument("--out-jsonl", type=Path, default=None)
    p.add_argument("--out-report", type=Path, default=None)
    args = p.parse_args()
    if args.out_jsonl is None:
        name = "green8" if args.source == "green8" else args.source.replace("/", "_")
        args.out_jsonl = _ROOT / "logs" / f"clarify_gate_{name}_replay.jsonl"
    if args.out_report is None:
        name = "green8" if args.source == "green8" else args.source.replace("/", "_")
        args.out_report = _ROOT / "logs" / f"clarify_gate_{name}_replay_report.txt"
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
