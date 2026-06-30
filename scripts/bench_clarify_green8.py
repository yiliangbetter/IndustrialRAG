#!/usr/bin/env python3
"""Benchmark clarify gate v4 on green8 (or other presets).

Times each phase separately (report/jsonl in seconds: ``gate_s``, ``answer_s``, ``total_s``):
  - ``gate_s``: original query → gate outcome (probe + candidate LLM + rerank probes)
  - ``answer_s``: picked recommendation → full ``aquery`` answer (offer/direct only)
  - ``total_s``: gate_s + answer_s (reject: gate only)

Also reports candidate ``final_score`` (rerank) and post-answer ``effective_answer`` heuristic.

Examples::

  uv run python scripts/bench_clarify_green8.py --source green8 --pick random --seed 42
  uv run python scripts/bench_clarify_green8.py --source green8 --gate-only
  uv run python scripts/bench_clarify_green8.py --source green8 --answer-all-candidates
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import random
import re
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

from raganything.clarify_gate import (  # noqa: E402
    GATE_VERSION,
    ClarifyBypass,
    ClarifyRequired,
    clarify_candidate_k,
    clarify_candidate_max_probes,
    clarify_candidate_max_rounds,
    clarify_candidate_min_rerank_score,
    clarify_candidate_skip_probe,
    clarify_direct_rerank_min,
    evaluate_clarify_gate,
    resolve_clarify_bundle,
)
from replay_clarify_gate_green8 import (  # noqa: E402
    _load_all_cases,
    _pick_candidate,
    _run_aquery,
)

_REFUSAL_RE = re.compile(
    r"(无法基于知识库|暂无法基于|知识库.*并未|文档中并未|并未提及|没有.*相关(信息|内容)|"
    r"无法找到.*依据|不足以回答|无法直接回答|未在.*记载|提供的知识库内容.*并未)",
    re.IGNORECASE,
)


def is_effective_answer(text: str) -> bool:
    """Heuristic: answer looks like a substantive KB response, not refusal/hedge."""
    ans = (text or "").strip()
    if len(ans) < 40:
        return False
    if _REFUSAL_RE.search(ans):
        return False
    if ans.startswith("抱歉") and ("无法" in ans[:120] or "不能" in ans[:120]):
        return False
    return True


def _ms_to_s(ms: int | float | None) -> float | None:
    if ms is None:
        return None
    return round(float(ms) / 1000.0, 1)


def _fmt_duration_s(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total_s = float(seconds)
    if total_s >= 60:
        minutes = int(total_s // 60)
        secs = round(total_s - minutes * 60, 1)
        return f"{minutes}min{secs:.1f}s"
    return f"{total_s:.1f}s"


def _finalize_timings(
    row: dict[str, Any], *, gate_ms: int, answer_ms: int | None
) -> None:
    row["gate_s"] = _ms_to_s(gate_ms)
    row["answer_s"] = _ms_to_s(answer_ms)
    row["total_s"] = _ms_to_s(gate_ms + (answer_ms or 0))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _answer_status(row: dict[str, Any]) -> str:
    outcome = row.get("gate_outcome")
    if outcome == "reject":
        return "未作答(reject)"
    if row.get("aquery_error"):
        return f"出错({row['aquery_error'][:40]})"
    if not row.get("has_answer"):
        return "无答案"
    if row.get("effective_answer"):
        return "LLM有效作答"
    return "LLM拒答/含糊"


def _extract_gate_timing(gate: ClarifyBypass | ClarifyRequired) -> dict[str, Any]:
    if isinstance(gate, ClarifyRequired):
        gen = gate.data.get("generation") or {}
        return dict(gen.get("gate_timing") or {})
    if isinstance(gate, ClarifyBypass) and gate.gate_timing:
        return dict(gate.gate_timing)
    return {}


def _format_gate_timing_lines(timing: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if not timing:
        return lines
    orig = timing.get("original_probe_s")
    if orig is not None:
        lines.append(f"  原问 probe: {_fmt_duration_s(float(orig))}")
    for gr in timing.get("candidate_gen_rounds") or []:
        lines.append(
            f"  推荐生成 round{gr.get('round')}: {_fmt_duration_s(gr.get('duration_s'))} "
            f"(请求{gr.get('lines_requested')}条, 返回{gr.get('lines_returned')}条)"
        )
    for pr in timing.get("candidate_probes") or []:
        passed = "过关" if pr.get("passed") else "未过"
        lines.append(
            f"  推荐 probe #{pr.get('index')} round{pr.get('round')}: "
            f"{_fmt_duration_s(pr.get('duration_s'))} final={pr.get('final_score')} {passed} | "
            f"{(pr.get('text') or '')[:50]}"
        )
    if timing.get("candidate_gen_total_s") is not None:
        lines.append(
            f"  小计 生成={_fmt_duration_s(timing.get('candidate_gen_total_s'))} "
            f"probe={_fmt_duration_s(timing.get('candidate_probe_total_s'))} "
            f"澄清循环={_fmt_duration_s(timing.get('clarify_loop_total_s'))}"
        )
    if timing.get("gate_total_s") is not None:
        lines.append(f"  门控合计(原问+澄清): {_fmt_duration_s(timing.get('gate_total_s'))}")
    return lines


def _filter_case_ids(
    cases: list[dict[str, Any]], ids_csv: str | None
) -> list[dict[str, Any]]:
    if not ids_csv:
        return cases
    want = {int(part.strip()) for part in ids_csv.split(",") if part.strip()}
    filtered = [case for case in cases if case.get("id") in want]
    missing = sorted(want - {case.get("id") for case in filtered})
    if missing:
        raise SystemExit(f"case id(s) not found in source: {missing}")
    order = {cid: idx for idx, cid in enumerate(sorted(want))}
    filtered.sort(key=lambda c: order.get(c.get("id"), 999))
    return filtered


async def _aquery_for_candidate(
    rag: Any,
    *,
    query: str,
    mode: str,
    clarification_id: str | None,
    candidate_id: str | None,
    parser_root: Path | None,
) -> tuple[int, str, str, str | None, bool]:
    bundle = resolve_clarify_bundle(
        clarification_id,
        "use_candidate",
        query,
        candidate_id,
    )
    t0 = time.perf_counter()
    _thinking, answer, err, _meta = await _run_aquery(
        rag, query, mode=mode, bundle=bundle, parser_root=parser_root
    )
    ms = int((time.perf_counter() - t0) * 1000)
    ans = (answer or "").strip()
    return ms, ans, answer or "", err, is_effective_answer(ans)


async def _bench_case(
    rag: Any,
    case: dict[str, Any],
    *,
    mode: str,
    pick_strategy: str,
    rng: random.Random,
    parser_root: Path | None,
    gate_only: bool,
    answer_all_candidates: bool,
) -> dict[str, Any]:
    orig_q = (case.get("query") or case.get("standard_question") or "").strip()
    if not orig_q:
        utterances = case.get("utterances") or []
        orig_q = (utterances[0] if utterances else "").strip()
    case_id = case.get("id", "?")

    t_gate = time.perf_counter()
    gate = await evaluate_clarify_gate(rag.lightrag, orig_q, mode=mode)
    gate_ms = int((time.perf_counter() - t_gate) * 1000)
    gate_timing = _extract_gate_timing(gate)

    orig_probe: dict[str, Any] = {}
    generation: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    if isinstance(gate, ClarifyRequired):
        orig_probe = gate.data.get("original_probe") or {}
        generation = dict(gate.data.get("generation") or {})
        candidates = [
            {
                "id": c.get("id"),
                "text": c.get("text"),
                "chunk_count": c.get("chunk_count"),
                "final_score": c.get("final_score"),
            }
            for c in (gate.data.get("candidates") or [])
        ]
    elif isinstance(gate, ClarifyBypass) and gate.probe is not None:
        orig_probe = gate.probe.as_stats()

    if isinstance(gate, ClarifyRequired):
        gate_outcome = gate.gate_outcome
        gate_reason = gate.data.get("gate_reason")
        clarification_id = gate.data.get("clarification_id")
    elif isinstance(gate, ClarifyBypass):
        gate_outcome = gate.reason
        gate_reason = gate.reason
        clarification_id = None
    else:
        gate_outcome = None
        gate_reason = None
        clarification_id = None

    row: dict[str, Any] = {
        "id": case_id,
        "category": case.get("category", ""),
        "source_set": case.get("source_set", ""),
        "original_query": orig_q,
        "original_final_score": orig_probe.get("final_score"),
        "original_chunk_count": orig_probe.get("chunk_count"),
        "gate_outcome": gate_outcome,
        "gate_reason": gate_reason,
        "gate_s": _ms_to_s(gate_ms),
        "answer_s": None,
        "total_s": _ms_to_s(gate_ms),
        "gate_timing": gate_timing,
        "generation": generation,
        "candidates": candidates,
        "pick_candidate_id": None,
        "pick_final_score": None,
        "picked_query": None,
        "candidate_answers": [],
        "has_answer": False,
        "effective_answer": False,
        "answer": "",
        "answer_preview": "",
        "aquery_error": None,
        "answer_status": "",
    }

    if gate_only or gate_outcome == "reject":
        _finalize_timings(row, gate_ms=gate_ms, answer_ms=None)
        row["answer_status"] = _answer_status(row)
        return row

    answer_ms = 0

    if isinstance(gate, ClarifyBypass) and gate.reason == "direct":
        if not gate_only:
            t0 = time.perf_counter()
            _thinking, answer, err, _meta = await _run_aquery(
                rag, orig_q, mode=mode, bundle=None, parser_root=parser_root
            )
            answer_ms = int((time.perf_counter() - t0) * 1000)
            ans = (answer or "").strip()
            row["picked_query"] = orig_q
            row["pick_kind"] = "direct"
            row["has_answer"] = bool(ans)
            row["effective_answer"] = is_effective_answer(ans)
            row["answer"] = ans
            row["answer_preview"] = ans[:240].replace("\n", " ")
            row["aquery_error"] = err
    elif gate_outcome == "offer" and isinstance(gate, ClarifyRequired):
        data = gate.data
        if answer_all_candidates:
            for cand in candidates:
                cid = cand.get("id")
                text = (cand.get("text") or "").strip()
                if not text or not cid:
                    continue
                bypass = await evaluate_clarify_gate(
                    rag.lightrag,
                    text,
                    mode=mode,
                    clarify_choice="use_candidate",
                    clarification_id=clarification_id,
                    candidate_id=cid,
                )
                if not isinstance(bypass, ClarifyBypass):
                    row["candidate_answers"].append(
                        {
                            "id": cid,
                            "final_score": cand.get("final_score"),
                            "text": text,
                            "answer_s": None,
                            "effective_answer": False,
                            "error": "bypass_validation_failed",
                        }
                    )
                    continue
                cms, ans, _raw, err, eff = await _aquery_for_candidate(
                    rag,
                    query=text,
                    mode=mode,
                    clarification_id=clarification_id,
                    candidate_id=cid,
                    parser_root=parser_root,
                )
                answer_ms += cms
                row["candidate_answers"].append(
                    {
                        "id": cid,
                        "final_score": cand.get("final_score"),
                        "text": text,
                        "answer_s": _ms_to_s(cms),
                        "effective_answer": eff,
                        "has_answer": bool(ans),
                        "answer_preview": ans[:200].replace("\n", " "),
                        "error": err,
                    }
                )
            if row["candidate_answers"]:
                pick = rng.choice(row["candidate_answers"])
                row["pick_candidate_id"] = pick.get("id")
                row["pick_final_score"] = pick.get("final_score")
                row["picked_query"] = pick.get("text")
                row["pick_kind"] = "all_candidates_sample"
                row["has_answer"] = bool(pick.get("has_answer"))
                row["effective_answer"] = bool(pick.get("effective_answer"))
                row["answer"] = pick.get("answer_preview", "")
                row["answer_preview"] = pick.get("answer_preview", "")
        else:
            pick_kind, pick_payload = _pick_candidate(
                data, strategy=pick_strategy, rng=rng
            )
            row["pick_kind"] = pick_kind
            if pick_kind == "candidate":
                final_q = (pick_payload.get("text") or "").strip()
                cid = pick_payload.get("id")
                row["picked_query"] = final_q
                row["pick_candidate_id"] = cid
                row["pick_final_score"] = pick_payload.get("final_score")
                bypass = await evaluate_clarify_gate(
                    rag.lightrag,
                    final_q,
                    mode=mode,
                    clarify_choice="use_candidate",
                    clarification_id=clarification_id,
                    candidate_id=cid,
                )
                if not isinstance(bypass, ClarifyBypass):
                    row["aquery_error"] = "bypass_validation_failed"
                else:
                    answer_ms, ans, raw, err, eff = await _aquery_for_candidate(
                        rag,
                        query=final_q,
                        mode=mode,
                        clarification_id=clarification_id,
                        candidate_id=cid,
                        parser_root=parser_root,
                    )
                    row["has_answer"] = bool(ans)
                    row["effective_answer"] = eff
                    row["answer"] = raw
                    row["answer_preview"] = ans[:240].replace("\n", " ")
                    row["aquery_error"] = err

    _finalize_timings(
        row, gate_ms=gate_ms, answer_ms=answer_ms if answer_ms else None
    )
    row["answer_status"] = _answer_status(row)
    return row


def _format_report(rows: list[dict[str, Any]], *, source: str) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"clarify gate bench v4 ({source})")
    lines.append("=" * 72)
    lines.append(f"time: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"gate_version: {GATE_VERSION}")
    lines.append(f"RERANK_MODEL: {os.getenv('RERANK_MODEL', '')}")
    lines.append(f"MIN_RERANK_SCORE: {clarify_candidate_min_rerank_score()}")
    lines.append(f"CLARIFY_DIRECT_RERANK_MIN: {clarify_direct_rerank_min()}")
    lines.append(f"CLARIFY_CANDIDATE_K: {clarify_candidate_k()}")
    max_probes = clarify_candidate_max_probes()
    lines.append(f"CLARIFY_CANDIDATE_MAX_ROUNDS: {clarify_candidate_max_rounds()}")
    lines.append(
        f"CLARIFY_CANDIDATE_MAX_PROBES: {max_probes if max_probes is not None else '(unset)'}"
    )
    lines.append(
        f"CLARIFY_CANDIDATE_SKIP_PROBE: {1 if clarify_candidate_skip_probe() else 0}"
    )
    lines.append(f"cases: {len(rows)}")
    lines.append("")

    gate_vals = [float(r["gate_s"]) for r in rows if r.get("gate_s") is not None]
    ans_vals = [float(r["answer_s"]) for r in rows if r.get("answer_s") is not None]
    tot_vals = [float(r["total_s"]) for r in rows if r.get("total_s") is not None]
    offer_gate = [
        float(r["gate_s"])
        for r in rows
        if r.get("gate_outcome") == "offer" and r.get("gate_s") is not None
    ]

    lines.append("汇总 timing:")
    if gate_vals:
        lines.append(
            f"  gate (原问→门控/推荐): n={len(gate_vals)} "
            f"min={_fmt_duration_s(min(gate_vals))} med={_fmt_duration_s(_median(gate_vals))} "
            f"max={_fmt_duration_s(max(gate_vals))} avg={_fmt_duration_s(sum(gate_vals)/len(gate_vals))}"
        )
    if offer_gate:
        lines.append(
            f"  gate (仅 offer): n={len(offer_gate)} "
            f"min={_fmt_duration_s(min(offer_gate))} med={_fmt_duration_s(_median(offer_gate))} "
            f"max={_fmt_duration_s(max(offer_gate))} avg={_fmt_duration_s(sum(offer_gate)/len(offer_gate))}"
        )
    if ans_vals:
        lines.append(
            f"  answer (点选→LLM答案): n={len(ans_vals)} "
            f"min={_fmt_duration_s(min(ans_vals))} med={_fmt_duration_s(_median(ans_vals))} "
            f"max={_fmt_duration_s(max(ans_vals))} avg={_fmt_duration_s(sum(ans_vals)/len(ans_vals))}"
        )
    if tot_vals:
        lines.append(
            f"  total (gate+answer): n={len(tot_vals)} "
            f"min={_fmt_duration_s(min(tot_vals))} med={_fmt_duration_s(_median(tot_vals))} "
            f"max={_fmt_duration_s(max(tot_vals))} avg={_fmt_duration_s(sum(tot_vals)/len(tot_vals))}"
        )
    lines.append("  (jsonl 内 gate_s/answer_s/total_s 为秒，保留 1 位小数)")
    lines.append("")

    n_direct = sum(1 for r in rows if r.get("gate_outcome") == "direct")
    n_offer = sum(1 for r in rows if r.get("gate_outcome") == "offer")
    n_reject = sum(1 for r in rows if r.get("gate_outcome") == "reject")
    n_eff = sum(1 for r in rows if r.get("effective_answer"))
    lines.append(f"outcomes: direct={n_direct} offer={n_offer} reject={n_reject}")
    lines.append(f"answers: has_answer={sum(1 for r in rows if r.get('has_answer'))} effective_answer={n_eff}")
    lines.append("")
    lines.append(
        f"{'id':>4} {'gate':>12} {'answer':>12} {'outcome':>8} {'cands':>5} {'LLM结果':<14} query"
    )
    for r in rows:
        lines.append(
            f"{r.get('id'):>4} {_fmt_duration_s(r.get('gate_s')):>12} "
            f"{_fmt_duration_s(r.get('answer_s')):>12} "
            f"{str(r.get('gate_outcome') or ''):>8} "
            f"{len(r.get('candidates') or []):>5} "
            f"{r.get('answer_status', ''):<14} "
            f"{(r.get('original_query') or '')[:24]}"
        )
    lines.append("")
    lines.append("逐题详情:")
    lines.append("-" * 72)
    for r in rows:
        gen = r.get("generation") or {}
        lines.append(
            f"#{r.get('id')} {r.get('original_query')} | "
            f"orig_final={r.get('original_final_score')} | {r.get('gate_outcome')}"
        )
        lines.append(
            f"  gate={_fmt_duration_s(r.get('gate_s'))} "
            f"answer={_fmt_duration_s(r.get('answer_s'))} "
            f"total={_fmt_duration_s(r.get('total_s'))}"
        )
        if gen:
            lines.append(
                f"  probes={gen.get('probes_used')} rounds={gen.get('rounds_used')} "
                f"reason={gen.get('reason')}"
            )
        for tl in _format_gate_timing_lines(r.get("gate_timing") or {}):
            lines.append(tl)
        for c in r.get("candidates") or []:
            lines.append(
                f"  候选 [{c.get('id')}] final={c.get('final_score')} "
                f"chunks={c.get('chunk_count')} | {c.get('text')}"
            )
        if r.get("candidate_answers"):
            lines.append("  各候选答题 (--answer-all-candidates):")
            for ca in r["candidate_answers"]:
                lines.append(
                    f"    [{ca.get('id')}] final={ca.get('final_score')} "
                    f"answer={_fmt_duration_s(ca.get('answer_s'))} "
                    f"effective={ca.get('effective_answer')} | {ca.get('text', '')[:60]}"
                )
                if ca.get("answer_preview"):
                    lines.append(f"      摘要: {ca.get('answer_preview')}")
        if r.get("picked_query") and r.get("pick_kind") not in ("all_candidates_sample",):
            lines.append(
                f"  点选: [{r.get('pick_candidate_id')}] final={r.get('pick_final_score')} | "
                f"{r.get('picked_query')}"
            )
        lines.append(f"  LLM: {r.get('answer_status')}")
        if r.get("answer_preview") and not r.get("candidate_answers"):
            lines.append(f"  摘要: {r.get('answer_preview')}")
        elif r.get("gate_outcome") == "reject":
            lines.append("  摘要: (门控拒答，未调用答题 LLM)")
        lines.append("-" * 72)
    return "\n".join(lines) + "\n"


async def _main(args: argparse.Namespace) -> None:
    os.environ.setdefault("RAG_CLARIFY_ENABLED", "1")
    wd = Path(
        os.getenv("RAG_WEB_WORKING_DIR") or (_ROOT / "data" / "rag_storage")
    ).resolve()
    pod = Path(
        os.getenv("RAG_WEB_PARSER_OUTPUT_DIR")
        or (_ROOT / "data" / "pipeline_parse")
    ).resolve()
    if not wd.is_dir():
        raise SystemExit(f"working_dir not found: {wd}")

    print(f"working_dir: {wd}", flush=True)
    rag, _, _ = await rpc._build_rag(wd, pod)
    cases, source_labels = _load_all_cases(args.source, limit=args.limit)
    cases = _filter_case_ids(cases, args.ids)
    source_label = args.source if args.source != "all" else "+".join(source_labels)
    rng = random.Random(args.seed if args.seed is not None else time.time_ns())

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = source_label.replace("+", "_")
    out_jsonl = args.out_jsonl or (_ROOT / "logs" / f"bench_clarify_{name}_{ts}.jsonl")
    out_report = args.out_report or (_ROOT / "logs" / f"bench_clarify_{name}_{ts}.txt")

    rows: list[dict[str, Any]] = []
    try:
        for i, case in enumerate(cases, 1):
            q = case.get("query") or case.get("standard_question")
            print(f"[{i}/{len(cases)}] #{case.get('id')} {q}", flush=True)
            row = await _bench_case(
                rag,
                case,
                mode=args.mode,
                pick_strategy=args.pick,
                rng=rng,
                parser_root=pod,
                gate_only=args.gate_only,
                answer_all_candidates=args.answer_all_candidates,
            )
            rows.append(row)
            print(
                f"  gate={_fmt_duration_s(row.get('gate_s'))} "
                f"answer={_fmt_duration_s(row.get('answer_s'))} "
                f"outcome={row.get('gate_outcome')} "
                f"status={row.get('answer_status')}",
                flush=True,
            )
    finally:
        await rag.finalize_storages()

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    report = _format_report(rows, source=source_label)
    out_report.write_text(report, encoding="utf-8")
    print(report)
    print(f"JSONL: {out_jsonl}")
    print(f"Report: {out_report}")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark clarify gate v4 with phase timings.")
    p.add_argument("--source", default="green8", help="green8, shili17, voice29, all")
    p.add_argument("--ids", default=None, help="Comma-separated case ids, e.g. 2,11,26")
    p.add_argument("--mode", default="mix")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--pick",
        choices=("first", "random"),
        default="random",
        help="Pick strategy when gate offers recommendations",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for --pick random")
    p.add_argument(
        "--gate-only",
        action="store_true",
        help="Only run evaluate_clarify_gate (no aquery)",
    )
    p.add_argument(
        "--answer-all-candidates",
        action="store_true",
        help="On offer, aquery every listed candidate (slow; checks each recommendation)",
    )
    p.add_argument("--out-jsonl", type=Path, default=None)
    p.add_argument("--out-report", type=Path, default=None)
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()
