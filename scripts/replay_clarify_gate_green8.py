#!/usr/bin/env python3
"""Replay question sets through clarify gate: pick first recommended candidate, then aquery.

Reports per-question scores, candidate scores, pick, and whether an answer was produced.

Examples::

  uv run python scripts/replay_clarify_gate_green8.py --source green8
  uv run python scripts/replay_clarify_gate_green8.py --source shili17
  uv run python scripts/replay_clarify_gate_green8.py --source voice29
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
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
from run_voice_script_tests import grade_response  # noqa: E402
from raganything.clarify_gate import (  # noqa: E402
    ClarifyBypass,
    ClarifyRequired,
    clarify_candidate_k,
    clarify_candidate_min_rerank_score,
    evaluate_clarify_gate,
    probe_llm_retrieval,
    resolve_clarify_bundle,
)


def _fscore(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _pick_first_candidate(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pick the first answerable recommended question (no random, no keep_original)."""
    candidates = list(data.get("candidates") or [])
    if candidates:
        return "candidate", candidates[0]
    return "none", {}


async def _run_aquery(
    rag: Any,
    query: str,
    *,
    mode: str,
    bundle: Any | None = None,
) -> tuple[str, str, str | None]:
    from query_progress_hooks import (  # noqa: E402
        clear_clarify_context_injection,
        query_progress_hooks,
        set_clarify_context_injection,
    )

    if bundle is not None:
        set_clarify_context_injection(bundle)
    started = time.perf_counter()
    error: str | None = None
    raw = ""
    try:
        async with query_progress_hooks():
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
    finally:
        clear_clarify_context_injection()
    thinking, answer = parse_complete_cot(raw or "")
    del started
    return thinking, answer, error if error else (None if not (answer or "").strip() else None)


async def _replay_case(
    rag: Any,
    case: dict[str, Any],
    *,
    mode: str,
    gate_only: bool = False,
) -> dict[str, Any]:
    orig_q = (case.get("query") or case.get("standard_question") or "").strip()
    if not orig_q:
        utterances = case.get("utterances") or []
        orig_q = (utterances[0] if utterances else "").strip()
    case_id = case.get("id", "?")

    orig_retrieval = await probe_llm_retrieval(rag.lightrag, orig_q, mode=mode)

    gate = await evaluate_clarify_gate(rag.lightrag, orig_q, mode=mode)
    clarify_triggered = isinstance(gate, ClarifyRequired)
    row: dict[str, Any] = {
        "id": case_id,
        "category": case.get("category", ""),
        "original_query": orig_q,
        "original_llm_chunks": int(orig_retrieval.get("chunk_count") or 0),
        "original_llm_total": int(orig_retrieval.get("llm_chunk_total") or 0),
        "original_max_rerank": orig_retrieval.get("max_rerank_score"),
        "original_scores_unavailable": bool(orig_retrieval.get("scores_unavailable")),
        "original_answerable": bool(orig_retrieval.get("answerable")),
        "min_rerank_threshold": clarify_candidate_min_rerank_score(),
        "k_required": clarify_candidate_k(),
        "source_set": case.get("source_set", ""),
        "clarify_triggered": clarify_triggered,
        "gate_outcome": gate.gate_outcome if isinstance(gate, ClarifyRequired) else None,
        "candidates": [],
        "pick_kind": None,
        "pick_text": None,
        "final_query": orig_q,
        "has_answer": False,
        "answer": "",
        "answer_chars": 0,
        "grade_ok": False,
        "grade_miss": [],
        "standard_answer": (case.get("standard_answer") or "").strip(),
        "aquery_error": None,
        "duration_ms": None,
    }

    bundle = None
    if clarify_triggered:
        data = gate.data
        row["clarification_id"] = data.get("clarification_id")
        row["gate_reason"] = data.get("gate_reason")
        row["unrelated"] = bool(data.get("unrelated"))
        row["unanswerable"] = bool(data.get("unanswerable"))
        row["candidates"] = [
            {
                "id": c.get("id"),
                "text": c.get("text"),
                "chunk_count": c.get("chunk_count"),
                "max_rerank_score": c.get("max_rerank_score"),
            }
            for c in (data.get("candidates") or [])
        ]
        if gate.gate_outcome == "reject":
            row["pick_kind"] = data.get("gate_reason") or "reject"
            row["pick_text"] = data.get("message")
            return row
        if gate_only:
            row["pick_kind"] = "gate_only"
            return row

        pick_kind, pick_payload = _pick_first_candidate(data)
        row["pick_kind"] = pick_kind

        if pick_kind == "candidate":
            final_q = (pick_payload.get("text") or "").strip()
            row["pick_text"] = final_q
            bypass = await evaluate_clarify_gate(
                rag.lightrag,
                final_q,
                mode=mode,
                clarify_choice="use_candidate",
                clarification_id=data.get("clarification_id"),
                candidate_id=pick_payload.get("id"),
            )
            bundle = resolve_clarify_bundle(
                data.get("clarification_id"),
                "use_candidate",
                final_q,
                pick_payload.get("id"),
            )
        else:
            row["aquery_error"] = "no_answerable_candidate"
            return row
        row["final_query"] = final_q
        if not isinstance(bypass, ClarifyBypass):
            row["aquery_error"] = "bypass_validation_failed"
            return row
    else:
        row["gate_bypass"] = gate.reason if isinstance(gate, ClarifyBypass) else None
        final_q = orig_q

    started = time.perf_counter()
    _thinking, answer, err = await _run_aquery(rag, final_q, mode=mode, bundle=bundle)
    row["duration_ms"] = int((time.perf_counter() - started) * 1000)
    row["aquery_error"] = err
    ans = (answer or "").strip()
    row["answer"] = ans
    row["has_answer"] = bool(ans)
    row["answer_chars"] = len(ans)
    ok, miss, _grade_mode = grade_response(ans, case)
    row["grade_ok"] = ok
    row["grade_miss"] = miss
    return row


_SOURCE_PRESETS: dict[str, Path | list[Path]] = {
    "green8": _ROOT / "data" / "voice_script_green8.json",
    "shili17": _ROOT / "docs" / "测试例.txt",
    "voice29": _ROOT / "data" / "voice_script_tests.json",
    "all": [
        _ROOT / "docs" / "测试例.txt",
        _ROOT / "data" / "voice_script_green8.json",
    ],
}


def _load_all_cases(source_key: str, *, limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    preset = _SOURCE_PRESETS.get(source_key, source_key)
    if isinstance(preset, list):
        rows: list[dict[str, Any]] = []
        labels: list[str] = []
        for path in preset:
            label = "shili17" if "测试例" in str(path) else path.stem
            chunk = _load_source_cases(path, limit=0)
            for row in chunk:
                row["source_set"] = label
            rows.extend(chunk)
            labels.append(f"{label}({len(chunk)})")
        if limit > 0:
            rows = rows[:limit]
        return rows, labels
    path = Path(preset) if not isinstance(preset, Path) else preset
    rows = _load_source_cases(path, limit=limit)
    for row in rows:
        row["source_set"] = source_key
    return rows, [source_key]


def _load_source_cases(source: Path, *, limit: int) -> list[dict[str, Any]]:
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = []
        for case in data.get("cases") or []:
            if case.get("skip"):
                continue
            q = (case.get("standard_question") or "").strip()
            if not q:
                utterances = case.get("utterances") or []
                q = (utterances[0] if utterances else "").strip()
            if not q:
                continue
            row = dict(case)
            row["query"] = q
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
        return rows
    slim = _load_cases(source.resolve(), limit=limit)
    return [dict(c) for c in slim]


def _format_report(
    rows: list[dict[str, Any]], *, source: str, wd: Path, gate_only: bool
) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"{source} 澄清门控批测 (v3 document-chunk gate)")
    lines.append("=" * 72)
    lines.append(f"time: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"working_dir: {wd}")
    lines.append(f"min_rerank: {clarify_candidate_min_rerank_score()}")
    lines.append(f"k_required: {clarify_candidate_k()}")
    lines.append(f"mode: gate_only={gate_only}")
    lines.append(f"cases: {len(rows)}")
    lines.append("")

    for r in rows:
        lines.append("-" * 72)
        src = r.get("source_set") or ""
        lines.append(
            f"#{r['id']} [{src}/{r.get('category','')}] {r['original_query']}"
        )
        lines.append(
            f"  gate_outcome={r.get('gate_outcome')} "
            f"clarify={'YES' if r['clarify_triggered'] else 'NO'}"
        )
        lines.append(
            f"  原问 LLM qualifying={r.get('original_llm_chunks')} "
            f"total={r.get('original_llm_total')} "
            f"max_rerank={r.get('original_max_rerank')} "
            f"answerable={r.get('original_answerable')}"
            + (" scores_unavailable" if r.get("original_scores_unavailable") else "")
        )
        if r["clarify_triggered"]:
            if r.get("gate_outcome") == "reject":
                lines.append(f"  拒答 ({r.get('gate_reason')}): {r.get('pick_text') or ''}")
            else:
                lines.append("  澄清推荐问法:")
                for c in r.get("candidates") or []:
                    rs = c.get("max_rerank_score")
                    cc = c.get("chunk_count")
                    lines.append(
                        f"    - [{c.get('id')}] chunks={cc} max_rerank={rs} | {c.get('text')}"
                    )
                if not gate_only and r.get("pick_kind") not in (None, "gate_only"):
                    lines.append(
                        f"  选用: {r['pick_kind']} | {r['pick_text']}"
                    )
        else:
            lines.append(f"  gate_bypass={r.get('gate_bypass')}")
        if not gate_only and r.get("duration_ms") is not None:
            lines.append(
                f"  最终问: {r['final_query']} "
                f"has_answer={r['has_answer']} grade={'PASS' if r.get('grade_ok') else 'FAIL'} "
                f"duration_ms={r['duration_ms']}"
            )
        if r.get("aquery_error"):
            lines.append(f"  error: {r['aquery_error']}")
        lines.append("")

    n_clarify = sum(1 for r in rows if r["clarify_triggered"])
    n_reject = sum(1 for r in rows if r.get("gate_outcome") == "reject")
    n_offer = sum(1 for r in rows if r.get("gate_outcome") == "offer")
    lines.append("=" * 72)
    lines.append(
        f"汇总: 总题 {len(rows)} | 触发门控 {n_clarify} | 拒答 {n_reject} | 澄清 Offer {n_offer}"
    )
    lines.append("")
    lines.append("set  id  orig_chunks  max_rerank  gate  cands  outcome  question")
    for r in rows:
        lines.append(
            f"{str(r.get('source_set','')):<7} "
            f"{str(r['id']):>3}  "
            f"{r.get('original_llm_chunks')!s:>11}  "
            f"{r.get('original_max_rerank')!s:>10}  "
            f"{'Y' if r['clarify_triggered'] else 'N':>4}  "
            f"{len(r.get('candidates') or []):>5}  "
            f"{str(r.get('gate_outcome') or '-'):<7} "
            f"{r['original_query']}"
        )
    lines.append("")
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
    print(f"parser_output_dir: {pod}", flush=True)
    rag, _, _ = await rpc._build_rag(wd, pod)
    cases, source_labels = _load_all_cases(args.source, limit=args.limit)
    source_label = args.source if args.source != "all" else "+".join(source_labels)

    rows: list[dict[str, Any]] = []
    try:
        for i, case in enumerate(cases, 1):
            q = case.get("query") or case.get("standard_question")
            print(f"[{i}/{len(cases)}] {case.get('source_set')}#{case.get('id')} {q}", flush=True)
            row = await _replay_case(
                rag,
                case,
                mode=args.mode,
                gate_only=args.gate_only,
            )
            rows.append(row)
            print(
                f"  chunks={row.get('original_llm_chunks')} rerank={row.get('original_max_rerank')} "
                f"gate={row.get('gate_outcome')} cands={len(row.get('candidates') or [])}",
                flush=True,
            )
    finally:
        await rag.finalize_storages()

    report = _format_report(
        rows,
        source=source_label,
        wd=wd,
        gate_only=args.gate_only,
    )
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
        help="Preset (green8, shili17, voice29, all) or path to question file",
    )
    p.add_argument("--mode", default="mix")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--gate-only",
        action="store_true",
        help="Only evaluate clarify gate (no aquery after pick)",
    )
    p.add_argument("--out-jsonl", type=Path, default=None)
    p.add_argument("--out-report", type=Path, default=None)
    args = p.parse_args()
    if args.out_jsonl is None:
        name = args.source.replace("/", "_")
        args.out_jsonl = _ROOT / "logs" / f"clarify_gate_{name}_replay.jsonl"
    if args.out_report is None:
        name = args.source.replace("/", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_report = _ROOT / "logs" / f"clarify_gate_{name}_replay_{ts}.txt"
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
