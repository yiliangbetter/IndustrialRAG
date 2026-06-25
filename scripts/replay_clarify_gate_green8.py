#!/usr/bin/env python3
"""Replay question sets through clarify gate: pick first recommended candidate, then aquery.

Reports per-question scores, candidate scores, pick, and whether an answer was produced.
Original-query probe stats come from the same single ``evaluate_clarify_gate`` call as Web
(``gate.data[\"original_probe\"]``), not a separate pre-probe.

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

from score_query_relevance import _load_cases  # noqa: E402
from stream_cot_parser import parse_complete_cot  # noqa: E402
from raganything.clarify_gate import (  # noqa: E402
    ClarifyBypass,
    ClarifyRequired,
    clarify_candidate_k,
    clarify_candidate_min_rerank_score,
    evaluate_clarify_gate,
    resolve_clarify_bundle,
    resolve_clarify_cached_response,
)


def _fscore(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _pick_candidate(
    data: dict[str, Any],
    *,
    strategy: str,
    rng: random.Random,
) -> tuple[str, dict[str, Any]]:
    """Pick a recommended question (first or random)."""
    candidates = list(data.get("candidates") or [])
    if not candidates:
        return "none", {}
    if strategy == "random":
        return "candidate", rng.choice(candidates)
    return "candidate", candidates[0]


def _answer_has_reference_markers(answer: str) -> bool:
    return bool(_extract_answer_reference_lines(answer))


def _extract_answer_reference_lines(answer: str) -> list[str]:
    """Parse ``### References`` block and ``[n]`` citation lines from LLM answer."""
    text = (answer or "").strip()
    if not text:
        return []
    lines: list[str] = []
    match = re.search(r"(?im)^#{1,3}\s*References\s*$", text)
    if match:
        tail = text[match.end() :].strip()
        for raw in tail.splitlines():
            line = raw.strip().lstrip("-*•").strip()
            if not line:
                continue
            if re.search(r"\[\d+\]", line) or ".pdf" in line.lower():
                lines.append(line)
    if not lines:
        for raw in text.splitlines():
            line = raw.strip().lstrip("-*•").strip()
            if re.search(r"^\[\d+\]", line):
                lines.append(line)
    # dedupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def _format_context_reference_lines(refs: list[dict[str, Any]] | None) -> list[str]:
    rows: list[str] = []
    for item in refs or []:
        if not isinstance(item, dict):
            continue
        rid = item.get("reference_id")
        fp = item.get("file_path") or item.get("doc_name")
        if rid is not None and fp:
            rows.append(f"[{rid}] {fp}")
        elif fp:
            rows.append(str(fp))
    return rows


def _refs_detail_from_llm_input(llm_input: dict[str, Any] | None) -> tuple[int, list[str]]:
    if not isinstance(llm_input, dict):
        return 0, []
    refs = llm_input.get("references")
    if isinstance(refs, list):
        lines = _format_context_reference_lines(refs)
        if lines:
            return len(lines), lines
    counts = llm_input.get("counts")
    count = int(counts.get("references") or 0) if isinstance(counts, dict) else 0
    return count, []


def _refs_from_llm_input(llm_input: dict[str, Any] | None) -> int:
    count, _ = _refs_detail_from_llm_input(llm_input)
    return count


def _refs_detail_from_bundle(bundle: Any | None) -> tuple[int, list[str]]:
    if bundle is None:
        return 0, []
    raw = getattr(bundle, "raw_data", None) or {}
    data = raw.get("data") if isinstance(raw, dict) else {}
    refs = (data or {}).get("references") if isinstance(data, dict) else []
    lines = _format_context_reference_lines(refs if isinstance(refs, list) else [])
    return len(lines), lines


def _refs_from_bundle(bundle: Any | None) -> int:
    count, _ = _refs_detail_from_bundle(bundle)
    return count


def _check_probe_score_logic(row: dict[str, Any], *, min_thr: float) -> tuple[bool, list[str]]:
    """C1: answerable iff qualifying chunks with rerank >= min_thr."""
    issues: list[str] = []
    chunks = int(row.get("original_llm_chunks") or 0)
    answerable = bool(row.get("original_answerable"))
    max_rs = _fscore(row.get("original_max_rerank"))

    if answerable != (chunks > 0):
        issues.append("answerable_ne_chunks_gt0_mismatch")
    if chunks > 0 and max_rs is not None and max_rs < min_thr:
        issues.append(f"qualifying_chunks_but_max_rerank_{max_rs}_lt_{min_thr}")
    if answerable and max_rs is not None and max_rs < min_thr:
        issues.append(f"answerable_but_max_rerank_{max_rs}_lt_{min_thr}")
    if chunks > 0 and max_rs is None and not row.get("original_scores_unavailable"):
        issues.append("qualifying_chunks_but_no_max_rerank")
    return not issues, issues


def _check_gate_outcome_logic(row: dict[str, Any], *, k: int) -> tuple[bool, list[str]]:
    """C2: reject/offer vs answerable and candidate count."""
    issues: list[str] = []
    outcome = row.get("gate_outcome")
    reason = row.get("gate_reason") or ""
    chunks = int(row.get("original_llm_chunks") or 0)
    answerable = bool(row.get("original_answerable"))
    cands = list(row.get("candidates") or [])

    if answerable != (chunks > 0):
        issues.append("gate_answerable_ne_chunks_gt0")

    if outcome == "reject":
        if reason == "no_document_chunks":
            if answerable or chunks > 0:
                issues.append("reject_no_document_chunks_but_answerable")
            if cands:
                issues.append("reject_no_document_chunks_but_has_candidates")
        elif reason == "cannot_fill_answerable_candidates":
            if not answerable:
                issues.append("cannot_fill_but_original_not_answerable")
            if len(cands) >= k:
                issues.append("cannot_fill_but_candidates_ge_k")
        if not row.get("unrelated"):
            issues.append("reject_should_mark_unrelated")
    elif outcome == "offer":
        if not answerable or chunks <= 0:
            issues.append("offer_but_original_not_answerable")
        if len(cands) != k:
            issues.append(f"offer_candidates_{len(cands)}_ne_k_{k}")
        if row.get("unrelated"):
            issues.append("offer_should_not_be_unrelated")
    elif row.get("clarify_triggered"):
        issues.append(f"unexpected_gate_outcome_{outcome}")

    return not issues, issues


def _check_candidates_logic(
    candidates: list[dict[str, Any]], *, min_thr: float
) -> tuple[bool, list[str]]:
    """C3a: each listed candidate probe stats meet threshold."""
    issues: list[str] = []
    for c in candidates:
        cid = c.get("id") or "?"
        cc = int(c.get("chunk_count") or 0)
        rs = _fscore(c.get("max_rerank_score"))
        if cc <= 0:
            issues.append(f"{cid}_chunk_count_zero")
        if rs is not None and rs < min_thr:
            issues.append(f"{cid}_max_rerank_{rs}_lt_{min_thr}")
        elif rs is None and cc > 0:
            issues.append(f"{cid}_missing_max_rerank")
    return not issues, issues


def _attach_design_checks(row: dict[str, Any]) -> None:
    min_thr = float(row.get("min_rerank_threshold") or clarify_candidate_min_rerank_score())
    k = int(row.get("k_required") or clarify_candidate_k())
    ok1, i1 = _check_probe_score_logic(row, min_thr=min_thr)
    ok2, i2 = _check_gate_outcome_logic(row, k=k)
    ok3, i3 = _check_candidates_logic(list(row.get("candidates") or []), min_thr=min_thr)

    pick_ok = True
    pick_issues: list[str] = []
    if row.get("gate_outcome") == "offer" and row.get("duration_ms") is not None:
        picked = {
            "id": row.get("pick_candidate_id"),
            "chunk_count": row.get("pick_chunk_count"),
            "max_rerank_score": row.get("pick_max_rerank_score"),
        }
        pick_ok, pick_issues = _check_candidates_logic([picked], min_thr=min_thr)
        if not row.get("has_answer"):
            pick_issues.append("picked_query_no_llm_answer")
        has_ref = bool(
            row.get("answer_has_reference")
            or int(row.get("bundle_reference_count") or 0) > 0
            or int(row.get("llm_input_reference_count") or 0) > 0
        )
        if not has_ref:
            pick_issues.append("picked_query_no_reference_evidence")
        pick_ok = pick_ok and not pick_issues

    row["design_checks"] = {
        "c1_probe_score": {"ok": ok1, "issues": i1},
        "c2_gate_outcome": {"ok": ok2, "issues": i2},
        "c3_candidates_listed": {"ok": ok3, "issues": i3},
        "c3_picked_answer": {"ok": pick_ok, "issues": pick_issues},
        "all_ok": ok1 and ok2 and ok3 and pick_ok,
    }


def _pick_first_candidate(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return _pick_candidate(data, strategy="first", rng=random.Random(0))


async def _run_aquery(
    rag: Any,
    query: str,
    *,
    mode: str,
    bundle: Any | None = None,
) -> tuple[str, str, str | None, dict[str, Any]]:
    from query_progress_hooks import (  # noqa: E402
        clear_clarify_context_injection,
        get_llm_input,
        get_query_debug_state,
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
    ans = (answer or "").strip()
    llm_input = get_llm_input() or (get_query_debug_state().get("llm_input") or {})
    if not isinstance(llm_input, dict):
        llm_input = {}
    meta = {
        "llm_input_reference_count": _refs_from_llm_input(llm_input),
        "llm_input_reference_lines": _refs_detail_from_llm_input(llm_input)[1],
        "llm_input_chunk_count": len(llm_input.get("chunks") or []),
        "answer_has_reference": _answer_has_reference_markers(ans),
        "answer_reference_lines": _extract_answer_reference_lines(ans),
    }
    del started
    return thinking, answer, error if error else (None if not ans else None), meta


async def _replay_case(
    rag: Any,
    case: dict[str, Any],
    *,
    mode: str,
    gate_only: bool = False,
    pick_strategy: str = "first",
    rng: random.Random | None = None,
) -> dict[str, Any]:
    orig_q = (case.get("query") or case.get("standard_question") or "").strip()
    if not orig_q:
        utterances = case.get("utterances") or []
        orig_q = (utterances[0] if utterances else "").strip()
    case_id = case.get("id", "?")

    gate = await evaluate_clarify_gate(rag.lightrag, orig_q, mode=mode)
    clarify_triggered = isinstance(gate, ClarifyRequired)
    orig_probe: dict[str, Any] = (
        (gate.data.get("original_probe") or {})
        if isinstance(gate, ClarifyRequired)
        else {}
    )
    row: dict[str, Any] = {
        "id": case_id,
        "category": case.get("category", ""),
        "original_query": orig_q,
        "original_llm_chunks": int(orig_probe.get("chunk_count") or 0),
        "original_llm_total": int(orig_probe.get("llm_chunk_total") or 0),
        "original_max_rerank": orig_probe.get("max_rerank_score"),
        "original_scores_unavailable": bool(orig_probe.get("scores_unavailable")),
        "original_answerable": bool(orig_probe.get("answerable")),
        "min_rerank_threshold": clarify_candidate_min_rerank_score(),
        "k_required": clarify_candidate_k(),
        "source_set": case.get("source_set", ""),
        "clarify_triggered": clarify_triggered,
        "gate_outcome": gate.gate_outcome if isinstance(gate, ClarifyRequired) else None,
        "candidates": [],
        "pick_kind": None,
        "pick_text": None,
        "pick_candidate_id": None,
        "pick_chunk_count": None,
        "pick_max_rerank_score": None,
        "final_query": orig_q,
        "has_answer": False,
        "answer_cached": False,
        "answer": "",
        "answer_chars": 0,
        "answer_has_reference": False,
        "answer_reference_lines": [],
        "llm_input_reference_count": 0,
        "llm_input_reference_lines": [],
        "bundle_reference_count": 0,
        "bundle_reference_lines": [],
        "standard_answer": (case.get("standard_answer") or "").strip(),
        "aquery_error": None,
        "duration_ms": None,
    }

    bundle = None
    cached: dict[str, Any] | None = None
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
            _attach_design_checks(row)
            return row
        if gate_only:
            row["pick_kind"] = "gate_only"
            _attach_design_checks(row)
            return row

        pick_rng = rng if rng is not None else random.Random()
        pick_kind, pick_payload = _pick_candidate(
            data, strategy=pick_strategy, rng=pick_rng
        )
        row["pick_kind"] = pick_kind

        if pick_kind == "candidate":
            final_q = (pick_payload.get("text") or "").strip()
            row["pick_text"] = final_q
            row["pick_candidate_id"] = pick_payload.get("id")
            row["pick_chunk_count"] = pick_payload.get("chunk_count")
            row["pick_max_rerank_score"] = pick_payload.get("max_rerank_score")
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
            row["bundle_reference_count"] = _refs_from_bundle(bundle)
            _, bundle_ref_lines = _refs_detail_from_bundle(bundle)
            row["bundle_reference_lines"] = bundle_ref_lines
        else:
            row["aquery_error"] = "no_answerable_candidate"
            _attach_design_checks(row)
            return row
        row["final_query"] = final_q
        if not isinstance(bypass, ClarifyBypass):
            row["aquery_error"] = "bypass_validation_failed"
            _attach_design_checks(row)
            return row
        if bypass.reason == "cached_answer":
            cached = resolve_clarify_cached_response(
                data.get("clarification_id"),
                "use_candidate",
                final_q,
                pick_payload.get("id"),
            )
    else:
        row["gate_bypass"] = gate.reason if isinstance(gate, ClarifyBypass) else None
        final_q = orig_q

    started = time.perf_counter()
    if cached is not None and cached.get("answer"):
        ans = str(cached.get("answer") or "").strip()
        row["answer"] = ans
        row["has_answer"] = bool(ans)
        row["answer_cached"] = True
        row["answer_chars"] = len(ans)
        row["answer_has_reference"] = _answer_has_reference_markers(ans)
        row["answer_reference_lines"] = _extract_answer_reference_lines(ans)
        row["llm_input_reference_count"] = int(row.get("bundle_reference_count") or 0)
        row["llm_input_reference_lines"] = list(row.get("bundle_reference_lines") or [])
        row["duration_ms"] = int((time.perf_counter() - started) * 1000)
    else:
        _thinking, answer, err, aquery_meta = await _run_aquery(
            rag, final_q, mode=mode, bundle=bundle
        )
        row["duration_ms"] = int((time.perf_counter() - started) * 1000)
        row["aquery_error"] = err
        ans = (answer or "").strip()
        row["answer"] = ans
        row["has_answer"] = bool(ans)
        row["answer_chars"] = len(ans)
        row["answer_has_reference"] = bool(aquery_meta.get("answer_has_reference"))
        row["answer_reference_lines"] = list(aquery_meta.get("answer_reference_lines") or [])
        row["llm_input_reference_count"] = int(
            aquery_meta.get("llm_input_reference_count") or 0
        )
        row["llm_input_reference_lines"] = list(
            aquery_meta.get("llm_input_reference_lines") or []
        )
    _attach_design_checks(row)
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
            dc = r.get("design_checks") or {}
            lines.append(
                f"  最终问: {r['final_query']} "
                f"has_answer={r['has_answer']} "
                f"design_ok={'PASS' if dc.get('all_ok') else 'FAIL'} "
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


def _format_design_report(
    rows: list[dict[str, Any]], *, source: str, wd: Path, pick_strategy: str
) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"{source} 澄清门控设计逻辑验证 (v3)")
    lines.append("=" * 72)
    lines.append(f"time: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"working_dir: {wd}")
    lines.append(f"min_rerank: {clarify_candidate_min_rerank_score()}")
    lines.append(f"k_required: {clarify_candidate_k()}")
    lines.append(f"pick_strategy: {pick_strategy}")
    lines.append(f"cases: {len(rows)}")
    lines.append("")
    lines.append("验证项:")
    lines.append("  C1 原问可答 <=> qualifying chunks>0 且 max_rerank>=阈值")
    lines.append("  C2 gate_outcome 与可答/推荐条数符合 v3 设计")
    lines.append("  C3 推荐问 probe 可答；随机点选后 LLM 有回答且 bundle/答案含 reference")
    lines.append("")

    n_all_ok = 0
    for r in rows:
        dc = r.get("design_checks") or {}
        lines.append("-" * 72)
        lines.append(
            f"#{r['id']} [{r.get('source_set','')}/{r.get('category','')}] {r['original_query']}"
        )
        lines.append(
            f"  gate={r.get('gate_outcome')} reason={r.get('gate_reason') or '-'} "
            f"qualifying={r.get('original_llm_chunks')} max_rerank={r.get('original_max_rerank')} "
            f"answerable={r.get('original_answerable')}"
        )
        for key, label in (
            ("c1_probe_score", "C1 probe/score"),
            ("c2_gate_outcome", "C2 gate/outcome"),
            ("c3_candidates_listed", "C3 candidates"),
            ("c3_picked_answer", "C3 picked+aquery"),
        ):
            block = dc.get(key) or {}
            ok = block.get("ok")
            issues = block.get("issues") or []
            if key == "c3_picked_answer" and r.get("gate_outcome") != "offer":
                lines.append(f"  {label}: SKIP (no offer pick)")
                continue
            if (
                key == "c3_picked_answer"
                and r.get("gate_outcome") == "offer"
                and r.get("duration_ms") is None
            ):
                lines.append(f"  {label}: SKIP (no aquery)")
                continue
            if ok is None and not r.get("clarify_triggered") and key.startswith("c2"):
                lines.append(f"  {label}: SKIP (gate bypass)")
                continue
            mark = "PASS" if ok else "FAIL"
            suffix = f" | {', '.join(issues)}" if issues else ""
            lines.append(f"  {label}: {mark}{suffix}")

        if r.get("gate_outcome") == "offer":
            lines.append("  推荐问:")
            for c in r.get("candidates") or []:
                lines.append(
                    f"    [{c.get('id')}] chunks={c.get('chunk_count')} "
                    f"max_rerank={c.get('max_rerank_score')} | {c.get('text')}"
                )
            if r.get("pick_kind") == "candidate":
                lines.append(
                    f"  随机选用: [{r.get('pick_candidate_id')}] {r.get('pick_text')}"
                )
                ans_refs = list(r.get("answer_reference_lines") or [])
                bundle_refs = list(r.get("bundle_reference_lines") or [])
                llm_refs = list(r.get("llm_input_reference_lines") or [])
                lines.append(
                    f"  LLM: has_answer={r.get('has_answer')} "
                    f"duration_ms={r.get('duration_ms')}"
                )
                lines.append("  References 证据:")
                if ans_refs:
                    lines.append("    答案 References 段落:")
                    for ref_line in ans_refs:
                        lines.append(f"      {ref_line}")
                else:
                    lines.append("    答案 References 段落: (无)")
                if bundle_refs:
                    lines.append("    点选 bundle context references:")
                    for ref_line in bundle_refs:
                        lines.append(f"      {ref_line}")
                else:
                    lines.append(
                        f"    点选 bundle context references: (无明细, count={r.get('bundle_reference_count')})"
                    )
                if llm_refs:
                    lines.append("    答题 llm_input references:")
                    for ref_line in llm_refs:
                        lines.append(f"      {ref_line}")
                else:
                    lines.append(
                        f"    答题 llm_input references: (无明细, count={r.get('llm_input_reference_count')})"
                    )
                ref_ok = bool(
                    ans_refs
                    or bundle_refs
                    or int(r.get("llm_input_reference_count") or 0) > 0
                )
                lines.append(
                    f"    判定: reference_evidence={'YES' if ref_ok else 'NO'} "
                    f"(answer_refs={len(ans_refs)} bundle_refs={len(bundle_refs)} "
                    f"llm_input_refs={r.get('llm_input_reference_count')})"
                )
                if r.get("has_answer"):
                    preview = (r.get("answer") or "").replace("\n", " ")[:160]
                    lines.append(f"  回答摘要: {preview}{'…' if len(r.get('answer') or '') > 160 else ''}")
        elif r.get("gate_outcome") == "reject":
            lines.append(f"  拒答: {r.get('pick_text') or ''}")

        if dc.get("all_ok"):
            n_all_ok += 1
        lines.append("")

    lines.append("=" * 72)
    lines.append(
        f"汇总: {n_all_ok}/{len(rows)} 题 design_checks.all_ok | "
        f"reject={sum(1 for r in rows if r.get('gate_outcome') == 'reject')} "
        f"offer={sum(1 for r in rows if r.get('gate_outcome') == 'offer')}"
    )
    failed = [r for r in rows if not (r.get("design_checks") or {}).get("all_ok")]
    if failed:
        lines.append("未通过:")
        for r in failed:
            dc = r.get("design_checks") or {}
            parts = []
            for k in ("c1_probe_score", "c2_gate_outcome", "c3_candidates_listed", "c3_picked_answer"):
                if not (dc.get(k) or {}).get("ok", True):
                    parts.extend((dc.get(k) or {}).get("issues") or [k])
            lines.append(f"  #{r['id']} {r['original_query']}: {', '.join(parts) or 'unknown'}")
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
    rng = random.Random(args.seed if args.seed is not None else time.time_ns())

    rows: list[dict[str, Any]] = []
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        args.out_jsonl.write_text("", encoding="utf-8")
    try:
        for i, case in enumerate(cases, 1):
            q = case.get("query") or case.get("standard_question")
            print(f"[{i}/{len(cases)}] {case.get('source_set')}#{case.get('id')} {q}", flush=True)
            row = await _replay_case(
                rag,
                case,
                mode=args.mode,
                gate_only=args.gate_only,
                pick_strategy=args.pick,
                rng=rng,
            )
            rows.append(row)
            if args.checkpoint:
                with args.out_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            dc = row.get("design_checks") or {}
            print(
                f"  chunks={row.get('original_llm_chunks')} rerank={row.get('original_max_rerank')} "
                f"gate={row.get('gate_outcome')} cands={len(row.get('candidates') or [])} "
                f"design={'OK' if dc.get('all_ok') else 'FAIL'}",
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
    design_report = _format_design_report(
        rows,
        source=source_label,
        wd=wd,
        pick_strategy=args.pick,
    )
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint:
        with args.out_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.out_report.write_text(report, encoding="utf-8")
    args.out_design_report.write_text(design_report, encoding="utf-8")
    print(report)
    print(design_report)
    print(f"JSONL: {args.out_jsonl}")
    print(f"Report: {args.out_report}")
    print(f"Design report: {args.out_design_report}")


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
    p.add_argument(
        "--pick",
        choices=("first", "random"),
        default="first",
        help="How to choose a recommended question when gate offers (default: first)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for --pick random (default: nondeterministic)",
    )
    p.add_argument(
        "--checkpoint",
        action="store_true",
        help="Append each case to --out-jsonl as it completes (crash-safe)",
    )
    p.add_argument("--out-jsonl", type=Path, default=None)
    p.add_argument("--out-report", type=Path, default=None)
    p.add_argument("--out-design-report", type=Path, default=None)
    args = p.parse_args()
    if args.out_jsonl is None:
        name = args.source.replace("/", "_")
        args.out_jsonl = _ROOT / "logs" / f"clarify_gate_{name}_replay.jsonl"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_report is None:
        name = args.source.replace("/", "_")
        args.out_report = _ROOT / "logs" / f"clarify_gate_{name}_replay_{ts}.txt"
    if args.out_design_report is None:
        name = args.source.replace("/", "_")
        args.out_design_report = (
            _ROOT / "logs" / f"clarify_gate_{name}_design_{ts}.txt"
        )
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
