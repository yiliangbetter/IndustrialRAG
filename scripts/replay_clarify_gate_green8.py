#!/usr/bin/env python3
"""Replay question sets through clarify gate, then aquery (optional LLM answer).

Reports per-question original ``final_score``, candidate ``final_score``s, random/first
pick among recommendations, and whether aquery produced an answer with references.

Examples::

  uv run python scripts/replay_clarify_gate_green8.py --source green8
  uv run python scripts/replay_clarify_gate_green8.py --source shili17
  uv run python scripts/replay_clarify_gate_green8.py --source voice29 --pick random --checkpoint
  uv run python scripts/replay_clarify_gate_green8.py --source voice29 --pick random --gate-only --no-dump

Each aquery (non ``--gate-only``) writes ``logs/query_dumps/{source}_{id}_*.json`` with
retrieval, ``clarify_gate`` replay metadata, and LLM input when ``--dump`` (default on).
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

from query_test_cases import _load_cases  # noqa: E402
from stream_cot_parser import parse_complete_cot  # noqa: E402
from raganything.clarify_gate import (  # noqa: E402
    ClarifyBypass,
    ClarifyRequired,
    clarify_candidate_k,
    clarify_candidate_min_rerank_score,
    clarify_direct_rerank_min,
    evaluate_clarify_gate,
    resolve_clarify_bundle,
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


def _clarify_gate_dump_meta(
    *,
    case_id: str,
    source_set: str,
    row: dict[str, Any],
    gate: ClarifyBypass | ClarifyRequired,
    clarify_triggered: bool,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "replay_id": case_id,
        "replay_source": source_set,
        "original_query": row.get("original_query"),
        "original_final_score": row.get("original_final_score"),
        "pick_kind": row.get("pick_kind"),
        "pick_candidate_id": row.get("pick_candidate_id"),
        "pick_final_score": row.get("pick_final_score"),
        "picked_query": row.get("final_query")
        if row.get("pick_kind") == "candidate"
        else None,
    }
    if isinstance(gate, ClarifyRequired):
        meta["required"] = True
        meta["gate_outcome"] = gate.gate_outcome
        if isinstance(gate.data, dict):
            for key, val in gate.data.items():
                if key not in meta:
                    meta[key] = val
    elif isinstance(gate, ClarifyBypass):
        meta["required"] = False
        meta["gate_skipped"] = gate.reason
        meta["gate_version"] = "v4"
        if gate.probe is not None:
            meta["original_probe"] = gate.probe.as_stats()
    meta["clarify_triggered"] = clarify_triggered
    return meta


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
    """C1: answerable iff final_score present (qualifying chunks exist)."""
    issues: list[str] = []
    final_score = _fscore(row.get("original_final_score") or row.get("original_max_rerank"))
    answerable = bool(row.get("original_answerable"))
    chunks = int(row.get("original_llm_chunks") or 0)

    if answerable != (final_score is not None):
        issues.append("answerable_ne_has_final_score")
    if answerable and chunks <= 0:
        issues.append("answerable_but_chunk_count_zero")
    if answerable and final_score is not None and final_score < min_thr:
        issues.append(f"answerable_but_final_{final_score}_lt_{min_thr}")
    if not answerable and final_score is not None:
        issues.append("not_answerable_but_has_final_score")
    return not issues, issues


def _check_gate_outcome_logic(
    row: dict[str, Any], *, k: int, direct_min: float
) -> tuple[bool, list[str]]:
    """C2: v4 reject / offer / direct vs final_score band."""
    issues: list[str] = []
    outcome = row.get("gate_outcome")
    reason = row.get("gate_reason") or ""
    bypass = row.get("gate_bypass")
    final_score = _fscore(row.get("original_final_score") or row.get("original_max_rerank"))
    cands = list(row.get("candidates") or [])

    effective = outcome or bypass
    if effective == "direct":
        if final_score is None or final_score <= direct_min:
            issues.append("direct_but_final_not_gt_direct_min")
        if cands:
            issues.append("direct_but_has_candidates")
    elif effective == "reject":
        if reason == "no_final_chunks":
            if final_score is not None:
                issues.append("reject_no_final_chunks_but_has_final_score")
            if cands:
                issues.append("reject_no_final_chunks_but_has_candidates")
        elif reason == "cannot_fill_high_confidence_candidates":
            if final_score is None or final_score > direct_min:
                issues.append("cannot_fill_but_final_not_mid_band")
            if len(cands) >= k:
                issues.append("cannot_fill_but_candidates_ge_k")
        if not row.get("unrelated"):
            issues.append("reject_should_mark_unrelated")
    elif effective == "offer":
        if final_score is None or final_score > direct_min:
            issues.append("offer_but_final_not_mid_band")
        if len(cands) != k:
            issues.append(f"offer_candidates_{len(cands)}_ne_k_{k}")
        if row.get("unrelated"):
            issues.append("offer_should_not_be_unrelated")
    elif row.get("clarify_triggered"):
        issues.append(f"unexpected_gate_outcome_{effective}")

    return not issues, issues


def _check_candidates_logic(
    candidates: list[dict[str, Any]], *, direct_min: float
) -> tuple[bool, list[str]]:
    """C3a: each listed candidate final_score > direct_min."""
    issues: list[str] = []
    for c in candidates:
        cid = c.get("id") or "?"
        fs = _fscore(c.get("final_score") or c.get("max_rerank_score"))
        cc = int(c.get("chunk_count") or 0)
        if cc <= 0:
            issues.append(f"{cid}_chunk_count_zero")
        if fs is None or fs <= direct_min:
            issues.append(f"{cid}_final_{fs}_not_gt_{direct_min}")
    return not issues, issues


def _attach_design_checks(row: dict[str, Any]) -> None:
    min_thr = float(row.get("min_rerank_threshold") or clarify_candidate_min_rerank_score())
    direct_min = float(row.get("direct_rerank_min") or clarify_direct_rerank_min())
    k = int(row.get("k_required") or clarify_candidate_k())
    ok1, i1 = _check_probe_score_logic(row, min_thr=min_thr)
    ok2, i2 = _check_gate_outcome_logic(row, k=k, direct_min=direct_min)
    ok3, i3 = _check_candidates_logic(list(row.get("candidates") or []), direct_min=direct_min)

    pick_ok = True
    pick_issues: list[str] = []
    if row.get("gate_outcome") == "offer" and row.get("duration_ms") is not None:
        picked = {
            "id": row.get("pick_candidate_id"),
            "chunk_count": row.get("pick_chunk_count"),
            "final_score": row.get("pick_final_score") or row.get("pick_max_rerank_score"),
            "max_rerank_score": row.get("pick_max_rerank_score"),
        }
        pick_ok, pick_issues = _check_candidates_logic([picked], direct_min=direct_min)
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
    parser_root: Path | None = None,
) -> tuple[str, str, str | None, dict[str, Any]]:
    from query_progress_hooks import (  # noqa: E402
        clear_clarify_context_injection,
        get_llm_input,
        get_query_debug_state,
        query_progress_hooks,
        set_clarify_context_injection,
        set_query_media_roots,
        set_query_text_for_images,
    )

    if bundle is not None:
        set_clarify_context_injection(bundle)
    if parser_root is not None:
        set_query_media_roots([parser_root.resolve()])
        set_query_text_for_images(query.strip())
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
    parser_root: Path | None = None,
    write_dumps: bool = False,
) -> dict[str, Any]:
    orig_q = (case.get("query") or case.get("standard_question") or "").strip()
    if not orig_q:
        utterances = case.get("utterances") or []
        orig_q = (utterances[0] if utterances else "").strip()
    case_id = case.get("id", "?")

    gate = await evaluate_clarify_gate(rag.lightrag, orig_q, mode=mode)
    clarify_triggered = isinstance(gate, ClarifyRequired)
    orig_probe: dict[str, Any] = {}
    if isinstance(gate, ClarifyRequired):
        orig_probe = gate.data.get("original_probe") or {}
    elif isinstance(gate, ClarifyBypass) and gate.probe is not None:
        orig_probe = gate.probe.as_stats()
    row: dict[str, Any] = {
        "id": case_id,
        "category": case.get("category", ""),
        "original_query": orig_q,
        "original_llm_chunks": int(orig_probe.get("chunk_count") or 0),
        "original_llm_total": int(orig_probe.get("llm_chunk_total") or 0),
        "original_final_score": orig_probe.get("final_score"),
        "original_max_rerank": orig_probe.get("max_rerank_score"),
        "original_scores_unavailable": bool(orig_probe.get("scores_unavailable")),
        "original_answerable": bool(orig_probe.get("answerable")),
        "min_rerank_threshold": clarify_candidate_min_rerank_score(),
        "direct_rerank_min": clarify_direct_rerank_min(),
        "k_required": clarify_candidate_k(),
        "source_set": case.get("source_set", ""),
        "clarify_triggered": clarify_triggered,
        "gate_outcome": (
            gate.gate_outcome
            if isinstance(gate, ClarifyRequired)
            else (gate.reason if isinstance(gate, ClarifyBypass) else None)
        ),
        "gate_bypass": gate.reason if isinstance(gate, ClarifyBypass) else None,
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
        "dump_path": None,
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
                "final_score": c.get("final_score"),
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
            row["pick_final_score"] = pick_payload.get("final_score")
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
    else:
        row["gate_bypass"] = gate.reason if isinstance(gate, ClarifyBypass) else None
        final_q = orig_q

    if isinstance(gate, ClarifyBypass) and gate.reason == "direct":
        row["clarify_triggered"] = False

    started = time.perf_counter()
    thinking, answer, err, aquery_meta = await _run_aquery(
        rag, final_q, mode=mode, bundle=bundle, parser_root=parser_root
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
    if write_dumps and parser_root is not None:
        from query_debug_dump import persist_query_debug_dump  # noqa: E402

        source_set = str(case.get("source_set") or "")
        dump_prefix = f"{source_set}_{case_id}".replace("/", "_")
        dump_path = persist_query_debug_dump(
            query=final_q,
            mode=mode,
            parser_root=parser_root,
            thinking=(thinking or None),
            answer=answer or None,
            error=err,
            duration_ms=row["duration_ms"],
            clarify_gate=_clarify_gate_dump_meta(
                case_id=str(case_id),
                source_set=source_set,
                row=row,
                gate=gate,
                clarify_triggered=clarify_triggered,
            ),
            enabled=True,
            name_prefix=dump_prefix,
        )
        if dump_path is not None:
            row["dump_path"] = str(dump_path)
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

# Builtin fallback when data/voice_script_green8.json is absent (bench / replay).
_BUILTIN_GREEN8_CASES: list[dict[str, Any]] = [
    {"id": 2, "standard_question": "漏胶", "category": "封边", "utterances": ["漏胶"]},
    {"id": 4, "standard_question": "仿形效果不好", "category": "封边", "utterances": ["仿形效果不好"]},
    {"id": 6, "standard_question": "气压报警？", "category": "所有", "utterances": ["气压报警？"]},
    {"id": 7, "standard_question": "靠板上限？", "category": "电脑锯", "utterances": ["靠板上限？"]},
    {"id": 11, "standard_question": "三相电异常", "category": "数控", "utterances": ["三相电异常"]},
    {"id": 14, "standard_question": "变频器异常报警", "category": "数控", "utterances": ["变频器异常报警"]},
    {"id": 26, "standard_question": "未检测到工作（板材）", "category": "数控", "utterances": ["未检测到工作（板材）"]},
    {"id": 28, "standard_question": "未检测到工件", "category": "数控", "utterances": ["未检测到工件"]},
]


def _missing_source_hint(path: Path) -> str:
    if path.name == "voice_script_tests.json":
        return (
            f"缺少用例文件: {path}\n"
            "请从 Markdown 生成: uv run python scripts/build_voice_script_tests_from_md.py\n"
            "或从 Excel 生成: uv run python scripts/build_voice_script_tests.py"
        )
    if path.name == "voice_script_green8.json":
        return (
            f"缺少用例文件: {path}\n"
            "可运行: uv run python scripts/extract_voice_green8.py "
            "(需先有 voice_script_tests.json)，或从 docs/voice_script_green8_tests.txt 恢复。"
        )
    return f"用例文件不存在: {path}"


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
        if not source.is_file():
            if source.name == "voice_script_green8.json":
                rows = [dict(c) for c in _BUILTIN_GREEN8_CASES]
                for row in rows:
                    row["query"] = (row.get("standard_question") or "").strip()
                if limit > 0:
                    rows = rows[:limit]
                return rows
            raise SystemExit(_missing_source_hint(source))
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
    if not source.is_file():
        raise SystemExit(_missing_source_hint(source))
    slim = _load_cases(source.resolve(), limit=limit)
    return [dict(c) for c in slim]


def _format_report(
    rows: list[dict[str, Any]], *, source: str, wd: Path, gate_only: bool
) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"{source} 澄清门控批测 (v4 final-score gate)")
    lines.append("=" * 72)
    lines.append(f"time: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"working_dir: {wd}")
    lines.append(f"min_rerank: {clarify_candidate_min_rerank_score()}")
    lines.append(f"direct_rerank_min: {clarify_direct_rerank_min()}")
    lines.append(f"k_required: {clarify_candidate_k()}")
    lines.append(f"mode: gate_only={gate_only}")
    if not gate_only:
        from query_debug_dump import get_query_dump_dir  # noqa: E402

        lines.append(f"query_dumps: {get_query_dump_dir()}")
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
            f"final={r.get('original_final_score')} "
            f"answerable={r.get('original_answerable')}"
            + (" scores_unavailable" if r.get("original_scores_unavailable") else "")
        )
        if r["clarify_triggered"]:
            if r.get("gate_outcome") == "reject":
                lines.append(f"  拒答 ({r.get('gate_reason')}): {r.get('pick_text') or ''}")
            else:
                lines.append("  澄清推荐问法:")
                for c in r.get("candidates") or []:
                    rs = c.get("final_score") if c.get("final_score") is not None else c.get("max_rerank_score")
                    cc = c.get("chunk_count")
                    lines.append(
                        f"    - [{c.get('id')}] chunks={cc} final={rs} | {c.get('text')}"
                    )
                if not gate_only and r.get("pick_kind") not in (None, "gate_only"):
                    lines.append(
                        f"  选用: {r['pick_kind']} id={r.get('pick_candidate_id')} "
                        f"final={r.get('pick_final_score')} | {r['pick_text']}"
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
        if r.get("dump_path"):
            lines.append(f"  dump: {r['dump_path']}")
        lines.append("")

    n_clarify = sum(1 for r in rows if r["clarify_triggered"])
    n_reject = sum(1 for r in rows if r.get("gate_outcome") == "reject")
    n_offer = sum(1 for r in rows if r.get("gate_outcome") == "offer")
    n_direct = sum(1 for r in rows if r.get("gate_outcome") == "direct")
    lines.append("=" * 72)
    lines.append(
        f"汇总: 总题 {len(rows)} | 直答 {n_direct} | 触发门控 {n_clarify} | "
        f"拒答 {n_reject} | 澄清 Offer {n_offer}"
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
    lines.append(f"{source} 澄清门控设计逻辑验证 (v4)")
    lines.append("=" * 72)
    lines.append(f"time: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"working_dir: {wd}")
    lines.append(f"min_rerank: {clarify_candidate_min_rerank_score()}")
    lines.append(f"direct_rerank_min: {clarify_direct_rerank_min()}")
    lines.append(f"k_required: {clarify_candidate_k()}")
    lines.append(f"pick_strategy: {pick_strategy}")
    lines.append(f"cases: {len(rows)}")
    lines.append("")
    lines.append("验证项:")
    lines.append("  C1 原问可答 <=> final_score 非空")
    lines.append("  C2 gate_outcome 与 final_score 三档符合 v4 设计")
    lines.append("  C3 推荐问 final>direct_min；点选后 LLM 有回答且含 reference")
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
            f"qualifying={r.get('original_llm_chunks')} final={r.get('original_final_score')} "
            f"max_rerank={r.get('original_max_rerank')} "
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
            if key == "c3_picked_answer" and r.get("gate_outcome") not in ("offer",):
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
        f"direct={sum(1 for r in rows if r.get('gate_outcome') == 'direct')} "
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
    write_dumps = args.dump and not args.gate_only

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
                parser_root=pod,
                write_dumps=write_dumps,
            )
            rows.append(row)
            if args.checkpoint:
                with args.out_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            dc = row.get("design_checks") or {}
            print(
                f"  chunks={row.get('original_llm_chunks')} rerank={row.get('original_max_rerank')} "
                f"gate={row.get('gate_outcome')} cands={len(row.get('candidates') or [])} "
                f"design={'OK' if dc.get('all_ok') else 'FAIL'}"
                + (f" dump={row.get('dump_path')}" if row.get("dump_path") else ""),
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
        "--dump",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write logs/query_dumps/*.json per aquery (default: on; skipped with --gate-only)",
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
