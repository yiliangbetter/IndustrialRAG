#!/usr/bin/env python3
"""Batch-test clarify gate on a question set; print scores and trigger status."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

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
from raganything.clarify_gate import (  # noqa: E402
    ClarifyBypass,
    ClarifyRequired,
    classify_query_relevance,
    clarify_threshold_lower,
    clarify_threshold_upper,
    evaluate_clarify_gate,
    probe_query_score,
)

_SOURCE_PRESETS = {
    "shili17": _ROOT / "docs" / "测试例.txt",
    "green8": _ROOT / "data" / "voice_script_green8.json",
}


async def _run(source: Path, *, limit: int, out: Path | None) -> None:
    os.environ.setdefault("RAG_CLARIFY_ENABLED", "1")
    wd = (_ROOT / "data" / "rag_storage").resolve()
    pod = (_ROOT / "data" / "pipeline_parse").resolve()
    rag, _, _ = await rpc._build_rag(wd, pod)
    cases = _load_cases(source, limit=limit)
    lo, hi = clarify_threshold_lower(), clarify_threshold_upper()
    lines: list[str] = []
    lines.append(f"Clarify gate batch test — {source.name}")
    lines.append(f"thresholds: lower={lo} upper={hi}")
    lines.append(f"rule: score >= {hi} -> pass; [{lo}, {hi}) -> case_b; < {lo} -> case_a")
    lines.append(f"cases: {len(cases)}")
    lines.append("")
    clarify_rows: list[tuple] = []
    pass_rows: list[tuple] = []
    for case in cases:
        q = case["query"]
        cid = case.get("id", "?")
        probe = await probe_query_score(rag.lightrag, q)
        sc = probe.get("max_cosine_similarity")
        try:
            score_f = float(sc) if sc is not None else None
        except (TypeError, ValueError):
            score_f = None
        band = classify_query_relevance(score_f, lower=lo, upper=hi).value
        result = await evaluate_clarify_gate(rag.lightrag, q, mode="mix")
        if isinstance(result, ClarifyRequired):
            n_cand = len(result.data.get("candidates") or [])
            unanswerable = bool(result.data.get("unanswerable"))
            triggered = "YES" if not unanswerable else "UNANSWERABLE"
            clarify_rows.append((cid, score_f, band, triggered, n_cand, q))
            line = (
                f"#{cid} score={score_f:.3f} band={band} CLARIFY={triggered} "
                f"candidates={n_cand}"
            )
        else:
            triggered = "NO"
            pass_rows.append((cid, score_f, band, triggered, result.reason, q))
            score_s = f"{score_f:.3f}" if score_f is not None else "None"
            line = f"#{cid} score={score_s} band={band} CLARIFY=NO bypass={result.reason}"
        lines.append(line)
        lines.append(f"  {q}")
        lines.append("")
        print(line)
        print(f"  {q}")
        print()
    lines.append("=" * 60)
    lines.append(f"Summary: clarify triggered {len(clarify_rows)}/{len(cases)}")
    lines.append("")
    lines.append("Triggered:")
    for row in clarify_rows:
        lines.append(f"  id={row[0]} score={row[1]:.3f} band={row[2]} candidates={row[4]} | {row[5]}")
    lines.append("")
    lines.append("Not triggered (pass):")
    for row in pass_rows:
        score_s = f"{row[1]:.3f}" if row[1] is not None else "None"
        lines.append(f"  id={row[0]} score={score_s} band={row[2]} bypass={row[4]} | {row[5]}")
    lines.append("")
    lines.append(f"Score < {hi} (user expectation: should clarify):")
    all_rows = clarify_rows + pass_rows
    for row in sorted(all_rows, key=lambda r: (r[1] is None, r[1] or 0)):
        score_f = row[1]
        if score_f is not None and score_f < hi:
            trig = row[3]
            band = row[2]
            qtxt = row[-1]
            lines.append(f"  score={score_f:.3f} clarify={trig} band={band} | {qtxt}")
    text = "\n".join(lines) + "\n"
    print("=" * 60)
    print(f"Summary: clarify triggered {len(clarify_rows)}/{len(cases)}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Report written: {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        default="green8",
        help="Preset (green8, shili17) or path to question file",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    src = _SOURCE_PRESETS.get(args.source, Path(args.source))
    out = args.out
    if out is None and args.source == "green8":
        out = _ROOT / "logs" / "clarify_gate_green8_report.txt"
    asyncio.run(_run(src.resolve(), limit=args.limit, out=out))


if __name__ == "__main__":
    main()
