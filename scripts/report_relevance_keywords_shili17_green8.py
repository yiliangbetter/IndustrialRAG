#!/usr/bin/env python3
"""Score shili17 + green8 (25 questions): query / high / low keywords + cosine.

Writes JSONL + human-readable report under logs/.

Example::

  uv run python scripts/report_relevance_keywords_shili17_green8.py
  uv run python scripts/report_relevance_keywords_shili17_green8.py -w data/rag_storage
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

load_dotenv(_ROOT / ".env", override=False)
if (os.getenv("HF_EMBED_OFFLINE") or "").strip().lower() in ("1", "true", "yes"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

spec = importlib.util.spec_from_file_location(
    "rag_pipeline_parse_graph_chat", _ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
)
rpc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rpc)

from score_query_relevance import _load_cases  # noqa: E402
from raganything.naive_relevance import (  # noqa: E402
    format_relevance_report,
    primary_relevance_score,
    score_naive_relevance,
)


def _load_all_cases() -> list[dict[str, Any]]:
    shili = _load_cases(_ROOT / "docs" / "测试例.txt", limit=0)
    green = _load_cases(_ROOT / "data" / "voice_script_green8.json", limit=0)
    rows: list[dict[str, Any]] = []
    for c in shili:
        rows.append(
            {
                "set": "shili17",
                "label": f"Q{c['id']}",
                "id": str(c["id"]),
                "query": c["query"],
                "category": "",
            }
        )
    for i, c in enumerate(green, start=1):
        rows.append(
            {
                "set": "green8",
                "label": f"G{i}",
                "id": str(c.get("id", i)),
                "voice_id": str(c.get("id", i)),
                "query": c["query"],
                "category": c.get("category", ""),
            }
        )
    return rows


def _fmt_keywords(words: list[str] | None) -> str:
    if not words:
        return "（空）"
    return ", ".join(str(w) for w in words)


def _score_block(payload: dict[str, Any], leg: str) -> dict[str, Any]:
    scores = payload.get("scores") or {}
    section = scores.get(leg) or {}
    return {
        "score": primary_relevance_score(payload, which=leg),
        "search_text": section.get("text") or "",
        "hits_above_threshold": section.get("hits_above_threshold"),
        "top_hits": section.get("top_hits") or [],
        "reason": section.get("reason"),
    }


def _stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "count": len(vals),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "avg": round(statistics.mean(vals), 4),
    }


def _build_report(rows: list[dict[str, Any]], *, thresholds: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 76)
    lines.append("Naive Relevance — shili17（17）+ green8（8）检索关键词与分数")
    lines.append("=" * 76)
    lines.append(f"time: {datetime.now(timezone.utc).isoformat()}")
    lines.append(
        f"thresholds: cosine={thresholds.get('cosine_threshold')} "
        f"probe_top_k={thresholds.get('probe_top_k')}"
    )
    lines.append(f"cases: {len(rows)}")
    lines.append("")

    for set_name in ("shili17", "green8"):
        subset = [r for r in rows if r["set"] == set_name]
        if not subset:
            continue
        q_vals = [r["scores"]["query"] for r in subset if r["scores"]["query"] is not None]
        h_vals = [r["scores"]["high"] for r in subset if r["scores"]["high"] is not None]
        l_vals = [r["scores"]["low"] for r in subset if r["scores"]["low"] is not None]
        qs, hs, ls = _stats(q_vals), _stats(h_vals), _stats(l_vals)
        lines.append("-" * 76)
        lines.append(f"【{set_name}】汇总（{len(subset)} 题）")
        lines.append("-" * 76)
        lines.append(
            f"query  avg={qs['avg']:.3f}  min={qs['min']:.3f}  max={qs['max']:.3f}"
        )
        lines.append(
            f"high   avg={hs['avg']:.3f}  min={hs['min']:.3f}  max={hs['max']:.3f}"
        )
        lines.append(
            f"low    avg={ls['avg']:.3f}  min={ls['min']:.3f}  max={ls['max']:.3f}"
        )
        lines.append("")
        lines.append(
            f"{'标签':<6} {'id':>4}  {'query':>6}  {'high':>6}  {'low':>6}  题目"
        )
        for r in subset:
            sc = r["scores"]
            q_s = f"{sc['query']:.3f}" if sc["query"] is not None else "  —  "
            h_s = f"{sc['high']:.3f}" if sc["high"] is not None else "  —  "
            l_s = f"{sc['low']:.3f}" if sc["low"] is not None else "  —  "
            vid = r.get("voice_id", r["id"])
            id_col = f"#{vid}" if set_name == "green8" else f" {r['id']:>2}"
            lines.append(
                f"{r['label']:<6}{id_col:>4}  {q_s:>6}  {h_s:>6}  {l_s:>6}  {r['query']}"
            )
        lines.append("")

    lines.append("-" * 76)
    lines.append("【全量 25 题】关键词一览")
    lines.append("-" * 76)
    lines.append(
        f"{'标签':<6} {'set':<8}  {'query':>6}  {'high':>6}  {'low':>6}  high keywords"
    )
    for r in rows:
        sc = r["scores"]
        q_s = f"{sc['query']:.3f}" if sc["query"] is not None else "  —  "
        h_s = f"{sc['high']:.3f}" if sc["high"] is not None else "  —  "
        l_s = f"{sc['low']:.3f}" if sc["low"] is not None else "  —  "
        hl = _fmt_keywords(r["keywords"].get("high_level"))
        if len(hl) > 48:
            hl = hl[:45] + "…"
        lines.append(
            f"{r['label']:<6} {r['set']:<8}  {q_s:>6}  {h_s:>6}  {l_s:>6}  {hl}"
        )
    lines.append("")

    lines.append("-" * 76)
    lines.append("逐条明细")
    lines.append("-" * 76)
    for r in rows:
        lines.append("")
        extra = ""
        if r.get("category"):
            extra = f" [{r['category']}]"
        if r["set"] == "green8":
            lines.append(f"[{r['label']} / voice #{r.get('voice_id', r['id'])}{extra}]")
        else:
            lines.append(f"[{r['label']}{extra}]")
        lines.append(format_relevance_report(r["payload"]).rstrip())
    lines.append("")
    return "\n".join(lines) + "\n"


async def _main(args: argparse.Namespace) -> None:
    wd = args.working_dir.expanduser().resolve()
    pod = args.parser_output_dir
    if pod is None:
        raw = (os.getenv("RAG_WEB_PARSER_OUTPUT_DIR") or "data/pipeline_parse").strip()
        pod = (_ROOT / raw).resolve()
    else:
        pod = pod.expanduser().resolve()

    cases = _load_all_cases()
    rag, _, _ = await rpc._build_rag(wd, pod)

    results: list[dict[str, Any]] = []
    thresholds: dict[str, Any] = {}

    try:
        for idx, case in enumerate(cases, start=1):
            q = case["query"]
            print(f"[{idx}/{len(cases)}] {case['label']} {q}", flush=True)
            payload = await score_naive_relevance(rag.lightrag, q)
            if not thresholds and isinstance(payload.get("thresholds"), dict):
                thresholds = dict(payload["thresholds"])
            payload["scored_at"] = datetime.now(timezone.utc).isoformat()
            kw = payload.get("keywords") or {}
            row = {
                **case,
                "payload": payload,
                "keywords": {
                    "high_level": list(kw.get("high_level") or []),
                    "low_level": list(kw.get("low_level") or []),
                },
                "scores": {
                    "query": primary_relevance_score(payload, which="query"),
                    "high": primary_relevance_score(payload, which="high_level"),
                    "low": primary_relevance_score(payload, which="low_level"),
                },
                "legs": {
                    "query": _score_block(payload, "query"),
                    "high_level": _score_block(payload, "high_level"),
                    "low_level": _score_block(payload, "low_level"),
                },
            }
            results.append(row)
            sc = row["scores"]
            print(
                f"  q={sc['query']} h={sc['high']} l={sc['low']}",
                flush=True,
            )
            print(f"  high: {row['keywords']['high_level']}", flush=True)
            print(f"  low:  {row['keywords']['low_level']}", flush=True)
    finally:
        await rag.finalize_storages()

    report = _build_report(results, thresholds=thresholds)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as fh:
        for row in results:
            export = {k: v for k, v in row.items() if k != "payload"}
            export["thresholds"] = thresholds
            fh.write(json.dumps(export, ensure_ascii=False) + "\n")
    args.out_report.write_text(report, encoding="utf-8")
    print(f"\nJSONL: {args.out_jsonl}")
    print(f"Report: {args.out_report}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-w", "--working-dir", type=Path, default=_ROOT / "data" / "rag_storage")
    p.add_argument("--parser-output-dir", type=Path, default=None)
    p.add_argument(
        "--out-jsonl",
        type=Path,
        default=_ROOT / "logs" / "relevance_shili17_green8_keywords.jsonl",
    )
    p.add_argument(
        "--out-report",
        type=Path,
        default=_ROOT / "logs" / "relevance_shili17_green8_keywords_report.txt",
    )
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()
