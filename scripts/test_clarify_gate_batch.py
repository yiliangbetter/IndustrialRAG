#!/usr/bin/env python3
"""Batch-test clarify gate v4 on a question set; print final_score and gate outcome."""

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
    clarify_direct_rerank_min,
    evaluate_clarify_gate,
    probe_llm_retrieval,
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
    direct_min = clarify_direct_rerank_min()
    lines: list[str] = []
    lines.append(f"Clarify gate v4 batch — {source.name}")
    lines.append(f"CLARIFY_DIRECT_RERANK_MIN={direct_min}")
    lines.append("rule: final=None reject; final>min direct; else offer/reject")
    lines.append(f"cases: {len(cases)}")
    lines.append("")

    for case in cases:
        q = case["query"]
        cid = case.get("id", "?")
        probe = await probe_llm_retrieval(rag.lightrag, q, mode="mix")
        final_score = probe.get("final_score")
        result = await evaluate_clarify_gate(rag.lightrag, q, mode="mix")
        if isinstance(result, ClarifyRequired):
            outcome = result.gate_outcome
            reason = result.data.get("gate_reason")
            n_cand = len(result.data.get("candidates") or [])
            lines.append(
                f"#{cid} final={final_score} {outcome}/{reason} candidates={n_cand}  {q[:50]}"
            )
        elif isinstance(result, ClarifyBypass):
            lines.append(
                f"#{cid} final={final_score} bypass={result.reason}  {q[:50]}"
            )
        else:
            lines.append(f"#{cid} final={final_score} unknown  {q[:50]}")

    text = "\n".join(lines) + "\n"
    print(text)
    if out:
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")

    await rag.finalize_storages()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="shili17", choices=sorted(_SOURCE_PRESETS))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    source = _SOURCE_PRESETS[args.source]
    asyncio.run(_run(source, limit=args.limit, out=args.out))


if __name__ == "__main__":
    main()
