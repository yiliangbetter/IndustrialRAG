#!/usr/bin/env python3
"""Score CrossEncoder rerank (mix path, no LLM) for shili17 / green8 / voice29−green8.

Uses ``aquery_data`` + the same hooks as Web-path query (steering, table_matrix boost).
Writes JSONL per case and a UTF-8 summary report with group stats.

Example::

  uv run python scripts/score_rerank_groups.py
  uv run python scripts/score_rerank_groups.py --sets shili17,green8
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
from lightrag import QueryParam

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
load_dotenv(_ROOT / ".env", override=False)

spec = importlib.util.spec_from_file_location(
    "rag_pipeline_parse_graph_chat",
    _ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py",
)
rpc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rpc)

from query_test_cases import _load_cases  # noqa: E402


def _green8_ids() -> set[str]:
    path = _ROOT / "data" / "voice_script_green8.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(c.get("id")) for c in (data.get("cases") or []) if not c.get("skip")}


def _load_voice29_non_green8() -> list[dict[str, Any]]:
    green_ids = _green8_ids()
    rows: list[dict[str, Any]] = []
    for c in _load_cases(_ROOT / "data" / "voice_script_tests.json", limit=0):
        vid = str(c.get("id", ""))
        if vid in green_ids:
            continue
        rows.append(
            {
                "set": "voice29_non_green8",
                "label": f"V{vid}",
                "id": vid,
                "query": c["query"],
                "category": c.get("category", ""),
            }
        )
    return rows


def _load_all_sets(names: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "shili17" in names:
        for c in _load_cases(_ROOT / "docs" / "测试例.txt", limit=0):
            rows.append(
                {
                    "set": "shili17",
                    "label": f"Q{c['id']}",
                    "id": str(c["id"]),
                    "query": c["query"],
                    "category": "",
                }
            )
    if "green8" in names:
        for i, c in enumerate(
            _load_cases(_ROOT / "data" / "voice_script_green8.json", limit=0), 1
        ):
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
    if "voice29_non_green8" in names:
        rows.extend(_load_voice29_non_green8())
    return rows


def _summarize_doc(doc: dict[str, Any]) -> dict[str, Any]:
    content = str(doc.get("content") or doc.get("text") or "")
    return {
        "chunk_id": doc.get("chunk_id") or doc.get("id"),
        "file_path": doc.get("file_path") or doc.get("source"),
        "rerank_score": doc.get("rerank_score"),
        "vector_score": doc.get("score"),
        "preview": content.replace("\n", " ")[:160],
    }


class _RerankCapture:
    merge_pool: list[dict[str, Any]] = []
    post_ce: list[dict[str, Any]] = []
    post_doc_filter: list[dict[str, Any]] = []
    final_llm: list[dict[str, Any]] = []


_CAPTURE = _RerankCapture()


def _install_rerank_recorders() -> tuple[Any, Any]:
    import lightrag.utils as ut
    import lightrag.operate as op

    native_rerank = ut.apply_rerank_if_enabled
    native_process = ut.process_chunks_unified

    async def _recording_rerank(
        query: str,
        retrieved_docs: list[dict],
        global_config: dict,
        enable_rerank: bool = True,
        top_n: int | None = None,
    ) -> list[dict]:
        _CAPTURE.merge_pool = [_summarize_doc(d) for d in (retrieved_docs or [])]
        docs = await native_rerank(
            query, retrieved_docs, global_config, enable_rerank, top_n
        )
        _CAPTURE.post_ce = [_summarize_doc(d) for d in (docs or [])]
        return docs

    async def _recording_process(
        query: str,
        unique_chunks: list[dict],
        query_param: Any,
        global_config: dict,
        source_type: str = "mixed",
        chunk_token_limit: int | None = None,
    ) -> list[dict]:
        if not _CAPTURE.merge_pool:
            _CAPTURE.merge_pool = [_summarize_doc(d) for d in (unique_chunks or [])]
        chunks = await native_process(
            query,
            unique_chunks,
            query_param,
            global_config,
            source_type,
            chunk_token_limit,
        )
        _CAPTURE.final_llm = [_summarize_doc(d) for d in (chunks or [])]
        return chunks

    ut.apply_rerank_if_enabled = _recording_rerank  # type: ignore[method-assign]
    ut.process_chunks_unified = _recording_process  # type: ignore[method-assign]
    op.process_chunks_unified = _recording_process  # type: ignore[method-assign]
    return native_rerank, native_process


def _restore_rerank_recorders(orig_rerank: Any, orig_process: Any) -> None:
    import lightrag.utils as ut
    import lightrag.operate as op

    ut.apply_rerank_if_enabled = orig_rerank  # type: ignore[method-assign]
    ut.process_chunks_unified = orig_process  # type: ignore[method-assign]
    op.process_chunks_unified = orig_process  # type: ignore[method-assign]


def _rerank_scores(rows: list[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for row in rows:
        raw = row.get("rerank_score")
        if raw is None:
            continue
        try:
            out.append(float(raw))
        except (TypeError, ValueError):
            continue
    return out


def _metrics_from_capture(*, min_thr: float) -> dict[str, Any]:
    ce = _rerank_scores(_CAPTURE.post_ce)
    fin = _rerank_scores(_CAPTURE.final_llm)
    filt = _rerank_scores(_CAPTURE.post_doc_filter or _CAPTURE.post_ce)

    def _pack(scores: list[float]) -> dict[str, Any]:
        if not scores:
            return {
                "count": 0,
                "max": None,
                "min": None,
                "avg": None,
                "qualifying": 0,
            }
        return {
            "count": len(scores),
            "max": max(scores),
            "min": min(scores),
            "avg": round(statistics.mean(scores), 4),
            "qualifying": sum(1 for s in scores if s >= min_thr),
        }

    top_ce = _CAPTURE.post_ce[0] if _CAPTURE.post_ce else {}
    top_fin = _CAPTURE.final_llm[0] if _CAPTURE.final_llm else {}
    return {
        "merge_pool_count": len(_CAPTURE.merge_pool),
        "post_ce": _pack(ce),
        "post_doc_filter": _pack(filt),
        "final_llm": _pack(fin),
        "max_rerank_ce": ce[0] if ce else None,
        "max_rerank_final": max(fin) if fin else None,
        "top_ce_chunk_id": top_ce.get("chunk_id"),
        "top_final_chunk_id": top_fin.get("chunk_id"),
        "top_ce_preview": top_ce.get("preview"),
    }


def _stats(vals: list[float]) -> dict[str, float | int]:
    if not vals:
        return {"n": 0, "min": 0.0, "max": 0.0, "avg": 0.0, "med": 0.0}
    s = sorted(vals)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {
        "n": n,
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
        "avg": round(statistics.mean(s), 4),
        "med": round(float(med), 4),
    }


def _build_report(rows: list[dict[str, Any]], *, config: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(
        "CrossEncoder rerank scores — shili17 / green8 / voice29−green8 (no LLM)"
    )
    lines.append("=" * 78)
    lines.append(f"time: {datetime.now(timezone.utc).isoformat()}")
    lines.append(
        f"config: mode={config.get('mode')} MIN_RERANK_SCORE={config.get('min_rerank_score')} "
        f"CHUNK_TOP_K={config.get('chunk_top_k')}"
    )
    lines.append(f"cases: {len(rows)}")
    lines.append("")

    by_set: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_set.setdefault(str(row["case"]["set"]), []).append(row)

    lines.append("## Group summary (max_rerank_ce = top score after CrossEncoder)")
    lines.append("")
    lines.append(
        "| set | n | CE max min | CE max max | CE max avg | CE max med | "
        "final max avg | CE qualifying avg |"
    )
    lines.append(
        "|-----|---|------------|------------|------------|------------|---------------|-------------------|"
    )
    for set_name in ("shili17", "green8", "voice29_non_green8"):
        group = by_set.get(set_name) or []
        if not group:
            continue
        ce_maxes = [
            float(m["max_rerank_ce"])
            for r in group
            if (m := r.get("metrics") or {}).get("max_rerank_ce") is not None
        ]
        fin_maxes = [
            float(m["max_rerank_final"])
            for r in group
            if (m := r.get("metrics") or {}).get("max_rerank_final") is not None
        ]
        qual = [
            int((m.get("post_ce") or {}).get("qualifying") or 0)
            for r in group
            if (m := r.get("metrics"))
        ]
        cs = _stats(ce_maxes)
        fs = _stats(fin_maxes)
        qavg = round(statistics.mean(qual), 2) if qual else 0.0
        lines.append(
            f"| {set_name} | {cs['n']} | {cs['min']} | {cs['max']} | {cs['avg']} | {cs['med']} | "
            f"{fs['avg']} | {qavg} |"
        )
    lines.append("")

    lines.append("## Per-question (CE max rerank)")
    lines.append("")
    for set_name in ("shili17", "green8", "voice29_non_green8"):
        group = by_set.get(set_name) or []
        if not group:
            continue
        lines.append(f"### {set_name}")
        for r in sorted(
            group,
            key=lambda x: float((x.get("metrics") or {}).get("max_rerank_ce") or 0),
        ):
            c = r["case"]
            m = r.get("metrics") or {}
            ce = m.get("max_rerank_ce")
            fin = m.get("max_rerank_final")
            q = (c.get("query") or "")[:50]
            lines.append(
                f"  {c.get('label'):6s} CE={ce!s:>8} final={fin!s:>8} "
                f"qual={(m.get('post_ce') or {}).get('qualifying', 0)}  {q}"
            )
        lines.append("")

    # Overlap hints
    lines.append("## Separation hints (CE max)")
    pools: dict[str, list[float]] = {}
    for set_name, group in by_set.items():
        pools[set_name] = [
            float(m["max_rerank_ce"])
            for r in group
            if (m := r.get("metrics") or {}).get("max_rerank_ce") is not None
        ]
    if pools.get("shili17") and pools.get("green8"):
        s, g = pools["shili17"], pools["green8"]
        lines.append(
            f"  shili17 [{min(s):.3f}, {max(s):.3f}]  vs  green8 [{min(g):.3f}, {max(g):.3f}]"
        )
        overlap_lo = max(min(s), min(g))
        overlap_hi = min(max(s), max(g))
        if overlap_lo <= overlap_hi:
            lines.append(f"  range overlap: [{overlap_lo:.3f}, {overlap_hi:.3f}]")
        else:
            lines.append("  ranges do not overlap")
    lines.append("")
    return "\n".join(lines)


async def _async_main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sets",
        default="shili17,green8,voice29_non_green8",
        help="Comma-separated: shili17, green8, voice29_non_green8",
    )
    p.add_argument("--limit", type=int, default=0, help="Max cases total (0=all)")
    p.add_argument("--out-jsonl", type=Path, default=None)
    p.add_argument("--out-report", type=Path, default=None)
    args = p.parse_args()

    names = {x.strip() for x in args.sets.split(",") if x.strip()}
    cases = _load_all_sets(names)
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No cases loaded")

    wd = Path(
        os.getenv("RAG_WEB_WORKING_DIR") or (_ROOT / "data" / "rag_storage")
    ).resolve()
    pod = Path(
        os.getenv("RAG_WEB_PARSER_OUTPUT_DIR") or (_ROOT / "data" / "pipeline_parse")
    ).resolve()
    mode = os.getenv("RAG_QUERY_MODE", "mix")
    min_thr = float(os.getenv("MIN_RERANK_SCORE") or "0.28")

    orig_rerank, orig_process = _install_rerank_recorders()
    rag, _, _ = await rpc._build_rag(wd, pod)
    config = {
        "mode": mode,
        "min_rerank_score": os.getenv("MIN_RERANK_SCORE"),
        "chunk_top_k": os.getenv("CHUNK_TOP_K"),
        "working_dir": str(wd),
    }

    from query_progress_hooks import get_llm_input_chunks, query_progress_hooks  # noqa: WPS433

    import query_doc_steering as qds

    orig_filter = qds.filter_retrieved_docs_by_query

    def _filter_and_record(q: str, docs: list[dict]) -> list[dict]:
        kept = orig_filter(q, docs)
        out = kept if kept else docs
        _CAPTURE.post_doc_filter = [_summarize_doc(d) for d in out]
        return out

    qds.filter_retrieved_docs_by_query = _filter_and_record  # type: ignore[method-assign]

    results: list[dict[str, Any]] = []
    try:
        for idx, case in enumerate(cases, start=1):
            global _CAPTURE
            _CAPTURE = _RerankCapture()
            q = case["query"]
            print(
                f"[{idx}/{len(cases)}] {case['set']} {case['label']} {q[:48]}",
                flush=True,
            )
            param = QueryParam(
                mode=mode,
                only_need_context=True,
                enable_rerank=True,
                **rpc._query_extras_from_env(q),
            )
            async with query_progress_hooks():
                await rag.lightrag.aquery_data(q, param)
                hook_final = get_llm_input_chunks()
                if hook_final and not _CAPTURE.final_llm:
                    _CAPTURE.final_llm = [_summarize_doc(d) for d in hook_final]
            metrics = _metrics_from_capture(min_thr=min_thr)
            row = {
                "case": case,
                "query": q,
                "metrics": metrics,
                "config": config,
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }
            results.append(row)
            print(
                f"  merge={metrics['merge_pool_count']} "
                f"CE_max={metrics['max_rerank_ce']} "
                f"final_max={metrics['max_rerank_final']} "
                f"qual={metrics['post_ce'].get('qualifying')}",
                flush=True,
            )
    finally:
        qds.filter_retrieved_docs_by_query = orig_filter  # type: ignore[method-assign]
        _restore_rerank_recorders(orig_rerank, orig_process)
        await rag.finalize_storages()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = args.out_jsonl or (_ROOT / "logs" / f"rerank_groups_{stamp}.jsonl")
    report_path = args.out_report or (_ROOT / "logs" / f"rerank_groups_{stamp}.txt")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path.write_text(_build_report(results, config=config), encoding="utf-8")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {report_path}")


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
