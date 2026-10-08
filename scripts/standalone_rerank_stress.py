#!/usr/bin/env python3
"""Standalone CrossEncoder rerank stress test (no LightRAG / clarify gate).

Simulates consecutive mix-mode rerank: one query x ~70 document chunks per round,
as in clarify probe (merge pool before CE top-N). Records per-round latency,
per-batch latency, and GPU memory. Use to compare Qwen3 vs BGE or diagnose slowdown.

Example::

  uv run python scripts/standalone_rerank_stress.py --model Qwen/Qwen3-Reranker-0.6B --rounds 16
  uv run python scripts/standalone_rerank_stress.py --model BAAI/bge-reranker-v2-m3 --rounds 16
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# green8 原问 + 部分推荐问（模拟 bench 连续 probe）
_DEFAULT_QUERIES = [
    "漏胶",
    "仿形效果不好",
    "气压报警？",
    "靠板上限？",
    "三相电异常",
    "变频器异常报警",
    "未检测到工作（板材）",
    "未检测到工件",
    "封边机出现漏胶现象的常见原因有哪些？",
    "如何检查电机三相电是否缺相？",
    "进给变频器“UF1”出现异常报警时应该如何排查故障原因？",
    "设备报警提示未检测到工件是什么原因导致的？",
    "封边机涂胶系统发生漏胶时应该如何排查和处理？",
    "涂胶电机变频器“UF2”显示故障代码或异常报警时怎么处理？",
    "机器提示“未检测到工作（板材）”并出现警告时，如何通过故障复位按钮解除报警？",
    "检测开关未检测到工件时应该如何排查故障？",
]


def _hub_snapshot(repo_id: str, hf_home: Path) -> Path | None:
    org, name = repo_id.split("/", 1)
    repo_dir = hf_home / "hub" / f"models--{org}--{name}"
    if not repo_dir.is_dir():
        return None
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        rev = ref.read_text(encoding="utf-8").strip()
        if rev:
            snap = repo_dir / "snapshots" / rev
            if (snap / "config.json").is_file():
                return snap
    snaps = sorted(
        (p for p in (repo_dir / "snapshots").iterdir() if (p / "config.json").is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return snaps[0] if snaps else None


def _nvidia_used_mb() -> float | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        parts = [x.strip() for x in out.strip().split(",")]
        return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        return None


def _torch_mem() -> dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 1),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
        }
    except Exception:
        return {}


def _load_docs(
    *,
    chunks_json: Path | None,
    pool_size: int,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    docs: list[str] = []
    if chunks_json and chunks_json.is_file():
        raw = json.loads(chunks_json.read_text(encoding="utf-8"))
        items = list(raw.values()) if isinstance(raw, dict) else list(raw)
        for item in items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("content") or item.get("text") or "").strip()
            if len(text) >= min_chars:
                docs.append(text[:max_chars])
        # 接近真实 merge pool：中长 chunk 为主，少量长表
        docs.sort(key=len)
        if len(docs) > pool_size:
            step = max(1, len(docs) // pool_size)
            docs = [docs[i] for i in range(0, len(docs), step)][:pool_size]
    if len(docs) < pool_size:
        # 合成兜底：模拟电气手册段落长度（200–1200 字）
        base = (
            "检查输入气源压力是否达到0.6MPa以上；检查电机三相电是否缺相；"
            "观察进给变频器UF1屏幕是否显示故障代码；按下黄色故障复位按钮；"
            "排查输送带电机转子是否卡死；检查相序保护开关X1.7与急停保护X1.6。"
        )
        while len(docs) < pool_size:
            n = len(docs)
            filler = base * (2 + (n % 5))
            docs.append(filler[: 200 + (n * 37) % 900])
    return docs[:pool_size]


def _predict_batched(
    ce: Any,
    pairs: list[tuple[str, str]],
    *,
    batch_size: int,
) -> tuple[list[float], list[float]]:
    """Mirror sentence-transformers internal batching; return scores + batch seconds."""
    batch_times: list[float] = []
    chunks: list[Any] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        t0 = time.perf_counter()
        scores = ce.predict(batch, show_progress_bar=False)
        batch_times.append(round(time.perf_counter() - t0, 3))
        if hasattr(scores, "tolist"):
            chunks.extend(scores.tolist())
        else:
            chunks.extend([float(s) for s in scores])
    return [float(s) for s in chunks], batch_times


def _slug_model(model: str) -> str:
    return model.split("/")[-1].replace(".", "_").lower()


def _format_report(payload: dict[str, Any], *, title: str = "Standalone rerank stress") -> str:
    s = payload.get("summary") or {}
    lines = [
        "=" * 72,
        title,
        "=" * 72,
        f"time: {payload.get('time', '')}",
        f"model: {s.get('model', '')}",
        f"pool_size: {s.get('pool_size')}  batch_size: {s.get('batch_size')}  rounds: {s.get('rounds')}",
        f"max_chunk_chars: {s.get('max_chunk_chars', '?')}",
        f"load_s: {s.get('load_s')}",
        "",
        "汇总:",
        f"  total_s: min={s.get('total_s_min')} med={s.get('total_s_median')} max={s.get('total_s_max')}",
        f"  round1={s.get('round1_s')}  round2={s.get('round2_s')}  "
        f"slowdown_r2/r1={s.get('slowdown_round2_vs_1')}  max/r1={s.get('slowdown_max_vs_round1')}",
    ]
    if s.get("first_batch_s_median") is not None:
        lines.append(
            f"  first_batch_s: med={s.get('first_batch_s_median')} max={s.get('first_batch_s_max')}"
        )
    if s.get("nvidia_peak_mb"):
        lines.append(f"  nvidia_peak_mb: {s.get('nvidia_peak_mb')}")
    lines.extend(["", "逐轮:", "-" * 72])
    for r in payload.get("rounds") or []:
        fb = r.get("first_batch_s")
        vram = r.get("nvidia_used_mb_after")
        vram_txt = f" vram={vram:.0f}MB" if vram else ""
        fb_txt = f" 1st_batch={fb}s" if fb is not None else ""
        lines.append(
            f"  [{r.get('round'):2d}] {r.get('total_s'):7.2f}s top={r.get('top_score')}"
            f"{fb_txt}{vram_txt} | {(r.get('query') or '')[:40]}"
        )
    lines.append("-" * 72)
    return "\n".join(lines) + "\n"


def _format_compare_report(comparison: dict[str, Any]) -> str:
    lines = [
        "=" * 72,
        "Rerank stress compare (BGE vs Qwen, sequential, isolated loads)",
        "=" * 72,
        f"time: {comparison.get('time', '')}",
        f"rounds: {comparison.get('rounds')}  pool_size: {comparison.get('pool_size')}  "
        f"batch_size: {comparison.get('batch_size')}",
        "",
        f"{'model':<32} {'r1(s)':>8} {'r2(s)':>8} {'med(s)':>8} {'max(s)':>8} {'r2/r1':>8} {'max/r1':>8}",
        "-" * 72,
    ]
    for row in comparison.get("models") or []:
        sm = row.get("summary") or {}
        lines.append(
            f"{sm.get('model', '')[:32]:<32} "
            f"{sm.get('round1_s', 0):>8} "
            f"{sm.get('round2_s') or '-':>8} "
            f"{sm.get('total_s_median', 0):>8} "
            f"{sm.get('total_s_max', 0):>8} "
            f"{sm.get('slowdown_round2_vs_1') or '-':>8} "
            f"{sm.get('slowdown_max_vs_round1') or '-':>8}"
        )
    lines.extend(["", "解读:", comparison.get("interpretation", ""), ""])
    for name, pl in (comparison.get("payloads") or {}).items():
        lines.append(_format_report(pl, title=f"Detail: {name}"))
    return "\n".join(lines)


def _run_stress(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from sentence_transformers import CrossEncoder

    hf_home = Path(args.hf_home).resolve()
    snap = _hub_snapshot(args.model, hf_home)
    if snap is None:
        raise SystemExit(f"No local snapshot for {args.model!r} under {hf_home}")

    load_path = str(snap.resolve())
    print(f"Model: {args.model} -> {load_path}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    docs = _load_docs(
        chunks_json=Path(args.chunks_json) if args.chunks_json else None,
        pool_size=args.pool_size,
        min_chars=args.min_chunk_chars,
        max_chars=args.max_chunk_chars,
    )
    print(f"Document pool: {len(docs)} chunks, chars min={min(len(d) for d in docs)} max={max(len(d) for d in docs)}", flush=True)

    queries = list(_DEFAULT_QUERIES)
    while len(queries) < args.rounds:
        queries.extend(_DEFAULT_QUERIES)
    queries = queries[: args.rounds]

    t_load = time.perf_counter()
    ce = CrossEncoder(load_path, local_files_only=True)
    load_s = round(time.perf_counter() - t_load, 2)
    print(f"CrossEncoder load: {load_s}s", flush=True)

    rounds_out: list[dict[str, Any]] = []
    for i, query in enumerate(queries, 1):
        pairs = [(query, d) for d in docs]
        if args.empty_cache_each and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        smi_before = _nvidia_used_mb()
        mem_before = _torch_mem()

        t0 = time.perf_counter()
        if args.per_batch_timing:
            scores, batch_times = _predict_batched(ce, pairs, batch_size=args.batch_size)
        else:
            scores = ce.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)
            batch_times = []
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            scores = [float(s) for s in scores]

        total_s = round(time.perf_counter() - t0, 3)
        smi_after = _nvidia_used_mb()
        mem_after = _torch_mem()

        top = max(scores) if scores else None
        row = {
            "round": i,
            "query": query,
            "pair_count": len(pairs),
            "total_s": total_s,
            "batch_times_s": batch_times,
            "first_batch_s": batch_times[0] if batch_times else None,
            "rest_batches_max_s": max(batch_times[1:], default=None) if len(batch_times) > 1 else None,
            "top_score": round(top, 4) if top is not None else None,
            "torch_mem_before": mem_before,
            "torch_mem_after": mem_after,
        }
        if smi_before:
            row["nvidia_used_mb_before"] = smi_before[0]
            row["nvidia_total_mb"] = smi_before[1]
            row["gpu_util_before"] = smi_before[2]
        if smi_after:
            row["nvidia_used_mb_after"] = smi_after[0]
            row["gpu_util_after"] = smi_after[2]
        rounds_out.append(row)

        bt = f" batches={batch_times}" if batch_times else ""
        smi = ""
        if smi_after:
            smi = f" vram={smi_after[0]:.0f}MB util={smi_after[2]:.0f}%"
        print(f"  [{i:2d}/{args.rounds}] {total_s:7.2f}s top={row['top_score']}{bt}{smi} | {query[:28]}", flush=True)

        if args.gc_each:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    totals = [r["total_s"] for r in rounds_out]
    first_batch = [r["first_batch_s"] for r in rounds_out if r.get("first_batch_s") is not None]
    summary = {
        "model": args.model,
        "load_path": load_path,
        "pool_size": len(docs),
        "max_chunk_chars": args.max_chunk_chars,
        "rounds": args.rounds,
        "batch_size": args.batch_size,
        "empty_cache_each": args.empty_cache_each,
        "gc_each": args.gc_each,
        "load_s": load_s,
        "total_s_min": min(totals),
        "total_s_median": round(statistics.median(totals), 2),
        "total_s_max": max(totals),
        "round1_s": totals[0],
        "round2_s": totals[1] if len(totals) > 1 else None,
        "slowdown_round2_vs_1": round(totals[1] / totals[0], 2) if len(totals) > 1 and totals[0] > 0 else None,
        "slowdown_max_vs_round1": round(max(totals) / totals[0], 2) if totals[0] > 0 else None,
        "nvidia_peak_mb": max(
            (r.get("nvidia_used_mb_after") or 0 for r in rounds_out),
            default=0,
        ),
    }
    if first_batch:
        summary["first_batch_s_median"] = round(statistics.median(first_batch), 2)
        summary["first_batch_s_max"] = max(first_batch)

    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "rounds": rounds_out,
    }


def _run_compare(args: argparse.Namespace) -> dict[str, Any]:
    """Run each model in a fresh process (unload GPU between models)."""
    models = [
        m.strip()
        for m in (args.compare_models or "BAAI/bge-reranker-v2-m3,Qwen/Qwen3-Reranker-0.6B").split(",")
        if m.strip()
    ]
    payloads: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []

    for model in models:
        print(f"\n{'=' * 60}\n>>> {model}\n{'=' * 60}", flush=True)
        sub = argparse.Namespace(**{**vars(args), "model": model})
        pl = _run_stress(sub)
        key = _slug_model(model)
        payloads[key] = pl
        summaries.append(pl["summary"])
        # 释放 GPU，避免上一模型影响下一模型
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        time.sleep(2)

    bge = summaries[0] if summaries else {}
    qwen = summaries[1] if len(summaries) > 1 else {}
    interp_lines = []
    if bge and qwen:
        bge_r2 = bge.get("slowdown_round2_vs_1")
        qwen_r2 = qwen.get("slowdown_round2_vs_1")
        if qwen_r2 and bge_r2 and qwen_r2 > max(3.0, bge_r2 * 3):
            interp_lines.append(
                f"- Qwen round2/round1={qwen_r2}x vs BGE {bge_r2}x → Qwen 同进程重复 predict 明显劣化"
            )
        elif qwen.get("total_s_max", 0) > bge.get("total_s_max", 1) * 5:
            interp_lines.append(
                f"- Qwen max={qwen.get('total_s_max')}s vs BGE max={bge.get('total_s_max')}s → 峰值耗时差显著"
            )
        else:
            interp_lines.append("- 两模型连续 rerank 耗时接近，未观察到 Qwen 特有劣化")
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "rounds": args.rounds,
        "pool_size": args.pool_size,
        "batch_size": args.batch_size,
        "max_chunk_chars": args.max_chunk_chars,
        "models": [{"name": k, "summary": payloads[k]["summary"]} for k in payloads],
        "payloads": payloads,
        "interpretation": "\n".join(interp_lines) if interp_lines else "(see per-model tables)",
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
    p.add_argument("--hf-home", default=os.getenv("HF_HOME") or str(root / "data" / "models"))
    p.add_argument(
        "--chunks-json",
        default=str(root / "data" / "rag_storage" / "kv_store_text_chunks.json"),
        help="Real chunk texts for realistic lengths (set '' to use synthetic only)",
    )
    p.add_argument("--pool-size", type=int, default=70, help="Pairs per round (~merge pool)")
    p.add_argument("--min-chunk-chars", type=int, default=80)
    p.add_argument(
        "--max-chunk-chars",
        type=int,
        default=1200,
        help="Cap doc length (mix pool 文本经灌库切分，通常远小于全文)",
    )
    p.add_argument("--rounds", type=int, default=16, help="Consecutive predict calls")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--per-batch-timing", action="store_true", default=True)
    p.add_argument("--no-per-batch-timing", action="store_false", dest="per_batch_timing")
    p.add_argument("--empty-cache-each", action="store_true", help="torch.cuda.empty_cache() before each round")
    p.add_argument("--gc-each", action="store_true")
    p.add_argument(
        "--compare",
        action="store_true",
        help="Run BGE + Qwen sequentially; write compare JSON + TXT report",
    )
    p.add_argument(
        "--compare-models",
        default="BAAI/bge-reranker-v2-m3,Qwen/Qwen3-Reranker-0.6B",
        help="Comma-separated models for --compare",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--out-report", type=Path, default=None, help="Human-readable .txt (default: same stem as JSON)")
    args = p.parse_args()
    if args.chunks_json == "":
        args.chunks_json = None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_logs = root / "logs"

    if args.compare:
        comparison = _run_compare(args)
        out_json = args.out or (root_logs / f"standalone_rerank_compare_{stamp}.json")
        out_txt = args.out_report or out_json.with_suffix(".txt")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        report = _format_compare_report(comparison)
        out_txt.write_text(report, encoding="utf-8")
        print(report, flush=True)
        print(f"\nWrote {out_json}", flush=True)
        print(f"Wrote {out_txt}", flush=True)
        return

    payload = _run_stress(args)
    s = payload["summary"]
    print("\n=== Summary ===", flush=True)
    print(json.dumps(s, ensure_ascii=False, indent=2), flush=True)

    out = args.out or (root_logs / f"standalone_rerank_stress_{_slug_model(args.model)}_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = _format_report(payload)
    out_txt = args.out_report or out.with_suffix(".txt")
    out_txt.write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"\nWrote {out}", flush=True)
    print(f"Wrote {out_txt}", flush=True)


if __name__ == "__main__":
    main()
