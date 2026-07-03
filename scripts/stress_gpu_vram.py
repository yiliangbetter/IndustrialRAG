#!/usr/bin/env python3
"""GPU VRAM stress test: isolate embedding resident vs rerank load/release accumulation.

Reads ``.env`` (same as batch scripts). Reports ``allocated`` / ``reserved`` / ``peak``
after each iteration. Does not call LLM.

Examples::

  .venv\\Scripts\\python.exe scripts/stress_gpu_vram.py --scenario all --iterations 20
  .venv\\Scripts\\python.exe scripts/stress_gpu_vram.py --scenario rerank_cycle --iterations 50
  .venv\\Scripts\\python.exe scripts/stress_gpu_vram.py --scenario rebuild_rag --iterations 5
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
load_dotenv(_ROOT / ".env", override=False)


def _gb(n: float) -> float:
    return round(n / (1024**3), 3)


def cuda_snapshot(reset_peak: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"cuda": False}
    try:
        import torch
    except ImportError:
        return out
    if not torch.cuda.is_available():
        return out
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    out["cuda"] = True
    out["device"] = torch.cuda.get_device_name(0)
    out["allocated_gb"] = _gb(torch.cuda.memory_allocated())
    out["reserved_gb"] = _gb(torch.cuda.memory_reserved())
    out["peak_allocated_gb"] = _gb(torch.cuda.max_memory_allocated())
    out["peak_reserved_gb"] = _gb(torch.cuda.max_memory_reserved())
    return out


def _merge_snap(label: str, snap: dict[str, Any], *, elapsed_s: float | None = None) -> dict[str, Any]:
    row = {"label": label, **snap}
    if elapsed_s is not None:
        row["elapsed_s"] = round(elapsed_s, 3)
    return row


def _print_row(row: dict[str, Any]) -> None:
    if not row.get("cuda"):
        print(f"  {row['label']}: (no CUDA)")
        return
    extra = f"  +{row['elapsed_s']:.2f}s" if row.get("elapsed_s") is not None else ""
    print(
        f"  {row['label']}: alloc={row['allocated_gb']:.3f}G "
        f"reserved={row['reserved_gb']:.3f}G "
        f"peak_alloc={row['peak_allocated_gb']:.3f}G "
        f"peak_reserved={row['peak_reserved_gb']:.3f}G{extra}"
    )


def _load_rpc():
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_parse_graph_chat", _ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
    )
    rpc = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(rpc)
    return rpc


def _sample_docs(n: int = 80) -> list[str]:
    base = (
        "高速智能封边机维护保养：检查电源开关外观与接地装置；机床外部清洁每天一次；"
        "清洁机床内部使用吸尘机擦拭油污；压带轮残胶用刮刀清理；输送链条加注润滑脂2#。"
    )
    return [f"{base} 片段{i}。" for i in range(n)]


async def scenario_embed_resident(*, iterations: int) -> list[dict[str, Any]]:
    from raganything.local_hf_embedding import make_local_hf_embedding_func

    dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    embed = make_local_hf_embedding_func(dim)
    rows: list[dict[str, Any]] = []
    rows.append(_merge_snap("embed_after_first_load", cuda_snapshot(reset_peak=True)))

    query = "高速智能封边机机床床身清洁周期"
    for i in range(1, iterations + 1):
        t0 = time.perf_counter()
        await embed.func([query])
        rows.append(
            _merge_snap(f"embed_encode_{i:03d}", cuda_snapshot(), elapsed_s=time.perf_counter() - t0)
        )
    return rows


async def scenario_rerank_cycle(*, iterations: int, pairs: int) -> list[dict[str, Any]]:
    from raganything.pipeline_rerank import hf_cross_encoder_rerank, release_cross_encoder

    docs = _sample_docs(pairs)
    query = "封边机维护保养手册适用哪些型号"
    rows: list[dict[str, Any]] = []
    rows.append(_merge_snap("rerank_cycle_baseline", cuda_snapshot(reset_peak=True)))

    for i in range(1, iterations + 1):
        t0 = time.perf_counter()
        await hf_cross_encoder_rerank(query, docs, top_n=24)
        rows.append(
            _merge_snap(f"rerank_cycle_{i:03d}", cuda_snapshot(), elapsed_s=time.perf_counter() - t0)
        )
    release_cross_encoder()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    rows.append(_merge_snap("rerank_cycle_after_final_release", cuda_snapshot()))
    return rows


async def scenario_rerank_resident(*, iterations: int, pairs: int) -> list[dict[str, Any]]:
    from raganything.pipeline_rerank import hf_cross_encoder_rerank, release_cross_encoder

    os.environ["RERANK_RELEASE_AFTER_PREDICT"] = "0"
    docs = _sample_docs(pairs)
    query = "封边机维护保养手册适用哪些型号"
    rows: list[dict[str, Any]] = []
    rows.append(_merge_snap("rerank_resident_baseline", cuda_snapshot(reset_peak=True)))

    for i in range(1, iterations + 1):
        t0 = time.perf_counter()
        await hf_cross_encoder_rerank(query, docs, top_n=24)
        rows.append(
            _merge_snap(f"rerank_resident_{i:03d}", cuda_snapshot(), elapsed_s=time.perf_counter() - t0)
        )
    release_cross_encoder()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    rows.append(_merge_snap("rerank_resident_after_release", cuda_snapshot()))
    return rows


async def scenario_rebuild_rag(*, iterations: int, wd: Path, pod: Path) -> list[dict[str, Any]]:
    rpc = _load_rpc()
    rows: list[dict[str, Any]] = []
    rows.append(_merge_snap("rebuild_rag_baseline", cuda_snapshot(reset_peak=True)))
    query = "高速智能封边机维护保养手册适用于哪些产品型号"

    for i in range(1, iterations + 1):
        t0 = time.perf_counter()
        rag, _, _ = await rpc._build_rag(wd, pod)
        # Trigger lazy embed load (mix retrieval would do this many times).
        if hasattr(rag, "lightrag") and rag.lightrag.embedding_func:
            await rag.lightrag.embedding_func.func([query])
        del rag
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        rows.append(
            _merge_snap(f"rebuild_rag_{i:03d}", cuda_snapshot(), elapsed_s=time.perf_counter() - t0)
        )
    return rows


async def scenario_query_pattern(*, iterations: int, pairs: int) -> list[dict[str, Any]]:
    """One question ≈ gate probe rerank + aquery rerank, embedding encode in between."""
    from raganything.local_hf_embedding import make_local_hf_embedding_func
    from raganything.pipeline_rerank import hf_cross_encoder_rerank

    dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    embed = make_local_hf_embedding_func(dim)
    docs = _sample_docs(pairs)
    query = "封边机维护保养手册适用哪些型号"
    rows: list[dict[str, Any]] = []
    rows.append(_merge_snap("query_pattern_baseline", cuda_snapshot(reset_peak=True)))

    for i in range(1, iterations + 1):
        t0 = time.perf_counter()
        await embed.func([query])
        await hf_cross_encoder_rerank(query, docs, top_n=24)  # gate-like
        await embed.func([query, query])
        await hf_cross_encoder_rerank(query, docs, top_n=24)  # aquery-like
        rows.append(
            _merge_snap(f"query_pattern_{i:03d}", cuda_snapshot(), elapsed_s=time.perf_counter() - t0)
        )
    return rows


SCENARIOS: dict[str, Callable[..., Any]] = {
    "embed_resident": scenario_embed_resident,
    "rerank_cycle": scenario_rerank_cycle,
    "rerank_resident": scenario_rerank_resident,
    "rebuild_rag": scenario_rebuild_rag,
    "query_pattern": scenario_query_pattern,
}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cuda_rows = [r for r in rows if r.get("cuda")]
    if not cuda_rows:
        return {"cuda": False}
    alloc = [r["allocated_gb"] for r in cuda_rows]
    reserved = [r["reserved_gb"] for r in cuda_rows]
    elapsed = [r["elapsed_s"] for r in cuda_rows if r.get("elapsed_s") is not None]
    return {
        "cuda": True,
        "samples": len(cuda_rows),
        "alloc_gb_first": alloc[0],
        "alloc_gb_last": alloc[-1],
        "alloc_gb_max": max(alloc),
        "alloc_gb_delta": round(alloc[-1] - alloc[0], 3),
        "reserved_gb_first": reserved[0],
        "reserved_gb_last": reserved[-1],
        "reserved_gb_max": max(reserved),
        "reserved_gb_delta": round(reserved[-1] - reserved[0], 3),
        "elapsed_s_last": elapsed[-1] if elapsed else None,
        "elapsed_s_first": elapsed[0] if elapsed else None,
        "elapsed_s_ratio": round(elapsed[-1] / elapsed[0], 2) if len(elapsed) >= 2 and elapsed[0] > 0 else None,
    }


async def _run_scenario(name: str, *, iterations: int, pairs: int, wd: Path, pod: Path) -> dict[str, Any]:
    fn = SCENARIOS[name]
    print(f"\n=== {name} (iterations={iterations}) ===", flush=True)
    if name in ("rerank_cycle", "rerank_resident", "query_pattern"):
        rows = await fn(iterations=iterations, pairs=pairs)
    elif name == "rebuild_rag":
        rows = await fn(iterations=iterations, wd=wd, pod=pod)
    else:
        rows = await fn(iterations=iterations)
    for row in rows:
        _print_row(row)
    summary = _summarize(rows)
    print(f"  >> summary: reserved_delta={summary.get('reserved_gb_delta')}G "
          f"alloc_delta={summary.get('alloc_gb_delta')}G "
          f"elapsed_ratio={summary.get('elapsed_s_ratio')}", flush=True)
    return {"scenario": name, "iterations": iterations, "rows": rows, "summary": summary}


async def main_async(args: argparse.Namespace) -> int:
    wd = Path(os.getenv("RAG_WEB_WORKING_DIR", str(_ROOT / "data" / "rag_storage"))).resolve()
    pod = Path(os.getenv("RAG_WEB_PARSER_OUTPUT_DIR", str(_ROOT / "data" / "pipeline_parse"))).resolve()

    names = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    env_snapshot = {
        "HF_EMBED_DEVICE": os.getenv("HF_EMBED_DEVICE"),
        "RERANK_HF_DEVICE": os.getenv("RERANK_HF_DEVICE"),
        "RERANK_RELEASE_AFTER_PREDICT": os.getenv("RERANK_RELEASE_AFTER_PREDICT"),
        "RERANK_RELEASE_AFTER_GATE": os.getenv("RERANK_RELEASE_AFTER_GATE"),
        "RERANK_TORCH_DTYPE": os.getenv("RERANK_TORCH_DTYPE"),
        "RERANK_BATCH_SIZE": os.getenv("RERANK_BATCH_SIZE"),
        "RERANK_MODEL": os.getenv("RERANK_MODEL"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
    }
    print("GPU VRAM stress test", flush=True)
    print(f"  env: {json.dumps(env_snapshot, ensure_ascii=False)}", flush=True)
    print(f"  baseline: {cuda_snapshot()}", flush=True)

    report: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).astimezone().isoformat(),
        "env": env_snapshot,
        "iterations": args.iterations,
        "pairs": args.pairs,
        "scenarios": [],
    }

    for name in names:
        block = await _run_scenario(
            name, iterations=args.iterations, pairs=args.pairs, wd=wd, pod=pod
        )
        report["scenarios"].append(block)

    out_dir = _ROOT / "logs" / "vram_stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stamp}_stress.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="GPU VRAM stress test (embedding vs rerank).")
    p.add_argument(
        "--scenario",
        choices=[*SCENARIOS.keys(), "all"],
        default="all",
        help="Which pattern to run (default: all).",
    )
    p.add_argument("--iterations", type=int, default=15, help="Loop count per scenario.")
    p.add_argument(
        "--pairs",
        type=int,
        default=80,
        help="Rerank pool size for rerank/query scenarios (default 80).",
    )
    raise SystemExit(asyncio.run(main_async(p.parse_args())))


if __name__ == "__main__":
    main()
