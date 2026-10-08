#!/usr/bin/env python3
"""Green8 probe-only timing + VRAM snapshot (diagnose unstable clarify gate).

Runs ``probe_llm_retrieval_full`` (mix + rerank, no LLM answer) in bench order.
Records merge-pool size, probe seconds, and GPU memory to separate:
  - rerank-model drift (see ``standalone_rerank_stress.py``)
  - full LightRAG mix path variance / cross-case interference

Examples::

  uv run python scripts/bench_probe_timing_green8.py
  uv run python scripts/bench_probe_timing_green8.py --repeat-each 2
  uv run python scripts/bench_probe_timing_green8.py --gc-between
  uv run python scripts/bench_probe_timing_green8.py --shuffle --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import importlib.util
import json
import os
import random
import subprocess
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

from replay_clarify_gate_green8 import _load_all_cases  # noqa: E402
from raganything.clarify_gate import probe_llm_retrieval_full  # noqa: E402

_merge_pool_size: int = 0


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
        return float(parts[0])
    except Exception:
        return None


def _torch_mem_mb() -> dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 1),
        }
    except Exception:
        return {}


def _install_merge_pool_hook() -> None:
    import lightrag.utils as ut

    native = ut.apply_rerank_if_enabled
    if getattr(native, "_bench_probe_pool_hook", False):
        return

    async def _wrapped(
        query: str,
        retrieved_docs: list[dict],
        global_config: dict,
        enable_rerank: bool = True,
        top_n: int | None = None,
    ):
        global _merge_pool_size
        _merge_pool_size = len(retrieved_docs or [])
        return await native(
            query, retrieved_docs, global_config, enable_rerank, top_n
        )

    _wrapped._bench_probe_pool_hook = True  # type: ignore[attr-defined]
    ut.apply_rerank_if_enabled = _wrapped  # type: ignore[method-assign]


async def _probe_once(
    lightrag: Any,
    query: str,
    *,
    mode: str,
    gc_between: bool,
) -> dict[str, Any]:
    global _merge_pool_size
    _merge_pool_size = 0

    if gc_between:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    vram_before = _nvidia_used_mb()
    torch_before = _torch_mem_mb()
    t0 = time.perf_counter()
    probe = await probe_llm_retrieval_full(lightrag, query, mode=mode)
    probe_s = round(time.perf_counter() - t0, 1)
    vram_after = _nvidia_used_mb()
    torch_after = _torch_mem_mb()

    return {
        "query": query,
        "probe_s": probe_s,
        "final_score": probe.final_score,
        "chunk_count": probe.chunk_count,
        "llm_chunk_total": probe.llm_chunk_total,
        "merge_pool_size": _merge_pool_size,
        "vram_mb_before": vram_before,
        "vram_mb_after": vram_after,
        "torch_after": torch_after,
        "torch_before": torch_before,
    }


def _format_report(rows: list[dict[str, Any]], *, meta: dict[str, Any]) -> str:
    lines = [
        "=" * 72,
        "green8 probe-only timing (mix + rerank, no answer LLM)",
        "=" * 72,
        f"time: {meta.get('time')}",
        f"RERANK_MODEL: {meta.get('rerank_model')}",
        f"repeat_each: {meta.get('repeat_each')}  gc_between: {meta.get('gc_between')}",
        f"shuffle: {meta.get('shuffle')}",
        "",
        f"{'id':>4} {'try':>3} {'probe_s':>8} {'pool':>5} {'chunks':>6} {'final':>8} {'vram':>8} query",
        "-" * 72,
    ]
    for row in rows:
        fs = row.get("final_score")
        fs_txt = f"{fs:.4f}" if isinstance(fs, (int, float)) else "-"
        vram = row.get("vram_mb_after")
        vram_txt = f"{vram:.0f}" if isinstance(vram, (int, float)) else "-"
        lines.append(
            f"{row.get('id', '?'):>4} {row.get('try', 1):>3} "
            f"{row.get('probe_s', 0):>8.1f} "
            f"{row.get('merge_pool_size') or 0:>5} "
            f"{row.get('chunk_count') or 0:>6} "
            f"{fs_txt:>8} {vram_txt:>8} "
            f"{(row.get('query') or '')[:36]}"
        )
    lines.append("-" * 72)

    by_id: dict[Any, list[float]] = {}
    for row in rows:
        by_id.setdefault(row.get("id"), []).append(float(row.get("probe_s") or 0))
    if any(len(v) > 1 for v in by_id.values()):
        lines.extend(["", "repeat drift (same id, probe_s):"])
        for cid, times in sorted(by_id.items(), key=lambda x: str(x[0])):
            if len(times) < 2:
                continue
            lines.append(
                f"  #{cid}: {times[0]:.1f}s -> {times[1]:.1f}s "
                f"(ratio={times[1]/times[0]:.2f})" if times[0] > 0 else f"  #{cid}: {times}"
            )

    probe_times = [float(r.get("probe_s") or 0) for r in rows if r.get("try") == 1]
    if probe_times:
        lines.extend(
            [
                "",
                f"first-pass probe_s: min={min(probe_times):.1f}s "
                f"med={sorted(probe_times)[len(probe_times)//2]:.1f}s "
                f"max={max(probe_times):.1f}s",
            ]
        )
    lines.append("")
    lines.append("对照: 若 probe 方差大但 standalone_rerank_stress 稳定 → 瓶颈在 mix 检索/多路 merge，")
    lines.append("      而非 rerank 模型显存泄漏。用 --repeat-each 2 区分「同问变慢」与「难问本身慢」。")
    return "\n".join(lines) + "\n"


async def _main(args: argparse.Namespace) -> None:
    _install_merge_pool_hook()
    wd = Path(os.getenv("RAG_WEB_WORKING_DIR") or (_ROOT / "data" / "rag_storage")).resolve()
    pod = Path(
        os.getenv("RAG_WEB_PARSER_OUTPUT_DIR") or (_ROOT / "data" / "pipeline_parse")
    ).resolve()
    cases, _ = _load_all_cases("green8", limit=0)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(cases)

    print(f"working_dir: {wd}", flush=True)
    rag, _, _ = await rpc._build_rag(wd, pod)
    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            cid = case.get("id", "?")
            q = (case.get("query") or case.get("standard_question") or "").strip()
            for attempt in range(1, args.repeat_each + 1):
                label = f"#{cid}" + (f" x{attempt}" if args.repeat_each > 1 else "")
                print(f"probe {label} {q[:40]!r} ...", flush=True)
                row = await _probe_once(
                    rag.lightrag,
                    q,
                    mode=args.mode,
                    gc_between=args.gc_between,
                )
                row["id"] = cid
                row["try"] = attempt
                rows.append(row)
                print(
                    f"  probe_s={row['probe_s']} pool={row['merge_pool_size']} "
                    f"chunks={row['chunk_count']} vram={row.get('vram_mb_after')}",
                    flush=True,
                )
    finally:
        await rag.finalize_storages()

    meta = {
        "time": datetime.now(timezone.utc).isoformat(),
        "rerank_model": os.getenv("RERANK_MODEL", ""),
        "repeat_each": args.repeat_each,
        "gc_between": args.gc_between,
        "shuffle": args.shuffle,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = args.out_json or (_ROOT / "logs" / f"bench_probe_timing_green8_{ts}.json")
    out_txt = args.out_report or (_ROOT / "logs" / f"bench_probe_timing_green8_{ts}.txt")
    payload = {"meta": meta, "rows": rows}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = _format_report(rows, meta=meta)
    out_txt.write_text(report, encoding="utf-8")
    print(report)
    print(f"JSON: {out_json}")
    print(f"Report: {out_txt}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="mix")
    p.add_argument("--repeat-each", type=int, default=1, help="Re-run each query N times")
    p.add_argument("--gc-between", action="store_true", help="gc + cuda.empty_cache before each probe")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-report", type=Path, default=None)
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()
