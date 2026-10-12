"""Benchmark the complete Web RAG response path against an isolated indexed KB."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

QUESTION = "比较 NB9-Smart 和 NB10-Smart 的电控板检查周期，并说明共同的安全要求。"
MANUAL = """# 高速智能封边机维护保养手册

## NB9-Smart 电控系统
NB9-Smart 的电控板每季度检查一次。检查前必须切断设备总电源，确认无电后再打开控制柜。检查内容包括端子松动、积尘与散热风扇状态。

## NB10-Smart 电控系统
NB10-Smart 的电控板每半年检查一次。检查前必须切断设备总电源，确认无电后再打开控制柜。检查内容包括接插件、积尘与风道状态。

## 液压系统
液压油每两年或运行四千小时更换，以先到者为准。换油前清洁油箱周边。

## 气动系统
每日开机前检查气压是否处于 0.6 至 0.8 MPa，并排出过滤器积水。

## 输送系统
输送链条每月检查张紧度。调整机械部件前应停机并挂牌。

## 清洁
设备外表面每班清洁，不得使用高压水直接冲洗电气柜。
"""


def _safe_benchmark_base(base: Path) -> Path:
    resolved = base.resolve()
    protected = {Path("/").resolve(), Path.home().resolve(), ROOT.resolve()}
    if (
        resolved in protected
        or base.is_symlink()
        or "benchmark" not in resolved.name.lower()
    ):
        raise ValueError(
            "--base must be a non-symlinked, dedicated directory whose name contains 'benchmark'"
        )
    return resolved


def _event_payload(line: str) -> dict[str, Any] | None:
    if not line.startswith("data: "):
        return None
    try:
        return json.loads(line[6:].strip())
    except json.JSONDecodeError:
        return None


async def _build_and_index(base: Path, *, rebuild: bool) -> tuple[Any, Any]:
    storage = base / "rag_storage"
    parser = base / "pipeline_parse"
    marker = storage / ".benchmark-indexed"
    if rebuild:
        for directory in (storage, parser):
            if directory.exists():
                shutil.rmtree(directory)
    storage.mkdir(parents=True, exist_ok=True)
    parser.mkdir(parents=True, exist_ok=True)

    from rag_pipeline_parse_graph_chat import _build_rag

    started = time.perf_counter()
    rag, config, _logger = await _build_rag(
        storage, parser, skip_multimodal=True, enable_llm_cache=False
    )
    build_s = time.perf_counter() - started
    if not marker.exists():
        started = time.perf_counter()
        await rag.lightrag.ainsert(MANUAL, file_paths="高速智能封边机维护保养手册.txt")
        marker.write_text("indexed\n", encoding="utf-8")
        index_s = time.perf_counter() - started
    else:
        index_s = 0.0
    return (rag, config), {
        "runtime_build_s": round(build_s, 3),
        "index_s": round(index_s, 3),
    }


async def _run_once(web: Any, *, label: str) -> dict[str, Any]:
    body = web.QueryBody(query=QUESTION, mode="mix", stream=True)
    started = time.perf_counter()
    first_event_s = None
    first_visible_s = None
    first_answer_s = None
    thinking: list[str] = []
    answer: list[str] = []
    events: list[dict[str, Any]] = []
    async for line in web._query_stream_events(QUESTION, "mix", body):
        now = time.perf_counter()
        payload = _event_payload(line)
        if payload is None:
            continue
        events.append(payload)
        if first_event_s is None:
            first_event_s = now - started
        kind = payload.get("type")
        if kind == "thinking_delta":
            if first_visible_s is None:
                first_visible_s = now - started
            thinking.append(str(payload.get("text") or ""))
        elif kind == "answer_delta":
            if first_visible_s is None:
                first_visible_s = now - started
            if first_answer_s is None:
                first_answer_s = now - started
            answer.append(str(payload.get("text") or ""))
    total_s = time.perf_counter() - started
    final = "".join(answer).strip()
    return {
        "label": label,
        "first_event_s": round(first_event_s or 0.0, 3),
        "first_visible_s": round(first_visible_s or 0.0, 3),
        "first_answer_s": round(first_answer_s or 0.0, 3),
        "total_s": round(total_s, 3),
        "thinking_chars": len("".join(thinking)),
        "answer": final,
        "correct": all(token in final for token in ("季度", "半年", "切断", "总电源")),
        "event_counts": dict(Counter(str(event.get("type")) for event in events)),
        "debug_dump": next(
            (
                event.get("path")
                for event in events
                if event.get("type") == "query_debug_saved"
            ),
            None,
        ),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    args.base = _safe_benchmark_base(args.base)
    benchmark_env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("EMBEDDING_", "HF_", "RERANK_", "RAG_", "TRANSFORMERS_"))
    }

    (rag, config), setup = await _build_and_index(args.base, rebuild=args.rebuild)

    import rag_web_server as web

    # Client-mode setup reloads .env during import. Explicit benchmark overrides
    # must remain authoritative for the query phase too.
    os.environ.update(benchmark_env)
    web.state.rag = rag
    web.state.config = config
    web.state.working_dir = args.base / "rag_storage"
    web.state.parser_output_dir = args.base / "pipeline_parse"
    web.state.query_mode = "mix"
    web.state.ready = True

    results = []
    for index in range(args.runs):
        results.append(await _run_once(web, label=f"run-{index + 1}"))
    payload = {
        "question": QUESTION,
        "source_document": MANUAL,
        "setup": setup,
        "environment": {
            key: os.getenv(key)
            for key in (
                "LLM_MODEL",
                "EMBEDDING_MODEL",
                "RERANK_MODEL",
                "RERANK_RELEASE_AFTER_QUERY",
                "RAG_CLARIFY_ENABLED",
                "RAG_QUERY_MODE",
            )
        },
        "runs": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    finalize = getattr(rag, "finalize_storages", None)
    if callable(finalize):
        await finalize()


if __name__ == "__main__":
    asyncio.run(main())
