#!/usr/bin/env python3
"""Benchmark the production Web/SSE RAG path against an existing Nanxing KB.

The script never inserts data. It refuses to start unless all seven source
documents are marked processed, disables answer caching and debug-dump I/O,
then records first-status, first-thinking, first-answer, and end-to-end time.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_BASE = ROOT / "data" / "nanxing_kb"
DEFAULT_CASES = ROOT / "tests" / "fixtures" / "nanxing_latency_cases.json"
EXPECTED_SOURCE_FILES = {
    "PC封边机电气报警排除方法.pdf",
    "加工中心产品维护保养说明-20240510.pdf",
    "双端封边机维护保养手册 新版8-24最终版.pdf",
    "数控六面钻产品维护保养说明-20240427.pdf",
    "自动封边机维护保养手册8-25.pdf",
    "高速智能封边机维护保养手册.pdf",
    "高速自动封边机维护保养手册8-24.pdf",
}


def _event_payload(line: str) -> dict[str, Any] | None:
    if not line.startswith("data: "):
        return None
    try:
        value = json.loads(line[6:].strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _load_cases(path: Path, selected: set[str] | None = None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No benchmark cases in {path}")
    result = [case for case in cases if not selected or str(case.get("id")) in selected]
    if selected:
        found = {str(case.get("id")) for case in result}
        missing = selected - found
        if missing:
            raise ValueError(f"Unknown case id(s): {', '.join(sorted(missing))}")
    return result


def _validate_index(base: Path, expected_docs: int = 7) -> dict[str, Any]:
    storage = base.resolve() / "rag_storage"
    parser = base.resolve() / "pipeline_parse"
    status_path = storage / "kv_store_doc_status.json"
    if not status_path.is_file():
        raise ValueError(f"Missing LightRAG document status: {status_path}")
    rows = json.loads(status_path.read_text(encoding="utf-8"))
    processed = [
        row
        for row in rows.values()
        if str(row.get("status") or "").lower().rsplit(".", 1)[-1] == "processed"
        and int(row.get("chunks_count") or 0) > 0
    ]
    if len(rows) != expected_docs or len(processed) != expected_docs:
        states = Counter(str(row.get("status") or "unknown") for row in rows.values())
        raise ValueError(
            f"Nanxing index is incomplete: {len(processed)}/{expected_docs} processed; "
            f"states={dict(states)}"
        )
    indexed_files = {
        Path(str(row.get("file_path") or "")).name for row in rows.values()
    }
    if indexed_files != EXPECTED_SOURCE_FILES:
        raise ValueError(
            "Nanxing index has the wrong source set: "
            f"missing={sorted(EXPECTED_SOURCE_FILES - indexed_files)}, "
            f"unexpected={sorted(indexed_files - EXPECTED_SOURCE_FILES)}"
        )
    if not parser.is_dir():
        raise ValueError(f"Missing parser output directory: {parser}")
    required = (
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
        "graph_chunk_entity_relation.graphml",
    )
    missing = [name for name in required if not (storage / name).is_file()]
    if missing:
        raise ValueError(f"Nanxing index is missing stores: {', '.join(missing)}")

    vector_counts: dict[str, int] = {}
    for name in required[:3]:
        data = json.loads((storage / name).read_text(encoding="utf-8"))
        vector_counts[name] = len(data.get("data") or [])
    if any(count == 0 for count in vector_counts.values()):
        raise ValueError(f"Nanxing vector stores are empty: {vector_counts}")

    import networkx as nx

    graph = nx.read_graphml(storage / "graph_chunk_entity_relation.graphml")
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError(
            "Nanxing graph is empty: "
            f"nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}"
        )
    content_lists = len(list(parser.rglob("*_content_list.json")))
    if content_lists < expected_docs:
        raise ValueError(
            f"Nanxing parser output is incomplete: {content_lists}/{expected_docs} content lists"
        )
    return {
        "base": str(base.resolve()),
        "storage": str(storage),
        "parser": str(parser),
        "documents": len(rows),
        "chunks": sum(int(row.get("chunks_count") or 0) for row in rows.values()),
        "files": sorted(str(row.get("file_path") or "") for row in rows.values()),
        "vectors": vector_counts,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "content_lists": content_lists,
    }


def _grade(answer: str, case: dict[str, Any]) -> tuple[bool, list[str]]:
    compact = "".join(answer.split()).lower()
    # Model answers commonly emphasize values (for example ``**9mm**``).
    # Evaluate relations against the rendered text, not Markdown punctuation.
    pattern_text = re.sub(r"[*_`~]+", "", compact)
    missing: list[str] = []
    for token in case.get("must_include") or []:
        if "".join(str(token).split()).lower() not in compact:
            missing.append(str(token))
    for choices in case.get("one_of") or []:
        if not any(
            "".join(str(choice).split()).lower() in compact for choice in choices
        ):
            missing.append("one_of:" + "|".join(map(str, choices)))
    for pattern in case.get("must_match") or []:
        if (
            re.search(str(pattern), pattern_text, flags=re.IGNORECASE | re.DOTALL)
            is None
        ):
            missing.append("pattern:" + str(pattern))
    return not missing, missing


def _summary_is_valid(summary: list[dict[str, Any]]) -> bool:
    """Require successful, grounded answers and visible thinking for every run."""
    return all(
        row["completed"] == row["runs"]
        and row["quality_passes"] == row["runs"]
        and row["thinking_responses"] == row["runs"]
        for row in summary
    )


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain=v1")),
    }


def _runtime_environment() -> dict[str, Any]:
    """Return reproducibility metadata without ever serializing API secrets."""
    packages: dict[str, str] = {}
    for name in ("lightrag-hku", "openai", "sentence-transformers", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"

    accelerator: dict[str, Any] = {}
    try:
        import torch

        accelerator = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except (ImportError, RuntimeError):
        accelerator = {"cuda_available": False, "cuda_version": None, "gpu": None}

    return {
        "command": [sys.executable, *sys.argv],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "accelerator": accelerator,
        "git": _git_metadata(),
        "resolved_llm_base_url": (
            os.getenv("LLM_BINDING_HOST")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
        ),
        "settings": {
            key: os.getenv(key)
            for key in (
                "LLM_MODEL",
                "LLM_BINDING_HOST",
                "OPENAI_BASE_URL",
                "LLM_BASE_URL",
                "KEYWORD_LLM_ENABLE_THINKING",
                "EMBEDDING_BACKEND",
                "EMBEDDING_MODEL",
                "HF_EMBED_DEVICE",
                "RERANK_BINDING",
                "RERANK_MODEL",
                "RERANK_HF_DEVICE",
                "RAG_QUERY_MODE",
                "RAG_QUERY_SUBJECT_STEER",
                "RAG_QUERY_ALIGN_MIN_CHARS",
                "TOP_K",
                "CHUNK_TOP_K",
                "MAX_TOTAL_TOKENS",
                "RERANK_BATCH_SIZE",
                "RERANK_RELEASE_AFTER_QUERY",
                "ENABLE_LLM_CACHE",
                "RAG_QUERY_DEBUG_DUMP",
                "RAG_CLARIFY_ENABLED",
                "CLARIFY_DIRECT_RERANK_MIN",
            )
        },
    }


def _source_basenames() -> list[str]:
    try:
        from query_progress_hooks import get_query_debug_state

        state = get_query_debug_state()
    except (ImportError, RuntimeError):
        return []
    llm_input = state.get("llm_input")
    chunks = llm_input.get("chunks") if isinstance(llm_input, dict) else None
    if not chunks:
        chunks = state.get("llm_chunks_for_images") or state.get("retrieved_docs") or []
    names: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for key in ("file_path", "source", "document_path"):
            value = chunk.get(key)
            if value:
                names.add(Path(str(value)).name)
                break
    return sorted(names)


async def _run_case(
    web: Any,
    case: dict[str, Any],
    *,
    round_number: int,
    warmup: bool,
    follow_clarification: bool = True,
) -> dict[str, Any]:
    question = str(case["question"])
    started = time.perf_counter()
    milestones: dict[str, float | None] = {
        "first_status_s": None,
        "first_clarification_s": None,
        "first_thinking_s": None,
        "first_answer_s": None,
        "done_s": None,
    }
    thinking: list[str] = []
    answer: list[str] = []
    events: list[dict[str, Any]] = []
    status_timeline: list[dict[str, Any]] = []
    clarification: dict[str, Any] | None = None

    async def consume(body: Any, *, stage: str) -> None:
        nonlocal clarification
        async for line in web._query_stream_events(body.query, "mix", body):
            payload = _event_payload(line)
            if payload is None:
                continue
            events.append({"benchmark_stage": stage, **payload})
            elapsed = time.perf_counter() - started
            kind = str(payload.get("type") or "")
            key = {
                "status": "first_status_s",
                "clarification_required": "first_clarification_s",
                "thinking_delta": "first_thinking_s",
                "answer_delta": "first_answer_s",
            }.get(kind)
            if key and milestones[key] is None:
                milestones[key] = elapsed
            if kind == "done":
                milestones["done_s"] = elapsed
            if kind in {"status", "retrieval_scope"}:
                status_timeline.append(
                    {"elapsed_s": round(elapsed, 3), "stage": stage, **payload}
                )
            if kind == "thinking_delta":
                thinking.append(str(payload.get("text") or ""))
            elif kind == "answer_delta":
                answer.append(str(payload.get("text") or ""))
            elif kind == "clarification_required":
                clarification = payload.get("data")

    initial_body = web.QueryBody(query=question, mode="mix", stream=True)
    await consume(initial_body, stage="initial")
    initial_done_s = milestones["done_s"]
    followup_started_s: float | None = None
    clarification_followed = False
    if follow_clarification and isinstance(clarification, dict):
        keep_original = clarification.get("keep_original")
        clarification_id = clarification.get("clarification_id")
        if isinstance(keep_original, dict) and clarification_id:
            followup_query = str(keep_original.get("query") or question)
            followup_started_s = time.perf_counter() - started
            followup_body = web.QueryBody(
                query=followup_query,
                mode="mix",
                stream=True,
                clarify_choice="keep_original",
                clarification_id=str(clarification_id),
            )
            clarification_followed = True
            await consume(followup_body, stage="keep_original")

    total_s = time.perf_counter() - started
    final_answer = "".join(answer).strip()
    answer_pass, missing = _grade(final_answer, case)
    source_files = _source_basenames()
    expected_sources = {
        source.strip()
        for source in str(case.get("source") or "").split("+")
        if source.strip()
    }
    source_pass = expected_sources.issubset(set(source_files))
    scopes = [event for event in events if event.get("type") == "retrieval_scope"]
    response_times = [
        value
        for value in (milestones["first_thinking_s"], milestones["first_answer_s"])
        if value is not None
    ]
    interaction_times = [
        value
        for value in (milestones["first_clarification_s"], *response_times)
        if value is not None
    ]
    error = next(
        (
            str(event.get("message") or "")
            for event in events
            if event.get("type") == "error"
        ),
        None,
    )
    clarification_required = clarification is not None
    clarification_only = clarification_required and not clarification_followed
    completed = milestones["done_s"] is not None and bool(final_answer) and not error

    def since_followup(value: float | None) -> float | None:
        if value is None or followup_started_s is None:
            return None
        return round(value - followup_started_s, 3)

    return {
        "case_id": str(case["id"]),
        "round": round_number,
        "warmup": warmup,
        "question": question,
        "expected_source": case.get("source"),
        **{
            key: round(value, 3) if value is not None else None
            for key, value in milestones.items()
        },
        "first_interaction_s": (
            round(min(interaction_times), 3) if interaction_times else None
        ),
        "first_response_s": round(min(response_times), 3) if response_times else None,
        "initial_done_s": round(initial_done_s, 3)
        if initial_done_s is not None
        else None,
        "followup_started_s": (
            round(followup_started_s, 3) if followup_started_s is not None else None
        ),
        "followup_first_response_s": since_followup(
            min(response_times) if response_times else None
        ),
        "followup_first_answer_s": since_followup(milestones["first_answer_s"]),
        "followup_done_s": since_followup(milestones["done_s"]),
        "total_s": round(total_s, 3),
        "thinking_chars": len("".join(thinking)),
        "answer_chars": len(final_answer),
        "answer": final_answer,
        "quality_pass": answer_pass and source_pass and completed,
        "answer_pass": answer_pass,
        "quality_missing": missing,
        "source_files": source_files,
        "source_pass": source_pass,
        "completed": completed,
        "clarification_required": clarification_required,
        "clarification_followed": clarification_followed,
        "clarification_only": clarification_only,
        "clarification_id": (
            clarification.get("clarification_id")
            if isinstance(clarification, dict)
            else None
        ),
        "clarification_gate_outcome": (
            clarification.get("gate_outcome")
            if isinstance(clarification, dict)
            else None
        ),
        "error": error,
        "event_counts": dict(Counter(str(event.get("type")) for event in events)),
        "retrieval_scopes": scopes,
        "status_timeline": status_timeline,
    }


def _median(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(statistics.median(values), 3) if values else None


def _summaries(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    measured = [row for row in trials if not row["warmup"]]
    result = []
    for case_id in dict.fromkeys(row["case_id"] for row in measured):
        rows = [row for row in measured if row["case_id"] == case_id]
        result.append(
            {
                "case_id": case_id,
                "runs": len(rows),
                "first_status_median_s": _median(rows, "first_status_s"),
                "first_interaction_median_s": _median(rows, "first_interaction_s"),
                "first_response_median_s": _median(rows, "first_response_s"),
                "first_thinking_median_s": _median(rows, "first_thinking_s"),
                "first_answer_median_s": _median(rows, "first_answer_s"),
                "initial_done_median_s": _median(rows, "initial_done_s"),
                "followup_first_response_median_s": _median(
                    rows, "followup_first_response_s"
                ),
                "followup_first_answer_median_s": _median(
                    rows, "followup_first_answer_s"
                ),
                "followup_done_median_s": _median(rows, "followup_done_s"),
                "done_median_s": _median(rows, "done_s"),
                "total_median_s": _median(rows, "total_s"),
                "quality_passes": sum(bool(row["quality_pass"]) for row in rows),
                "answer_passes": sum(bool(row["answer_pass"]) for row in rows),
                "source_passes": sum(bool(row["source_pass"]) for row in rows),
                "completed": sum(bool(row["completed"]) for row in rows),
                "errors": sum(bool(row["error"]) for row in rows),
                "clarifications": sum(
                    bool(row["clarification_required"]) for row in rows
                ),
                "thinking_responses": sum(
                    int(row["thinking_chars"] > 0) for row in rows
                ),
            }
        )
    return result


def _aggregate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in trials if not row["warmup"]]
    return {
        "runs": len(measured),
        "first_status_median_s": _median(measured, "first_status_s"),
        "first_interaction_median_s": _median(measured, "first_interaction_s"),
        "first_response_median_s": _median(measured, "first_response_s"),
        "first_thinking_median_s": _median(measured, "first_thinking_s"),
        "first_answer_median_s": _median(measured, "first_answer_s"),
        "initial_done_median_s": _median(measured, "initial_done_s"),
        "followup_first_response_median_s": _median(
            measured, "followup_first_response_s"
        ),
        "followup_first_answer_median_s": _median(measured, "followup_first_answer_s"),
        "followup_done_median_s": _median(measured, "followup_done_s"),
        "done_median_s": _median(measured, "done_s"),
        "quality_passes": sum(bool(row["quality_pass"]) for row in measured),
        "thinking_responses": sum(int(row["thinking_chars"] > 0) for row in measured),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    aggregate = payload["aggregate"]
    lines = [
        "# Nanxing complete-RAG latency benchmark",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        (
            "Aggregate measured medians: "
            f"first interaction `{aggregate['first_interaction_median_s']}s`, "
            f"final-stream response `{aggregate['first_response_median_s']}s`, "
            f"first final answer `{aggregate['first_answer_median_s']}s`, "
            f"done `{aggregate['done_median_s']}s`; "
            f"quality `{aggregate['quality_passes']}/{aggregate['runs']}`, "
            f"thinking `{aggregate['thinking_responses']}/{aggregate['runs']}`."
        ),
        "",
        "| Case | Runs | First status p50 | First interaction p50 | Initial done p50 | Final response p50 | First answer p50 | Done p50 | Quality | Gate offers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        fmt = lambda value: "—" if value is None else f"{value:.3f}s"
        lines.append(
            f"| {row['case_id']} | {row['runs']} | {fmt(row['first_status_median_s'])} | "
            f"{fmt(row['first_interaction_median_s'])} | "
            f"{fmt(row['initial_done_median_s'])} | "
            f"{fmt(row['first_response_median_s'])} | "
            f"{fmt(row['first_answer_median_s'])} | {fmt(row['done_median_s'])} | "
            f"{row['quality_passes']}/{row['runs']} | {row['clarifications']}/{row['runs']} |"
        )
    for trial in payload["trials"]:
        label = "warmup" if trial["warmup"] else f"round {trial['round']}"
        lines.extend(
            [
                "",
                f"## {trial['case_id']} — {label}",
                "",
                f"Question: {trial['question']}",
                "",
                (
                    f"Timings: status={trial['first_status_s']}s, "
                    f"interaction={trial['first_interaction_s']}s, "
                    f"initial_done={trial['initial_done_s']}s, "
                    f"response={trial['first_response_s']}s, "
                    f"thinking={trial['first_thinking_s']}s, "
                    f"answer={trial['first_answer_s']}s, done={trial['done_s']}s"
                ),
                "",
                "Answer:",
                "",
                trial["answer"] or "_(no final answer)_",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _main(args: argparse.Namespace) -> tuple[Path, Path, bool]:
    load_dotenv(ROOT / ".env", override=False)
    index = _validate_index(args.base)
    benchmark_env = {
        "ENABLE_LLM_CACHE": "false",
        "RAG_QUERY_DEBUG_DUMP": "0",
        "RAG_QUERY_TIMING_DUMP": "0",
        "RERANK_RELEASE_AFTER_QUERY": "0",
        "RAG_WEB_WORKING_DIR": index["storage"],
        "RAG_WEB_PARSER_OUTPUT_DIR": index["parser"],
    }
    os.environ.update(benchmark_env)
    # The Web server's client-mode bootstrap reloads .env during import. Keep
    # explicit shell overrides (model paths, gate settings, A/B knobs) stable.
    isolated_environment = dict(os.environ)
    cases = _load_cases(args.cases, set(args.case_ids or []))

    from rag_pipeline_parse_graph_chat import _build_rag

    build_started = time.perf_counter()
    rag, config, _logger = await _build_rag(
        Path(index["storage"]),
        Path(index["parser"]),
        skip_multimodal=True,
        enable_llm_cache=False,
    )
    runtime_build_s = time.perf_counter() - build_started

    import rag_web_server as web

    # rag_web_server applies the development/client env at import time. Restore
    # the isolated benchmark settings before any query-time helpers read them.
    os.environ.update(isolated_environment)

    web.state.rag = rag
    web.state.config = config
    web.state.working_dir = index["storage"]
    web.state.parser_output_dir = index["parser"]
    web.state.query_mode = "mix"
    web.state.ready = True

    trials: list[dict[str, Any]] = []
    try:
        if args.warmup:
            for case in cases:
                trials.append(
                    await _run_case(
                        web,
                        case,
                        round_number=0,
                        warmup=True,
                        follow_clarification=args.follow_clarification,
                    )
                )
        for round_number in range(1, args.runs + 1):
            offset = (round_number - 1) % len(cases)
            for case in cases[offset:] + cases[:offset]:
                trials.append(
                    await _run_case(
                        web,
                        case,
                        round_number=round_number,
                        warmup=False,
                        follow_clarification=args.follow_clarification,
                    )
                )
    finally:
        finalize = getattr(rag, "finalize_storages", None)
        if callable(finalize):
            await finalize()

    summary = _summaries(trials)
    valid = _summary_is_valid(summary)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "protocol": {
            "path": "rag_web_server._query_stream_events (same SSE path as Web UI)",
            "transport": "in-process; excludes FastAPI HTTP and browser rendering",
            "llm_cache": False,
            "debug_dump": False,
            "thinking_enabled": True,
            "auto_follow_keep_original": bool(args.follow_clarification),
            "warmup_per_case": bool(args.warmup),
            "measured_rounds": args.runs,
        },
        "environment": _runtime_environment(),
        "index": index,
        "runtime_build_s": round(runtime_build_s, 3),
        "valid": valid,
        "aggregate": _aggregate(trials),
        "summary": summary,
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = args.output.with_suffix(".md")
    _write_markdown(markdown, payload)
    return args.output, markdown, valid


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--case-ids", nargs="*", default=[])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--follow-clarification",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically emulate the UI's keep-original action after a gate offer.",
    )
    default_output = (
        ROOT
        / "logs"
        / "benchmarks"
        / f"nanxing_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    )
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    return args


if __name__ == "__main__":
    output_json, output_md, benchmark_valid = asyncio.run(_main(_parse_args()))
    print(output_json)
    print(output_md)
    if not benchmark_valid:
        raise SystemExit(2)
