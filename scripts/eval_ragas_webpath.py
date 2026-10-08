#!/usr/bin/env python3
"""Read-only RAGAS evaluation on the Web-path query stack (Q1–Q17).

Uses the same in-process pipeline as ``run_web_path_q1_17.py`` (``query_progress_hooks``
+ ``aquery``), **does not** call ingest APIs or modify the knowledge base.

Workflow:
  1. ``build_ragas_ground_truth_shili17.py`` → ``data/ragas_shili17_ground_truth.json``
  2. This script collects (question, answer, contexts, ground_truth) per case
  3. RAGAS scores faithfulness / answer relevancy / context recall / context precision

Install eval deps first (keeps sentence-transformers for HF embedding)::

  uv sync --extra eval

Examples::

  uv run python scripts/eval_ragas_webpath.py
  uv run python scripts/eval_ragas_webpath.py --ids 1,3,5 --collect-only
  uv run python scripts/eval_ragas_webpath.py --run-jsonl logs/ragas_webpath/runs/latest.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import json
import math
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env", override=False)

_DEFAULT_DATASET = _ROOT / "data" / "ragas_shili17_ground_truth.json"
_RESULTS_DIR = _ROOT / "logs" / "ragas_webpath"

warnings.filterwarnings(
    "ignore",
    message=".*LangchainLLMWrapper is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Unexpected type for token usage.*",
    category=UserWarning,
)


def _load_rpc():
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_parse_graph_chat", _SCRIPTS / "rag_pipeline_parse_graph_chat.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _parse_ids(raw: str | None, valid: set[int]) -> list[int]:
    if not raw or not raw.strip():
        return sorted(valid)
    out: list[int] = []
    for part in re.split(r"[\s,]+", raw.strip()):
        if not part:
            continue
        cid = int(part.lstrip("Qq"))
        if cid not in valid:
            raise SystemExit(f"Unknown case id Q{cid}; valid: {sorted(valid)}")
        out.append(cid)
    return out


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = list(data.get("test_cases") or [])
    if not cases:
        raise SystemExit(f"No test_cases in {path}")
    return cases


def _contexts_from_docs(docs: list[dict[str, Any]] | None) -> list[str]:
    """One context string per retrieved chunk (dedupe by file_path + prefix)."""
    out: list[str] = []
    seen: set[tuple[str, str]] = set()
    for doc in docs or []:
        if not isinstance(doc, dict):
            continue
        text = ""
        for key in ("content", "text", "chunk_content", "page_content"):
            val = doc.get(key)
            if isinstance(val, str) and val.strip():
                text = val.strip()
                break
        if not text:
            continue
        fp = str(doc.get("file_path") or doc.get("source") or "")
        sig = (fp, text[:120])
        if sig in seen:
            continue
        seen.add(sig)
        out.append(text)
    return out


async def _collect_webpath_row(
    rag: Any,
    rpc: Any,
    *,
    case: dict[str, Any],
    mode: str,
    parser_root: Path,
) -> dict[str, Any]:
    from query_doc_steering import strip_manual_circled_step_markers  # noqa: WPS433
    from query_progress_hooks import (  # noqa: WPS433
        get_query_debug_state,
        query_progress_hooks,
        set_query_media_roots,
        set_query_text_for_images,
    )
    from stream_cot_parser import parse_complete_cot  # noqa: WPS433

    qid = case["id"]
    question = case["question"]
    ground_truth = case.get("ground_truth") or ""
    t0 = time.perf_counter()
    error: str | None = None
    answer = ""
    contexts: list[str] = []

    try:
        async with query_progress_hooks():
            set_query_media_roots([parser_root.resolve()])
            set_query_text_for_images(question.strip())
            raw = await rag.aquery(
                question,
                mode=mode,
                vlm_enhanced=False,
                **rpc._query_extras_from_env(question),
            )
            if not isinstance(raw, str):
                parts: list[str] = []
                async for chunk in raw:
                    if chunk:
                        parts.append(chunk if isinstance(chunk, str) else str(chunk))
                raw = "".join(parts)
            _thinking, answer = parse_complete_cot(raw or "")
            answer = strip_manual_circled_step_markers((answer or "").strip())
            state = get_query_debug_state()
            contexts = _contexts_from_docs(state.get("retrieved_docs"))
    except Exception as exc:
        error = str(exc)

    return {
        "id": qid,
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "contexts": contexts,
        "context_count": len(contexts),
        "duration_ms": int((time.perf_counter() - t0) * 1000),
        "error": error,
    }


async def collect_rows(
    cases: list[dict[str, Any]],
    *,
    mode: str,
    wd: Path,
    pod: Path,
) -> list[dict[str, Any]]:
    rpc = _load_rpc()
    rag, _, _ = await rpc._build_rag(wd, pod)
    rows: list[dict[str, Any]] = []
    try:
        for i, case in enumerate(cases, 1):
            print(f"[collect {i}/{len(cases)}] Q{case['id']}", flush=True)
            row = await _collect_webpath_row(
                rag, rpc, case=case, mode=mode, parser_root=pod
            )
            rows.append(row)
            print(
                f"  answer_chars={len(row['answer'])} contexts={row['context_count']} "
                f"ms={row['duration_ms']}",
                flush=True,
            )
            if row.get("error"):
                print(f"  error: {row['error']}", flush=True)
    finally:
        await rag.finalize_storages()
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


_KNOWN_RAGAS_METRICS = frozenset(
    {
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
        "context_utilization",
        "answer_similarity",
        "answer_correctness",
    }
)
_RAGAS_INPUT_COLS = frozenset(
    {
        "question",
        "answer",
        "contexts",
        "ground_truth",
        "user_input",
        "response",
        "retrieved_contexts",
        "reference",
    }
)


def _parse_metric_value(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def _metric_columns(df) -> list[str]:
    cols = [c for c in df.columns if c in _KNOWN_RAGAS_METRICS]
    if cols:
        return cols
    inferred: list[str] = []
    for c in df.columns:
        if c in _RAGAS_INPUT_COLS:
            continue
        if any(_parse_metric_value(row[c]) is not None for _, row in df.iterrows()):
            inferred.append(c)
    return inferred


def _resolve_hf_model_path(model_id: str) -> str:
    from raganything.local_hf_embedding import _hub_snapshot_dir  # noqa: WPS433

    hf_home = (os.getenv("HF_HOME") or "").strip() or str(_ROOT / "data" / "models")
    snap = _hub_snapshot_dir(model_id, hf_home)
    return str(snap) if snap else model_id


def _build_ragas_llm():
    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise ImportError(
            "RAGAS eval deps missing or incompatible. Install with: uv sync --extra eval"
        ) from exc

    api_key = (
        os.getenv("EVAL_LLM_BINDING_API_KEY")
        or os.getenv("LLM_BINDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise EnvironmentError(
            "Set EVAL_LLM_BINDING_API_KEY, LLM_BINDING_API_KEY, or OPENAI_API_KEY"
        )

    base_url = os.getenv("EVAL_LLM_BINDING_HOST") or os.getenv("LLM_BINDING_HOST")
    model = os.getenv("EVAL_LLM_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "max_retries": int(os.getenv("EVAL_LLM_MAX_RETRIES", "5")),
        "request_timeout": int(os.getenv("EVAL_LLM_TIMEOUT", "180")),
    }
    if base_url:
        llm_kwargs["base_url"] = base_url

    base_llm = ChatOpenAI(**llm_kwargs)
    try:
        eval_llm = LangchainLLMWrapper(langchain_llm=base_llm, bypass_n=True)
    except Exception:
        eval_llm = base_llm
    return eval_llm, model


def _build_ragas_embeddings():
    """Embeddings for answer_relevancy only. Never send local HF model id to OpenAI API."""
    eval_model = (os.getenv("EVAL_EMBEDDING_MODEL") or "").strip()
    eval_backend = (os.getenv("EVAL_EMBEDDING_BACKEND") or "").strip().lower()

    if not eval_backend:
        if eval_model.startswith("text-embedding"):
            eval_backend = "api"
        elif eval_model:
            eval_backend = "hf"
        elif (os.getenv("EMBEDDING_BACKEND") or "").strip().lower() == "hf":
            eval_backend = "hf"
        else:
            eval_backend = "api"

    if eval_backend == "hf":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Local RAGAS embeddings need langchain-community + sentence-transformers"
            ) from exc
        from raganything.local_hf_embedding import _resolve_embedding_device  # noqa: WPS433

        model_id = eval_model or (os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3").strip()
        model_path = _resolve_hf_model_path(model_id)
        device = _resolve_embedding_device()
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        return embeddings, f"hf:{model_id}"

    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise ImportError(
            "RAGAS eval deps missing or incompatible. Install with: uv sync --extra eval"
        ) from exc

    api_key = (
        os.getenv("EVAL_EMBEDDING_BINDING_API_KEY")
        or os.getenv("LLM_BINDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise EnvironmentError(
            "API embeddings need EVAL_EMBEDDING_BINDING_API_KEY or LLM_BINDING_API_KEY"
        )
    base_url = os.getenv("EVAL_EMBEDDING_BINDING_HOST") or os.getenv("LLM_BINDING_HOST")
    # Do not fall back to EMBEDDING_MODEL (BAAI/bge-m3) — that is local HF, not an API id.
    model = eval_model or "text-embedding-v3"
    embed_kwargs: dict[str, Any] = {"model": model, "api_key": api_key}
    if base_url:
        embed_kwargs["base_url"] = base_url
    return OpenAIEmbeddings(**embed_kwargs), model


def run_ragas_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise ImportError(
            "RAGAS eval deps missing or incompatible. Install with: uv sync --extra eval"
        ) from exc

    usable = [r for r in rows if (r.get("answer") or "").strip() and r.get("contexts")]
    if not usable:
        raise RuntimeError("No rows with both answer and contexts; cannot run RAGAS")

    eval_llm, llm_model = _build_ragas_llm()
    embeddings, embed_model = _build_ragas_embeddings()
    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in usable],
            "answer": [r["answer"] for r in usable],
            "contexts": [r["contexts"] for r in usable],
            "ground_truth": [r.get("ground_truth") or "" for r in usable],
        }
    )
    metrics = [
        Faithfulness(llm=eval_llm),
        AnswerRelevancy(llm=eval_llm, embeddings=embeddings),
        ContextRecall(llm=eval_llm),
        ContextPrecision(llm=eval_llm),
    ]
    print(
        f"Running RAGAS on {len(usable)}/{len(rows)} rows "
        f"(llm={llm_model}, embed={embed_model})…",
        flush=True,
    )
    result = evaluate(dataset, metrics=metrics)
    df = result.to_pandas()

    per_case: list[dict[str, Any]] = []
    metric_cols = _metric_columns(df)
    for idx, row in df.iterrows():
        case_row = usable[int(idx)]
        scores = {}
        for col in metric_cols:
            scores[col] = _parse_metric_value(row[col])
        per_case.append(
            {
                "id": case_row.get("id"),
                "question": case_row.get("question"),
                "scores": scores,
            }
        )

    summary: dict[str, float | None] = {}
    for col in metric_cols:
        vals = [r["scores"].get(col) for r in per_case if r["scores"].get(col) is not None]
        summary[col] = round(sum(vals) / len(vals), 4) if vals else None

    ragas_score_vals = [v for v in summary.values() if v is not None]
    summary["ragas_score"] = (
        round(sum(ragas_score_vals) / len(ragas_score_vals), 4) if ragas_score_vals else None
    )

    return {
        "evaluated": len(usable),
        "skipped": len(rows) - len(usable),
        "llm_model": llm_model,
        "embedding_model": embed_model,
        "summary": summary,
        "per_case": per_case,
        "raw_columns": metric_cols,
    }


def _write_results(
    *,
    stamp: str,
    rows: list[dict[str, Any]],
    ragas: dict[str, Any] | None,
    meta: dict[str, Any],
) -> tuple[Path, Path | None]:
    out_dir = _RESULTS_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / "run.jsonl"
    _write_jsonl(run_path, rows)
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path: Path | None = None
    if ragas:
        ragas_path = out_dir / "ragas.json"
        ragas_path.write_text(json.dumps(ragas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        csv_path = out_dir / "ragas_scores.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            cols = ["id", "question"] + list(ragas.get("raw_columns") or [])
            writer.writerow(cols)
            for item in ragas.get("per_case") or []:
                scores = item.get("scores") or {}
                writer.writerow(
                    [item.get("id"), item.get("question")]
                    + [scores.get(c) for c in ragas.get("raw_columns") or []]
                )
    return run_path, csv_path


def _format_report(
    *,
    stamp: str,
    rows: list[dict[str, Any]],
    ragas: dict[str, Any] | None,
    meta: dict[str, Any],
    run_path: Path,
    csv_path: Path | None,
) -> str:
    lines = [
        "=" * 72,
        "RAGAS Web-path eval (read-only)",
        "=" * 72,
        f"time: {meta.get('time')}",
        f"working_dir: {meta.get('working_dir')}",
        f"parser_output_dir: {meta.get('parser_output_dir')}",
        f"mode: {meta.get('mode')}",
        f"dataset: {meta.get('dataset')}",
        f"cases: {len(rows)}",
        f"run_jsonl: {run_path}",
        "",
    ]
    if ragas:
        lines.append("--- RAGAS summary ---")
        for key, val in (ragas.get("summary") or {}).items():
            lines.append(f"  {key}: {val}")
        lines.append("")
        lines.append("id   faithfulness  answer_relevancy  context_recall  context_precision")
        for item in ragas.get("per_case") or []:
            s = item.get("scores") or {}

            def _fmt(key: str) -> str:
                val = s.get(key)
                return f"{val:.3f}" if isinstance(val, (int, float)) else "—"

            lines.append(
                f"{str(item.get('id')):>3}  "
                f"{_fmt('faithfulness'):>13}  "
                f"{_fmt('answer_relevancy'):>16}  "
                f"{_fmt('context_recall'):>14}  "
                f"{_fmt('context_precision'):>17}"
            )
        if csv_path:
            lines.append("")
            lines.append(f"csv: {csv_path}")
    else:
        lines.append("(RAGAS scoring skipped — use without --collect-only to evaluate)")
    lines.append("")
    return "\n".join(lines) + "\n"


async def _async_main(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).resolve()
    cases_all = _load_dataset(dataset_path)
    valid_ids = {int(c["id"]) for c in cases_all}
    want_ids = _parse_ids(args.ids, valid_ids)
    cases = [c for c in cases_all if int(c["id"]) in want_ids]
    cases.sort(key=lambda c: int(c["id"]))

    wd = Path(
        os.getenv("RAG_WEB_WORKING_DIR") or (_ROOT / "data" / "kb_new" / "rag_storage")
    ).resolve()
    pod = Path(
        os.getenv("RAG_WEB_PARSER_OUTPUT_DIR")
        or (_ROOT / "data" / "kb_new" / "pipeline_parse")
    ).resolve()
    mode = (args.mode or os.getenv("RAG_QUERY_MODE") or "mix").strip()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    meta = {
        "time": datetime.now(timezone.utc).isoformat(),
        "working_dir": str(wd),
        "parser_output_dir": str(pod),
        "mode": mode,
        "dataset": str(dataset_path),
        "case_ids": want_ids,
        "read_only": True,
    }

    if args.run_jsonl:
        rows = _read_jsonl(args.run_jsonl.resolve())
        stamp = args.run_jsonl.resolve().parent.name or stamp
        print(f"Loaded {len(rows)} rows from {args.run_jsonl}", flush=True)
    else:
        rows = await collect_rows(cases, mode=mode, wd=wd, pod=pod)
        # Persist collection before RAGAS (scoring can fail on dep/API issues)
        out_dir = _RESULTS_DIR / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        collect_path = out_dir / "run.jsonl"
        _write_jsonl(collect_path, rows)
        (out_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved collection -> {collect_path}", flush=True)

    ragas: dict[str, Any] | None = None
    if not args.collect_only:
        ragas = run_ragas_eval(rows)

    run_path, csv_path = _write_results(stamp=stamp, rows=rows, ragas=ragas, meta=meta)
    report = _format_report(
        stamp=stamp,
        rows=rows,
        ragas=ragas,
        meta=meta,
        run_path=run_path,
        csv_path=csv_path,
    )
    report_path = run_path.parent / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report: {report_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help="Ground-truth JSON (default: data/ragas_shili17_ground_truth.json)",
    )
    p.add_argument(
        "--ids",
        default="",
        help="Comma-separated case ids (default: all in dataset)",
    )
    p.add_argument("--mode", default="", help="LightRAG mode (default: RAG_QUERY_MODE or mix)")
    p.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect answers/contexts; skip RAGAS scoring",
    )
    p.add_argument(
        "--run-jsonl",
        type=Path,
        default=None,
        help="Score an existing run.jsonl instead of querying again",
    )
    args = p.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
