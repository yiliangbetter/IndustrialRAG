#!/usr/bin/env python3
"""Read-only RAGAS evaluation on the exact production Web query path (Q1–Q17).

Uses the same clarification, retrieval, reranking, streaming, and image-finalization
orchestration as ``/api/query/stream``. It does not call ingest APIs or modify the
knowledge base. Cases are deliberately executed sequentially because the current
model-stream lifecycle is not isolated for parallel in-process queries.

Workflow:
  1. Load the tracked, versioned Shili17 evaluation dataset
  2. Collect (question, answer, exact model contexts, reference) per case
  3. Score faithfulness, relevance, correctness, and retrieval precision/recall

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
import uuid
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env", override=False)

_DEFAULT_DATASET = _ROOT / "tests" / "fixtures" / "evaluation" / "shili17.json"
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
    if data.get("schema_version") == "rag-evaluation-dataset/v1":
        from raganything.evaluation_legacy import (  # noqa: WPS433
            ragas_shili17_cases,
        )
        from raganything.evaluation_schema import (  # noqa: WPS433
            load_evaluation_dataset,
        )

        return ragas_shili17_cases(load_evaluation_dataset(path))
    cases = list(data.get("test_cases") or [])
    if not cases:
        raise SystemExit(f"No test_cases in {path}")
    return cases


async def _collect_webpath_row(
    web: Any,
    *,
    case: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    qid = case["id"]
    question = case["question"]
    ground_truth = case.get("ground_truth") or ""
    body = web.QueryBody(query=question, mode=mode, stream=True)
    trace = await web.execute_query_with_trace(body)
    contexts = list(trace.get("contexts") or [])

    return {
        "id": qid,
        "question": question,
        "ground_truth": ground_truth,
        "answer": str(trace.get("answer") or ""),
        "contexts": contexts,
        "context_count": len(contexts),
        "duration_ms": int(trace.get("duration_ms") or 0),
        "outcome": trace.get("outcome"),
        "error": trace.get("error"),
        "trace": trace,
    }


async def collect_rows(
    cases: list[dict[str, Any]],
    *,
    mode: str,
    wd: Path,
    pod: Path,
    on_progress: Callable[[list[dict[str, Any]]], None] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    original_environment = dict(os.environ)
    rag: Any = None
    rows: list[dict[str, Any]] = []
    try:
        os.environ.update(
            {
                "RAG_WEB_WORKING_DIR": str(wd),
                "RAG_WEB_PARSER_OUTPUT_DIR": str(pod),
            }
        )
        evaluation_environment = dict(os.environ)
        rpc = _load_rpc()

        # Dynamic imports can load .env-backed modules and mutate os.environ.
        # Keep setup and every query pinned to the explicit evaluation values.
        os.environ.clear()
        os.environ.update(evaluation_environment)
        rag, config, _logger = await rpc._build_rag(
            wd,
            pod,
            enable_llm_cache=False,
        )

        # Import after the explicit evaluation environment is set. The Web
        # module's client bootstrap reloads configuration at import time, so
        # restore our frozen values before query helpers read them.
        import rag_web_server as web  # noqa: WPS433

        os.environ.clear()
        os.environ.update(evaluation_environment)
        web._RPC = rpc
        web.state.rag = rag
        web.state.config = config
        web.state.working_dir = str(wd)
        web.state.parser_output_dir = str(pod)
        web.state.query_mode = mode
        web.state.ready = True

        for i, case in enumerate(cases, 1):
            print(f"[collect {i}/{len(cases)}] Q{case['id']}", flush=True)
            try:
                row = await _collect_webpath_row(web, case=case, mode=mode)
            except Exception as exc:  # noqa: BLE001 - preserve the paid batch
                row = {
                    "id": case.get("id"),
                    "question": case.get("question"),
                    "ground_truth": case.get("ground_truth") or "",
                    "answer": "",
                    "contexts": [],
                    "context_count": 0,
                    "duration_ms": 0,
                    "outcome": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": {},
                }
            rows.append(row)
            if on_progress is not None:
                on_progress(rows)
            print(
                f"  outcome={row['outcome']} answer_chars={len(row['answer'])} "
                f"contexts={row['context_count']} ms={row['duration_ms']}",
                flush=True,
            )
            if row.get("error"):
                print(f"  error: {row['error']}", flush=True)
    finally:
        try:
            if rag is not None:
                try:
                    await rag.finalize_storages()
                except Exception as exc:  # noqa: BLE001 - rows are still valuable
                    message = f"{type(exc).__name__}: {exc}"
                    if diagnostics is not None:
                        diagnostics["finalization_error"] = message
                    print(
                        f"WARNING: storage finalization failed after collection: {message}",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            os.environ.clear()
            os.environ.update(original_environment)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _atomic_write_text(path, content)


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace one artifact atomically so a failed checkpoint keeps its predecessor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


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
    if isinstance(val, (int, float)):
        parsed = float(val)
        return parsed if math.isfinite(parsed) else None
    if isinstance(val, str):
        try:
            parsed = float(val)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
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
        raise OSError(
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
    except Exception:  # noqa: BLE001 - compatibility varies across RAGAS releases
        eval_llm = base_llm
    return eval_llm, model


def _build_ragas_embeddings():
    """Embeddings for answer_relevancy only. Never send local HF model id to OpenAI API."""
    eval_model = (os.getenv("EVAL_EMBEDDING_MODEL") or "").strip()
    eval_backend = (os.getenv("EVAL_EMBEDDING_BACKEND") or "").strip().lower()

    if not eval_backend:
        if eval_model.startswith("text-embedding"):
            eval_backend = "api"
        elif (
            eval_model or (os.getenv("EMBEDDING_BACKEND") or "").strip().lower() == "hf"
        ):
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
        from raganything.local_hf_embedding import (  # noqa: WPS433
            _resolve_embedding_device,
        )

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
        raise OSError(
            "API embeddings need EVAL_EMBEDDING_BINDING_API_KEY or LLM_BINDING_API_KEY"
        )
    base_url = os.getenv("EVAL_EMBEDDING_BINDING_HOST") or os.getenv("LLM_BINDING_HOST")
    # Do not fall back to EMBEDDING_MODEL (BAAI/bge-m3) — that is local HF, not an API id.
    model = eval_model or "text-embedding-v3"
    embed_kwargs: dict[str, Any] = {"model": model, "api_key": api_key}
    if base_url:
        embed_kwargs["base_url"] = base_url
    return OpenAIEmbeddings(**embed_kwargs), model


def _row_error(row: dict[str, Any]) -> Any:
    """Return an error exposed either by the row or its underlying trace."""
    error = row.get("error")
    if error:
        return error
    trace = row.get("trace")
    if isinstance(trace, dict):
        return trace.get("error")
    return None


def _unevaluated_reason(row: dict[str, Any]) -> str | None:
    """Explain why a collected row is unsafe or incomplete for RAGAS."""
    answer = str(row.get("answer") or "").strip()
    contexts = row.get("contexts")
    outcome = str(row.get("outcome") or "").strip()
    error = _row_error(row)

    # A stream can emit answer deltas before its provider fails. Treating that
    # partial text as a completed answer would produce deceptively valid scores.
    if answer and (error or outcome == "error"):
        return "partial_answer_ended_in_error"
    if error or outcome == "error":
        return "query_error"
    if outcome != "answer":
        return "outcome_not_answer"

    trace = row.get("trace")
    if isinstance(trace, dict) and "terminal_event" in trace:
        terminal = trace.get("terminal_event")
        if not isinstance(terminal, dict) or terminal.get("type") != "done":
            return "terminal_not_done"
    if isinstance(trace, dict) and trace.get("contexts_complete") is False:
        return "incomplete_context_capture"

    if not answer and not contexts:
        return "missing_answer_and_contexts"
    if not answer:
        return "missing_answer"
    if not contexts:
        return "missing_contexts"
    return None


def _classify_ragas_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usable: list[dict[str, Any]] = []
    unevaluated: list[dict[str, Any]] = []
    for row in rows:
        reason = _unevaluated_reason(row)
        if reason is None:
            usable.append(row)
            continue
        unevaluated.append(
            {
                "id": row.get("id"),
                "outcome": row.get("outcome"),
                "error": _row_error(row),
                "reason": reason,
            }
        )
    return usable, unevaluated


def _finalize_ragas_scores(
    *,
    rows: list[dict[str, Any]],
    usable: list[dict[str, Any]],
    unevaluated: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    metric_cols: list[str],
    llm_model: str | None,
    embedding_model: str | None,
) -> dict[str, Any]:
    """Account for judge failures at case and metric granularity."""
    per_case: list[dict[str, Any]] = []
    for index, case_row in enumerate(usable):
        raw_scores = score_rows[index] if index < len(score_rows) else {}
        scores = {
            column: _parse_metric_value(raw_scores.get(column))
            for column in metric_cols
        }
        missing_metrics = [column for column, value in scores.items() if value is None]
        if not metric_cols:
            missing_metrics = ["<no_metric_columns>"]
        per_case.append(
            {
                "id": case_row.get("id"),
                "question": case_row.get("question"),
                "scores": scores,
                "missing_metrics": missing_metrics,
                "fully_evaluated": not missing_metrics,
            }
        )

    metric_coverage: dict[str, dict[str, Any]] = {}
    summary: dict[str, float | None] = {}
    for column in metric_cols:
        values = [
            item["scores"][column]
            for item in per_case
            if item["scores"].get(column) is not None
        ]
        evaluated_count = len(values)
        metric_coverage[column] = {
            "evaluated": evaluated_count,
            "missing": len(usable) - evaluated_count,
            "coverage": (round(evaluated_count / len(rows), 4) if rows else 0.0),
            "eligible_coverage": (
                round(evaluated_count / len(usable), 4) if usable else 0.0
            ),
        }
        summary[column] = round(sum(values) / len(values), 4) if values else None

    fully_evaluated = sum(item["fully_evaluated"] for item in per_case)
    partially_evaluated = sum(
        any(value is not None for value in item["scores"].values())
        and not item["fully_evaluated"]
        for item in per_case
    )
    judge_incomplete = [
        {
            "id": item.get("id"),
            "missing_metrics": item["missing_metrics"],
        }
        for item in per_case
        if not item["fully_evaluated"]
    ]
    return {
        "eligible": len(usable),
        "evaluated": fully_evaluated,
        "partially_evaluated": partially_evaluated,
        "skipped": len(rows) - fully_evaluated,
        "coverage": round(fully_evaluated / len(rows), 4) if rows else 0.0,
        "eligibility_coverage": (round(len(usable) / len(rows), 4) if rows else 0.0),
        "unevaluated": unevaluated,
        "judge_incomplete": judge_incomplete,
        "metric_coverage": metric_coverage,
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "summary": summary,
        "per_case": per_case,
        "raw_columns": metric_cols,
    }


def run_ragas_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable, unevaluated = _classify_ragas_rows(rows)
    if not usable:
        # Return a normal result so the caller can still persist ragas.json and
        # report.txt for failed/clarified/empty batches.
        return {
            "eligible": 0,
            "evaluated": 0,
            "partially_evaluated": 0,
            "skipped": len(rows),
            "coverage": 0.0,
            "eligibility_coverage": 0.0,
            "unevaluated": unevaluated,
            "judge_incomplete": [],
            "metric_coverage": {},
            "llm_model": None,
            "embedding_model": None,
            "summary": {},
            "per_case": [],
            "raw_columns": [],
        }

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise ImportError(
            "RAGAS eval deps missing or incompatible. Install with: uv sync --extra eval"
        ) from exc

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
        AnswerCorrectness(llm=eval_llm, embeddings=embeddings),
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

    metric_cols = _metric_columns(df)
    score_rows = [dict(row) for _, row in df.iterrows()]
    return _finalize_ragas_scores(
        rows=rows,
        usable=usable,
        unevaluated=unevaluated,
        score_rows=score_rows,
        metric_cols=metric_cols,
        llm_model=llm_model,
        embedding_model=embed_model,
    )


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
    _atomic_write_text(
        meta_path,
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
    )

    csv_path: Path | None = None
    if ragas:
        ragas_path = out_dir / "ragas.json"
        _atomic_write_text(
            ragas_path,
            json.dumps(ragas, ensure_ascii=False, indent=2) + "\n",
        )
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
    collection_diagnostics = meta.get("collection_diagnostics") or {}
    if collection_diagnostics:
        lines.append("--- Collection diagnostics ---")
        for key, value in collection_diagnostics.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    if ragas:
        lines.append("--- RAGAS summary ---")
        lines.append(f"eligible: {ragas.get('eligible', 0)}")
        lines.append(f"eligibility_coverage: {ragas.get('eligibility_coverage', 0.0)}")
        lines.append(f"evaluated: {ragas.get('evaluated', 0)}")
        lines.append(f"partially_evaluated: {ragas.get('partially_evaluated', 0)}")
        lines.append(f"skipped: {ragas.get('skipped', 0)}")
        lines.append(f"coverage: {ragas.get('coverage', 0.0)}")
        for key, val in (ragas.get("summary") or {}).items():
            lines.append(f"  {key}: {val}")
        metric_coverage = ragas.get("metric_coverage") or {}
        if metric_coverage:
            lines.append("metric_coverage:")
            for key, values in metric_coverage.items():
                lines.append(
                    f"  {key}: {values.get('evaluated', 0)}/{len(rows)} "
                    f"({values.get('coverage', 0.0)})"
                )
        lines.append("")
        lines.append(
            "id   faithfulness  answer_relevancy  answer_correctness  "
            "context_recall  context_precision"
        )

        def _fmt(scores: dict[str, Any], key: str) -> str:
            val = scores.get(key)
            return f"{val:.3f}" if isinstance(val, (int, float)) else "—"

        for item in ragas.get("per_case") or []:
            scores = item.get("scores") or {}

            lines.append(
                f"{item.get('id')!s:>3}  "
                f"{_fmt(scores, 'faithfulness'):>13}  "
                f"{_fmt(scores, 'answer_relevancy'):>16}  "
                f"{_fmt(scores, 'answer_correctness'):>18}  "
                f"{_fmt(scores, 'context_recall'):>14}  "
                f"{_fmt(scores, 'context_precision'):>17}"
            )
        if csv_path:
            lines.append("")
            lines.append(f"csv: {csv_path}")
        unevaluated = ragas.get("unevaluated") or []
        if unevaluated:
            lines.append("")
            lines.append("--- Unevaluated rows ---")
            for item in unevaluated:
                detail = f"outcome={item.get('outcome') or '—'}"
                if item.get("error"):
                    detail += f", error={item['error']}"
                lines.append(f"  {item.get('id')}: {item.get('reason')} ({detail})")
        judge_incomplete = ragas.get("judge_incomplete") or []
        if judge_incomplete:
            lines.append("")
            lines.append("--- Incomplete judge rows ---")
            for item in judge_incomplete:
                missing = ", ".join(item.get("missing_metrics") or [])
                lines.append(f"  {item.get('id')}: missing {missing}")
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
        out_dir = _RESULTS_DIR / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        collect_path = out_dir / "run.jsonl"
        collection_diagnostics: dict[str, Any] = {}

        def _checkpoint(partial_rows: list[dict[str, Any]]) -> None:
            _write_jsonl(collect_path, partial_rows)

        # Persist metadata before the first paid call and checkpoint after each
        # case, so an interruption or later scoring failure cannot erase work.
        _atomic_write_text(
            out_dir / "meta.json",
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        )
        rows = await collect_rows(
            cases,
            mode=mode,
            wd=wd,
            pod=pod,
            on_progress=_checkpoint,
            diagnostics=collection_diagnostics,
        )
        if collection_diagnostics:
            meta["collection_diagnostics"] = collection_diagnostics
            _atomic_write_text(
                out_dir / "meta.json",
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            )
        _write_jsonl(collect_path, rows)
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
    _atomic_write_text(report_path, report)
    print(report)
    print(f"Report: {report_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help="Versioned evaluation dataset (default: canonical Shili17 fixture)",
    )
    p.add_argument(
        "--ids",
        default="",
        help="Comma-separated case ids (default: all in dataset)",
    )
    p.add_argument(
        "--mode", default="", help="LightRAG mode (default: RAG_QUERY_MODE or mix)"
    )
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
