#!/usr/bin/env python3
"""Run a versioned evaluation dataset through the production Web query path.

This runner is intentionally sequential and deterministic at the orchestration
layer. It records every answer, clarification, rejection, empty response, and
error in the versioned evaluation schema. The built-in scorer is lexical and
structural; LLM-judge scoring can be added to the resulting run artifact later.

Examples::

  uv run python scripts/run_rag_evaluation.py --validate-only
  uv run python scripts/run_rag_evaluation.py --ids shili17:q01,shili17:q02
  uv run python scripts/run_rag_evaluation.py --fail-on-hard-gate
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for import_path in (ROOT, SCRIPTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from raganything.evaluation_legacy_metrics import score_legacy_case
from raganything.evaluation_metrics import score_evaluation_case
from raganything.evaluation_schema import (
    EvaluationCase,
    EvaluationRun,
    EvaluationTraceResult,
    EvidenceAnchor,
    ExpectedAction,
    RetrievedEvidence,
    RunManifest,
    SchemaValidationError,
    TraceImage,
    TraceStatus,
    load_evaluation_dataset,
)

DEFAULT_DATASET = ROOT / "tests" / "fixtures" / "evaluation" / "shili17.json"
DEFAULT_RESULTS = ROOT / "logs" / "evaluation_runs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Detach runtime objects into strict JSON without losing trace structure."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, set):
        return [_json_safe(child) for child in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return str(value)


def _git_revision() -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision + ("-dirty" if dirty else ""), dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _selected_cases(
    cases: tuple[EvaluationCase, ...], raw_ids: str
) -> tuple[EvaluationCase, ...]:
    if not raw_ids.strip():
        return cases
    requested = {item.strip() for item in raw_ids.split(",") if item.strip()}
    selected = tuple(case for case in cases if case.case_id in requested)
    missing = requested - {case.case_id for case in selected}
    if missing:
        raise ValueError(f"Unknown case id(s): {', '.join(sorted(missing))}")
    return selected


def _ensure_supported_cases(cases: tuple[EvaluationCase, ...]) -> None:
    """Fail rather than silently dropping interaction history."""
    unsupported = [case.case_id for case in cases if case.conversation]
    if unsupported:
        raise ValueError(
            "The production QueryBody path does not accept conversation history; "
            "multi-turn cases require an explicit session-replay runner. Unsupported "
            f"case(s): {', '.join(unsupported)}"
        )


def _first_text(document: dict[str, Any]) -> str | None:
    for key in ("content", "text", "chunk_content", "page_content"):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _positive_page(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page >= 1 else None


def _retrieved_evidence(raw_trace: dict[str, Any]) -> tuple[RetrievedEvidence, ...]:
    rows: list[RetrievedEvidence] = []
    documents = raw_trace.get("context_documents")
    if not isinstance(documents, list):
        return ()
    for rank, raw_document in enumerate(documents, start=1):
        if not isinstance(raw_document, dict):
            continue
        document = _json_safe(raw_document)
        doc_id = str(
            document.get("document_id")
            or document.get("manual_id")
            or document.get("source_document_id")
            or document.get("file_path")
            or document.get("source")
            or document.get("title")
            or "unknown-document"
        )
        chunk_id = str(
            document.get("chunk_id")
            or document.get("id")
            or document.get("reference_id")
            or f"runtime-rank-{rank}"
        )
        digest = document.get("hash") or document.get("document_hash")
        score = document.get("rerank_score", document.get("score"))
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            score = None
        page = _positive_page(document.get("page") or document.get("page_number"))
        anchor_values = {
            "doc_id": doc_id,
            "hash": str(digest) if digest not in (None, "") else None,
            "page": page,
            "span": document.get("span") or document.get("text_span"),
            "bbox": document.get("bbox") if page is not None else None,
            "chunk_id": chunk_id,
            "modality": str(document.get("modality") or "text"),
        }
        try:
            anchor = EvidenceAnchor(**anchor_values)
        except SchemaValidationError:
            # Keep stable identity/rank even if optional runtime locator metadata
            # is malformed. Exact-anchor scoring will correctly treat omitted
            # locators as unsupported rather than crashing the paid run.
            anchor_values["span"] = None
            anchor_values["bbox"] = None
            anchor = EvidenceAnchor(**anchor_values)
        rows.append(
            RetrievedEvidence(
                anchor=anchor,
                rank=rank,
                score=float(score) if score is not None else None,
                text=_first_text(document),
            )
        )
    return tuple(rows)


def _runtime_image_evidence(image: dict[str, Any]) -> tuple[EvidenceAnchor, ...]:
    explicit = image.get("evidence")
    anchors: list[EvidenceAnchor] = []
    if isinstance(explicit, list):
        for value in explicit:
            try:
                anchors.append(
                    EvidenceAnchor.from_dict(value, "runtime_image.evidence")
                )
            except (SchemaValidationError, TypeError):
                continue
        return tuple(anchors)

    doc_id = (
        image.get("document_id")
        or image.get("manual_id")
        or image.get("source_document_id")
        or image.get("file_path")
        or image.get("source")
    )
    page = _positive_page(image.get("page") or image.get("page_number"))
    chunk_id = image.get("chunk_id") or image.get("reference_id")
    span = image.get("span") or image.get("text_span")
    bbox = image.get("bbox") if page is not None else None
    if not doc_id or all(value is None for value in (page, chunk_id, span, bbox)):
        return ()
    try:
        return (
            EvidenceAnchor(
                doc_id=str(doc_id),
                hash=(
                    str(image.get("hash") or image.get("document_hash"))
                    if image.get("hash") or image.get("document_hash")
                    else None
                ),
                page=page,
                span=span,
                bbox=bbox,
                modality=str(image.get("modality") or "image"),
                chunk_id=str(chunk_id) if chunk_id else None,
            ),
        )
    except SchemaValidationError:
        return ()


def _trace_images(raw_trace: dict[str, Any]) -> tuple[TraceImage, ...]:
    rows: list[TraceImage] = []
    images = raw_trace.get("images")
    if not isinstance(images, list):
        return ()
    for index, raw_image in enumerate(images, start=1):
        if not isinstance(raw_image, dict):
            continue
        image = _json_safe(raw_image)
        uri = image.get("url") or image.get("path") or image.get("image_path")
        image_id = str(
            image.get("image_id")
            or image.get("target_id")
            or image.get("id")
            or image.get("reference_id")
            or uri
            or f"runtime-image-{index}"
        )
        rows.append(
            TraceImage(
                image_id=image_id,
                uri=str(uri) if uri not in (None, "") else None,
                evidence=_runtime_image_evidence(image),
                metadata=image,
            )
        )
    return tuple(rows)


def _observed_action(raw_trace: dict[str, Any]) -> ExpectedAction | None:
    outcome = str(raw_trace.get("outcome") or "")
    if outcome == "answer":
        return ExpectedAction.ANSWER
    if outcome == "clarify":
        return ExpectedAction.CLARIFY
    if outcome in {"reject", "empty"}:
        return ExpectedAction.ABSTAIN
    return None


def _schema_trace(
    case: EvaluationCase,
    raw_trace: dict[str, Any],
    *,
    started_at: str,
    completed_at: str,
) -> EvaluationTraceResult:
    error = str(raw_trace.get("error") or "").strip() or None
    base = EvaluationTraceResult(
        case_id=case.case_id,
        status=TraceStatus.ERROR if error else TraceStatus.COMPLETED,
        observed_action=_observed_action(raw_trace),
        answer=str(raw_trace.get("answer") or "").strip() or None,
        started_at=started_at,
        completed_at=completed_at,
        retrieved_evidence=_retrieved_evidence(raw_trace),
        images=_trace_images(raw_trace),
        timings_ms={"total": float(raw_trace.get("duration_ms") or 0)},
        error=error,
        trace=_json_safe(raw_trace),
    )
    generic_score = score_evaluation_case(case, base)
    metrics = dict(generic_score.metrics)
    diagnostics: dict[str, Any] = {
        "generic": generic_score.diagnostics,
        "legacy": None,
    }
    if isinstance(case.metadata.get("legacy"), dict):
        legacy_score = score_legacy_case(case, raw_trace)
        generic_hard_gate = metrics.get("hard_gate_passed") is True
        generic_conditions = dict(
            generic_score.diagnostics.get("hard_gate", {}).get("conditions", {})
        )
        # Migrated suites carry their correctness-bearing checks in the
        # compatibility scorer rather than typed vital_claims. All other
        # generic conditions (status, action, real evidence, typed anchors and
        # images when present) remain mandatory.
        generic_conditions["answer_has_graded_content"] = True
        legacy_augmented_generic_gate = all(generic_conditions.values())
        legacy_hard_gate = legacy_score.metrics.get("legacy_hard_gate_passed") is True
        metrics["generic_hard_gate_passed"] = generic_hard_gate
        metrics["legacy_augmented_generic_gate_passed"] = legacy_augmented_generic_gate
        metrics.update(legacy_score.metrics)
        metrics["hard_gate_passed"] = legacy_augmented_generic_gate and legacy_hard_gate
        diagnostics["legacy"] = legacy_score.diagnostics
        diagnostics["legacy_augmented_generic_conditions"] = generic_conditions
    diagnostics["combined_hard_gate_passed"] = metrics.get("hard_gate_passed") is True
    return replace(
        base,
        metrics=metrics,
        trace={
            **dict(base.trace),
            "deterministic_diagnostics": _json_safe(diagnostics),
        },
    )


def _summary(traces: tuple[EvaluationTraceResult, ...]) -> dict[str, Any]:
    metric_names = sorted(
        {
            name
            for trace in traces
            for name in trace.metrics
            if name != "hard_gate_passed"
        }
    )
    metric_means: dict[str, float | None] = {}
    for name in metric_names:
        values = [
            float(value)
            for trace in traces
            if (value := trace.metrics.get(name)) is not None
            and not isinstance(value, bool)
        ]
        metric_means[name] = round(sum(values) / len(values), 4) if values else None
    hard_gate_passes = sum(
        trace.metrics.get("hard_gate_passed") is True for trace in traces
    )
    return {
        "cases": len(traces),
        "completed": sum(trace.status is TraceStatus.COMPLETED for trace in traces),
        "errors": sum(trace.status is TraceStatus.ERROR for trace in traces),
        "hard_gate_passes": hard_gate_passes,
        "hard_gate_failures": len(traces) - hard_gate_passes,
        "metric_means": metric_means,
    }


def _error_trace(
    case: EvaluationCase,
    *,
    started_at: str,
    error: BaseException,
    raw_trace: dict[str, Any] | None = None,
) -> EvaluationTraceResult:
    message = f"{type(error).__name__}: {error}"
    trace = {
        "schema_version": "rag_query_trace_v1",
        "query": case.query,
        "outcome": "error",
        "error": message,
        "runner_exception": True,
    }
    if raw_trace is not None:
        trace["partial_trace"] = _json_safe(raw_trace)
    return EvaluationTraceResult(
        case_id=case.case_id,
        status=TraceStatus.ERROR,
        observed_action=None,
        answer=None,
        started_at=started_at,
        completed_at=_utc_now(),
        metrics={
            "status_completed": False,
            "action_match": False,
            "hard_gate_passed": False,
        },
        error=message,
        trace=trace,
    )


def _run_snapshot(
    manifest: RunManifest,
    traces: list[EvaluationTraceResult],
    *,
    finalization_error: str | None = None,
) -> EvaluationRun:
    trace_tuple = tuple(traces)
    summary = _summary(trace_tuple)
    if finalization_error is not None:
        summary["finalization_error"] = finalization_error
    return EvaluationRun(
        manifest=manifest,
        traces=trace_tuple,
        completed_at=_utc_now(),
        summary=summary,
    )


def _safe_endpoint(value: str | None) -> str | None:
    """Record endpoint identity without credentials, query strings, or fragments."""
    if not value:
        return None
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        return "configured"
    host = parsed.hostname
    try:
        port = parsed.port
    except ValueError:
        return "configured"
    if port is not None:
        host = f"{host}:{port}"
    return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))


def _models_from_environment() -> dict[str, Any]:
    """Return resolved runtime model defaults, never an ambiguous empty mapping."""
    generator = (os.getenv("LLM_MODEL") or "gpt-4o-mini").strip()
    embedding_backend = (os.getenv("EMBEDDING_BACKEND") or "openai").strip().lower()
    if embedding_backend == "hf":
        default_embedding_model = "BAAI/bge-m3"
        default_embedding_dim = 1024
    else:
        default_embedding_model = "text-embedding-3-small"
        default_embedding_dim = 1536
    reranker_binding = (os.getenv("RERANK_BINDING") or "none").strip().lower()
    return {
        "generator": {
            "model": generator,
            "endpoint": _safe_endpoint(
                os.getenv("LLM_BINDING_HOST") or os.getenv("OPENAI_BASE_URL")
            ),
        },
        "vision": {
            "model": (os.getenv("VISION_MODEL") or generator).strip(),
        },
        "embedding": {
            "backend": embedding_backend,
            "model": (os.getenv("EMBEDDING_MODEL") or default_embedding_model).strip(),
            "dimension": int(os.getenv("EMBEDDING_DIM") or str(default_embedding_dim)),
            "endpoint": _safe_endpoint(
                os.getenv("EMBEDDING_BINDING_HOST")
                or os.getenv("LLM_BINDING_HOST")
                or os.getenv("OPENAI_BASE_URL")
            ),
        },
        "reranker": {
            "binding": reranker_binding,
            "model": (os.getenv("RERANK_MODEL") or "").strip() or None,
            "endpoint": _safe_endpoint(os.getenv("RERANK_BINDING_HOST")),
        },
    }


def _safe_query_configuration() -> dict[str, Any]:
    """Capture behavior-affecting query settings without API credentials."""
    defaults: dict[str, str] = {
        "RERANK_BY_DEFAULT": "true",
        "CLARIFY_GATE_ENABLED": "",
        "CLARIFY_CANDIDATE_MIN_RERANK_SCORE": "",
        "CLARIFY_DIRECT_RERANK_MIN": "",
        "PROMPT_LANGUAGE": "",
        "KEYWORD_LLM_ENABLE_THINKING": "",
    }
    return {
        name: os.getenv(name, default)
        for name, default in defaults.items()
        if os.getenv(name, default) != ""
    }


def _directory_contains_file(path: Path) -> bool:
    return (
        path.is_dir()
        and next((item for item in path.rglob("*") if item.is_file()), None) is not None
    )


def _ensure_release_preconditions(args: argparse.Namespace) -> None:
    """Prevent a release gate from running against an unidentified/empty corpus."""
    if not args.fail_on_hard_gate:
        return
    corpus_hash = str(args.corpus_hash or "").strip()
    if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", corpus_hash):
        raise ValueError(
            "--fail-on-hard-gate requires --corpus-hash sha256:<64 hex chars>"
        )
    working_dir = args.working_dir.resolve()
    if not _directory_contains_file(working_dir):
        raise ValueError(
            "--fail-on-hard-gate requires a non-empty existing RAG working "
            f"directory: {working_dir}"
        )


def _evaluation_artifact_paths(output: Path) -> tuple[Path, Path, Path]:
    """Return non-colliding artifact paths for a JSON envelope target."""
    if output.suffix.lower() != ".json":
        raise ValueError(
            "Evaluation output must end in '.json' so JSON, JSONL, and Markdown "
            f"artifacts remain distinct: {output}"
        )
    return output, output.with_suffix(".jsonl"), output.with_suffix(".md")


def _preflight_output(output: Path, *, overwrite: bool = False) -> None:
    paths = _evaluation_artifact_paths(output)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite evaluation artifact(s): "
            + ", ".join(str(path) for path in existing)
        )


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace one artifact atomically so a failed checkpoint keeps its predecessor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _output_argument(value: str) -> Path:
    output = Path(value)
    try:
        _evaluation_artifact_paths(output)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return output


def _write_run(
    output: Path,
    run: EvaluationRun,
    cases: tuple[EvaluationCase, ...],
    *,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    output, jsonl, report = _evaluation_artifact_paths(output)
    _preflight_output(output, overwrite=overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        jsonl,
        "\n".join(
            json.dumps(record, ensure_ascii=False) for record in run.to_jsonl_records()
        )
        + "\n",
    )
    _atomic_write_text(
        output,
        json.dumps(run.to_dict(), ensure_ascii=False, indent=2) + "\n",
    )
    summary = run.summary
    lines = [
        "# Industrial RAG evaluation",
        "",
        f"- Run: `{run.manifest.run_id}`",
        f"- Dataset: `{run.manifest.dataset_id}` / `{run.manifest.dataset_version}`",
        f"- Cases: {summary.get('cases', 0)}",
        f"- Hard gates: {summary.get('hard_gate_passes', 0)}/{summary.get('cases', 0)}",
        f"- Errors: {summary.get('errors', 0)}",
        "",
        "| Case | Status | Expected | Observed | Hard gate | Total ms |",
        "|---|---|---|---|---:|---:|",
    ]
    expected_actions = {case.case_id: case.expected_action.value for case in cases}
    for trace in run.traces:
        lines.append(
            f"| {trace.case_id} | {trace.status.value} | "
            f"{expected_actions[trace.case_id]} | "
            f"{trace.observed_action.value if trace.observed_action else '—'} | "
            f"{'✓' if trace.metrics.get('hard_gate_passed') else '✗'} | "
            f"{trace.timings_ms.get('total', 0):.0f} |"
        )
    _atomic_write_text(report, "\n".join(lines) + "\n")
    return output, jsonl, report


async def _execute(
    args: argparse.Namespace,
    *,
    on_progress: Callable[[EvaluationRun], None] | None = None,
) -> EvaluationRun:
    if args.seed is not None:
        random.seed(args.seed)
    dataset_path = args.dataset.resolve()
    dataset = load_evaluation_dataset(dataset_path)
    cases = _selected_cases(dataset.cases, args.ids)
    if not cases:
        raise ValueError("No evaluation cases selected")
    _ensure_supported_cases(cases)

    working_dir = args.working_dir.resolve()
    parser_output_dir = args.parser_output_dir.resolve()
    mode = args.mode.strip()
    started_at = _utc_now()
    revision, dirty = _git_revision()
    manifest = RunManifest(
        run_id=f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        dataset_hash=_sha256_file(dataset_path),
        started_at=started_at,
        runner="production-web-semantic-events/v1",
        corpus_id=args.corpus_id or str(working_dir),
        corpus_hash=args.corpus_hash or None,
        code_revision=revision,
        random_seed=args.seed,
        config={
            "mode": mode,
            "working_dir": str(working_dir),
            "parser_output_dir": str(parser_output_dir),
            "case_ids": [case.case_id for case in cases],
            "parallelism": 1,
            "auto_follow_clarification": False,
            "llm_cache": False,
            "query_environment": _safe_query_configuration(),
        },
        models=_models_from_environment(),
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_worktree_dirty": dirty,
        },
    )

    original_environment = dict(os.environ)
    rag: Any = None
    traces: list[EvaluationTraceResult] = []
    finalization_error: str | None = None
    try:
        os.environ.update(
            {
                "RAG_WEB_WORKING_DIR": str(working_dir),
                "RAG_WEB_PARSER_OUTPUT_DIR": str(parser_output_dir),
            }
        )
        frozen_environment = dict(os.environ)
        import rag_pipeline_parse_graph_chat as rpc  # noqa: WPS433

        os.environ.clear()
        os.environ.update(frozen_environment)
        rag, config, _logger = await rpc._build_rag(
            working_dir,
            parser_output_dir,
            enable_llm_cache=False,
        )
        import rag_web_server as web  # noqa: WPS433

        os.environ.clear()
        os.environ.update(frozen_environment)
        web._RPC = rpc
        web.state.rag = rag
        web.state.config = config
        web.state.working_dir = str(working_dir)
        web.state.parser_output_dir = str(parser_output_dir)
        web.state.query_mode = mode
        web.state.ready = True

        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case.case_id}: {case.query}", flush=True)
            case_started = _utc_now()
            raw_trace: dict[str, Any] | None = None
            try:
                raw_trace = await web.execute_query_with_trace(
                    web.QueryBody(
                        query=case.query,
                        mode=mode,
                        stream=True,
                        clarify_choice=None,
                        clarification_id=None,
                        candidate_id=None,
                    )
                )
                trace = _schema_trace(
                    case,
                    raw_trace,
                    started_at=case_started,
                    completed_at=_utc_now(),
                )
            except Exception as exc:  # noqa: BLE001 - preserve the paid case result
                trace = _error_trace(
                    case,
                    started_at=case_started,
                    error=exc,
                    raw_trace=raw_trace,
                )
            traces.append(trace)
            if on_progress is not None:
                on_progress(_run_snapshot(manifest, traces))
            print(
                f"  status={trace.status.value} "
                f"action={trace.observed_action.value if trace.observed_action else 'none'} "
                f"hard_gate={trace.metrics.get('hard_gate_passed')} "
                f"ms={trace.timings_ms.get('total', 0):.0f}",
                flush=True,
            )
    finally:
        try:
            if rag is not None:
                try:
                    await rag.finalize_storages()
                except Exception as exc:  # noqa: BLE001 - report without losing traces
                    finalization_error = f"{type(exc).__name__}: {exc}"
                    print(
                        f"warning: storage finalization failed: {finalization_error}",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            os.environ.clear()
            os.environ.update(original_environment)

    run = _run_snapshot(
        manifest,
        traces,
        finalization_error=finalization_error,
    )
    if on_progress is not None and finalization_error is not None:
        on_progress(run)
    return run


def _parse_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ids", default="", help="Comma-separated canonical case IDs")
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path(
            os.getenv("RAG_WEB_WORKING_DIR") or ROOT / "data" / "kb_new" / "rag_storage"
        ),
    )
    parser.add_argument(
        "--parser-output-dir",
        type=Path,
        default=Path(
            os.getenv("RAG_WEB_PARSER_OUTPUT_DIR")
            or ROOT / "data" / "kb_new" / "pipeline_parse"
        ),
    )
    parser.add_argument("--mode", default=os.getenv("RAG_QUERY_MODE") or "mix")
    parser.add_argument("--corpus-id", default=os.getenv("RAG_EVAL_CORPUS_ID") or "")
    parser.add_argument(
        "--corpus-hash", default=os.getenv("RAG_EVAL_CORPUS_HASH") or ""
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional runner RNG seed (does not override provider-side sampling)",
    )
    parser.add_argument(
        "--output",
        type=_output_argument,
        default=None,
        help="JSON envelope path; sibling .jsonl and .md artifacts are also written",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fail-on-hard-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset = load_evaluation_dataset(args.dataset.resolve())
    cases = _selected_cases(dataset.cases, args.ids)
    _ensure_supported_cases(cases)
    if args.validate_only:
        print(
            f"Validated {len(cases)} case(s) from {dataset.dataset_id} "
            f"version {dataset.dataset_version}"
        )
        return 0

    _ensure_release_preconditions(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = (args.output or DEFAULT_RESULTS / stamp / "run.json").resolve()
    _preflight_output(output, overwrite=args.overwrite)
    checkpoint_written = False

    def persist_checkpoint(snapshot: EvaluationRun) -> None:
        nonlocal checkpoint_written
        _write_run(
            output,
            snapshot,
            cases,
            overwrite=True if checkpoint_written else args.overwrite,
        )
        checkpoint_written = True

    run = asyncio.run(_execute(args, on_progress=persist_checkpoint))
    json_path, jsonl_path, report_path = _write_run(
        output,
        run,
        cases,
        overwrite=True if checkpoint_written else args.overwrite,
    )
    print(f"JSON: {json_path}")
    print(f"JSONL: {jsonl_path}")
    print(f"Report: {report_path}")
    if args.fail_on_hard_gate and run.summary.get("hard_gate_failures"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
