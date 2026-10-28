"""Pure conversion and artifact tests for the canonical evaluation runner."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_rag_evaluation as runner

from raganything.evaluation_legacy import load_canonical_fixture
from raganything.evaluation_schema import (
    EvaluationCase,
    EvaluationRun,
    RunManifest,
    load_evaluation_run,
)


def _case() -> EvaluationCase:
    return EvaluationCase.from_dict(
        {
            "schema_version": "rag-evaluation-case/v1",
            "case_id": "manual:q1",
            "query": "润滑周期？",
            "expected_action": "answer",
            "vital_claims": ["每周润滑"],
            "evidence": [{"doc_id": "manual-a", "page": 1}],
            "expected_images": [
                {
                    "image_id": "figure-a",
                    "evidence": [
                        {
                            "doc_id": "manual-a",
                            "page": 1,
                            "bbox": [0, 0, 10, 10],
                            "modality": "image",
                        }
                    ],
                }
            ],
        }
    )


def test_schema_trace_converts_product_trace_and_scores_hard_gate(tmp_path) -> None:
    raw_trace = {
        "schema_version": "rag_query_trace_v1",
        "outcome": "answer",
        "answer": "主轴需要每周润滑。",
        "duration_ms": 125,
        "error": None,
        "contexts_complete": True,
        "terminal_event": {"type": "done"},
        "context_documents": [
            {
                "source": "manual-a",
                "chunk_id": "chunk-1",
                "page": 1,
                "hash": "sha256:manual-a",
                "modality": "text",
                "rerank_score": 0.91,
                "content": "每周润滑",
            }
        ],
        "images": [
            {
                "target_id": "figure-a",
                "path": tmp_path / "figure.png",
                "evidence": [
                    {
                        "doc_id": "manual-a",
                        "page": 1,
                        "bbox": [0, 0, 10, 10],
                        "modality": "image",
                    }
                ],
            }
        ],
        "debug_state": {"parser_root": tmp_path},
    }

    trace = runner._schema_trace(
        _case(),
        raw_trace,
        started_at="2026-10-26T01:00:00Z",
        completed_at="2026-10-26T01:00:01Z",
    )

    assert trace.observed_action.value == "answer"
    assert trace.retrieved_evidence[0].anchor.doc_id == "manual-a"
    assert trace.images[0].image_id == "figure-a"
    assert trace.metrics["hard_gate_passed"] is True
    assert trace.trace["debug_state"]["parser_root"] == str(tmp_path)


def test_canonical_legacy_gate_prevents_wrong_answer_false_green() -> None:
    case = load_canonical_fixture("shili17").cases[0]

    trace = runner._schema_trace(
        case,
        {
            "outcome": "answer",
            "answer": "完全错误但非空",
            "duration_ms": 1,
            "context_documents": [],
            "images": [],
            "placements": [],
        },
        started_at="2026-10-26T01:00:00Z",
        completed_at="2026-10-26T01:00:01Z",
    )

    assert trace.metrics["generic_hard_gate_passed"] is False
    assert trace.metrics["legacy_augmented_generic_gate_passed"] is False
    assert trace.metrics["legacy_hard_gate_passed"] is False
    assert trace.metrics["hard_gate_passed"] is False
    assert trace.trace["deterministic_diagnostics"]["legacy"]["text"]["missing"] == [
        "missing:NB9-Smart",
        "missing:NB10-Smart",
    ]


def test_canonical_legacy_gate_still_requires_typed_gold_evidence() -> None:
    case = load_canonical_fixture("shili17").cases[0]
    raw_trace = {
        "outcome": "answer",
        "answer": "适用于 NB9-Smart 和 NB10-Smart。",
        "duration_ms": 1,
        "contexts_complete": True,
        "terminal_event": {"type": "done"},
        "context_documents": [
            {
                "source": "manual.pdf",
                "chunk_id": "runtime-1",
                "content": "NB9-Smart 和 NB10-Smart",
            }
        ],
        "images": [],
        "placements": [],
    }

    trace = runner._schema_trace(
        case,
        raw_trace,
        started_at="2026-10-26T01:00:00Z",
        completed_at="2026-10-26T01:00:01Z",
    )

    assert trace.metrics["generic_hard_gate_passed"] is False
    assert trace.metrics["answer_evidence_present"] is True
    assert trace.metrics["answer_has_gold_evidence"] is False
    assert trace.metrics["legacy_augmented_generic_gate_passed"] is False
    assert trace.metrics["legacy_hard_gate_passed"] is True
    assert trace.metrics["hard_gate_passed"] is False


def test_write_run_emits_schema_valid_json_and_jsonl(tmp_path) -> None:
    case = _case()
    trace = runner._schema_trace(
        case,
        {
            "outcome": "answer",
            "answer": "每周润滑",
            "duration_ms": 5,
            "contexts_complete": True,
            "terminal_event": {"type": "done"},
            "context_documents": [
                {
                    "source": "manual-a",
                    "chunk_id": "c1",
                    "page": 1,
                    "hash": "sha256:manual-a",
                    "content": "每周润滑",
                }
            ],
            "images": [
                {
                    "image_id": "figure-a",
                    "evidence": [
                        {
                            "doc_id": "manual-a",
                            "page": 1,
                            "bbox": [0, 0, 10, 10],
                            "modality": "image",
                        }
                    ],
                }
            ],
        },
        started_at="2026-10-26T01:00:00Z",
        completed_at="2026-10-26T01:00:01Z",
    )
    run = EvaluationRun(
        manifest=RunManifest(
            run_id="run-1",
            dataset_id="manual",
            dataset_version="1",
            dataset_hash="sha256:test",
            started_at="2026-10-26T01:00:00Z",
            runner="test",
        ),
        traces=(trace,),
        completed_at="2026-10-26T01:00:02Z",
        summary=runner._summary((trace,)),
    )

    json_path, jsonl_path, report_path = runner._write_run(
        tmp_path / "run.json", run, (case,)
    )

    assert (json_path, jsonl_path, report_path) == (
        tmp_path / "run.json",
        tmp_path / "run.jsonl",
        tmp_path / "run.md",
    )
    assert len({json_path, jsonl_path, report_path}) == 3
    assert load_evaluation_run(json_path).to_dict() == run.to_dict()
    assert load_evaluation_run(jsonl_path).to_dict() == run.to_dict()
    assert "manual:q1" in report_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        runner._write_run(json_path, run, (case,))

    unsafe_path = tmp_path / "unsafe" / "run.jsonl"
    unsafe_path.parent.mkdir()
    unsafe_path.write_text("do not overwrite\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must end in '.json'"):
        runner._write_run(unsafe_path, run, (case,), overwrite=True)
    assert unsafe_path.read_text(encoding="utf-8") == "do not overwrite\n"
    assert not (unsafe_path.parent / "run.md").exists()


@pytest.mark.asyncio
async def test_execute_disables_persistent_llm_cache_and_restores_environment(
    monkeypatch, tmp_path
) -> None:
    calls: list[object] = []

    class FakeRag:
        async def finalize_storages(self) -> None:
            calls.append("finalized")

    fake_rag = FakeRag()

    async def build_rag(wd, pod, *, enable_llm_cache=None):
        calls.append((wd, pod, enable_llm_cache))
        return fake_rag, SimpleNamespace(), SimpleNamespace()

    fake_rpc = ModuleType("rag_pipeline_parse_graph_chat")
    fake_rpc._build_rag = build_rag
    fake_web = ModuleType("rag_web_server")
    fake_web.state = SimpleNamespace()
    fake_web.QueryBody = SimpleNamespace

    async def execute_query_with_trace(body):
        calls.append(body.query)
        return {
            "outcome": "answer",
            "answer": "NB9-Smart、NB10-Smart",
            "duration_ms": 1,
            "context_documents": [],
            "images": [],
            "placements": [],
            "terminal_event": {"type": "done"},
        }

    fake_web.execute_query_with_trace = execute_query_with_trace
    monkeypatch.setitem(sys.modules, "rag_pipeline_parse_graph_chat", fake_rpc)
    monkeypatch.setitem(sys.modules, "rag_web_server", fake_web)
    monkeypatch.setattr(runner, "_git_revision", lambda: ("test-revision", False))
    monkeypatch.setenv("EVAL_RUNNER_SENTINEL", "preserved")
    before = dict(os.environ)

    checkpoints = []
    run = await runner._execute(
        SimpleNamespace(
            seed=7,
            dataset=runner.DEFAULT_DATASET,
            ids="shili17:q01",
            working_dir=tmp_path / "storage",
            parser_output_dir=tmp_path / "parser",
            mode="mix",
            corpus_id="test-corpus",
            corpus_hash="sha256:test",
        ),
        on_progress=checkpoints.append,
    )

    assert calls[0] == (
        tmp_path / "storage",
        tmp_path / "parser",
        False,
    )
    assert calls[-1] == "finalized"
    assert len(run.traces) == 1
    assert len(checkpoints) == 1
    assert len(checkpoints[0].traces) == 1
    assert dict(os.environ) == before


@pytest.mark.asyncio
async def test_execute_preserves_case_and_finalizer_errors(
    monkeypatch, tmp_path
) -> None:
    class FakeRag:
        async def finalize_storages(self) -> None:
            raise RuntimeError("finalizer failed")

    async def build_rag(*_args, **_kwargs):
        return FakeRag(), SimpleNamespace(), SimpleNamespace()

    fake_rpc = ModuleType("rag_pipeline_parse_graph_chat")
    fake_rpc._build_rag = build_rag
    fake_web = ModuleType("rag_web_server")
    fake_web.state = SimpleNamespace()
    fake_web.QueryBody = SimpleNamespace

    async def execute_query_with_trace(_body):
        raise ValueError("provider failed")

    fake_web.execute_query_with_trace = execute_query_with_trace
    monkeypatch.setitem(sys.modules, "rag_pipeline_parse_graph_chat", fake_rpc)
    monkeypatch.setitem(sys.modules, "rag_web_server", fake_web)
    monkeypatch.setattr(runner, "_git_revision", lambda: ("test-revision", False))
    checkpoints = []

    run = await runner._execute(
        SimpleNamespace(
            seed=7,
            dataset=runner.DEFAULT_DATASET,
            ids="shili17:q01",
            working_dir=tmp_path / "storage",
            parser_output_dir=tmp_path / "parser",
            mode="mix",
            corpus_id="test-corpus",
            corpus_hash="sha256:test",
        ),
        on_progress=checkpoints.append,
    )

    assert len(run.traces) == 1
    assert run.traces[0].status.value == "error"
    assert run.traces[0].error == "ValueError: provider failed"
    assert run.summary["finalization_error"] == "RuntimeError: finalizer failed"
    assert len(checkpoints) == 2
    assert "finalization_error" not in checkpoints[0].summary
    assert checkpoints[1].summary["finalization_error"] == (
        "RuntimeError: finalizer failed"
    )


def test_conversation_cases_fail_fast_instead_of_dropping_history() -> None:
    case = EvaluationCase.from_dict(
        {
            "schema_version": "rag-evaluation-case/v1",
            "case_id": "multi-turn:q1",
            "query": "那它多久一次？",
            "conversation": [
                {"role": "user", "content": "主轴如何保养？"},
                {"role": "assistant", "content": "请问具体哪项要求？"},
            ],
            "expected_action": "answer",
            "vital_claims": ["每周一次"],
        }
    )

    with pytest.raises(ValueError, match="does not accept conversation history"):
        runner._ensure_supported_cases((case,))


def test_resolved_model_manifest_records_defaults_and_redacts_endpoint(
    monkeypatch,
) -> None:
    for name in (
        "LLM_MODEL",
        "VISION_MODEL",
        "EMBEDDING_BACKEND",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIM",
        "RERANK_BINDING",
        "RERANK_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "LLM_BINDING_HOST",
        "https://user:secret@example.test/v1?api_key=hidden",
    )

    models = runner._models_from_environment()

    assert models["generator"] == {
        "model": "gpt-4o-mini",
        "endpoint": "https://example.test/v1",
    }
    assert models["vision"]["model"] == "gpt-4o-mini"
    assert models["embedding"]["model"] == "text-embedding-3-small"
    assert models["embedding"]["dimension"] == 1536
    assert models["reranker"]["binding"] == "none"
    assert "secret" not in str(models)
    assert "hidden" not in str(models)
    assert runner._safe_endpoint("https://example.test:invalid/v1") == "configured"


def test_release_gate_requires_frozen_nonempty_corpus(tmp_path) -> None:
    args = SimpleNamespace(
        fail_on_hard_gate=True,
        corpus_hash="",
        working_dir=tmp_path / "storage",
    )
    with pytest.raises(ValueError, match="requires --corpus-hash"):
        runner._ensure_release_preconditions(args)

    args.corpus_hash = "sha256:" + "a" * 64
    with pytest.raises(ValueError, match="non-empty existing"):
        runner._ensure_release_preconditions(args)

    args.working_dir.mkdir()
    (args.working_dir / "kv_store.json").write_text("{}", encoding="utf-8")
    runner._ensure_release_preconditions(args)

    args.fail_on_hard_gate = False
    args.corpus_hash = ""
    runner._ensure_release_preconditions(args)
