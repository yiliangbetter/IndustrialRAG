from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from scripts import benchmark_nanxing_rag as bench


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def test_grade_normalizes_spacing_and_checks_choice_groups():
    case = {
        "must_include": ["0.6", "压力继电器"],
        "one_of": [["输入气源", "气源压力"]],
    }
    assert bench._grade("输入 气源至少 0.6 MPa，并检查压力继电器。", case) == (
        True,
        [],
    )
    passed, missing = bench._grade("压力为 0.6 MPa。", case)
    assert not passed
    assert "压力继电器" in missing


def test_grade_matches_relations_across_markdown_emphasis():
    case = {"must_match": [r"(低于|小于)9(mm|毫米)"]}

    assert bench._grade("磨损厚度低于 **9mm** 时必须更换。", case) == (True, [])


def test_summary_validity_requires_thinking_for_every_run():
    row = {
        "runs": 3,
        "completed": 3,
        "quality_passes": 3,
        "thinking_responses": 3,
    }
    assert bench._summary_is_valid([row])
    assert not bench._summary_is_valid([{**row, "thinking_responses": 2}])


@pytest.mark.asyncio
async def test_run_case_captures_stream_milestones_and_answer(monkeypatch):
    class FakeWeb:
        QueryBody = SimpleNamespace

        @staticmethod
        async def _query_stream_events(_question, _mode, _body):
            yield _sse({"type": "status", "text": "检索"})
            yield _sse({"type": "thinking_delta", "text": "分析"})
            yield _sse({"type": "answer_delta", "text": "至少0.6MPa，"})
            yield _sse({"type": "answer_delta", "text": "检查压力继电器和输入气源。"})
            yield _sse({"type": "done", "mode": "mix"})

    case = {
        "id": "pc",
        "question": "question",
        "source": "manual.pdf",
        "must_include": ["0.6", "压力继电器"],
        "one_of": [["输入气源"]],
    }
    monkeypatch.setattr(bench, "_source_basenames", lambda: ["manual.pdf"])
    row = await bench._run_case(FakeWeb, case, round_number=1, warmup=False)
    assert row["quality_pass"] is True
    assert row["completed"] is True
    assert row["source_pass"] is True
    assert row["thinking_chars"] == 2
    assert row["first_status_s"] is not None
    assert row["first_response_s"] is not None
    assert row["first_thinking_s"] is not None
    assert row["first_answer_s"] is not None
    assert row["done_s"] is not None
    assert row["event_counts"] == {
        "status": 1,
        "thinking_delta": 1,
        "answer_delta": 2,
        "done": 1,
    }


def test_load_cases_rejects_unknown_selection(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [{"id": "known", "question": "q"}]}))
    with pytest.raises(ValueError, match="Unknown case"):
        bench._load_cases(path, {"missing"})


@pytest.mark.asyncio
async def test_run_case_handles_answer_without_thinking(monkeypatch):
    class FakeWeb:
        QueryBody = SimpleNamespace

        @staticmethod
        async def _query_stream_events(_question, _mode, _body):
            yield _sse({"type": "answer_delta", "text": "合格答案"})
            yield _sse({"type": "done"})

    monkeypatch.setattr(bench, "_source_basenames", lambda: ["manual.pdf"])
    row = await bench._run_case(
        FakeWeb,
        {
            "id": "answer-only",
            "question": "q",
            "source": "manual.pdf",
            "must_include": ["合格"],
        },
        round_number=1,
        warmup=False,
    )

    assert row["first_thinking_s"] is None
    assert row["first_response_s"] == row["first_answer_s"]
    assert row["completed"] is True


@pytest.mark.asyncio
async def test_run_case_marks_error_invalid(monkeypatch):
    class FakeWeb:
        QueryBody = SimpleNamespace

        @staticmethod
        async def _query_stream_events(_question, _mode, _body):
            yield _sse({"type": "error", "message": "provider failed"})
            yield _sse({"type": "done", "error": True})

    monkeypatch.setattr(bench, "_source_basenames", lambda: ["manual.pdf"])
    row = await bench._run_case(
        FakeWeb,
        {"id": "error", "question": "q", "source": "manual.pdf"},
        round_number=1,
        warmup=False,
    )

    assert row["error"] == "provider failed"
    assert row["completed"] is False
    assert row["quality_pass"] is False


@pytest.mark.asyncio
async def test_run_case_follows_keep_original_clarification(monkeypatch):
    class FakeWeb:
        QueryBody = SimpleNamespace
        calls = 0

        @classmethod
        async def _query_stream_events(cls, _question, _mode, body):
            cls.calls += 1
            if cls.calls == 1:
                yield _sse({"type": "status", "phase": "clarify"})
                yield _sse(
                    {
                        "type": "clarification_required",
                        "data": {
                            "clarification_id": "clarify-1",
                            "gate_outcome": "offer",
                            "keep_original": {"query": "original question"},
                        },
                    }
                )
                yield _sse({"type": "done", "clarification_only": True})
                return
            assert body.clarify_choice == "keep_original"
            assert body.clarification_id == "clarify-1"
            yield _sse({"type": "thinking_delta", "text": "reason"})
            yield _sse({"type": "answer_delta", "text": "grounded answer"})
            yield _sse({"type": "done"})

    monkeypatch.setattr(bench, "_source_basenames", lambda: ["manual.pdf"])
    row = await bench._run_case(
        FakeWeb,
        {
            "id": "clarify",
            "question": "original question",
            "source": "manual.pdf",
            "must_include": ["grounded"],
        },
        round_number=1,
        warmup=False,
    )

    assert FakeWeb.calls == 2
    assert row["clarification_required"] is True
    assert row["clarification_followed"] is True
    assert row["initial_done_s"] is not None
    assert row["followup_first_response_s"] is not None
    assert row["completed"] is True
    assert row["quality_pass"] is True


def _write_complete_index(base: Path) -> None:
    storage = base / "rag_storage"
    parser = base / "pipeline_parse"
    storage.mkdir(parents=True)
    parser.mkdir(parents=True)
    status = {
        f"doc-{index}": {
            "status": "processed",
            "chunks_count": 1,
            "file_path": filename,
        }
        for index, filename in enumerate(sorted(bench.EXPECTED_SOURCE_FILES))
    }
    (storage / "kv_store_doc_status.json").write_text(
        json.dumps(status, ensure_ascii=False), encoding="utf-8"
    )
    for name in ("vdb_chunks.json", "vdb_entities.json", "vdb_relationships.json"):
        (storage / name).write_text(json.dumps({"data": [{"id": "one"}]}))
    graph = nx.Graph()
    graph.add_edge("a", "b")
    nx.write_graphml(graph, storage / "graph_chunk_entity_relation.graphml")
    for index in range(7):
        (parser / f"doc-{index}_content_list.json").write_text("[]")


def test_validate_index_checks_real_store_contents(tmp_path):
    _write_complete_index(tmp_path)

    result = bench._validate_index(tmp_path)

    assert result["documents"] == 7
    assert result["graph_nodes"] == 2
    assert result["vectors"]["vdb_chunks.json"] == 1

    (tmp_path / "rag_storage" / "vdb_chunks.json").write_text(json.dumps({"data": []}))
    with pytest.raises(ValueError, match="vector stores are empty"):
        bench._validate_index(tmp_path)
