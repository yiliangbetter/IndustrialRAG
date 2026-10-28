"""The offline collector must preserve the browser query path semantics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import query_progress_hooks
import rag_web_server


def _decode_sse(line: str) -> dict:
    assert line.startswith("data: ") and line.endswith("\n\n")
    return json.loads(line.removeprefix("data: ").strip())


@pytest.mark.asyncio
async def test_sse_adapter_preserves_semantic_event_payload(monkeypatch) -> None:
    payloads = [
        {"type": "status", "text": "retrieving"},
        {"type": "answer_delta", "text": "answer"},
        {"type": "done", "mode": "mix"},
    ]

    async def fake_events(_query, _mode, _body):
        for payload in payloads:
            yield payload

    monkeypatch.setattr(rag_web_server, "_query_event_payloads", fake_events)
    body = rag_web_server.QueryBody(query="question", mode="mix")

    encoded = [
        event
        async for event in rag_web_server._query_stream_events(body.query, "mix", body)
    ]

    assert [_decode_sse(event) for event in encoded] == payloads


@pytest.mark.asyncio
async def test_trace_prefers_model_input_chunks_and_is_deep_copied(monkeypatch) -> None:
    snapshot = {
        "llm_input": {
            "chunks": [
                {"chunk_id": "model-1", "content": "model context"},
                {"chunk_id": "model-2", "text": "second context"},
            ]
        },
        "retrieved_docs": [
            {"chunk_id": "retrieved-only", "content": "retrieved context"}
        ],
        "parser_root": Path("/tmp/parser"),
    }

    async def fake_events(query, mode, _body):
        assert query == "trim me"
        assert mode == "hybrid"
        yield {"type": "status", "phase": "retrieve", "text": "retrieving"}
        yield {"type": "thinking_delta", "text": "reason"}
        yield {"type": "answer_delta", "text": "grounded "}
        yield {"type": "answer_delta", "text": "answer"}
        yield {
            "type": "inline_images",
            "images": [{"id": "image-1"}],
            "placements": [{"image_id": "image-1", "after": 0}],
        }
        yield {"type": "done", "mode": mode}

    monkeypatch.setattr(rag_web_server, "_query_event_payloads", fake_events)
    monkeypatch.setattr(query_progress_hooks, "get_query_debug_state", lambda: snapshot)
    rag_web_server.state.query_mode = "hybrid"

    trace = await rag_web_server.execute_query_with_trace(
        rag_web_server.QueryBody(query="  trim me  ")
    )

    assert trace["schema_version"] == "rag_query_trace_v1"
    assert trace["outcome"] == "answer"
    assert trace["answer"] == "grounded answer"
    assert trace["thinking"] == "reason"
    assert trace["contexts"] == ["model context", "second context"]
    assert [row["chunk_id"] for row in trace["context_documents"]] == [
        "model-1",
        "model-2",
    ]
    assert trace["context_source"] == "llm_input.chunks"
    assert trace["contexts_complete"] is False
    assert trace["images"] == [{"id": "image-1"}]
    assert trace["terminal_event"] == {"type": "done", "mode": "hybrid"}
    assert trace["debug_state"]["parser_root"] == "/tmp/parser"

    snapshot["llm_input"]["chunks"][0]["content"] = "mutated"
    assert trace["debug_state"]["llm_input"]["chunks"][0]["content"] == (
        "model context"
    )


@pytest.mark.asyncio
async def test_trace_prefers_complete_untruncated_model_chunk_batch(
    monkeypatch,
) -> None:
    full_content = "x" * 5001
    exact_chunks = [
        {
            "chunk_id": f"exact-{index}",
            "content": full_content,
            "document_id": "manual-v3",
            "hash": "sha256:manual-v3",
            "page": index + 1,
            "modality": "table" if index == 0 else "text",
        }
        for index in range(40)
    ]
    snapshot = {
        "llm_chunks_for_images": exact_chunks,
        "llm_input": {
            "chunks": [{"chunk_id": "truncated", "content": full_content[:4000]}]
        },
        "retrieved_docs": [{"chunk_id": "retrieved", "content": "retrieved"}],
    }

    async def fake_events(_query, mode, _body):
        yield {"type": "answer_delta", "text": "answer"}
        yield {"type": "done", "mode": mode}

    monkeypatch.setattr(rag_web_server, "_query_event_payloads", fake_events)
    monkeypatch.setattr(query_progress_hooks, "get_query_debug_state", lambda: snapshot)

    trace = await rag_web_server.execute_query_with_trace(
        rag_web_server.QueryBody(query="question", mode="mix")
    )

    assert trace["context_source"] == "llm_chunks_for_images"
    assert trace["contexts_complete"] is True
    assert len(trace["context_documents"]) == 40
    assert len(trace["context_documents"][0]["content"]) == 5001
    assert trace["context_documents"][0]["hash"] == "sha256:manual-v3"
    assert trace["context_documents"][0]["page"] == 1
    assert trace["context_documents"][0]["modality"] == "table"

    exact_chunks[0]["content"] = "mutated"
    assert len(trace["context_documents"][0]["content"]) == 5001


@pytest.mark.asyncio
async def test_trace_does_not_report_retrieval_as_context_when_model_saw_none(
    monkeypatch,
) -> None:
    snapshot = {
        "llm_input": {"chunks": []},
        "retrieved_docs": [
            {"chunk_id": "retrieved-only", "content": "not sent to model"}
        ],
    }

    async def fake_events(_query, mode, _body):
        yield {"type": "answer_delta", "text": "answer without context"}
        yield {"type": "done", "mode": mode}

    monkeypatch.setattr(rag_web_server, "_query_event_payloads", fake_events)
    monkeypatch.setattr(query_progress_hooks, "get_query_debug_state", lambda: snapshot)

    trace = await rag_web_server.execute_query_with_trace(
        rag_web_server.QueryBody(query="question", mode="mix")
    )

    assert trace["contexts"] == []
    assert trace["context_documents"] == []
    assert trace["context_source"] == "llm_input.chunks"
    assert trace["contexts_complete"] is False
    assert trace["debug_state"]["retrieved_docs"][0]["chunk_id"] == "retrieved-only"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payloads", "expected_outcome"),
    [
        (
            [
                {
                    "type": "clarification_required",
                    "data": {"gate_outcome": "offer", "clarification_id": "c1"},
                },
                {"type": "done", "clarification_only": True},
            ],
            "clarify",
        ),
        (
            [
                {"type": "error", "message": "provider failed"},
                {"type": "done", "error": True},
            ],
            "error",
        ),
    ],
)
async def test_trace_keeps_non_answer_outcomes(
    monkeypatch, payloads, expected_outcome
) -> None:
    async def fake_events(_query, _mode, _body):
        for payload in payloads:
            yield payload

    monkeypatch.setattr(rag_web_server, "_query_event_payloads", fake_events)
    monkeypatch.setattr(query_progress_hooks, "get_query_debug_state", dict)

    trace = await rag_web_server.execute_query_with_trace(
        rag_web_server.QueryBody(query="question", mode="mix")
    )

    assert trace["outcome"] == expected_outcome
    assert trace["contexts"] == []
    assert trace["answer"] == ""
    assert [event["type"] for event in trace["events"]] == [
        payload["type"] for payload in payloads
    ]
