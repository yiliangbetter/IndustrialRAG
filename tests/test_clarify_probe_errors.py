"""Retrieval failures must not be reported as irrelevant user questions."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from raganything.clarify_gate import ClarifyProbeError, probe_llm_retrieval_full

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rag_web_server  # noqa: E402


class _BrokenLightRAG:
    async def aquery_data(self, *_args, **_kwargs):
        raise RuntimeError("retrieval backend unavailable")


@pytest.mark.asyncio
async def test_probe_propagates_backend_failure() -> None:
    with pytest.raises(ClarifyProbeError) as raised:
        await probe_llm_retrieval_full(_BrokenLightRAG(), "valid question")

    assert isinstance(raised.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_web_maps_probe_failure_to_retryable_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fail(*_args, **_kwargs):
        raise ClarifyProbeError("probe failed")

    monkeypatch.setattr("raganything.clarify_gate.evaluate_clarify_gate", _fail)
    monkeypatch.setattr(
        rag_web_server.state,
        "rag",
        SimpleNamespace(lightrag=SimpleNamespace()),
    )

    body = rag_web_server.QueryBody(query="valid question")
    with pytest.raises(HTTPException) as raised:
        await rag_web_server._evaluate_clarify_gate(body, "mix")

    assert raised.value.status_code == 503
    assert "稍后重试" in raised.value.detail
