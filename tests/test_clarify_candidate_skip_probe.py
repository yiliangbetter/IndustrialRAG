"""Unit tests for clarify candidate collection (LLM-only, no probe)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from raganything.clarify_gate import _collect_high_confidence_candidates


def test_collect_candidates_llm_only_accepts_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidates are accepted purely from LLM generation without probing."""
    monkeypatch.setenv("CLARIFY_CANDIDATE_K", "2")
    monkeypatch.setenv("CLARIFY_CANDIDATE_MAX_ROUNDS", "1")

    requested: list[int] = []

    async def _fake_generate(*_args, **kwargs):
        requested.append(int(kwargs.get("count") or 0))
        return ["推荐问 A？", "推荐问 B？"]

    monkeypatch.setattr(
        "raganything.clarify_gate._generate_candidate_lines",
        _fake_generate,
    )
    probe_mock = AsyncMock()
    monkeypatch.setattr(
        "raganything.clarify_gate.probe_llm_retrieval_full",
        probe_mock,
    )

    candidates, options, meta = asyncio.run(
        _collect_high_confidence_candidates(
            MagicMock(),
            query="原问？",
            original_bundle=None,
            mode="mix",
        )
    )

    assert len(candidates) == 2
    assert candidates[0]["text"] == "推荐问 A？"
    assert candidates[0]["final_score"] is None
    assert options["c1"]["bundle"] is None
    assert meta["candidate_validation"] == "llm_only"
    assert meta["probes_used"] == 0
    assert requested == [2]
    gen_rounds = (meta.get("gate_timing") or {}).get("candidate_gen_rounds") or []
    assert gen_rounds[0]["lines_requested"] == 2
    probe_mock.assert_not_called()
