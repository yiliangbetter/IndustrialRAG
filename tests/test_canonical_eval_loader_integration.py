"""Canonical fixture defaults for legacy diagnostic entry points."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import query_test_cases
import replay_clarify_gate_green8 as replay
import score_rerank_groups as rerank


def test_shared_voice_preset_uses_all_canonical_voice_cases() -> None:
    rows = query_test_cases._load_cases(
        query_test_cases._SOURCE_PRESETS["voice29"], limit=0
    )

    assert len(rows) == 29
    assert [row["id"] for row in rows] == [str(case_id) for case_id in range(1, 30)]


def test_shared_green8_preset_filters_canonical_voice_cases() -> None:
    rows = query_test_cases._load_preset_cases("green8")

    assert len(rows) == 8
    assert [row["id"] for row in rows] == ["2", "4", "6", "7", "11", "14", "26", "28"]


def test_rerank_views_partition_canonical_voice29() -> None:
    green = rerank._load_all_sets({"green8"})
    remainder = rerank._load_all_sets({"voice29_non_green8"})

    assert len(green) == 8
    assert len(remainder) == 21
    assert {row["id"] for row in green}.isdisjoint({row["id"] for row in remainder})


def test_replay_all_is_shili17_plus_canonical_green8() -> None:
    rows, labels = replay._load_all_cases("all", limit=0)

    assert len(rows) == 25
    assert labels == ["shili17(17)", "green8(8)"]
