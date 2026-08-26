"""PromptRegistry snapshot/swap isolation.

Language switches replace the prompt dict in one step. Readers that hold a
snapshot must not see later mutations, and swap must copy the incoming
mapping so callers cannot mutate the live registry by accident.
"""

import pytest

from raganything.prompt import PromptRegistry


def test_swap_replaces_contents_and_copies_input():
    registry = PromptRegistry()
    incoming = {"A": "one", "B": "two"}
    registry.swap(incoming)

    incoming["A"] = "mutated"
    incoming["C"] = "extra"

    assert registry["A"] == "one"
    assert registry["B"] == "two"
    assert "C" not in registry
    assert len(registry) == 2


def test_snapshot_is_isolated_from_later_swaps_and_item_writes():
    registry = PromptRegistry()
    registry.swap({"A": "english-a", "B": "english-b"})
    snapshot = registry.snapshot()

    registry.swap({"A": "zh-a"})
    registry["B"] = "zh-b"

    assert snapshot == {"A": "english-a", "B": "english-b"}
    assert registry["A"] == "zh-a"
    assert registry["B"] == "zh-b"
    snapshot["A"] = "local-only"
    assert registry["A"] == "zh-a"


def test_mapping_helpers_and_missing_key():
    registry = PromptRegistry()
    registry.swap({"IMAGE": "img", "TABLE": "tbl"})

    assert "IMAGE" in registry
    assert list(registry) == ["IMAGE", "TABLE"]
    assert list(registry.keys()) == ["IMAGE", "TABLE"]
    assert registry.get("IMAGE") == "img"
    assert registry.get("missing", "fallback") == "fallback"
    with pytest.raises(KeyError):
        _ = registry["missing"]

    del registry["TABLE"]
    assert "TABLE" not in registry
    assert "PromptRegistry" in repr(registry)
