"""Regression tests for PromptRegistry atomic snapshot swapping.

Language switches replace the active prompt dict via PromptRegistry.swap while
callers keep a stable PROMPTS reference. A broken swap/snapshot contract can
leave mid-query readers with a partial or mixed language prompt set.
"""

from raganything.prompt import PromptRegistry


def test_prompt_registry_swap_replaces_snapshot_atomically():
    registry = PromptRegistry()
    registry.swap({"A": "one", "B": "two"})

    before = registry.snapshot()
    registry.swap({"A": "alpha", "C": "gamma"})

    assert before == {"A": "one", "B": "two"}
    assert registry.snapshot() == {"A": "alpha", "C": "gamma"}
    assert "B" not in registry
    assert registry["A"] == "alpha"
    assert registry.get("C") == "gamma"
    assert registry.get("missing") is None


def test_prompt_registry_snapshot_is_independent_of_later_mutation():
    registry = PromptRegistry()
    registry.swap({"IMAGE_ANALYSIS_SYSTEM": "english"})
    snap = registry.snapshot()

    registry["IMAGE_ANALYSIS_SYSTEM"] = "mutated-in-place"
    registry["NEW_KEY"] = "added"

    assert snap == {"IMAGE_ANALYSIS_SYSTEM": "english"}
    assert registry["IMAGE_ANALYSIS_SYSTEM"] == "mutated-in-place"
    assert "NEW_KEY" in registry
    assert len(registry) == 2
    assert list(registry.keys()) == ["IMAGE_ANALYSIS_SYSTEM", "NEW_KEY"]


def test_prompt_registry_dict_protocol_basics():
    registry = PromptRegistry()
    registry.swap({"k": "v"})

    assert "k" in registry
    assert list(registry) == ["k"]
    assert list(registry.values()) == ["v"]
    assert list(registry.items()) == [("k", "v")]

    del registry["k"]
    assert "k" not in registry
    assert len(registry) == 0
    assert "PromptRegistry" in repr(registry)
