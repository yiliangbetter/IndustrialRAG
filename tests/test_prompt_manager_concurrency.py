"""Regression tests for concurrent prompt language switches.

`set_prompt_language` / `reset_prompts` must atomically swap the global
`PROMPTS` registry so concurrent readers never observe a cleared or mixed
EN/ZH snapshot mid-query.
"""

from __future__ import annotations

import threading
import time

import pytest

from raganything.prompt import PROMPTS
from raganything.prompt_manager import (
    get_prompt_language,
    register_prompt_language,
    reset_prompts,
    set_prompt_language,
)


SAMPLE_KEY = "IMAGE_ANALYSIS_SYSTEM"
VALID_MARKERS = frozenset({"LANG_AA_MARKER", "LANG_BB_MARKER"})


@pytest.fixture(autouse=True)
def _reset_language():
    yield
    reset_prompts()


def _register_fixture_languages():
    register_prompt_language("aa", {SAMPLE_KEY: "LANG_AA_MARKER"})
    register_prompt_language("bb", {SAMPLE_KEY: "LANG_BB_MARKER"})


def test_concurrent_language_switches_never_expose_empty_or_partial_snapshots():
    _register_fixture_languages()
    errors: list[str] = []
    stop = threading.Event()

    def writer_aa():
        while not stop.is_set():
            set_prompt_language("aa")

    def writer_bb():
        while not stop.is_set():
            set_prompt_language("bb")

    def reader():
        for _ in range(400):
            snap = PROMPTS.snapshot()
            if not snap:
                errors.append("empty PROMPTS snapshot observed")
                return
            if SAMPLE_KEY not in snap:
                errors.append("canonical key missing from snapshot")
                return
            # Wholesale swap must retain English-fallback keys that the
            # fixture languages did not override.
            if "TABLE_ANALYSIS_SYSTEM" not in snap:
                errors.append("fallback key missing during concurrent swap")
                return
            value = snap[SAMPLE_KEY]
            # While writers race, the sample key must be a complete language
            # snapshot — never a torn/partial write or empty string.
            if not isinstance(value, str) or not value:
                errors.append(f"invalid sample value: {value!r}")
                return
            if value not in VALID_MARKERS and "expert" not in value.lower():
                # English default is also acceptable if a reset raced in.
                if "image" not in value.lower():
                    errors.append(f"unexpected sample value: {value!r}")
                    return
            time.sleep(0)

    threads = [
        threading.Thread(target=writer_aa),
        threading.Thread(target=writer_bb),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    time.sleep(0.35)
    stop.set()
    for t in threads:
        t.join(timeout=2)
        assert not t.is_alive(), "worker thread did not stop"

    assert not errors, errors
    reset_prompts()
    assert get_prompt_language() == "en"


def test_reset_prompts_under_concurrent_set_restores_consistent_english():
    _register_fixture_languages()
    set_prompt_language("aa")
    assert PROMPTS[SAMPLE_KEY] == "LANG_AA_MARKER"

    stop = threading.Event()
    errors: list[str] = []

    def flipper():
        while not stop.is_set():
            set_prompt_language("aa")
            set_prompt_language("bb")

    def resetter():
        for _ in range(80):
            reset_prompts()
            snap = PROMPTS.snapshot()
            if not snap:
                errors.append("empty after reset")
                return
            if "TABLE_ANALYSIS_SYSTEM" not in snap:
                errors.append("English fallback missing after reset")
                return
            time.sleep(0)

    flip = threading.Thread(target=flipper)
    reset = threading.Thread(target=resetter)
    flip.start()
    reset.start()
    time.sleep(0.25)
    stop.set()
    flip.join(timeout=2)
    reset.join(timeout=2)

    assert not errors, errors
    reset_prompts()
    assert get_prompt_language() == "en"
    sample = PROMPTS[SAMPLE_KEY].lower()
    assert "expert" in sample or "image" in sample


def test_set_prompt_language_falls_back_to_english_for_missing_keys():
    register_prompt_language("cc", {SAMPLE_KEY: "LANG_CC_ONLY"})
    set_prompt_language("cc")
    assert PROMPTS[SAMPLE_KEY] == "LANG_CC_ONLY"
    assert PROMPTS["TABLE_ANALYSIS_SYSTEM"]
    assert "LANG_CC" not in PROMPTS["TABLE_ANALYSIS_SYSTEM"]
