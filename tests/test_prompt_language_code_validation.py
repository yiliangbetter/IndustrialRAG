"""Prompt-language code checks and partial-translation merge.

Invalid codes must fail before they replace the active templates. A partial
translation may set a key to an empty string, and that empty value has to win
over English. Keys that are not in the English set must not appear in the
active snapshot, and later edits to the caller's dict must not change it.
"""

import pytest

from raganything.prompt import PROMPTS
from raganything.prompt_manager import (
    get_prompt_language,
    register_prompt_language,
    reset_prompts,
    set_prompt_language,
)

_IMAGE_KEY = "IMAGE_ANALYSIS_SYSTEM"
_TABLE_KEY = "TABLE_ANALYSIS_SYSTEM"


@pytest.fixture(autouse=True)
def _reset_language():
    yield
    reset_prompts()


def test_set_prompt_language_rejects_non_string_and_blank_codes():
    english = PROMPTS[_IMAGE_KEY]

    with pytest.raises(TypeError, match="non-empty string"):
        set_prompt_language(None)
    with pytest.raises(TypeError, match="got int"):
        set_prompt_language(12)
    with pytest.raises(ValueError, match="non-empty string"):
        set_prompt_language("   ")
    with pytest.raises(ValueError, match="non-empty string"):
        register_prompt_language("", {_IMAGE_KEY: "should-not-register"})

    assert get_prompt_language() == "en"
    assert PROMPTS[_IMAGE_KEY] == english


def test_whitespace_language_code_selects_canonical_language():
    set_prompt_language("  ZH  ")
    assert get_prompt_language() == "zh"
    assert PROMPTS[_IMAGE_KEY] != ""


def test_empty_translation_is_kept_and_unknown_keys_are_dropped():
    english_table = PROMPTS[_TABLE_KEY]
    source = {
        _IMAGE_KEY: "",
        "NOT_A_PROMPT": "must-not-leak",
    }

    register_prompt_language("  JA  ", source)
    source[_IMAGE_KEY] = "mutated-after-register"
    source[_TABLE_KEY] = "mutated-table"

    set_prompt_language("ja")

    assert get_prompt_language() == "ja"
    assert PROMPTS[_IMAGE_KEY] == ""
    assert PROMPTS[_TABLE_KEY] == english_table
    assert "NOT_A_PROMPT" not in PROMPTS
    assert "must-not-leak" not in PROMPTS.values()
    assert "mutated-after-register" not in PROMPTS.values()
    assert "mutated-table" not in PROMPTS.values()
