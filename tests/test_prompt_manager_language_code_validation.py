"""Prompt language codes must fail closed on empty or non-string values.

set_prompt_language / register_prompt_language share _normalize_language_code.
A silent coerce-to-empty key would corrupt the process-global PROMPTS registry.
"""

import pytest

from raganything.prompt_manager import (
    get_prompt_language,
    register_prompt_language,
    reset_prompts,
    set_prompt_language,
)


@pytest.fixture(autouse=True)
def _reset_language():
    yield
    reset_prompts()


class TestLanguageCodeValidation:
    def test_non_string_raises_typeerror(self):
        with pytest.raises(TypeError, match="non-empty string"):
            set_prompt_language(None)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="non-empty string"):
            register_prompt_language(123, {"A": "x"})  # type: ignore[arg-type]

    def test_empty_and_whitespace_raise_valueerror(self):
        with pytest.raises(ValueError, match="non-empty string"):
            set_prompt_language("")
        with pytest.raises(ValueError, match="non-empty string"):
            set_prompt_language("   ")
        with pytest.raises(ValueError, match="non-empty string"):
            register_prompt_language("\t", {"A": "x"})

    def test_surrounding_whitespace_is_stripped(self):
        set_prompt_language("  ZH  ")
        assert get_prompt_language() == "zh"
