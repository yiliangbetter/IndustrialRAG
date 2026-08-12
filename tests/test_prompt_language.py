"""Tests for multilingual prompt template system (addresses #85)."""

import pytest
import raganything.prompt_manager as prompt_manager_module

from raganything.prompt import PROMPTS
from raganything.prompt_manager import (
    set_prompt_language,
    get_prompt_language,
    reset_prompts,
    register_prompt_language,
    get_available_languages,
)


@pytest.fixture(autouse=True)
def _reset_language():
    """Ensure prompts are reset to English after each test."""
    yield
    reset_prompts()


class TestSetPromptLanguage:
    def test_switch_to_chinese(self):
        set_prompt_language("zh")
        assert get_prompt_language() == "zh"
        assert (
            "请" in PROMPTS["QUERY_IMAGE_DESCRIPTION"]
            or "图" in PROMPTS["QUERY_IMAGE_DESCRIPTION"]
        )

    def test_switch_back_to_english(self):
        set_prompt_language("zh")
        set_prompt_language("en")
        assert get_prompt_language() == "en"
        assert (
            "Please" in PROMPTS["QUERY_IMAGE_DESCRIPTION"]
            or "briefly" in PROMPTS["QUERY_IMAGE_DESCRIPTION"]
        )

    def test_unknown_language_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt language"):
            set_prompt_language("xx")

    def test_case_insensitive(self):
        set_prompt_language("ZH")
        assert get_prompt_language() == "zh"

    def test_chinese_prompts_have_all_keys(self):
        """Chinese templates should cover all English keys."""
        from raganything.prompts_zh import PROMPTS_ZH

        english_keys = set(PROMPTS.keys())
        chinese_keys = set(PROMPTS_ZH.keys())
        missing = english_keys - chinese_keys
        assert not missing, f"Chinese prompts missing keys: {missing}"


class TestResetPrompts:
    def test_reset_restores_english(self):
        original = PROMPTS["IMAGE_ANALYSIS_SYSTEM"]
        set_prompt_language("zh")
        assert PROMPTS["IMAGE_ANALYSIS_SYSTEM"] != original
        reset_prompts()
        assert PROMPTS["IMAGE_ANALYSIS_SYSTEM"] == original
        assert get_prompt_language() == "en"


class TestRegisterLanguage:
    def test_register_custom_language(self):
        custom = {"IMAGE_ANALYSIS_SYSTEM": "Analyse d'image"}
        register_prompt_language("fr", custom)
        set_prompt_language("fr")
        assert PROMPTS["IMAGE_ANALYSIS_SYSTEM"] == "Analyse d'image"
        # Missing keys should fallback to English
        assert (
            "expert" in PROMPTS["TABLE_ANALYSIS_SYSTEM"].lower()
            or "analyst" in PROMPTS["TABLE_ANALYSIS_SYSTEM"].lower()
        )

    def test_mixed_case_registration(self):
        custom = {"IMAGE_ANALYSIS_SYSTEM": "Analyse d'image"}
        register_prompt_language("FR", custom)
        # Case-insensitive selection should still work
        set_prompt_language("fr")
        assert PROMPTS["IMAGE_ANALYSIS_SYSTEM"] == "Analyse d'image"


class TestGetAvailableLanguages:
    def test_includes_defaults(self):
        langs = get_available_languages()
        assert "en" in langs
        assert "zh" in langs


class TestZhLazyLoadRegistry:
    """Lazy zh registration side effects (distinct from normalize / concurrency)."""

    def test_first_zh_switch_populates_registry(self):
        prompt_manager_module._PROMPT_LANGUAGES.pop("zh", None)
        assert "zh" not in prompt_manager_module._PROMPT_LANGUAGES

        set_prompt_language("zh")

        assert "zh" in prompt_manager_module._PROMPT_LANGUAGES
        assert get_prompt_language() == "zh"
        from raganything.prompts_zh import PROMPTS_ZH

        assert prompt_manager_module._PROMPT_LANGUAGES["zh"] is PROMPTS_ZH

    def test_second_zh_switch_does_not_reimport(self, monkeypatch):
        # Ensure zh is already registered so the lazy path is skipped.
        from raganything.prompts_zh import PROMPTS_ZH

        prompt_manager_module._PROMPT_LANGUAGES["zh"] = PROMPTS_ZH

        calls = {"count": 0}
        real_lazy = prompt_manager_module._lazy_load_language

        def counting_lazy(lang):
            calls["count"] += 1
            return real_lazy(lang)

        monkeypatch.setattr(prompt_manager_module, "_lazy_load_language", counting_lazy)

        set_prompt_language("zh")
        set_prompt_language("zh")

        assert calls["count"] == 0
        assert get_prompt_language() == "zh"


class TestAtomicPromptSwitches:
    def test_set_and_reset_use_atomic_swap(self, monkeypatch):
        class FakePrompts:
            def __init__(self, initial):
                self.data = dict(initial)
                self.swap_calls = []

            def snapshot(self):
                return dict(self.data)

            def swap(self, prompts):
                snapshot = dict(prompts)
                self.swap_calls.append(snapshot)
                self.data = snapshot

        english = {"A": "english-a", "B": "english-b"}
        fake_prompts = FakePrompts(english)

        monkeypatch.setattr(prompt_manager_module, "PROMPTS", fake_prompts)
        monkeypatch.setattr(prompt_manager_module, "_ENGLISH_PROMPTS", dict(english))
        monkeypatch.setattr(
            prompt_manager_module,
            "_PROMPT_LANGUAGES",
            {
                "en": dict(english),
                "fr": {"A": "francais-a"},
            },
        )
        monkeypatch.setattr(prompt_manager_module, "_current_language", "en")

        prompt_manager_module.set_prompt_language("fr")
        assert fake_prompts.data == {"A": "francais-a", "B": "english-b"}
        assert prompt_manager_module.get_prompt_language() == "fr"

        prompt_manager_module.reset_prompts()
        assert fake_prompts.data == english
        assert prompt_manager_module.get_prompt_language() == "en"
        assert fake_prompts.swap_calls == [
            {"A": "francais-a", "B": "english-b"},
            english,
        ]
