"""Prompt snapshots must stay independent of the dicts that produced them.

``reset_prompts`` installs the canonical English set through ``PromptRegistry.swap``.
``register_prompt_language`` stores the caller's templates. Both copy. If either
keeps the caller's dict, a later in-place edit permanently changes English
fallbacks or a language that was already registered.
"""

import raganything.prompt_manager as prompt_manager_module
from raganything.prompt import PROMPTS
from raganything.prompt_manager import (
    register_prompt_language,
    reset_prompts,
    set_prompt_language,
)


def test_reset_prompts_does_not_alias_the_english_canonical_set():
    canonical = prompt_manager_module._ENGLISH_PROMPTS
    saved = dict(canonical)
    key = "IMAGE_ANALYSIS_SYSTEM"
    try:
        reset_prompts()
        PROMPTS[key] = "tampered-for-this-process"
        reset_prompts()

        assert PROMPTS[key] == saved[key]
        assert canonical[key] == saved[key]
        assert "tampered-for-this-process" not in canonical.values()
    finally:
        canonical.clear()
        canonical.update(saved)
        reset_prompts()


def test_register_prompt_language_does_not_alias_the_caller_dict():
    language = "snapshot-isolation"
    languages = prompt_manager_module._PROMPT_LANGUAGES
    custom = {"IMAGE_ANALYSIS_SYSTEM": "french analysis"}
    try:
        register_prompt_language(language, custom)
        custom["IMAGE_ANALYSIS_SYSTEM"] = "mutated after register"
        custom["EXTRA_PROMPT"] = "not registered"

        set_prompt_language(language)

        assert PROMPTS["IMAGE_ANALYSIS_SYSTEM"] == "french analysis"
        assert "EXTRA_PROMPT" not in PROMPTS
    finally:
        languages.pop(language, None)
        reset_prompts()
