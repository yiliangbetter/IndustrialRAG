"""Regression tests for VLM message assembly and vision-model dispatch.

`_process_image_paths_for_vlm` (path sandbox) is covered elsewhere. These
tests lock the marker→base64 alignment and the string-vs-messages call
shape used by `aquery_vlm_enhanced`.
"""

from types import SimpleNamespace

import pytest

from raganything.query import QueryMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _mixin(**attrs):
    query = QueryMixin()
    query.logger = FakeLogger()
    for key, value in attrs.items():
        setattr(query, key, value)
    return query


class TestBuildVlmMessagesWithImages:
    def test_no_cached_images_returns_plain_text_user_message(self):
        query = _mixin()
        messages = query._build_vlm_messages_with_images(
            enhanced_prompt="retrieved context",
            user_query="What is the rated current?",
            system_prompt="Stay concise.",
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], str)
        assert "retrieved context" in messages[0]["content"]
        assert "What is the rated current?" in messages[0]["content"]

    def test_single_marker_inserts_image_between_surrounding_text(self):
        query = _mixin(_current_images_base64=["AAA"])
        messages = query._build_vlm_messages_with_images(
            enhanced_prompt="Before [VLM_IMAGE_1] after the figure.",
            user_query="Describe the figure.",
            system_prompt=None,
        )
        assert messages[0]["role"] == "system"
        parts = messages[1]["content"]
        types = [part["type"] for part in parts]
        assert "image_url" in types
        image = next(part for part in parts if part["type"] == "image_url")
        assert image["image_url"]["url"] == "data:image/jpeg;base64,AAA"
        text = "".join(part["text"] for part in parts if part["type"] == "text")
        assert "Before" in text
        assert "after the figure." in text
        assert "Describe the figure." in text

    def test_two_markers_keep_image_order(self):
        query = _mixin(_current_images_base64=["ONE", "TWO"])
        messages = query._build_vlm_messages_with_images(
            enhanced_prompt="A [VLM_IMAGE_1] then B [VLM_IMAGE_2] end",
            user_query="Compare both images.",
            system_prompt=None,
        )
        images = [
            part["image_url"]["url"]
            for part in messages[1]["content"]
            if part["type"] == "image_url"
        ]
        assert images == [
            "data:image/jpeg;base64,ONE",
            "data:image/jpeg;base64,TWO",
        ]

    def test_out_of_range_marker_skips_image_and_keeps_remaining_text(self):
        query = _mixin(_current_images_base64=["ONLY"])
        messages = query._build_vlm_messages_with_images(
            enhanced_prompt="lead [VLM_IMAGE_9] leftover caption",
            user_query="Q",
            system_prompt=None,
        )
        parts = messages[1]["content"]
        assert not any(part["type"] == "image_url" for part in parts)
        text = "".join(part["text"] for part in parts if part["type"] == "text")
        assert "leftover caption" in text

    def test_custom_system_prompt_is_appended_to_default(self):
        query = _mixin(_current_images_base64=["AAA"])
        messages = query._build_vlm_messages_with_images(
            enhanced_prompt="ctx [VLM_IMAGE_1] more",
            user_query="Q",
            system_prompt="Answer in Chinese.",
        )
        system = messages[0]["content"]
        assert "helpful assistant" in system
        assert "Answer in Chinese." in system


class TestCallVlmWithMultimodalContent:
    @pytest.mark.asyncio
    async def test_string_content_passes_prompt_and_system_prompt(self):
        captured = {}

        async def vision_model_func(prompt, system_prompt=None, **kwargs):
            captured["prompt"] = prompt
            captured["system_prompt"] = system_prompt
            captured["kwargs"] = kwargs
            return "text-mode-answer"

        query = _mixin(vision_model_func=vision_model_func)
        result = await query._call_vlm_with_multimodal_content(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "plain context"},
            ]
        )
        assert result == "text-mode-answer"
        assert captured["prompt"] == "plain context"
        assert captured["system_prompt"] == "sys"
        assert "messages" not in captured["kwargs"]

    @pytest.mark.asyncio
    async def test_list_content_passes_full_messages_payload(self):
        captured = {}
        messages = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "see image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,AAA"},
                    },
                ],
            },
        ]

        async def vision_model_func(prompt, messages=None, **kwargs):
            captured["prompt"] = prompt
            captured["messages"] = messages
            return "vision-answer"

        query = _mixin(vision_model_func=vision_model_func)
        result = await query._call_vlm_with_multimodal_content(messages)
        assert result == "vision-answer"
        assert captured["prompt"] == ""
        assert captured["messages"] is messages

    @pytest.mark.asyncio
    async def test_vision_failure_is_reraised(self):
        async def vision_model_func(*args, **kwargs):
            raise RuntimeError("vision backend down")

        query = _mixin(vision_model_func=vision_model_func)
        with pytest.raises(RuntimeError, match="vision backend down"):
            await query._call_vlm_with_multimodal_content(
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "plain"},
                ]
            )


class TestAqueryVlmEnhancedGuards:
    @pytest.mark.asyncio
    async def test_missing_vision_model_raises(self):
        query = _mixin(vision_model_func=None)
        with pytest.raises(ValueError, match="vision_model_func"):
            await query.aquery_vlm_enhanced("What is shown?")

    @pytest.mark.asyncio
    async def test_lightrag_init_failure_raises(self):
        async def fail_init():
            return {"success": False, "error": "missing llm"}

        query = _mixin(
            vision_model_func=lambda *a, **k: "unused",
            _ensure_lightrag_initialized=fail_init,
        )
        with pytest.raises(RuntimeError, match="missing llm"):
            await query.aquery_vlm_enhanced("What is shown?")

    @pytest.mark.asyncio
    async def test_no_images_falls_back_to_normal_aquery(self):
        calls = []

        class FakeLightRAG:
            async def aquery(self, query, param, system_prompt=None):
                calls.append(
                    SimpleNamespace(
                        query=query,
                        only_need_prompt=getattr(param, "only_need_prompt", False),
                        system_prompt=system_prompt,
                    )
                )
                if getattr(param, "only_need_prompt", False):
                    return "prompt without images"
                return "fallback-answer"

        async def ok_init():
            return {"success": True}

        query = _mixin(
            vision_model_func=lambda *a, **k: "unused",
            _ensure_lightrag_initialized=ok_init,
            lightrag=FakeLightRAG(),
        )

        async def no_images(raw_prompt, extra_safe_dirs=None):
            return raw_prompt, 0

        query._process_image_paths_for_vlm = no_images
        result = await query.aquery_vlm_enhanced(
            "rated current?", system_prompt="Keep units."
        )
        assert result == "fallback-answer"
        assert len(calls) == 2
        assert calls[0].only_need_prompt is True
        assert calls[1].only_need_prompt is False
        assert calls[1].system_prompt == "Keep units."
