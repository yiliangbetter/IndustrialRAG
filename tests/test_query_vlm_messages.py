"""Regression tests for VLM message construction and vision_model_func kwargs.

Multimodal answers depend on `_build_vlm_messages_with_images` placing images at
marker positions and `_call_vlm_with_multimodal_content` forwarding the full
messages list via `messages=` (not only a plain prompt). A regression here
silently drops figures or breaks vision backends that require the OpenAI-style
messages format.
"""

import base64

import pytest

from raganything.query import QueryMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class DummyQuery(QueryMixin):
    def __init__(self):
        self.logger = FakeLogger()
        self.vision_calls = []

    async def vision_model_func(self, prompt, **kwargs):
        self.vision_calls.append({"prompt": prompt, **kwargs})
        return "vlm-answer"


def test_build_vlm_messages_text_only_when_no_images():
    query = DummyQuery()
    query._current_images_base64 = []

    messages = query._build_vlm_messages_with_images(
        "Context without images",
        "What happened?",
        "Stay concise.",
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Context without images" in messages[0]["content"]
    assert "What happened?" in messages[0]["content"]


def test_build_vlm_messages_interleaves_images_and_appends_system_prompt():
    query = DummyQuery()
    img_a = base64.b64encode(b"image-a").decode("utf-8")
    img_b = base64.b64encode(b"image-b").decode("utf-8")
    query._current_images_base64 = [img_a, img_b]

    messages = query._build_vlm_messages_with_images(
        "Before [VLM_IMAGE_1] middle [VLM_IMAGE_2] after",
        "Compare the figures",
        "Answer as a safety reviewer",
    )

    assert messages[0]["role"] == "system"
    assert "helpful assistant" in messages[0]["content"]
    assert "Answer as a safety reviewer" in messages[0]["content"]

    assert messages[1]["role"] == "user"
    content = messages[1]["content"]
    assert isinstance(content, list)

    image_parts = [part for part in content if part.get("type") == "image_url"]
    assert len(image_parts) == 2
    assert image_parts[0]["image_url"]["url"] == f"data:image/jpeg;base64,{img_a}"
    assert image_parts[1]["image_url"]["url"] == f"data:image/jpeg;base64,{img_b}"

    text_blob = " ".join(
        part["text"] for part in content if part.get("type") == "text"
    )
    assert "Before" in text_blob
    assert "middle" in text_blob
    assert "after" in text_blob
    assert "Compare the figures" in text_blob


@pytest.mark.asyncio
async def test_call_vlm_passes_messages_kwarg_for_multimodal_content():
    query = DummyQuery()
    messages = [
        {"role": "system", "content": "system text"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "see figure"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,abc"},
                },
            ],
        },
    ]

    result = await query._call_vlm_with_multimodal_content(messages)

    assert result == "vlm-answer"
    assert len(query.vision_calls) == 1
    call = query.vision_calls[0]
    assert call["prompt"] == ""
    assert call["messages"] is messages
    assert "system_prompt" not in call


@pytest.mark.asyncio
async def test_call_vlm_uses_system_prompt_for_pure_text_content():
    query = DummyQuery()
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "plain text context"},
    ]

    result = await query._call_vlm_with_multimodal_content(messages)

    assert result == "vlm-answer"
    assert len(query.vision_calls) == 1
    call = query.vision_calls[0]
    assert call["prompt"] == "plain text context"
    assert call["system_prompt"] == "be brief"
    assert "messages" not in call
