"""Regression tests for QueryMixin._call_vlm_with_multimodal_content.

Open PRs cover VLM message construction (#73), sandbox path filtering (#64),
and aquery_vlm_enhanced orchestration (#95). The final vision-model call still
owns the pure-text vs messages= branch; a regression silently drops images or
system prompts from VLM answers.
"""

from __future__ import annotations

import logging

import pytest

from raganything.query import QueryMixin


class FakeLogger:
    def __init__(self):
        self.errors = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))

    def debug(self, *args, **kwargs):
        pass


class DummyQuery(QueryMixin):
    pass


def _make_query(vision_func):
    query = DummyQuery()
    query.logger = FakeLogger()
    query.vision_model_func = vision_func
    return query


@pytest.mark.asyncio
async def test_call_vlm_pure_text_passes_content_and_system_prompt():
    """String user content must call vision with prompt + system_prompt."""
    calls = []

    async def vision_model_func(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return "text-answer"

    query = _make_query(vision_model_func)
    messages = [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Context:\nplain text\n\nUser Question: Q?"},
    ]

    result = await query._call_vlm_with_multimodal_content(messages)

    assert result == "text-answer"
    assert len(calls) == 1
    prompt, kwargs = calls[0]
    assert prompt == messages[1]["content"]
    assert kwargs == {"system_prompt": "Be precise."}
    assert "messages" not in kwargs


@pytest.mark.asyncio
async def test_call_vlm_multimodal_passes_messages_kwarg():
    """List user content must call vision with empty prompt and messages=."""
    calls = []

    async def vision_model_func(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return "vision-answer"

    query = _make_query(vision_model_func)
    messages = [
        {"role": "system", "content": "Analyze images."},
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

    assert result == "vision-answer"
    assert len(calls) == 1
    prompt, kwargs = calls[0]
    assert prompt == ""
    assert kwargs == {"messages": messages}


@pytest.mark.asyncio
async def test_call_vlm_rethrows_vision_failures(caplog):
    """Vision failures must surface to callers after logging."""

    async def vision_model_func(prompt, **kwargs):
        raise RuntimeError("vision down")

    query = _make_query(vision_model_func)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="vision down"):
            await query._call_vlm_with_multimodal_content(messages)

    assert any("VLM call failed" in msg for msg in query.logger.errors)
