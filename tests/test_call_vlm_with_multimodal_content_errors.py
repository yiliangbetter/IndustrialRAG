"""Regression tests for VLM call failure handling.

Open PR #73 covers happy-path kwargs for multimodal vs pure-text content.
This file locks the fail-closed contract: vision backend errors must propagate
so callers do not treat a failed VLM answer as an empty success.
"""

from __future__ import annotations

import pytest

from raganything.query import QueryMixin


class FakeLogger:
    def __init__(self):
        self.errors = []

    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        self.errors.append(msg % args if args else msg)


class DummyQuery(QueryMixin):
    def __init__(self, vision_model_func):
        self.logger = FakeLogger()
        self.vision_model_func = vision_model_func


@pytest.mark.asyncio
async def test_call_vlm_reraises_when_multimodal_vision_backend_fails():
    async def boom(prompt, **kwargs):
        raise RuntimeError("vision provider unavailable")

    query = DummyQuery(vision_model_func=boom)
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

    with pytest.raises(RuntimeError, match="vision provider unavailable"):
        await query._call_vlm_with_multimodal_content(messages)

    assert any("VLM call failed" in msg for msg in query.logger.errors)


@pytest.mark.asyncio
async def test_call_vlm_reraises_when_pure_text_vision_backend_fails():
    async def boom(prompt, **kwargs):
        raise TimeoutError("vision timed out")

    query = DummyQuery(vision_model_func=boom)
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "plain text context"},
    ]

    with pytest.raises(TimeoutError, match="vision timed out"):
        await query._call_vlm_with_multimodal_content(messages)

    assert any("VLM call failed" in msg for msg in query.logger.errors)
