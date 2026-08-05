"""Regression tests for QueryMixin.aquery_vlm_enhanced orchestration.

Open PRs cover routing into this method (#82), message/kwargs construction (#73),
and path sandboxing (#64). This file covers the end-to-end glue: vision/init
gates, image-cache reset, no-image text fallback, and the happy path that
encodes in-workspace images then calls the vision model.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

pytest.importorskip("lightrag")

from raganything.query import QueryMixin


class FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg % args if args else msg)

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg % args if args else msg)

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class FakeLightRAG:
    def __init__(self, prompts_by_call=None):
        self.calls = []
        self._prompts = list(prompts_by_call or [])

    async def aquery(self, query, param, system_prompt=None):
        self.calls.append(
            {
                "query": query,
                "mode": getattr(param, "mode", None),
                "only_need_prompt": getattr(param, "only_need_prompt", False),
                "system_prompt": system_prompt,
            }
        )
        if self._prompts:
            return self._prompts.pop(0)
        return "fallback-text-answer"


class DummyQuery(QueryMixin):
    def __init__(self, vision_model_func=None, lightrag=None, config=None):
        self.lightrag = lightrag if lightrag is not None else FakeLightRAG()
        self.logger = FakeLogger()
        self.vision_model_func = vision_model_func
        self.config = config
        self.init_result = {"success": True}
        self.vision_calls = []

    async def _ensure_lightrag_initialized(self):
        return self.init_result

    async def vision_model_func_impl(self, prompt, **kwargs):
        self.vision_calls.append({"prompt": prompt, **kwargs})
        return "vlm-answer"


@pytest.mark.asyncio
async def test_aquery_vlm_enhanced_requires_vision_model():
    query = DummyQuery(vision_model_func=None)

    with pytest.raises(ValueError, match="requires vision_model_func"):
        await query.aquery_vlm_enhanced("what is in the figure?")


@pytest.mark.asyncio
async def test_aquery_vlm_enhanced_requires_successful_lightrag_init():
    async def vision(*args, **kwargs):
        return "unused"

    query = DummyQuery(vision_model_func=vision)
    query.init_result = {"success": False, "error": "missing llm"}

    with pytest.raises(RuntimeError, match="LightRAG initialization failed"):
        await query.aquery_vlm_enhanced("what is in the figure?")


@pytest.mark.asyncio
async def test_aquery_vlm_enhanced_falls_back_when_no_valid_images():
    async def vision(*args, **kwargs):
        return "should-not-run"

    lightrag = FakeLightRAG(
        prompts_by_call=["Context without any figures.", "text-fallback-answer"]
    )
    query = DummyQuery(vision_model_func=vision, lightrag=lightrag)
    query._current_images_base64 = ["stale-cache"]

    result = await query.aquery_vlm_enhanced(
        "summarize", mode="hybrid", system_prompt="Be brief."
    )

    assert result == "text-fallback-answer"
    assert len(lightrag.calls) == 2
    assert lightrag.calls[0]["only_need_prompt"] is True
    assert lightrag.calls[1]["only_need_prompt"] is False
    assert lightrag.calls[1]["system_prompt"] == "Be brief."
    assert lightrag.calls[1]["mode"] == "hybrid"
    assert any("No valid images found" in msg for msg in query.logger.infos)
    # Cache is re-initialized empty by image processing, not left stale.
    assert query._current_images_base64 == []


@pytest.mark.asyncio
async def test_aquery_vlm_enhanced_happy_path_calls_vision_with_images(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    image = Path("figure.png")
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"figure-bytes"
    image.write_bytes(image_bytes)

    async def vision(prompt, **kwargs):
        return "vlm-answer"

    prompt_with_image = f"Evidence\nImage Path: {image.resolve()}\nEnd"
    lightrag = FakeLightRAG(prompts_by_call=[prompt_with_image])
    query = DummyQuery(vision_model_func=vision, lightrag=lightrag)
    # Stale cache must be cleared before processing the new prompt.
    query._current_images_base64 = ["old-base64"]

    async def capture_vision(prompt, **kwargs):
        query.vision_calls.append({"prompt": prompt, **kwargs})
        return "vlm-answer"

    query.vision_model_func = capture_vision

    result = await query.aquery_vlm_enhanced("Describe the figure", mode="mix")

    assert result == "vlm-answer"
    assert len(lightrag.calls) == 1
    assert lightrag.calls[0]["only_need_prompt"] is True
    assert len(query.vision_calls) == 1
    call = query.vision_calls[0]
    assert call["prompt"] == ""
    assert "messages" in call
    messages = call["messages"]
    assert messages[0]["role"] == "system"
    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    image_parts = [p for p in user_content if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    expected_b64 = base64.b64encode(image_bytes).decode("utf-8")
    assert image_parts[0]["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"
    assert query._current_images_base64 == [expected_b64]
