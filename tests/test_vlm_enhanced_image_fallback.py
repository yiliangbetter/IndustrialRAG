"""Regression tests for QueryMixin.aquery_vlm_enhanced image fallback.

Path sandboxing and the VLM message helper are covered elsewhere. These tests
lock the orchestration around them: a missing vision function or failed
LightRAG init must stop before retrieval, a blocked image must return the
text answer, and an in-workspace image must reach the vision model.
"""

from __future__ import annotations

import base64

import pytest

pytest.importorskip("lightrag")

from raganything.query import QueryMixin


class FakeLogger:
    def __init__(self):
        self.infos = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg % args if args else msg)

    def warning(self, *args, **kwargs):
        pass

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
        return "text-answer"


class DummyQuery(QueryMixin):
    def __init__(self, vision_model_func=None, lightrag=None):
        self.lightrag = lightrag if lightrag is not None else FakeLightRAG()
        self.logger = FakeLogger()
        self.vision_model_func = vision_model_func
        self.config = None
        self.init_calls = 0
        self.init_result = {"success": True}
        self.vision_calls = []

    async def _ensure_lightrag_initialized(self):
        self.init_calls += 1
        return self.init_result


@pytest.mark.asyncio
async def test_missing_vision_func_raises_before_init():
    query = DummyQuery(vision_model_func=None)

    with pytest.raises(ValueError, match="requires vision_model_func"):
        await query.aquery_vlm_enhanced("what is the torque spec?")

    assert query.init_calls == 0
    assert query.lightrag.calls == []


@pytest.mark.asyncio
async def test_failed_init_raises_before_retrieval():
    async def vision(*args, **kwargs):
        return "unused"

    query = DummyQuery(vision_model_func=vision)
    query.init_result = {"success": False, "error": "missing llm"}

    with pytest.raises(RuntimeError, match="missing llm"):
        await query.aquery_vlm_enhanced("what is the torque spec?")

    assert query.init_calls == 1
    assert query.lightrag.calls == []


@pytest.mark.asyncio
async def test_blocked_image_falls_back_to_text_answer(tmp_path, monkeypatch):
    """An image outside the workspace must not be sent to the vision model."""
    cwd = tmp_path / "cwd"
    outside = tmp_path / "secret"
    cwd.mkdir()
    outside.mkdir()
    monkeypatch.chdir(cwd)

    image = outside / "figure.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nsecret-bytes")

    async def vision(*args, **kwargs):
        raise AssertionError("vision model must not run for a blocked image")

    user_query = "summarize the torque spec"
    lightrag = FakeLightRAG(
        prompts_by_call=[
            f"Retrieved context\nImage Path: {image}\nEnd",
            "text-fallback-answer",
        ]
    )
    query = DummyQuery(vision_model_func=vision, lightrag=lightrag)
    query._current_images_base64 = ["stale-cache"]

    result = await query.aquery_vlm_enhanced(
        user_query, mode="hybrid", system_prompt="Be brief."
    )

    assert result == "text-fallback-answer"
    assert len(lightrag.calls) == 2
    assert lightrag.calls[0]["query"] == user_query
    assert lightrag.calls[0]["only_need_prompt"] is True
    assert lightrag.calls[0]["system_prompt"] is None
    assert lightrag.calls[1]["query"] == user_query
    assert lightrag.calls[1]["only_need_prompt"] is False
    assert lightrag.calls[1]["mode"] == "hybrid"
    assert lightrag.calls[1]["system_prompt"] == "Be brief."
    assert query._current_images_base64 == []
    assert any("No valid images found" in msg for msg in query.logger.infos)


@pytest.mark.asyncio
async def test_in_workspace_image_is_sent_to_vision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image = tmp_path / "figure.png"
    image_bytes = b"\x89PNG\r\n\x1a\nfigure-bytes"
    image.write_bytes(image_bytes)

    user_query = "describe the figure"
    lightrag = FakeLightRAG(
        prompts_by_call=[f"Evidence\nImage Path: {image.resolve()}\nEnd"]
    )
    query = DummyQuery(lightrag=lightrag)
    query._current_images_base64 = ["old-base64"]

    async def vision(prompt, **kwargs):
        query.vision_calls.append({"prompt": prompt, **kwargs})
        return "vlm-answer"

    query.vision_model_func = vision

    result = await query.aquery_vlm_enhanced(user_query, mode="mix")

    assert result == "vlm-answer"
    assert len(lightrag.calls) == 1
    assert lightrag.calls[0]["only_need_prompt"] is True
    assert lightrag.calls[0]["query"] == user_query
    assert len(query.vision_calls) == 1
    call = query.vision_calls[0]
    assert call["prompt"] == ""
    user_content = call["messages"][1]["content"]
    image_parts = [part for part in user_content if part.get("type") == "image_url"]
    assert len(image_parts) == 1
    expected_b64 = base64.b64encode(image_bytes).decode("utf-8")
    assert (
        image_parts[0]["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"
    )
    assert query._current_images_base64 == [expected_b64]
