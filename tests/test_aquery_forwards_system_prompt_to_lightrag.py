"""Text query must forward system_prompt into LightRAG, including empty multimodal fallback."""

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


class CapturingLightRAG:
    def __init__(self):
        self.calls = []

    async def aquery(self, query, param, system_prompt=None):
        self.calls.append(
            {
                "query": query,
                "mode": param.mode,
                "system_prompt": system_prompt,
            }
        )
        return "plant-answer"


def _query_mixin():
    class Dummy(QueryMixin):
        pass

    dummy = Dummy()
    dummy.lightrag = CapturingLightRAG()
    dummy.logger = FakeLogger()
    dummy.callback_manager = None
    dummy.vision_model_func = None
    dummy.modal_processors = {}
    return dummy


@pytest.mark.asyncio
async def test_aquery_forwards_system_prompt_and_mode():
    dummy = _query_mixin()

    result = await dummy.aquery(
        "What is the torque spec?",
        mode="hybrid",
        system_prompt="Answer as a plant technician.",
        vlm_enhanced=False,
    )

    assert result == "plant-answer"
    assert dummy.lightrag.calls == [
        {
            "query": "What is the torque spec?",
            "mode": "hybrid",
            "system_prompt": "Answer as a plant technician.",
        }
    ]


@pytest.mark.asyncio
async def test_aquery_with_multimodal_empty_content_forwards_system_prompt():
    dummy = _query_mixin()

    async def fake_ensure():
        return {"success": True}

    dummy._ensure_lightrag_initialized = fake_ensure

    result = await dummy.aquery_with_multimodal(
        "List the procedure steps",
        multimodal_content=[],
        mode="mix",
        system_prompt="Use only the retrieved manual.",
        vlm_enhanced=False,
    )

    assert result == "plant-answer"
    assert dummy.lightrag.calls == [
        {
            "query": "List the procedure steps",
            "mode": "mix",
            "system_prompt": "Use only the retrieved manual.",
        }
    ]
