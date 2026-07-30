"""Regression tests for multimodal query content description generation.

`aquery_with_multimodal` enriches the user query from images/tables/equations.
Missing-image fallbacks, caption aliases, and per-item error isolation decide
whether retrieval sees useful context or silently drops related multimodal
evidence.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from raganything.prompt import PROMPTS
from raganything.query import QueryMixin


class FakeLogger:
    def __init__(self):
        self.errors = []
        self.infos = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg))

    def warning(self, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))

    def debug(self, *args, **kwargs):
        pass


class FakeProcessor:
    def __init__(self, encode_result="encoded=="):
        self.encode_result = encode_result
        self.caption_calls = []
        self.modal_caption_func = AsyncMock(side_effect=self._caption)

    def _encode_image_to_base64(self, image_path):
        return self.encode_result

    async def _caption(self, prompt, **kwargs):
        self.caption_calls.append({"prompt": prompt, **kwargs})
        return f"desc:{prompt[:24]}"


def _make_query(processors=None):
    class DummyQuery(QueryMixin):
        pass

    query = DummyQuery()
    query.logger = FakeLogger()
    query.modal_processors = processors or {}
    return query


@pytest.mark.asyncio
async def test_describe_image_uses_vision_when_file_exists(tmp_path):
    img = tmp_path / "part.png"
    img.write_bytes(b"fake-image")
    processor = FakeProcessor(encode_result="base64img")
    query = _make_query({"image": processor})

    description = await query._describe_image_for_query(
        processor,
        {"img_path": str(img), "image_caption": ["ignored when vision works"]},
    )

    assert description.startswith("desc:")
    assert len(processor.caption_calls) == 1
    call = processor.caption_calls[0]
    assert call["image_data"] == "base64img"
    assert call["system_prompt"] == PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"]
    assert call["prompt"] == PROMPTS["QUERY_IMAGE_DESCRIPTION"]


@pytest.mark.asyncio
async def test_describe_image_falls_back_to_path_and_caption_aliases(tmp_path):
    processor = FakeProcessor()
    query = _make_query({"image": processor})
    missing = tmp_path / "gone.png"

    description = await query._describe_image_for_query(
        processor,
        {
            "img_path": str(missing),
            "img_caption": ["legacy caption"],
            "img_footnote": ["legacy footnote"],
        },
    )

    assert processor.caption_calls == []
    assert str(missing) in description
    assert "legacy caption" in description
    assert "legacy footnote" in description


@pytest.mark.asyncio
async def test_describe_image_incomplete_when_no_metadata():
    processor = FakeProcessor()
    query = _make_query({"image": processor})

    description = await query._describe_image_for_query(processor, {"type": "image"})

    assert description == "Image content information incomplete"
    assert processor.caption_calls == []


@pytest.mark.asyncio
async def test_describe_table_equation_and_generic_call_caption_func():
    processor = FakeProcessor()
    query = _make_query(
        {
            "table": processor,
            "equation": processor,
            "generic": processor,
        }
    )

    table_desc = await query._describe_table_for_query(
        processor,
        {"table_data": "A,B\n1,2", "table_caption": "dims"},
    )
    eq_desc = await query._describe_equation_for_query(
        processor,
        {"latex": "a^2+b^2", "equation_caption": "pythagoras"},
    )
    generic_desc = await query._describe_generic_for_query(
        processor,
        {"content": "waveform"},
        "audio",
    )

    assert table_desc.startswith("desc:")
    assert eq_desc.startswith("desc:")
    assert generic_desc.startswith("desc:")
    assert len(processor.caption_calls) == 3
    assert "A,B\n1,2" in processor.caption_calls[0]["prompt"]
    assert "a^2+b^2" in processor.caption_calls[1]["prompt"]
    assert "audio" in processor.caption_calls[2]["prompt"]


@pytest.mark.asyncio
async def test_process_multimodal_query_content_routes_types_and_isolates_errors(
    monkeypatch,
):
    good = FakeProcessor()
    query = _make_query({"table": good, "image": good})

    from raganything import query as query_module

    real_get = query_module.get_processor_for_type

    def flaky_get(processors, content_type):
        if content_type == "equation":
            raise RuntimeError("processor registry boom")
        return real_get(processors, content_type)

    monkeypatch.setattr(query_module, "get_processor_for_type", flaky_get)

    enhanced = await query._process_multimodal_query_content(
        "What are the limits?",
        [
            {"type": "table", "table_data": "x,y", "table_caption": ""},
            {"type": "equation", "latex": "x=1", "equation_caption": ""},
            {"type": "unknown_widget", "payload": "no processor"},
        ],
    )

    assert "User query: What are the limits?" in enhanced
    assert "Related table content:" in enhanced
    # hard failure in processor lookup is isolated; processing continues
    assert "Related equation content:" not in enhanced
    assert any("Error processing multimodal content" in e for e in query.logger.errors)
    # unknown type without processor uses basic str fallback
    assert "Related unknown_widget content:" in enhanced
    assert PROMPTS["QUERY_ENHANCEMENT_SUFFIX"] in enhanced


@pytest.mark.asyncio
async def test_process_multimodal_query_soft_fallback_still_included():
    bad = FakeProcessor()
    bad.modal_caption_func = AsyncMock(side_effect=RuntimeError("caption failed"))
    query = _make_query({"equation": bad})

    enhanced = await query._process_multimodal_query_content(
        "Explain the formula",
        [{"type": "equation", "latex": "x=1", "equation_caption": ""}],
    )

    # description helper soft-falls back instead of dropping the item
    assert "Related equation content: equation content:" in enhanced
    assert "x=1" in enhanced
    assert any("Error generating equation description" in e for e in query.logger.errors)


@pytest.mark.asyncio
async def test_generate_query_content_description_falls_back_on_error():
    processor = FakeProcessor()
    processor.modal_caption_func = AsyncMock(side_effect=RuntimeError("boom"))
    query = _make_query({"table": processor})

    description = await query._generate_query_content_description(
        processor,
        {"type": "table", "table_data": "secret-ish"},
        "table",
    )

    assert description.startswith("table content:")
    assert "secret-ish" in description
    assert any("Error generating table description" in e for e in query.logger.errors)


@pytest.mark.asyncio
async def test_aquery_with_multimodal_falls_back_to_text_when_empty():
    query = _make_query()
    query.aquery = AsyncMock(return_value="text-only")

    async def ok_init():
        return {"success": True}

    query._ensure_lightrag_initialized = ok_init
    query.lightrag = SimpleNamespace(llm_response_cache=None)

    result = await query.aquery_with_multimodal(
        "plain question",
        multimodal_content=None,
        mode="hybrid",
        system_prompt="be brief",
    )

    assert result == "text-only"
    query.aquery.assert_awaited_once_with(
        "plain question",
        mode="hybrid",
        system_prompt="be brief",
    )
