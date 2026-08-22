"""Regression tests for multimodal query enhancement.

`aquery_with_multimodal` rewrites the user query from image/table/equation
payloads before LightRAG retrieval. Wrong fallbacks drop intended context
or abort the whole query when one item fails.
"""

import pytest

from raganything.prompt import PROMPTS
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


class FakeProcessor:
    def __init__(self, caption_result="captioned", image_base64="YmFzZTY0"):
        self.caption_result = caption_result
        self.image_base64 = image_base64
        self.calls = []

    def _encode_image_to_base64(self, image_path):
        self.calls.append(("encode", image_path))
        return self.image_base64

    async def modal_caption_func(self, prompt, **kwargs):
        self.calls.append(("caption", prompt, kwargs))
        return self.caption_result


def _mixin(processors=None):
    query = QueryMixin()
    query.logger = FakeLogger()
    query.modal_processors = processors or {}
    return query


class TestDescribeImageForQuery:
    @pytest.mark.asyncio
    async def test_missing_file_uses_captions_and_footnotes(self):
        query = _mixin()
        processor = FakeProcessor()
        description = await query._describe_image_for_query(
            processor,
            {
                "img_path": "/missing/does-not-exist.png",
                "image_caption": ["Front view"],
                "image_footnote": ["See §3.2"],
            },
        )
        assert "Image path: /missing/does-not-exist.png" in description
        assert "Front view" in description
        assert "See §3.2" in description
        assert processor.calls == []

    @pytest.mark.asyncio
    async def test_existing_image_calls_vision_caption(self, tmp_path):
        image = tmp_path / "figure.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        query = _mixin()
        processor = FakeProcessor(caption_result="A nameplate photo")
        description = await query._describe_image_for_query(
            processor, {"img_path": str(image)}
        )
        assert description == "A nameplate photo"
        assert processor.calls[0] == ("encode", str(image))
        assert processor.calls[1][0] == "caption"
        assert processor.calls[1][1] == PROMPTS["QUERY_IMAGE_DESCRIPTION"]
        assert processor.calls[1][2]["image_data"] == "YmFzZTY0"

    @pytest.mark.asyncio
    async def test_incomplete_image_payload_returns_explicit_fallback(self):
        query = _mixin()
        description = await query._describe_image_for_query(FakeProcessor(), {})
        assert description == "Image content information incomplete"


class TestDescribeStructuredContentForQuery:
    @pytest.mark.asyncio
    async def test_table_prompt_includes_body_and_caption(self):
        query = _mixin()
        processor = FakeProcessor(caption_result="ratings table")
        description = await query._describe_table_for_query(
            processor,
            {"table_data": "A|B\n1|2", "table_caption": "Rated values"},
        )
        assert description == "ratings table"
        prompt = processor.calls[0][1]
        assert "A|B" in prompt
        assert "Rated values" in prompt

    @pytest.mark.asyncio
    async def test_equation_prompt_includes_latex(self):
        query = _mixin()
        processor = FakeProcessor(caption_result="ohms law")
        description = await query._describe_equation_for_query(
            processor, {"latex": "V=IR", "equation_caption": "Ohm"}
        )
        assert description == "ohms law"
        prompt = processor.calls[0][1]
        assert "V=IR" in prompt
        assert "Ohm" in prompt

    @pytest.mark.asyncio
    async def test_generic_prompt_includes_content_type(self):
        query = _mixin()
        processor = FakeProcessor(caption_result="chart summary")
        description = await query._describe_generic_for_query(
            processor, {"title": "Trend"}, "chart"
        )
        assert description == "chart summary"
        prompt = processor.calls[0][1]
        assert "chart" in prompt
        assert "Trend" in prompt


class TestProcessMultimodalQueryContent:
    @pytest.mark.asyncio
    async def test_unknown_type_without_processor_uses_truncated_repr(self):
        query = _mixin(processors={})
        enhanced = await query._process_multimodal_query_content(
            "What is the rating?",
            [{"type": "audio", "blob": "x" * 250}],
        )
        assert enhanced.startswith("User query: What is the rating?")
        assert "Related audio content:" in enhanced
        assert PROMPTS["QUERY_ENHANCEMENT_SUFFIX"] in enhanced
        related = enhanced.split("Related audio content:", 1)[1]
        # Basic fallback is str(content)[:200], not the full payload.
        raw = str({"type": "audio", "blob": "x" * 250})
        assert raw[:200] in related
        assert raw not in related

    @pytest.mark.asyncio
    async def test_item_error_is_skipped_and_remaining_items_still_enhance(self):
        query = _mixin()
        original_generate = QueryMixin._generate_query_content_description

        async def fail_image_only(processor, content, content_type):
            if content_type == "image":
                raise RuntimeError("vision failed")
            return await original_generate(query, processor, content, content_type)

        query._generate_query_content_description = fail_image_only
        query.modal_processors = {
            "image": FakeProcessor(),
            "table": FakeProcessor(caption_result="table ok"),
        }

        enhanced = await query._process_multimodal_query_content(
            "Explain the figure",
            [
                {"type": "image", "img_path": "/missing.png"},
                {"type": "table", "table_data": "1|2", "table_caption": "T"},
            ],
        )
        assert "Related image content:" not in enhanced
        assert "Related table content: table ok" in enhanced
        assert enhanced.startswith("User query: Explain the figure")

    @pytest.mark.asyncio
    async def test_description_helper_error_returns_truncated_fallback(self):
        query = _mixin()

        async def fail(*args, **kwargs):
            raise RuntimeError("processor exploded")

        query._describe_table_for_query = fail
        fallback = await query._generate_query_content_description(
            FakeProcessor(),
            {"table_data": "x" * 150},
            "table",
        )
        assert fallback.startswith("table content:")
        assert len(fallback) <= len("table content: ") + 100
