"""Query-time descriptions must keep caption sources and empty formulas distinct."""

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


class RecordingProcessor:
    def __init__(self, encode_result="", caption_result="described"):
        self.encode_result = encode_result
        self.caption_result = caption_result
        self.calls = []

    def _encode_image_to_base64(self, image_path):
        self.calls.append(("encode", image_path))
        return self.encode_result

    async def modal_caption_func(self, prompt, **kwargs):
        self.calls.append(("caption", prompt, kwargs))
        return self.caption_result


def _mixin():
    query = QueryMixin()
    query.logger = FakeLogger()
    query.modal_processors = {}
    return query


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_value", [[], None])
async def test_empty_canonical_caption_and_footnote_hide_v1_aliases(empty_value):
    query = _mixin()
    description = await query._describe_image_for_query(
        RecordingProcessor(),
        {
            "img_path": "/missing/figure.png",
            "image_caption": empty_value,
            "img_caption": ["Alias nameplate"],
            "image_footnote": empty_value,
            "img_footnote": ["Alias note"],
        },
    )

    assert description == "Image path: /missing/figure.png"
    assert "Alias nameplate" not in description
    assert "Alias note" not in description
    assert "Image captions:" not in description
    assert "Image footnotes:" not in description


@pytest.mark.asyncio
async def test_footnotes_only_fallback_omits_an_empty_caption_clause():
    query = _mixin()
    description = await query._describe_image_for_query(
        RecordingProcessor(),
        {
            "img_path": "/missing/figure.png",
            "image_footnote": ["See section 4", "Torque 12 Nm"],
        },
    )

    assert (
        description
        == "Image path: /missing/figure.png; Image footnotes: See section 4, Torque 12 Nm"
    )


@pytest.mark.asyncio
async def test_encoded_image_description_replaces_captions(tmp_path):
    image = tmp_path / "panel.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    processor = RecordingProcessor(
        encode_result="encoded-bytes",
        caption_result="A bolt on the panel.",
    )
    query = _mixin()
    query.modal_processors = {"image": processor}
    content = {
        "type": "image",
        "img_path": str(image),
        "image_caption": ["SHOULD NOT LEAK"],
        "image_footnote": ["SHOULD NOT LEAK EITHER"],
    }

    description = await query._describe_image_for_query(processor, content)
    enhanced = await query._process_multimodal_query_content(
        "Where is the bolt?",
        [content],
    )

    assert description == "A bolt on the panel."
    assert "Related image content: A bolt on the panel." in enhanced
    assert "SHOULD NOT LEAK" not in description
    assert "SHOULD NOT LEAK" not in enhanced
    assert processor.calls[0] == ("encode", str(image))
    assert processor.calls[1] == (
        "caption",
        PROMPTS["QUERY_IMAGE_DESCRIPTION"],
        {
            "image_data": "encoded-bytes",
            "system_prompt": PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"],
        },
    )


@pytest.mark.asyncio
async def test_equation_query_still_asks_the_model_when_latex_is_empty():
    processor = RecordingProcessor(caption_result="No formula was supplied.")
    query = _mixin()

    description = await query._describe_equation_for_query(
        processor,
        {"equation_caption": "Rated torque"},
    )

    assert description == "No formula was supplied."
    assert processor.calls == [
        (
            "caption",
            PROMPTS["QUERY_EQUATION_ANALYSIS"].format(
                latex="",
                equation_caption="Rated torque",
            ),
            {"system_prompt": PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"]},
        )
    ]
