"""An existing query image with an empty base64 encode keeps its captions.

``_describe_image_for_query`` only sends an image to the vision model when
encoding returns data. An existing file that encodes to an empty string must
fall back to the path and captions. That fallback has to stay in the enhanced
query, and the next multimodal item still has to be described.
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
    def __init__(self, caption_result="described", encode_result=""):
        self.caption_result = caption_result
        self.encode_result = encode_result
        self.calls = []

    def _encode_image_to_base64(self, image_path):
        self.calls.append(("encode", image_path))
        return self.encode_result

    async def modal_caption_func(self, prompt, **kwargs):
        self.calls.append(("caption", prompt, kwargs))
        return self.caption_result


def _query(processors=None):
    query = QueryMixin()
    query.logger = FakeLogger()
    query.modal_processors = processors or {}
    return query


@pytest.mark.asyncio
async def test_existing_image_with_empty_encode_keeps_captions(tmp_path):
    image = tmp_path / "nameplate.png"
    image.write_bytes(b"not-a-decodable-image")
    processor = FakeProcessor(caption_result="SHOULD-NOT-USE", encode_result="")
    description = await _query()._describe_image_for_query(
        processor,
        {
            "img_path": str(image),
            "image_caption": ["Nameplate", "Model NX-7"],
            "image_footnote": ["See drawing A"],
        },
    )

    assert processor.calls == [("encode", str(image))]
    assert "SHOULD-NOT-USE" not in description
    assert f"Image path: {image}" in description
    assert "Image captions: Nameplate, Model NX-7" in description
    assert "Image footnotes: See drawing A" in description


@pytest.mark.asyncio
async def test_empty_encode_still_reaches_the_enhanced_query(tmp_path):
    image = tmp_path / "seal.png"
    image.write_bytes(b"unreadable")
    image_processor = FakeProcessor(caption_result="SHOULD-NOT-USE", encode_result="")
    table_processor = FakeProcessor(caption_result="torque table")
    query = _query({"image": image_processor, "table": table_processor})

    enhanced = await query._process_multimodal_query_content(
        "What torque?",
        [
            {
                "type": "image",
                "img_path": str(image),
                "image_caption": ["Seal face"],
            },
            {"type": "table", "table_data": "Nm|12", "table_caption": "Bolt"},
        ],
    )

    assert enhanced.startswith("User query: What torque?")
    assert "SHOULD-NOT-USE" not in enhanced
    assert "Image captions: Seal face" in enhanced
    assert "Related table content: torque table" in enhanced
    assert PROMPTS["QUERY_ENHANCEMENT_SUFFIX"] in enhanced
    assert image_processor.calls == [("encode", str(image))]
    assert table_processor.calls[0][0] == "caption"
    assert "Nm|12" in table_processor.calls[0][1]
