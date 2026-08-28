"""Query image aliases and description-generation fail-closed fallbacks.

MinerU v1 fields are img_caption/img_footnote. Dropping those aliases
silently omits figure context from retrieval queries. Caption errors must
still contribute a truncated payload rather than aborting the whole query.
"""

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


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def _encode_image_to_base64(self, image_path):
        self.calls.append(("encode", image_path))
        return ""

    async def modal_caption_func(self, prompt, **kwargs):
        self.calls.append(("caption", prompt, kwargs))
        raise RuntimeError("caption exploded")


def _mixin():
    query = QueryMixin()
    query.logger = FakeLogger()
    query.modal_processors = {}
    return query


@pytest.mark.asyncio
async def test_missing_image_uses_mineru_v1_caption_aliases():
    query = _mixin()
    description = await query._describe_image_for_query(
        FakeProcessor(),
        {
            "img_path": "/missing/figure.png",
            "img_caption": ["Nameplate"],
            "img_footnote": ["See §4.1"],
        },
    )
    assert "Image path: /missing/figure.png" in description
    assert "Nameplate" in description
    assert "See §4.1" in description


@pytest.mark.asyncio
async def test_empty_encode_falls_back_to_path_and_captions(tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    processor = FakeProcessor()
    query = _mixin()

    description = await query._describe_image_for_query(
        processor,
        {
            "img_path": str(image),
            "image_caption": ["Front view"],
        },
    )

    assert processor.calls == [("encode", str(image))]
    assert f"Image path: {image}" in description
    assert "Front view" in description


@pytest.mark.asyncio
async def test_description_exception_returns_truncated_content_fallback():
    query = _mixin()
    processor = FakeProcessor()

    payload = {"type": "table", "table_data": "x" * 80, "extra": "keep-prefix"}
    description = await query._generate_query_content_description(
        processor, payload, "table"
    )

    assert description.startswith("table content: ")
    assert str(payload)[:100] in description
    assert str(payload) not in description
    assert processor.calls and processor.calls[0][0] == "caption"
