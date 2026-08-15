"""Regression tests for image parse fallback to MinerU.

Docling and custom parsers may raise NotImplementedError from parse_image.
parse_document must fall back to MinerU instead of failing the whole ingest.
A supported parser must not take that fallback path.
"""

import pytest

import raganything.processor as processor_module
from raganything.processor import ProcessorMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _make_processor(tmp_path, parser_name="docling"):
    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.config = type(
        "Config",
        (),
        {
            "parser": parser_name,
            "parser_output_dir": str(tmp_path / "output"),
            "parse_method": "auto",
            "display_content_stats": False,
            "use_full_path": False,
        },
    )()
    dummy.logger = FakeLogger()
    dummy.parse_cache = None
    return dummy


@pytest.mark.asyncio
async def test_parse_document_image_falls_back_to_mineru(monkeypatch, tmp_path):
    class UnsupportedImageParser:
        def parse_image(self, **kwargs):
            raise NotImplementedError("image parsing not supported")

    mineru_calls = []

    class FakeMineruParser:
        def parse_image(self, **kwargs):
            mineru_calls.append(kwargs)
            return [{"type": "text", "text": "mineru image fallback", "page_idx": 0}]

    monkeypatch.setattr(
        processor_module,
        "get_parser",
        lambda parser_name: UnsupportedImageParser(),
    )
    monkeypatch.setattr(processor_module, "MineruParser", FakeMineruParser)

    dummy = _make_processor(tmp_path)

    async def fake_store_cached_result(*args, **kwargs):
        return None

    monkeypatch.setattr(
        dummy,
        "_store_cached_result",
        fake_store_cached_result,
    )
    monkeypatch.setattr(
        dummy,
        "_generate_content_based_doc_id",
        lambda content_list: "doc-image",
    )

    fake_png = tmp_path / "figure.png"
    fake_png.write_bytes(b"\x89PNG\r\n\x1a\n")

    content_list, doc_id = await dummy.parse_document(str(fake_png), lang="ch")

    assert doc_id == "doc-image"
    assert content_list == [
        {"type": "text", "text": "mineru image fallback", "page_idx": 0}
    ]
    assert len(mineru_calls) == 1
    assert mineru_calls[0]["image_path"] == fake_png
    assert mineru_calls[0]["output_dir"] == str(tmp_path / "output")
    assert mineru_calls[0]["lang"] == "ch"


@pytest.mark.asyncio
async def test_parse_document_image_does_not_fallback_when_parser_supports_it(
    monkeypatch, tmp_path
):
    mineru_calls = []

    class SupportingImageParser:
        def parse_image(self, **kwargs):
            return [{"type": "text", "text": "native image parse", "page_idx": 0}]

    class FakeMineruParser:
        def parse_image(self, **kwargs):
            mineru_calls.append(kwargs)
            return [{"type": "text", "text": "should not run", "page_idx": 0}]

    monkeypatch.setattr(
        processor_module,
        "get_parser",
        lambda parser_name: SupportingImageParser(),
    )
    monkeypatch.setattr(processor_module, "MineruParser", FakeMineruParser)

    dummy = _make_processor(tmp_path, parser_name="paddleocr")

    async def fake_store_cached_result(*args, **kwargs):
        return None

    monkeypatch.setattr(dummy, "_store_cached_result", fake_store_cached_result)
    monkeypatch.setattr(
        dummy,
        "_generate_content_based_doc_id",
        lambda content_list: "doc-native",
    )

    fake_png = tmp_path / "figure.png"
    fake_png.write_bytes(b"\x89PNG\r\n\x1a\n")

    content_list, doc_id = await dummy.parse_document(str(fake_png))

    assert doc_id == "doc-native"
    assert content_list == [
        {"type": "text", "text": "native image parse", "page_idx": 0}
    ]
    assert mineru_calls == []
