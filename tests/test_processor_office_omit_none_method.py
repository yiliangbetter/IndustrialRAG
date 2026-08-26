"""Office/HTML parse must not forward method=None into parser kwargs.

Production fix: ProcessorMixin.parse_document strips kwargs['method'] and
only sets method when parse_method/config.parse_method is non-None. Passing
None used to break Office parsers that treat method as a required string.
"""

import pytest

pytest.importorskip("lightrag")

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


class FakeDocParser:
    def __init__(self):
        self.calls = []

    def parse_office_doc(self, doc_path, output_dir, **kwargs):
        self.calls.append(
            {
                "doc_path": doc_path,
                "output_dir": output_dir,
                "kwargs": dict(kwargs),
            }
        )
        return [{"type": "text", "text": "from-office"}]


class DummyConfig:
    def __init__(self, tmp_path, parse_method="auto"):
        self.parser = "mineru"
        self.parse_method = parse_method
        self.parser_output_dir = str(tmp_path)
        self.display_content_stats = False
        self.use_full_path = False


class DummyProcessor(ProcessorMixin):
    pass


def _make_processor(tmp_path, parse_method="auto"):
    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = DummyConfig(tmp_path, parse_method=parse_method)
    processor.doc_parser = FakeDocParser()

    async def no_cache(*args, **kwargs):
        return None

    async def no_store(*args, **kwargs):
        return None

    processor._get_cached_result = no_cache
    processor._store_cached_result = no_store
    processor._generate_content_based_doc_id = lambda content_list: "doc-office"
    return processor


@pytest.mark.asyncio
async def test_office_parse_omits_none_method(tmp_path):
    processor = _make_processor(tmp_path, parse_method=None)
    docx = tmp_path / "manual.docx"
    docx.write_bytes(b"PK")

    content_list, doc_id = await processor.parse_document(
        str(docx), display_stats=False, lang="en"
    )

    assert doc_id == "doc-office"
    assert content_list[0]["text"] == "from-office"
    call = processor.doc_parser.calls[0]
    assert "method" not in call["kwargs"]
    assert call["kwargs"]["lang"] == "en"


@pytest.mark.asyncio
async def test_office_parse_forwards_non_none_method_and_strips_kwargs_method(
    tmp_path,
):
    processor = _make_processor(tmp_path, parse_method="auto")
    docx = tmp_path / "manual.docx"
    docx.write_bytes(b"PK")

    await processor.parse_document(
        str(docx),
        parse_method="ocr",
        display_stats=False,
        method="should-be-stripped",
        lang="ch",
        backend="pipeline",
    )

    call = processor.doc_parser.calls[0]
    assert call["kwargs"]["method"] == "ocr"
    assert call["kwargs"]["lang"] == "ch"
    assert call["kwargs"]["backend"] == "pipeline"
    assert "should-be-stripped" not in call["kwargs"].values()


@pytest.mark.asyncio
async def test_html_parse_uses_office_entrypoint(tmp_path):
    processor = _make_processor(tmp_path, parse_method="txt")
    html = tmp_path / "page.html"
    html.write_text("<html><body>hi</body></html>", encoding="utf-8")

    content_list, _doc_id = await processor.parse_document(
        str(html), display_stats=False
    )

    assert content_list[0]["text"] == "from-office"
    assert len(processor.doc_parser.calls) == 1
    assert processor.doc_parser.calls[0]["kwargs"]["method"] == "txt"
