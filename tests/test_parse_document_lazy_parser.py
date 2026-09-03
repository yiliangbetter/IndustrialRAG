"""parse_document must construct a parser when doc_parser is unset.

Embedding-only init can leave doc_parser as None. A later full parse that
skips get_parser crashes before OCR, so manuals never enter the graph.
"""

import pytest

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


class RecordingParser:
    def __init__(self):
        self.pdf_calls = []

    def parse_pdf(self, **kwargs):
        self.pdf_calls.append(kwargs)
        return [{"type": "text", "text": "from-lazy-parser"}]


def _make_processor(tmp_path):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "parser": "mineru",
            "parser_output_dir": str(tmp_path / "output"),
            "parse_method": "ocr",
            "display_content_stats": False,
            "use_full_path": False,
        },
    )()
    processor.doc_parser = None
    processor.parse_cache = None
    return processor


@pytest.mark.asyncio
async def test_parse_document_constructs_parser_when_missing(monkeypatch, tmp_path):
    processor = _make_processor(tmp_path)
    parser = RecordingParser()
    get_parser_calls = []

    def fake_get_parser(name):
        get_parser_calls.append(name)
        return parser

    async def no_store(*args, **kwargs):
        return None

    monkeypatch.setattr("raganything.processor.get_parser", fake_get_parser)
    processor._store_cached_result = no_store
    processor._generate_content_based_doc_id = lambda content_list: "doc-lazy"

    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    content_list, doc_id = await processor.parse_document(
        str(pdf), display_stats=False, lang="ch"
    )

    assert get_parser_calls == ["mineru"]
    assert processor.doc_parser is parser
    assert content_list == [{"type": "text", "text": "from-lazy-parser"}]
    assert doc_id == "doc-lazy"
    call = parser.pdf_calls[0]
    assert str(call["pdf_path"]).endswith("manual.pdf")
    assert call["method"] == "ocr"
    assert call["lang"] == "ch"


@pytest.mark.asyncio
async def test_parse_document_reuses_existing_parser(monkeypatch, tmp_path):
    processor = _make_processor(tmp_path)
    parser = RecordingParser()
    processor.doc_parser = parser
    get_parser_calls = []

    def fake_get_parser(name):
        get_parser_calls.append(name)
        raise AssertionError("existing doc_parser must not call get_parser")

    async def no_store(*args, **kwargs):
        return None

    monkeypatch.setattr("raganything.processor.get_parser", fake_get_parser)
    processor._store_cached_result = no_store
    processor._generate_content_based_doc_id = lambda content_list: "doc-reuse"

    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    content_list, doc_id = await processor.parse_document(str(pdf), display_stats=False)

    assert get_parser_calls == []
    assert processor.doc_parser is parser
    assert doc_id == "doc-reuse"
    assert content_list == [{"type": "text", "text": "from-lazy-parser"}]
    assert len(parser.pdf_calls) == 1
