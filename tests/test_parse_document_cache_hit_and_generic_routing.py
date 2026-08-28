"""parse_document cache hits must skip the parser and still emit complete.

Stale re-parses waste OCR; skipping parse_complete after a cache hit leaves
downstream callbacks thinking the document never finished.
"""

import pytest

from raganything.callbacks import CallbackManager, ProcessingCallback
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


class RecordingParseCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_parse_start(self, file_path, **kwargs):
        self.events.append(("start", file_path))

    def on_parse_complete(self, file_path, content_blocks=0, doc_id="", **kwargs):
        self.events.append(("complete", file_path, content_blocks, doc_id))

    def on_parse_error(self, file_path, error="", **kwargs):
        self.events.append(("error", file_path, str(error)))


class DummyProcessor(ProcessorMixin):
    pass


class RecordingParser:
    def __init__(self):
        self.pdf_calls = []
        self.document_calls = []

    def parse_pdf(self, **kwargs):
        self.pdf_calls.append(kwargs)
        raise AssertionError("cache hit must not parse PDF")

    def parse_document(self, **kwargs):
        self.document_calls.append(kwargs)
        return [{"type": "text", "text": "from-generic"}]


def _make_processor(tmp_path):
    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "parser": "mineru",
            "parser_output_dir": str(tmp_path / "output"),
            "parse_method": "auto",
            "display_content_stats": False,
            "use_full_path": False,
        },
    )()
    processor.callback_manager = CallbackManager()
    processor.parse_cache = object()
    processor.doc_parser = RecordingParser()
    return processor


@pytest.mark.asyncio
async def test_parse_document_cache_hit_skips_parser_and_store(tmp_path):
    processor = _make_processor(tmp_path)
    callback = RecordingParseCallback()
    processor.callback_manager.register(callback)

    cached = ([{"type": "text", "text": "cached body"}], "doc-cached")
    store_calls = []

    async def fake_cached(*args, **kwargs):
        return cached

    async def fake_store(*args, **kwargs):
        store_calls.append((args, kwargs))

    processor._get_cached_result = fake_cached
    processor._store_cached_result = fake_store

    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    content_list, doc_id = await processor.parse_document(str(pdf), display_stats=False)

    assert content_list == cached[0]
    assert doc_id == "doc-cached"
    assert processor.doc_parser.pdf_calls == []
    assert store_calls == []
    kinds = [event[0] for event in callback.events]
    assert kinds == ["start", "complete"]
    assert callback.events[1][2] == 1
    assert callback.events[1][3] == "doc-cached"


@pytest.mark.asyncio
async def test_parse_document_routes_text_files_to_generic_parser(tmp_path):
    processor = _make_processor(tmp_path)

    async def no_cache(*args, **kwargs):
        return None

    async def no_store(*args, **kwargs):
        return None

    processor._get_cached_result = no_cache
    processor._store_cached_result = no_store
    processor._generate_content_based_doc_id = lambda content_list: "doc-txt"

    notes = tmp_path / "notes.txt"
    notes.write_text("plain notes", encoding="utf-8")

    content_list, doc_id = await processor.parse_document(
        str(notes), parse_method="ocr", display_stats=False, lang="en"
    )

    assert doc_id == "doc-txt"
    assert content_list == [{"type": "text", "text": "from-generic"}]
    assert processor.doc_parser.pdf_calls == []
    call = processor.doc_parser.document_calls[0]
    assert str(call["file_path"]).endswith("notes.txt")
    assert call["method"] == "ocr"
    assert call["lang"] == "en"
