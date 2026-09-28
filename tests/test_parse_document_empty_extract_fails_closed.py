"""Empty parser extracts must not become successful or cached documents.

``parse_document`` raises when a parser returns no blocks, and the parse cache
treats an empty ``content_list`` as a miss. If either check regresses, a blank
extract is stored and later reads skip the parser entirely.
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


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_parse_start(self, file_path, **kw):
        self.events.append("start")

    def on_parse_complete(self, file_path, content_blocks=0, **kw):
        self.events.append("complete")


class FakeParseCache:
    def __init__(self):
        self.records = {}
        self.upserts = []

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.upserts.append(data)
        self.records.update(data)

    async def index_done_callback(self):
        return None


class FakeParser:
    def __init__(self, content_list):
        self.content_list = content_list
        self.calls = []

    def parse_pdf(self, **kwargs):
        self.calls.append("parse_pdf")
        return self.content_list

    def parse_image(self, **kwargs):
        self.calls.append("parse_image")
        return self.content_list

    def parse_office_doc(self, **kwargs):
        self.calls.append("parse_office_doc")
        return self.content_list

    def parse_document(self, **kwargs):
        self.calls.append("parse_document")
        return self.content_list


def _processor(tmp_path, parser, cache):
    class DummyProcessor(ProcessorMixin):
        pass

    callback = RecordingCallback()
    manager = CallbackManager()
    manager.register(callback)

    dummy = DummyProcessor()
    dummy.logger = FakeLogger()
    dummy.config = type(
        "Config",
        (),
        {
            "parser": "mineru",
            "parser_output_dir": str(tmp_path / "output"),
            "parse_method": "auto",
            "display_content_stats": False,
        },
    )()
    dummy.doc_parser = parser
    dummy.parse_cache = cache
    dummy.callback_manager = manager
    return dummy, callback


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "parser_method"),
    [
        ("scan.pdf", "parse_pdf"),
        ("photo.png", "parse_image"),
        ("notes.txt", "parse_document"),
        ("sheet.docx", "parse_office_doc"),
    ],
)
async def test_empty_extract_raises_and_is_not_cached(
    tmp_path, filename, parser_method
):
    source = tmp_path / filename
    source.write_bytes(b"not-empty-bytes")
    parser = FakeParser([])
    cache = FakeParseCache()
    dummy, callback = _processor(tmp_path, parser, cache)

    with pytest.raises(ValueError, match="No content was extracted"):
        await dummy.parse_document(str(source))

    assert parser.calls == [parser_method]
    assert cache.upserts == []
    assert cache.records == {}
    assert callback.events == ["start"]


@pytest.mark.asyncio
async def test_cached_empty_content_list_is_a_miss_and_reparse_replaces_it(tmp_path):
    source = tmp_path / "manual.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    recovered = [{"type": "text", "text": "recovered body", "page_idx": 0}]
    parser = FakeParser(recovered)
    cache = FakeParseCache()
    dummy, callback = _processor(tmp_path, parser, cache)

    cache_key = dummy._generate_cache_key(source, "auto")
    cache.records[cache_key] = {
        "content_list": [],
        "doc_id": "doc-stale-empty",
        "mtime": source.stat().st_mtime,
        "parse_config": {"parser": "mineru", "parse_method": "auto"},
    }

    content_list, doc_id = await dummy.parse_document(str(source))

    assert parser.calls == ["parse_pdf"]
    assert content_list == recovered
    assert doc_id != "doc-stale-empty"
    assert doc_id.startswith("doc-")
    stored = cache.records[cache_key]
    assert stored["content_list"] == recovered
    assert stored["doc_id"] == doc_id
    assert "complete" in callback.events
