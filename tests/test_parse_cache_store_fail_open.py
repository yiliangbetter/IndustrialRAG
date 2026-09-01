"""Parse-cache writes must not abort a successful parse.

_get_cached_result already swallows backend errors (claimed elsewhere).
If upsert or index_done_callback fails, ingest must still return the
extracted content_list — otherwise a flaky cache backend drops documents
that already parsed successfully.
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


def _make_processor(parse_cache, tmp_path):
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
            "parse_method": "auto",
            "display_content_stats": False,
            "use_full_path": False,
        },
    )()
    processor.parse_cache = parse_cache
    return processor


@pytest.mark.asyncio
async def test_store_cached_result_swallows_upsert_errors(tmp_path):
    class BrokenUpsert:
        async def upsert(self, data):
            raise RuntimeError("cache backend down")

        async def index_done_callback(self):
            raise AssertionError("must not flush after upsert failure")

    processor = _make_processor(BrokenUpsert(), tmp_path)
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    await processor._store_cached_result(
        "cache-key",
        [{"type": "text", "text": "hello"}],
        "doc-123",
        file_path,
    )


@pytest.mark.asyncio
async def test_store_cached_result_swallows_index_done_errors(tmp_path):
    class BrokenFlush:
        def __init__(self):
            self.upserts = []

        async def upsert(self, data):
            self.upserts.append(data)

        async def index_done_callback(self):
            raise RuntimeError("disk full during cache flush")

    cache = BrokenFlush()
    processor = _make_processor(cache, tmp_path)
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    await processor._store_cached_result(
        "cache-key",
        [{"type": "text", "text": "hello"}],
        "doc-123",
        file_path,
    )

    assert cache.upserts, "upsert should still have been attempted"


@pytest.mark.asyncio
async def test_parse_document_returns_content_when_cache_store_fails(
    monkeypatch, tmp_path
):
    import raganything.processor as processor_module

    class FakeParser:
        def parse_pdf(self, **kwargs):
            return [{"type": "text", "text": "parsed body", "page_idx": 0}]

    monkeypatch.setattr(
        processor_module, "get_parser", lambda parser_name: FakeParser()
    )

    class BrokenCache:
        async def upsert(self, data):
            raise RuntimeError("parse cache unavailable")

        async def index_done_callback(self):
            return None

    processor = _make_processor(BrokenCache(), tmp_path)
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    content_list, doc_id = await processor.parse_document(str(pdf))

    assert content_list == [{"type": "text", "text": "parsed body", "page_idx": 0}]
    assert isinstance(doc_id, str) and doc_id
