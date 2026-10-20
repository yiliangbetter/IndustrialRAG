"""Parse-cache identity for falsy formula flags and page zero.

``formula=False`` turns formula OCR off. ``start_page=0`` and ``end_page=0``
mean the first page only. All three values are falsy, so a filter that keeps
only truthy parser options stores the same config as a full-document parse and
replays the wrong extract on the next run.
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


class FakeKVStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


def _make_processor():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {"parser": "mineru", "parse_method": "auto"},
    )()
    processor.parse_cache = FakeKVStorage()
    return processor


@pytest.mark.asyncio
async def test_falsy_formula_and_page_zero_stay_in_the_cache_config(tmp_path):
    processor = _make_processor()
    file_path = tmp_path / "manual.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")
    content = [{"type": "text", "text": "cover"}]

    await processor._store_cached_result(
        "first-page",
        content,
        "doc-cover",
        file_path,
        formula=False,
        table=False,
        start_page=0,
        end_page=0,
    )

    stored = processor.parse_cache.records["first-page"]["parse_config"]
    assert stored == {
        "parser": "mineru",
        "parse_method": "auto",
        "formula": False,
        "table": False,
        "start_page": 0,
        "end_page": 0,
    }

    hit = await processor._get_cached_result(
        "first-page",
        file_path,
        formula=False,
        table=False,
        start_page=0,
        end_page=0,
    )
    assert hit == (content, "doc-cover")

    # A full-document lookup must not reuse the first-page, formula-off extract.
    assert await processor._get_cached_result("first-page", file_path) is None


@pytest.mark.asyncio
async def test_page_zero_lookup_does_not_reuse_a_full_document_cache(tmp_path):
    processor = _make_processor()
    file_path = tmp_path / "manual.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")
    content = [{"type": "text", "text": "whole manual"}]

    await processor._store_cached_result(
        "full-doc",
        content,
        "doc-full",
        file_path,
    )

    assert (
        await processor._get_cached_result(
            "full-doc",
            file_path,
            end_page=0,
        )
        is None
    )
    assert (
        await processor._get_cached_result(
            "full-doc",
            file_path,
            start_page=0,
        )
        is None
    )
    assert (
        await processor._get_cached_result(
            "full-doc",
            file_path,
            formula=False,
        )
        is None
    )
    assert await processor._get_cached_result("full-doc", file_path) == (
        content,
        "doc-full",
    )
