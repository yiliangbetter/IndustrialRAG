"""Regression tests for parse-result cache store, hit, and invalidation.

The parse cache is the only guard against re-parsing. Silent regressions here
serve stale OCR for changed files or settings, or lose durability when the
index flush is skipped.
"""

import os

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


def _make_processor(parse_method="auto"):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "parser": "mineru",
            "parse_method": parse_method,
        },
    )()
    processor.parse_cache = FakeKVStorage()
    return processor


@pytest.mark.asyncio
async def test_store_cached_result_persists_config_and_flushes(tmp_path):
    processor = _make_processor()
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    await processor._store_cached_result(
        "cache-key",
        [{"type": "text", "text": "hello"}],
        "doc-123",
        file_path,
        lang="en",
        ignored="value",
    )

    cached = processor.parse_cache.records["cache-key"]
    assert cached["doc_id"] == "doc-123"
    assert cached["content_list"] == [{"type": "text", "text": "hello"}]
    assert cached["mtime"] == file_path.stat().st_mtime
    assert cached["parse_config"] == {
        "parser": "mineru",
        "parse_method": "auto",
        "lang": "en",
    }
    # Irrelevant kwargs must not pollute the cache config key space.
    assert "ignored" not in cached["parse_config"]
    assert processor.parse_cache.index_done_calls == 1


@pytest.mark.asyncio
async def test_get_cached_result_hit_with_matching_mtime_and_config(tmp_path):
    processor = _make_processor()
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    await processor._store_cached_result(
        "cache-key",
        [{"type": "text", "text": "cached"}],
        "doc-hit",
        file_path,
        lang="en",
    )

    result = await processor._get_cached_result(
        "cache-key",
        file_path,
        lang="en",
    )

    assert result == ([{"type": "text", "text": "cached"}], "doc-hit")


@pytest.mark.asyncio
async def test_get_cached_result_miss_when_file_mtime_changes(tmp_path):
    processor = _make_processor()
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    await processor._store_cached_result(
        "cache-key",
        [{"type": "text", "text": "stale"}],
        "doc-stale",
        file_path,
        lang="en",
    )

    new_mtime = file_path.stat().st_mtime + 5
    os.utime(file_path, (new_mtime, new_mtime))

    result = await processor._get_cached_result(
        "cache-key",
        file_path,
        lang="en",
    )

    assert result is None


@pytest.mark.asyncio
async def test_get_cached_result_miss_when_lang_config_changes(tmp_path):
    processor = _make_processor()
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    await processor._store_cached_result(
        "cache-key",
        [{"type": "text", "text": "en-ocr"}],
        "doc-en",
        file_path,
        lang="en",
    )

    result = await processor._get_cached_result(
        "cache-key",
        file_path,
        lang="ch",
    )

    assert result is None


@pytest.mark.asyncio
async def test_get_cached_result_miss_when_parse_method_changes(tmp_path):
    processor = _make_processor(parse_method="auto")
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    await processor._store_cached_result(
        "cache-key",
        [{"type": "text", "text": "auto-parse"}],
        "doc-auto",
        file_path,
        parse_method="auto",
    )

    result = await processor._get_cached_result(
        "cache-key",
        file_path,
        parse_method="ocr",
    )

    assert result is None


@pytest.mark.asyncio
async def test_get_cached_result_miss_when_content_or_doc_id_missing(tmp_path):
    processor = _make_processor()
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")
    mtime = file_path.stat().st_mtime

    processor.parse_cache.records["empty-content"] = {
        "content_list": [],
        "doc_id": "doc-empty",
        "mtime": mtime,
        "parse_config": {"parser": "mineru", "parse_method": "auto"},
    }
    processor.parse_cache.records["missing-doc"] = {
        "content_list": [{"type": "text", "text": "x"}],
        "doc_id": None,
        "mtime": mtime,
        "parse_config": {"parser": "mineru", "parse_method": "auto"},
    }

    assert await processor._get_cached_result("empty-content", file_path) is None
    assert await processor._get_cached_result("missing-doc", file_path) is None


@pytest.mark.asyncio
async def test_get_cached_result_returns_none_without_cache(tmp_path):
    processor = _make_processor()
    processor.parse_cache = None
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    assert await processor._get_cached_result("any", file_path) is None


@pytest.mark.asyncio
async def test_store_cached_result_noops_without_cache(tmp_path):
    processor = _make_processor()
    processor.parse_cache = None
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    await processor._store_cached_result(
        "cache-key",
        [{"type": "text", "text": "hello"}],
        "doc-123",
        file_path,
    )


@pytest.mark.asyncio
async def test_get_cached_result_swallows_storage_errors(tmp_path):
    processor = _make_processor()
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    class BrokenStorage:
        async def get_by_id(self, key):
            raise RuntimeError("cache backend down")

    processor.parse_cache = BrokenStorage()

    assert await processor._get_cached_result("cache-key", file_path) is None
