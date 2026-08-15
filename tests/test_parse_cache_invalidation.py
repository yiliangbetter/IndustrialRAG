"""Parse-cache identity and invalidation for ProcessorMixin.

Stale or over-broad parse cache reuse would serve the wrong document blocks
as a successful parse. These tests lock cache-key inputs, mtime/config
invalidation, and parse_document cache hits without invoking real parsers.
"""

from __future__ import annotations

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


class FakeParseCache:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


def _make_processor(tmp_path, *, parser="mineru", parse_method="auto", parse_cache=None):
    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.config = type(
        "Config",
        (),
        {
            "parser": parser,
            "parser_output_dir": str(tmp_path / "output"),
            "parse_method": parse_method,
            "display_content_stats": False,
            "use_full_path": False,
        },
    )()
    dummy.logger = FakeLogger()
    dummy.parse_cache = parse_cache
    dummy.callback_manager = None
    return dummy


def test_cache_key_stable_for_same_file_and_config(tmp_path):
    dummy = _make_processor(tmp_path)
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    key1 = dummy._generate_cache_key(pdf, "auto")
    key2 = dummy._generate_cache_key(pdf, "auto")
    assert key1 == key2
    assert len(key1) == 32


def test_cache_key_changes_with_parse_method_parser_and_relevant_kwargs(tmp_path):
    dummy = _make_processor(tmp_path, parser="mineru", parse_method="auto")
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    base = dummy._generate_cache_key(pdf, "auto")
    assert dummy._generate_cache_key(pdf, "ocr") != base
    assert dummy._generate_cache_key(pdf, "auto", lang="ch") != base
    assert dummy._generate_cache_key(pdf, "auto", device="cuda") != base
    assert dummy._generate_cache_key(pdf, "auto", start_page=2, end_page=5) != base

    dummy.config.parser = "docling"
    assert dummy._generate_cache_key(pdf, "auto") != base


def test_cache_key_ignores_irrelevant_kwargs(tmp_path):
    dummy = _make_processor(tmp_path)
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    base = dummy._generate_cache_key(pdf, "auto")
    assert dummy._generate_cache_key(pdf, "auto", method="ocr", foo="bar") == base
    assert dummy._generate_cache_key(pdf, "auto", output_dir="/tmp") == base


@pytest.mark.asyncio
async def test_get_cached_result_returns_none_without_cache(tmp_path):
    dummy = _make_processor(tmp_path, parse_cache=None)
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    assert await dummy._get_cached_result("abc", pdf, "auto") is None


@pytest.mark.asyncio
async def test_store_and_get_cached_result_roundtrip(tmp_path):
    cache = FakeParseCache()
    dummy = _make_processor(tmp_path, parse_cache=cache)
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    content = [{"type": "text", "text": "pump spec", "page_idx": 0}]
    await dummy._store_cached_result("key-1", content, "doc-1", pdf, "auto")

    assert cache.index_done_calls == 1
    hit = await dummy._get_cached_result("key-1", pdf, "auto")
    assert hit == (content, "doc-1")


@pytest.mark.asyncio
async def test_cached_result_invalidated_when_mtime_changes(tmp_path):
    cache = FakeParseCache()
    dummy = _make_processor(tmp_path, parse_cache=cache)
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    await dummy._store_cached_result(
        "key-1",
        [{"type": "text", "text": "old", "page_idx": 0}],
        "doc-1",
        pdf,
        "auto",
    )
    os.utime(pdf, (pdf.stat().st_atime, pdf.stat().st_mtime + 10))

    assert await dummy._get_cached_result("key-1", pdf, "auto") is None


@pytest.mark.asyncio
async def test_cached_result_invalidated_when_parse_config_changes(tmp_path):
    cache = FakeParseCache()
    dummy = _make_processor(tmp_path, parse_cache=cache, parse_method="auto")
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    await dummy._store_cached_result(
        "key-1",
        [{"type": "text", "text": "body", "page_idx": 0}],
        "doc-1",
        pdf,
        "auto",
    )
    # Same cache key, different method — second-line config check must miss.
    assert await dummy._get_cached_result("key-1", pdf, "ocr") is None


@pytest.mark.asyncio
async def test_cached_result_incomplete_without_doc_id(tmp_path):
    cache = FakeParseCache()
    dummy = _make_processor(tmp_path, parse_cache=cache)
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    cache.records["key-1"] = {
        "content_list": [{"type": "text", "text": "x"}],
        "doc_id": None,
        "mtime": pdf.stat().st_mtime,
        "parse_config": {"parser": "mineru", "parse_method": "auto"},
    }
    assert await dummy._get_cached_result("key-1", pdf, "auto") is None


@pytest.mark.asyncio
async def test_get_cached_result_swallows_cache_errors(tmp_path):
    class BrokenCache:
        async def get_by_id(self, key):
            raise RuntimeError("disk full")

    dummy = _make_processor(tmp_path, parse_cache=BrokenCache())
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    assert await dummy._get_cached_result("key-1", pdf, "auto") is None


@pytest.mark.asyncio
async def test_store_cached_result_swallows_cache_errors(tmp_path):
    class BrokenCache:
        async def upsert(self, data):
            raise RuntimeError("disk full")

        async def index_done_callback(self):
            return None

    dummy = _make_processor(tmp_path, parse_cache=BrokenCache())
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    await dummy._store_cached_result(
        "key-1",
        [{"type": "text", "text": "x"}],
        "doc-1",
        pdf,
        "auto",
    )


@pytest.mark.asyncio
async def test_parse_document_uses_cache_and_skips_parser(tmp_path):
    cache = FakeParseCache()
    dummy = _make_processor(tmp_path, parse_cache=cache)
    calls = {"parse_pdf": 0}

    class FakeParser:
        def parse_pdf(self, **kwargs):
            calls["parse_pdf"] += 1
            return [{"type": "text", "text": f"pass-{calls['parse_pdf']}", "page_idx": 0}]

    dummy.doc_parser = FakeParser()

    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    first, first_id = await dummy.parse_document(str(pdf))
    second, second_id = await dummy.parse_document(str(pdf))

    assert calls["parse_pdf"] == 1
    assert first == [{"type": "text", "text": "pass-1", "page_idx": 0}]
    assert second == first
    assert first_id == second_id
    assert first_id.startswith("doc-")

    os.utime(pdf, (pdf.stat().st_atime, pdf.stat().st_mtime + 10))
    third, _ = await dummy.parse_document(str(pdf))
    assert calls["parse_pdf"] == 2
    assert third == [{"type": "text", "text": "pass-2", "page_idx": 0}]
