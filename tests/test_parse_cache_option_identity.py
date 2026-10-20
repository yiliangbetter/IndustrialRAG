"""Parse-cache identity for parser options that are easy to drop.

``formula=False`` and ``start_page=0`` are real parse settings. They must
change the cache key and reject a stored entry that does not carry the same
values. Unrelated keyword arguments must not.
"""

import logging
import os
from types import SimpleNamespace

import pytest

from raganything.processor import ProcessorMixin


class _MemoryCache:
    def __init__(self):
        self.rows = {}

    async def upsert(self, data):
        self.rows.update(data)

    async def index_done_callback(self):
        return None

    async def get_by_id(self, key):
        return self.rows.get(key)


class _RaisingCache:
    async def get_by_id(self, key):
        raise RuntimeError("parse cache unavailable")

    async def upsert(self, data):
        raise RuntimeError("parse cache unavailable")

    async def index_done_callback(self):
        raise AssertionError("index should not run after a failed upsert")


def _processor(parse_method="auto", parser="mineru"):
    proc = ProcessorMixin()
    proc.config = SimpleNamespace(parser=parser, parse_method=parse_method)
    proc.logger = logging.getLogger("test_parse_cache_option_identity")
    return proc


def _pdf(tmp_path):
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"%PDF-1.4")
    return path


def test_cache_key_keeps_false_and_zero_parser_options(tmp_path):
    path = _pdf(tmp_path)
    proc = _processor()

    base = proc._generate_cache_key(path, "auto")
    formula_off = proc._generate_cache_key(path, "auto", formula=False)
    first_page = proc._generate_cache_key(path, "auto", start_page=0)
    chinese = proc._generate_cache_key(path, "auto", lang="zh")

    assert formula_off != base
    assert first_page != base
    assert chinese != base
    assert len({formula_off, first_page, chinese}) == 3


def test_cache_key_ignores_unrelated_kwargs_and_uses_config_method(tmp_path):
    path = _pdf(tmp_path)
    proc = _processor(parse_method="ocr")

    plain = proc._generate_cache_key(path, "ocr")
    assert proc._generate_cache_key(path, None) == plain
    assert proc._generate_cache_key(path, "ocr", extra_flag=1) == plain
    assert proc._generate_cache_key(path, "txt") != plain


def test_cache_key_changes_when_mtime_changes(tmp_path):
    path = _pdf(tmp_path)
    proc = _processor()
    before = proc._generate_cache_key(path, "auto")
    stamp = path.stat().st_mtime + 10
    os.utime(path, (stamp, stamp))
    assert proc._generate_cache_key(path, "auto") != before


@pytest.mark.asyncio
async def test_stored_false_and_zero_options_must_match_on_lookup(tmp_path):
    path = _pdf(tmp_path)
    proc = _processor(parse_method="ocr")
    proc.parse_cache = _MemoryCache()
    content = [{"type": "text", "text": "torque spec"}]
    key = proc._generate_cache_key(
        path, "ocr", formula=False, table=False, start_page=0, lang="zh"
    )

    await proc._store_cached_result(
        key,
        content,
        "doc-1",
        path,
        None,
        formula=False,
        table=False,
        start_page=0,
        lang="zh",
        extra_flag="noise",
    )

    hit = await proc._get_cached_result(
        key,
        path,
        None,
        formula=False,
        table=False,
        start_page=0,
        lang="zh",
        extra_flag="different",
    )
    assert hit == (content, "doc-1")

    omitted = await proc._get_cached_result(key, path, "ocr", lang="zh")
    assert omitted is None

    formula_on = await proc._get_cached_result(
        key,
        path,
        "ocr",
        formula=True,
        table=False,
        start_page=0,
        lang="zh",
    )
    assert formula_on is None


@pytest.mark.asyncio
async def test_parse_cache_misses_stale_mtime_empty_payload_and_cache_errors(
    tmp_path,
):
    path = _pdf(tmp_path)
    proc = _processor()
    proc.parse_cache = _MemoryCache()
    content = [{"type": "text", "text": "kept"}]
    key = proc._generate_cache_key(path, "auto")
    await proc._store_cached_result(key, content, "doc-1", path, "auto")

    stamp = path.stat().st_mtime + 5
    os.utime(path, (stamp, stamp))
    assert await proc._get_cached_result(key, path, "auto") is None

    proc.parse_cache.rows[key]["mtime"] = path.stat().st_mtime
    assert await proc._get_cached_result(key, path, "auto") == (content, "doc-1")

    await proc._store_cached_result(key, [], "doc-1", path, "auto")
    assert await proc._get_cached_result(key, path, "auto") is None

    await proc._store_cached_result(key, content, "", path, "auto")
    assert await proc._get_cached_result(key, path, "auto") is None

    proc.parse_cache = None
    assert await proc._get_cached_result(key, path, "auto") is None

    proc.parse_cache = _RaisingCache()
    assert await proc._get_cached_result(key, path, "auto") is None
    await proc._store_cached_result(key, content, "doc-1", path, "auto")
