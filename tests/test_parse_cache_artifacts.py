"""Regression: parse cache must respect output_dir and missing image artifacts."""

import pytest
from pathlib import Path

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
        self.store = {}
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.store.get(key)

    async def upsert(self, data):
        # data is {cache_key: payload}
        self.store.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


def _make_processor(tmp_path, parser_output_dir=None):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    out = parser_output_dir or str(tmp_path / "out_a")
    processor.config = type(
        "Config",
        (),
        {"parser": "mineru", "parse_method": "auto", "parser_output_dir": out},
    )()
    processor.parse_cache = FakeParseCache()
    return processor


@pytest.mark.asyncio
async def test_parse_cache_misses_when_output_dir_changes(tmp_path):
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4")
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    out_a.mkdir()
    out_b.mkdir()

    processor = _make_processor(tmp_path, parser_output_dir=str(out_a))
    content_list = [{"type": "text", "text": "hello"}]
    doc_id = "doc-test"

    key_a = processor._generate_cache_key(src, "auto", output_dir=str(out_a))
    await processor._store_cached_result(
        key_a, content_list, doc_id, src, "auto", output_dir=str(out_a)
    )

    # Same file/method but different output_dir must not reuse the entry
    hit = await processor._get_cached_result(
        key_a, src, "auto", output_dir=str(out_b)
    )
    assert hit is None

    key_b = processor._generate_cache_key(src, "auto", output_dir=str(out_b))
    assert key_a != key_b


@pytest.mark.asyncio
async def test_parse_cache_misses_when_image_artifact_deleted(tmp_path):
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4")
    out = tmp_path / "out"
    out.mkdir()
    img = out / "figure.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    processor = _make_processor(tmp_path, parser_output_dir=str(out))
    content_list = [
        {"type": "text", "text": "caption"},
        {"type": "image", "img_path": str(img)},
    ]
    doc_id = "doc-img"

    key = processor._generate_cache_key(src, "auto", output_dir=str(out))
    await processor._store_cached_result(
        key, content_list, doc_id, src, "auto", output_dir=str(out)
    )

    hit = await processor._get_cached_result(key, src, "auto", output_dir=str(out))
    assert hit is not None
    assert hit[1] == doc_id

    img.unlink()
    miss = await processor._get_cached_result(key, src, "auto", output_dir=str(out))
    assert miss is None


def test_cached_content_artifacts_exist_helper(tmp_path):
    present = tmp_path / "ok.png"
    present.write_bytes(b"x")
    assert ProcessorMixin._cached_content_artifacts_exist(
        [{"type": "image", "img_path": str(present)}]
    )
    assert not ProcessorMixin._cached_content_artifacts_exist(
        [{"type": "image", "img_path": str(tmp_path / "missing.png")}]
    )
    assert ProcessorMixin._cached_content_artifacts_exist(
        [{"type": "text", "text": "no images"}]
    )
