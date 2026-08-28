"""Embedding-only ingest must split on blank lines and skip empty paragraphs.

Without this, a 20-page manual becomes one vector or keeps blank chunks that
pollute nearest-neighbor retrieval. Tokenizer fallback is covered elsewhere.
"""

import pytest

from raganything.base import DocStatus
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


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class FakeStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeLightRAG:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.text_chunks = FakeStorage()
        self.chunks_vdb = FakeStorage()
        self.doc_status = FakeStorage()
        self.insert_done_calls = 0

        async def _insert_done():
            self.insert_done_calls += 1

        self._insert_done = _insert_done


class DummyProcessor(ProcessorMixin):
    pass


def _processor():
    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = FakeLightRAG()
    return processor


@pytest.mark.asyncio
async def test_embedding_only_splits_paragraphs_and_skips_blank_chunks():
    processor = _processor()
    text = "First paragraph has words.\n\n   \n\nSecond paragraph is distinct."

    await processor._insert_text_content_embedding_only(
        text_content=text, file_ref="manual.pdf", doc_id="doc-split"
    )

    chunks = processor.lightrag.text_chunks.records
    vdb = processor.lightrag.chunks_vdb.records
    assert chunks == vdb
    ordered = sorted(chunks.values(), key=lambda c: c["chunk_order_index"])
    assert [c["content"] for c in ordered] == [
        "First paragraph has words.",
        "Second paragraph is distinct.",
    ]
    assert [c["chunk_order_index"] for c in ordered] == [0, 1]
    assert [c["tokens"] for c in ordered] == [4, 4]
    assert all(c["full_doc_id"] == "doc-split" for c in ordered)
    assert all(c["file_path"] == "manual.pdf" for c in ordered)

    status = processor.lightrag.doc_status.records["doc-split"]
    assert status["status"] == DocStatus.PROCESSED
    assert status["chunks_count"] == 2
    assert status["multimodal_processed"] is True
    assert status["file_path"] == "manual.pdf"
    assert set(status["chunks_list"]) == set(chunks)
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1
