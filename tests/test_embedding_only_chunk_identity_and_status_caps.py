"""Embedding-only chunk identity and doc-status size caps.

Repeated safety text must not collapse to one vector, and shared wording
across manuals must not overwrite another document's chunk. The status row
keeps only a bounded prefix of the raw document; the chunk store keeps the
full paragraphs.
"""

import pytest

from lightrag.utils import compute_mdhash_id

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

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        return None


class FakeLightRAG:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.text_chunks = FakeStorage()
        self.chunks_vdb = FakeStorage()
        self.doc_status = FakeStorage()

        async def _insert_done():
            return None

        self._insert_done = _insert_done


class DummyProcessor(ProcessorMixin):
    pass


def _processor():
    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = FakeLightRAG()
    return processor


@pytest.mark.asyncio
async def test_repeated_and_cross_document_text_keeps_distinct_chunk_ids():
    processor = _processor()
    warning = "DANGER: lock out the breaker before service."

    await processor._insert_text_content_embedding_only(
        text_content=f"{warning}\n\n{warning}",
        file_ref="line-a.pdf",
        doc_id="doc-a",
    )
    await processor._insert_text_content_embedding_only(
        text_content=warning,
        file_ref="line-b.pdf",
        doc_id="doc-b",
    )

    expected = {
        compute_mdhash_id(f"doc-a:0:{warning}", prefix="chunk-"): (
            "doc-a",
            0,
            "line-a.pdf",
        ),
        compute_mdhash_id(f"doc-a:1:{warning}", prefix="chunk-"): (
            "doc-a",
            1,
            "line-a.pdf",
        ),
        compute_mdhash_id(f"doc-b:0:{warning}", prefix="chunk-"): (
            "doc-b",
            0,
            "line-b.pdf",
        ),
    }
    assert len(expected) == 3

    chunks = processor.lightrag.text_chunks.records
    assert set(chunks) == set(expected)
    assert set(processor.lightrag.chunks_vdb.records) == set(expected)
    for chunk_id, (doc_id, index, file_ref) in expected.items():
        chunk = chunks[chunk_id]
        assert chunk["content"] == warning
        assert chunk["full_doc_id"] == doc_id
        assert chunk["chunk_order_index"] == index
        assert chunk["file_path"] == file_ref

    assert processor.lightrag.doc_status.records["doc-a"]["chunks_count"] == 2
    assert processor.lightrag.doc_status.records["doc-b"]["chunks_count"] == 1
    assert (
        processor.lightrag.doc_status.records["doc-a"]["status"] == DocStatus.PROCESSED
    )


@pytest.mark.asyncio
async def test_doc_status_caps_raw_text_while_chunks_keep_full_paragraphs():
    processor = _processor()
    first = "A" * 1200
    second = "B" * 1200
    # Leading and trailing spaces survive in the status prefix and are
    # stripped from the stored chunks.
    raw = f"  {first}  \n\n  {second}"

    await processor._insert_text_content_embedding_only(
        text_content=raw,
        file_ref="long-manual.pdf",
        doc_id="doc-long",
    )

    status = processor.lightrag.doc_status.records["doc-long"]
    assert status["content"] == raw[:2000]
    assert status["content_summary"] == raw[:500]
    assert status["content"].startswith("  ")
    assert "\n\n" in status["content"]
    assert status["error_msg"] == ""
    assert status["chunks_count"] == 2
    assert status["file_path"] == "long-manual.pdf"
    assert status["multimodal_processed"] is True

    ordered = sorted(
        processor.lightrag.text_chunks.records.values(),
        key=lambda chunk: chunk["chunk_order_index"],
    )
    assert [chunk["content"] for chunk in ordered] == [first, second]
    assert (
        processor.lightrag.chunks_vdb.records == processor.lightrag.text_chunks.records
    )
