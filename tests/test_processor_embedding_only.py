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


class FakeStorage:
    def __init__(self):
        self.records = {}
        self.upsert_calls = []
        self.index_done_calls = 0

    async def upsert(self, data):
        self.records.update(data)
        self.upsert_calls.append(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class FakeLightRAG:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.text_chunks = FakeStorage()
        self.chunks_vdb = FakeStorage()
        self.doc_status = FakeStorage()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


@pytest.mark.asyncio
async def test_insert_text_content_embedding_only_persists_chunks_and_doc_status():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = FakeLightRAG()

    await processor._insert_text_content_embedding_only(
        "para one\n\npara two", "doc.pdf", "doc-abc"
    )

    assert len(processor.lightrag.text_chunks.upsert_calls) == 1
    assert len(processor.lightrag.chunks_vdb.upsert_calls) == 1

    text_chunks = processor.lightrag.text_chunks.upsert_calls[0]
    vector_chunks = processor.lightrag.chunks_vdb.upsert_calls[0]
    assert text_chunks == vector_chunks
    assert len(text_chunks) == 2
    assert [chunk["content"] for chunk in text_chunks.values()] == [
        "para one",
        "para two",
    ]
    assert {chunk["full_doc_id"] for chunk in text_chunks.values()} == {"doc-abc"}
    assert [chunk["chunk_order_index"] for chunk in text_chunks.values()] == [0, 1]
    assert [chunk["file_path"] for chunk in text_chunks.values()] == [
        "doc.pdf",
        "doc.pdf",
    ]

    doc_status = processor.lightrag.doc_status.records["doc-abc"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["multimodal_processed"] is True
    assert doc_status["chunks_count"] == 2
    assert doc_status["chunks_list"] == list(text_chunks.keys())
    assert doc_status["file_path"] == "doc.pdf"

    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1
    assert processor.lightrag.doc_status.index_done_calls == 1
    assert processor.lightrag.insert_done_calls == 1
