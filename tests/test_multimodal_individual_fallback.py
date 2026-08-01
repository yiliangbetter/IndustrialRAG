"""Regression tests for multimodal batch→individual fallback processing."""

import pytest

from raganything.base import DocStatus
from raganything.processor import ProcessorMixin


class FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []
        self.debugs = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg))

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))

    def debug(self, msg, *args, **kwargs):
        self.debugs.append(str(msg))


class FakeDocStatusStorage:
    def __init__(self, records=None):
        self.records = records or {}
        self.upserts = []
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.upserts.append(data)
        for doc_id, payload in data.items():
            self.records[doc_id] = payload

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeProcessor:
    def __init__(self, *, fail=False, chunk_id="chunk-mm-1"):
        self.fail = fail
        self.chunk_id = chunk_id
        self.calls = []

    async def process_multimodal_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("processor boom")
        return (
            "caption",
            {"entity_name": "Table 1 (table)", "chunk_id": self.chunk_id},
            [{"nodes": {}, "edges": {}}],
        )


def _make_processor(*, records=None, modal_processors=None, use_full_path=False):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type("Config", (), {"use_full_path": use_full_path})()
    processor.modal_processors = modal_processors or {}
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "doc_status": FakeDocStatusStorage(records),
            "chunk_entity_relation_graph": object(),
            "entities_vdb": object(),
            "relationships_vdb": object(),
            "full_entities": object(),
            "full_relations": object(),
            "llm_response_cache": object(),
            "entity_chunks": object(),
            "relation_chunks": object(),
        },
    )()

    async def fake_ensure():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure
    return processor


@pytest.mark.asyncio
async def test_process_multimodal_falls_back_to_individual_when_batch_fails():
    processor = _make_processor()
    processor._batch_calls = 0
    processor._individual_calls = 0
    processor._mark_calls = []

    async def failing_batch(**kwargs):
        processor._batch_calls += 1
        raise RuntimeError("batch exploded")

    async def fake_individual(items, file_path, doc_id):
        processor._individual_calls += 1
        processor._individual_args = (items, file_path, doc_id)

    async def fake_mark(doc_id):
        processor._mark_calls.append(doc_id)

    processor._process_multimodal_content_batch_type_aware = failing_batch
    processor._process_multimodal_content_individual = fake_individual
    processor._mark_multimodal_processing_complete = fake_mark

    items = [{"type": "table", "table_body": "| a |"}]
    await processor._process_multimodal_content(items, "manual.pdf", "doc-1")

    assert processor._batch_calls == 1
    assert processor._individual_calls == 1
    assert processor._individual_args == (items, "manual.pdf", "doc-1")
    assert processor._mark_calls == ["doc-1"]
    assert any(
        "Falling back to individual multimodal processing" in w
        for w in processor.logger.warnings
    )
    assert any("Error in multimodal processing" in e for e in processor.logger.errors)


@pytest.mark.asyncio
async def test_individual_processing_updates_doc_status_and_merges(monkeypatch):
    table_processor = FakeProcessor(chunk_id="chunk-mm-table")
    processor = _make_processor(
        records={
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "chunks_list": ["chunk-text"],
                "chunks_count": 1,
            }
        },
        modal_processors={"table": table_processor},
    )
    processor._marked = None

    async def fake_mark(doc_id):
        processor._marked = doc_id

    processor._mark_multimodal_processing_complete = fake_mark

    merge_calls = []

    async def fake_merge(**kwargs):
        merge_calls.append(kwargs)

    async def fake_get_namespace_data(name):
        return {"latest_message": "", "history_messages": []}

    def fake_get_pipeline_status_lock():
        return object()

    async def fake_insert_done():
        processor._insert_done = True

    import lightrag.kg.shared_storage as shared_mod
    import lightrag.operate as operate_mod

    monkeypatch.setattr(operate_mod, "merge_nodes_and_edges", fake_merge)
    monkeypatch.setattr(shared_mod, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(
        shared_mod, "get_pipeline_status_lock", fake_get_pipeline_status_lock
    )
    processor.lightrag._insert_done = fake_insert_done

    await processor._process_multimodal_content_individual(
        [{"type": "table", "table_body": "| a | b |", "page_idx": 2}],
        "/docs/manual.pdf",
        "doc-1",
    )

    assert len(table_processor.calls) == 1
    call = table_processor.calls[0]
    assert call["file_path"] == "manual.pdf"
    assert call["doc_id"] == "doc-1"
    assert call["chunk_order_index"] == 1
    assert call["item_info"] == {"page_idx": 2, "index": 0, "type": "table"}
    assert call["batch_mode"] is True

    storage = processor.lightrag.doc_status
    assert storage.upserts[0]["doc-1"]["chunks_list"] == [
        "chunk-text",
        "chunk-mm-table",
    ]
    assert storage.upserts[0]["doc-1"]["chunks_count"] == 2
    assert storage.index_done_calls == 1
    assert len(merge_calls) == 1
    assert merge_calls[0]["doc_id"] == "doc-1"
    assert processor._insert_done is True
    assert processor._marked == "doc-1"
    assert any(
        "Individual multimodal content processing complete" in i
        for i in processor.logger.infos
    )


@pytest.mark.asyncio
async def test_individual_processing_isolates_per_item_failures(monkeypatch):
    good = FakeProcessor(chunk_id="chunk-ok")
    bad = FakeProcessor(fail=True)
    processor = _make_processor(
        records={
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "chunks_list": [],
                "chunks_count": 0,
            }
        },
        modal_processors={"table": good, "image": bad},
    )

    async def fake_mark(doc_id):
        processor._marked = doc_id

    processor._mark_multimodal_processing_complete = fake_mark

    async def fake_merge(**kwargs):
        return None

    async def fake_get_namespace_data(name):
        return {}

    def fake_get_pipeline_status_lock():
        return object()

    async def fake_insert_done():
        return None

    import lightrag.kg.shared_storage as shared_mod
    import lightrag.operate as operate_mod

    monkeypatch.setattr(operate_mod, "merge_nodes_and_edges", fake_merge)
    monkeypatch.setattr(shared_mod, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(
        shared_mod, "get_pipeline_status_lock", fake_get_pipeline_status_lock
    )
    processor.lightrag._insert_done = fake_insert_done

    await processor._process_multimodal_content_individual(
        [
            {"type": "image", "img_path": "/tmp/a.png"},
            {"type": "table", "table_body": "| ok |"},
        ],
        "manual.pdf",
        "doc-1",
    )

    assert len(good.calls) == 1
    assert len(bad.calls) == 1
    assert processor.lightrag.doc_status.records["doc-1"]["chunks_list"] == ["chunk-ok"]
    assert processor._marked == "doc-1"
    assert any(
        "Error processing multimodal content" in e for e in processor.logger.errors
    )
