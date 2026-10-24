"""Regression tests for the individual multimodal fallback's chunk merge.

Batch extraction can fail and drop into ``_process_multimodal_content_individual``.
That path is what still attaches figure and table chunks to an existing text
document. Distinct from the type-aware status helper and from the outer
skip/fallback gates.
"""

import pytest

from raganything.base import DocStatus
from raganything.processor import ProcessorMixin


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))

    def debug(self, *args, **kwargs):
        pass


class FakeDocStatusStorage:
    def __init__(self, records=None):
        self.records = dict(records or {})
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


class RaisingUpsertStorage(FakeDocStatusStorage):
    async def upsert(self, data):
        raise RuntimeError("status disk full")


class FakeModalProcessor:
    def __init__(self, *, chunk_id="chunk-mm", fail=False, include_chunk_id=True):
        self.chunk_id = chunk_id
        self.fail = fail
        self.include_chunk_id = include_chunk_id
        self.calls = []

    async def process_multimodal_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("processor boom")
        entity_info = {"entity_name": "Widget"}
        if self.include_chunk_id:
            entity_info["chunk_id"] = self.chunk_id
        return ("caption", entity_info, [({"entity": self.chunk_id}, {})])


def _make_processor(*, records=None, modal_processors=None, storage=None):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type("Config", (), {"use_full_path": False})()
    processor.modal_processors = modal_processors or {}
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "doc_status": storage
            if storage is not None
            else FakeDocStatusStorage(records),
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
    processor._marked = []

    async def fake_mark(doc_id):
        processor._marked.append(doc_id)

    processor._mark_multimodal_processing_complete = fake_mark
    return processor


def _patch_merge(monkeypatch, merge_calls):
    async def fake_merge(**kwargs):
        merge_calls.append(kwargs)

    async def fake_get_namespace_data(name):
        return {"name": name}

    def fake_get_pipeline_status_lock():
        return object()

    import lightrag.kg.shared_storage as shared_mod
    import lightrag.operate as operate_mod

    monkeypatch.setattr(operate_mod, "merge_nodes_and_edges", fake_merge)
    monkeypatch.setattr(shared_mod, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(
        shared_mod, "get_pipeline_status_lock", fake_get_pipeline_status_lock
    )


@pytest.mark.asyncio
async def test_individual_appends_chunk_ids_without_dropping_text(monkeypatch):
    table = FakeModalProcessor(chunk_id="chunk-table")
    equation = FakeModalProcessor(chunk_id="chunk-equation", include_chunk_id=False)
    processor = _make_processor(
        records={
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "chunks_list": ["chunk-text"],
                "chunks_count": 1,
                "file_path": "manual.pdf",
                "error_msg": "",
                "content_summary": "keep me",
            }
        },
        modal_processors={"table": table, "equation": equation},
    )
    merge_calls = []
    _patch_merge(monkeypatch, merge_calls)

    async def fake_insert_done():
        processor._insert_done = True

    processor.lightrag._insert_done = fake_insert_done

    await processor._process_multimodal_content_individual(
        [
            {"type": "table", "table_body": "| a |", "page_idx": 2},
            {"type": "equation", "latex": "E=mc^2"},
        ],
        "/docs/manual.pdf",
        "doc-1",
    )

    assert table.calls[0]["file_path"] == "manual.pdf"
    assert table.calls[0]["chunk_order_index"] == 1
    assert table.calls[0]["batch_mode"] is True
    assert table.calls[0]["doc_id"] == "doc-1"
    assert table.calls[0]["item_info"] == {"page_idx": 2, "index": 0, "type": "table"}
    assert equation.calls[0]["chunk_order_index"] == 2
    assert equation.calls[0]["item_info"]["page_idx"] == 0

    stored = processor.lightrag.doc_status.records["doc-1"]
    assert stored["chunks_list"] == ["chunk-text", "chunk-table"]
    assert stored["chunks_count"] == 2
    assert stored["status"] == DocStatus.PROCESSED
    assert stored["file_path"] == "manual.pdf"
    assert stored["error_msg"] == ""
    assert stored["content_summary"] == "keep me"
    assert isinstance(stored["updated_at"], str) and stored["updated_at"]
    assert processor.lightrag.doc_status.index_done_calls == 1

    assert len(merge_calls) == 1
    assert merge_calls[0]["doc_id"] == "doc-1"
    assert merge_calls[0]["file_path"] == "manual.pdf"
    assert merge_calls[0]["chunk_results"] == [
        ({"entity": "chunk-table"}, {}),
        ({"entity": "chunk-equation"}, {}),
    ]
    assert processor._insert_done is True
    assert processor._marked == ["doc-1"]


@pytest.mark.asyncio
async def test_individual_keeps_sibling_when_one_item_fails(monkeypatch):
    image = FakeModalProcessor(fail=True)
    table = FakeModalProcessor(chunk_id="chunk-table")
    processor = _make_processor(
        records={
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "chunks_list": ["chunk-a", "chunk-b"],
                "chunks_count": 2,
                "file_path": "manual.pdf",
            }
        },
        modal_processors={"image": image, "table": table},
    )
    merge_calls = []
    _patch_merge(monkeypatch, merge_calls)
    processor.lightrag._insert_done = _async_noop()

    await processor._process_multimodal_content_individual(
        [
            {"type": "image", "img_path": "/docs/fig.png"},
            {"type": "chart", "content": "unused"},
            {"type": "table", "table_body": "| ok |", "page_idx": 4},
        ],
        "/docs/manual.pdf",
        "doc-1",
    )

    assert len(image.calls) == 1
    assert image.calls[0]["chunk_order_index"] == 2
    assert len(table.calls) == 1
    assert table.calls[0]["chunk_order_index"] == 4
    stored = processor.lightrag.doc_status.records["doc-1"]
    assert stored["chunks_list"] == ["chunk-a", "chunk-b", "chunk-table"]
    assert stored["chunks_count"] == 3
    assert stored["file_path"] == "manual.pdf"
    assert merge_calls[0]["chunk_results"] == [({"entity": "chunk-table"}, {})]
    assert processor._marked == ["doc-1"]
    assert any(
        "Error processing multimodal content" in msg for msg in processor.logger.errors
    )
    assert any("chart" in msg for msg in processor.logger.warnings)


@pytest.mark.asyncio
async def test_individual_missing_status_does_not_create_a_row(monkeypatch):
    table = FakeModalProcessor(chunk_id="chunk-table")
    storage = FakeDocStatusStorage()
    processor = _make_processor(
        storage=storage,
        modal_processors={"table": table},
    )
    merge_calls = []
    _patch_merge(monkeypatch, merge_calls)
    processor.lightrag._insert_done = _async_noop()

    await processor._process_multimodal_content_individual(
        [{"type": "table", "table_body": "| a |"}],
        "manual.pdf",
        "doc-missing",
    )

    assert table.calls[0]["chunk_order_index"] == 0
    assert storage.upserts == []
    assert storage.records == {}
    assert storage.index_done_calls == 0
    assert len(merge_calls) == 1
    assert merge_calls[0]["doc_id"] == "doc-missing"
    assert processor._marked == ["doc-missing"]


@pytest.mark.asyncio
async def test_individual_status_write_error_still_merges(monkeypatch):
    table = FakeModalProcessor(chunk_id="chunk-table")
    storage = RaisingUpsertStorage(
        {
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "chunks_list": ["chunk-text"],
                "chunks_count": 1,
            }
        }
    )
    processor = _make_processor(
        storage=storage,
        modal_processors={"table": table},
    )
    merge_calls = []
    _patch_merge(monkeypatch, merge_calls)
    processor.lightrag._insert_done = _async_noop()

    await processor._process_multimodal_content_individual(
        [{"type": "table", "table_body": "| a |"}],
        "manual.pdf",
        "doc-1",
    )

    assert storage.records["doc-1"]["chunks_list"] == ["chunk-text"]
    assert storage.records["doc-1"]["chunks_count"] == 1
    assert storage.index_done_calls == 0
    assert len(merge_calls) == 1
    assert processor._marked == ["doc-1"]
    assert any(
        "Error updating doc_status with multimodal chunks" in msg
        for msg in processor.logger.warnings
    )


@pytest.mark.asyncio
async def test_individual_all_item_failures_skip_merge_and_mark_complete(monkeypatch):
    image = FakeModalProcessor(fail=True)
    processor = _make_processor(
        records={
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "chunks_list": ["chunk-text"],
                "chunks_count": 1,
            }
        },
        modal_processors={"image": image},
    )
    merge_calls = []
    _patch_merge(monkeypatch, merge_calls)
    processor.lightrag._insert_done = _async_noop()

    await processor._process_multimodal_content_individual(
        [{"type": "image", "img_path": "/docs/fig.png"}],
        "manual.pdf",
        "doc-1",
    )

    assert merge_calls == []
    assert processor.lightrag.doc_status.upserts == []
    assert processor.lightrag.doc_status.records["doc-1"]["chunks_list"] == [
        "chunk-text"
    ]
    assert processor._marked == ["doc-1"]


def _async_noop():
    async def _noop():
        return None

    return _noop
