"""Runtime contracts for multimodal full_entities storage.

Existing source-shape checks do not execute the helper. A swallowed storage
error drops graph entities while ingest still reports success; a merge that
replaces (instead of extending) entity_names loses text-pipeline metadata.
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


class MemoryFullEntities:
    def __init__(self, records=None, get_error=None, upsert_error=None):
        self.records = records or {}
        self.get_error = get_error
        self.upsert_error = upsert_error
        self.index_done_calls = 0

    async def get_by_id(self, key):
        if self.get_error is not None:
            raise self.get_error
        return self.records.get(key)

    async def upsert(self, data):
        if self.upsert_error is not None:
            raise self.upsert_error
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


def _processor(storage):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = type("FakeLightRAG", (), {"full_entities": storage})()
    return processor


@pytest.mark.asyncio
async def test_creates_entry_when_document_has_no_full_entities():
    storage = MemoryFullEntities()
    processor = _processor(storage)

    await processor._store_multimodal_entities_to_full_entities(
        {
            "ent-1": {"entity_name": "Figure 1 (image)"},
            "ent-2": {"entity_name": "Table 2 (table)"},
        },
        "doc-1",
    )

    stored = storage.records["doc-1"]
    assert stored["entity_names"] == ["Figure 1 (image)", "Table 2 (table)"]
    assert stored["count"] == 2
    assert "update_time" in stored
    assert storage.index_done_calls == 1


@pytest.mark.asyncio
async def test_merge_preserves_text_pipeline_metadata_and_dedupes():
    storage = MemoryFullEntities(
        {
            "doc-1": {
                "entity_names": ["Alice", "Figure 1 (image)"],
                "count": 2,
                "update_time": 10,
                "source": "text_pipeline",
                "doc_status": "indexed",
            }
        }
    )
    processor = _processor(storage)

    await processor._store_multimodal_entities_to_full_entities(
        {
            "ent-fig": {"entity_name": "Figure 1 (image)"},
            "ent-tbl": {"entity_name": "Table 2 (table)"},
        },
        "doc-1",
    )

    stored = storage.records["doc-1"]
    assert stored["entity_names"] == ["Alice", "Figure 1 (image)", "Table 2 (table)"]
    assert stored["count"] == 3
    assert stored["source"] == "text_pipeline"
    assert stored["doc_status"] == "indexed"
    assert stored["update_time"] != 10
    assert storage.index_done_calls == 1


@pytest.mark.asyncio
async def test_storage_get_error_is_reraised():
    processor = _processor(
        MemoryFullEntities(get_error=RuntimeError("full_entities down"))
    )

    with pytest.raises(RuntimeError, match="full_entities down"):
        await processor._store_multimodal_entities_to_full_entities(
            {"ent-1": {"entity_name": "Figure 1 (image)"}},
            "doc-1",
        )


@pytest.mark.asyncio
async def test_storage_upsert_error_is_reraised():
    processor = _processor(
        MemoryFullEntities(upsert_error=RuntimeError("upsert failed"))
    )

    with pytest.raises(RuntimeError, match="upsert failed"):
        await processor._store_multimodal_entities_to_full_entities(
            {"ent-1": {"entity_name": "Figure 1 (image)"}},
            "doc-1",
        )
