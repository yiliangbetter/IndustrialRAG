"""Regression tests for multimodal main-entity KG / VDB / full_entities storage."""

import pytest
from lightrag.utils import compute_mdhash_id

from raganything.processor import ProcessorMixin


class FakeLogger:
    def __init__(self):
        self.errors = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))

    def debug(self, *args, **kwargs):
        pass


class FakeGraph:
    def __init__(self, fail=False):
        self.fail = fail
        self.nodes = []

    async def upsert_node(self, entity_name, node_data):
        if self.fail:
            raise RuntimeError("graph failed")
        self.nodes.append((entity_name, node_data))


class FakeEntitiesVDB:
    def __init__(self):
        self.upserts = []
        self.index_done_calls = 0

    async def upsert(self, data):
        self.upserts.append(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeFullEntities:
    def __init__(self, records=None, fail=False):
        self.records = records or {}
        self.fail = fail
        self.upserts = []
        self.index_done_calls = 0

    async def get_by_id(self, key):
        if self.fail:
            raise RuntimeError("full_entities failed")
        return self.records.get(key)

    async def upsert(self, data):
        self.upserts.append(data)
        for doc_id, payload in data.items():
            self.records[doc_id] = payload

    async def index_done_callback(self):
        self.index_done_calls += 1


def _make_processor(
    *, graph=None, entities_vdb=None, full_entities=None, use_full_path=False
):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type("Config", (), {"use_full_path": use_full_path})()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "chunk_entity_relation_graph": graph if graph is not None else FakeGraph(),
            "entities_vdb": entities_vdb
            if entities_vdb is not None
            else FakeEntitiesVDB(),
            "full_entities": full_entities
            if full_entities is not None
            else FakeFullEntities(),
        },
    )()
    return processor


def _table_data(entity_name="Spec Table (table)"):
    return {
        "entity_info": {
            "entity_name": entity_name,
            "entity_type": "table",
            "summary": "Specification table summary",
        },
        "description": "Enhanced table description",
        "content_type": "table",
        "original_item": {
            "type": "table",
            "table_body": "| a | b |",
            "table_caption": ["Spec"],
            "img_path": "/tmp/table.png",
        },
    }


@pytest.mark.asyncio
async def test_store_multimodal_main_entities_noop_when_empty():
    graph = FakeGraph()
    entities_vdb = FakeEntitiesVDB()
    processor = _make_processor(graph=graph, entities_vdb=entities_vdb)

    await processor._store_multimodal_main_entities([], {}, "manual.pdf", "doc-1")

    assert graph.nodes == []
    assert entities_vdb.upserts == []


@pytest.mark.asyncio
async def test_store_multimodal_main_entities_writes_graph_vdb_and_full_entities():
    graph = FakeGraph()
    entities_vdb = FakeEntitiesVDB()
    full_entities = FakeFullEntities()
    processor = _make_processor(
        graph=graph, entities_vdb=entities_vdb, full_entities=full_entities
    )
    data = _table_data()
    entity_name = data["entity_info"]["entity_name"]

    await processor._store_multimodal_main_entities(
        [data], {}, "/docs/manual.pdf", "doc-1"
    )

    expected_entity_id = compute_mdhash_id(entity_name, prefix="ent-")
    formatted = processor._apply_chunk_template(
        "table", data["original_item"], data["description"]
    )
    expected_chunk_id = compute_mdhash_id(formatted, prefix="chunk-")

    assert len(graph.nodes) == 1
    assert graph.nodes[0][0] == entity_name
    node = graph.nodes[0][1]
    assert node["entity_id"] == entity_name
    assert node["entity_type"] == "table"
    assert node["description"] == "Specification table summary"
    assert node["source_id"] == expected_chunk_id
    assert node["file_path"] == "manual.pdf"
    assert "created_at" in node

    assert len(entities_vdb.upserts) == 1
    stored = entities_vdb.upserts[0]
    assert expected_entity_id in stored
    assert stored[expected_entity_id]["entity_name"] == entity_name
    assert stored[expected_entity_id]["source_id"] == expected_chunk_id
    assert stored[expected_entity_id]["file_path"] == "manual.pdf"
    assert entities_vdb.index_done_calls == 1

    assert len(full_entities.upserts) == 1
    payload = full_entities.upserts[0]["doc-1"]
    assert payload["entity_names"] == [entity_name]
    assert payload["count"] == 1
    assert full_entities.index_done_calls == 1


@pytest.mark.asyncio
async def test_store_multimodal_main_entities_merges_full_entities_without_dupes():
    full_entities = FakeFullEntities(
        {
            "doc-1": {
                "entity_names": ["Spec Table (table)", "Alice"],
                "count": 2,
                "source": "text_pipeline",
            }
        }
    )
    processor = _make_processor(full_entities=full_entities)

    await processor._store_multimodal_main_entities(
        [_table_data(), _table_data("New Figure (image)")],
        {},
        "manual.pdf",
        "doc-1",
    )

    payload = full_entities.upserts[0]["doc-1"]
    assert payload["entity_names"] == [
        "Spec Table (table)",
        "Alice",
        "New Figure (image)",
    ]
    assert payload["count"] == 3
    assert payload["source"] == "text_pipeline"


@pytest.mark.asyncio
async def test_store_multimodal_main_entities_uses_full_path_when_configured():
    graph = FakeGraph()
    processor = _make_processor(graph=graph, use_full_path=True)

    await processor._store_multimodal_main_entities(
        [_table_data()], {}, "/docs/manual.pdf", "doc-1"
    )

    assert graph.nodes[0][1]["file_path"] == "/docs/manual.pdf"


@pytest.mark.asyncio
async def test_store_multimodal_main_entities_skips_full_entities_without_doc_id():
    full_entities = FakeFullEntities()
    processor = _make_processor(full_entities=full_entities)

    await processor._store_multimodal_main_entities(
        [_table_data()], {}, "manual.pdf", doc_id=None
    )

    assert full_entities.upserts == []


@pytest.mark.asyncio
async def test_store_multimodal_main_entities_reraises_storage_errors():
    processor = _make_processor(graph=FakeGraph(fail=True))

    with pytest.raises(RuntimeError, match="graph failed"):
        await processor._store_multimodal_main_entities(
            [_table_data()], {}, "manual.pdf", "doc-1"
        )

    assert any(
        "Error storing multimodal main entities" in e for e in processor.logger.errors
    )
