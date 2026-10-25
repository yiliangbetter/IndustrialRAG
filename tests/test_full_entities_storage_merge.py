"""Regression tests for the real multimodal full_entities merge.

``tests/test_full_entities_merge.py`` checks a copied helper and a source
snippet. These tests call ``ProcessorMixin`` so a change to the storage
method fails here. Distinct from the disabled-storage skip, which never
writes this index.
"""

import pytest
from lightrag.utils import compute_mdhash_id

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


class FakeFullEntities:
    def __init__(self, records=None, get_error=None, upsert_error=None):
        self.records = dict(records or {})
        self.upserts = []
        self.index_done_calls = 0
        self.get_error = get_error
        self.upsert_error = upsert_error

    async def get_by_id(self, key):
        if self.get_error:
            raise self.get_error
        return self.records.get(key)

    async def upsert(self, data):
        if self.upsert_error:
            raise self.upsert_error
        self.upserts.append(data)
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeGraph:
    def __init__(self, error=None):
        self.nodes = []
        self.error = error

    async def upsert_node(self, entity_name, node_data):
        if self.error:
            raise self.error
        self.nodes.append((entity_name, node_data))


class FakeEntitiesVdb:
    def __init__(self):
        self.upserts = []
        self.index_done_calls = 0

    async def upsert(self, data):
        self.upserts.append(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


def _processor(full_entities, *, graph=None, entities_vdb=None, use_full_path=False):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type("Config", (), {"use_full_path": use_full_path})()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "full_entities": full_entities,
            "chunk_entity_relation_graph": graph or FakeGraph(),
            "entities_vdb": entities_vdb or FakeEntitiesVdb(),
        },
    )()
    return processor


def _entities():
    return {
        "ent-bolt": {"entity_name": "Bolt"},
        "ent-pump": {"entity_name": "Pump"},
        "ent-bolt-again": {"entity_name": "Bolt"},
        "ent-valve": {"entity_name": "Valve"},
    }


@pytest.mark.asyncio
async def test_merge_appends_new_names_without_mutating_or_reordering(monkeypatch):
    monkeypatch.setattr("raganything.processor.time.time", lambda: 1700000000.9)
    original_names = ["Alice", "Bolt"]
    storage = FakeFullEntities(
        {
            "doc-1": {
                "entity_names": original_names,
                "count": 2,
                "update_time": 10,
                "source": "text_pipeline",
                "scheme_name": "line-a",
            }
        }
    )
    processor = _processor(storage)

    await processor._store_multimodal_entities_to_full_entities(_entities(), "doc-1")

    assert original_names == ["Alice", "Bolt"]
    stored = storage.records["doc-1"]
    assert stored["entity_names"] == ["Alice", "Bolt", "Pump", "Valve"]
    assert stored["count"] == 4
    assert stored["update_time"] == 1700000000
    assert stored["source"] == "text_pipeline"
    assert stored["scheme_name"] == "line-a"
    assert storage.upserts == [{"doc-1": stored}]
    assert storage.index_done_calls == 1


@pytest.mark.asyncio
async def test_missing_document_creates_name_list_in_insertion_order(monkeypatch):
    monkeypatch.setattr("raganything.processor.time.time", lambda: 1700000001.2)
    storage = FakeFullEntities()
    processor = _processor(storage)

    await processor._store_multimodal_entities_to_full_entities(
        {
            "ent-bolt": {"entity_name": "Bolt"},
            "ent-pump": {"entity_name": "Pump"},
            "ent-valve": {"entity_name": "Valve"},
        },
        "doc-new",
    )

    assert storage.records["doc-new"] == {
        "entity_names": ["Bolt", "Pump", "Valve"],
        "count": 3,
        "update_time": 1700000001,
    }
    assert storage.index_done_calls == 1


@pytest.mark.asyncio
async def test_missing_entity_names_keeps_other_metadata(monkeypatch):
    monkeypatch.setattr("raganything.processor.time.time", lambda: 1700000002.4)
    storage = FakeFullEntities(
        {"doc-1": {"source": "text_pipeline", "content_summary": "manual"}}
    )
    processor = _processor(storage)

    await processor._store_multimodal_entities_to_full_entities(
        {"ent-pump": {"entity_name": "Pump"}}, "doc-1"
    )

    stored = storage.records["doc-1"]
    assert stored["entity_names"] == ["Pump"]
    assert stored["count"] == 1
    assert stored["source"] == "text_pipeline"
    assert stored["content_summary"] == "manual"
    assert stored["update_time"] == 1700000002


@pytest.mark.asyncio
async def test_full_entities_storage_errors_propagate():
    processor = _processor(
        FakeFullEntities(get_error=RuntimeError("status unreadable"))
    )
    with pytest.raises(RuntimeError, match="status unreadable"):
        await processor._store_multimodal_entities_to_full_entities(
            _entities(), "doc-1"
        )

    storage = FakeFullEntities(
        {"doc-1": {"entity_names": ["Alice"], "source": "text_pipeline"}},
        upsert_error=RuntimeError("disk full"),
    )
    processor = _processor(storage)
    with pytest.raises(RuntimeError, match="disk full"):
        await processor._store_multimodal_entities_to_full_entities(
            _entities(), "doc-1"
        )

    assert storage.index_done_calls == 0
    assert storage.records["doc-1"]["entity_names"] == ["Alice"]
    assert storage.records["doc-1"]["source"] == "text_pipeline"


@pytest.mark.asyncio
async def test_main_entities_fall_back_for_summary_and_type(monkeypatch):
    monkeypatch.setattr("raganything.processor.time.time", lambda: 1700000003.8)
    storage = FakeFullEntities()
    graph = FakeGraph()
    vdb = FakeEntitiesVdb()
    processor = _processor(storage, graph=graph, entities_vdb=vdb, use_full_path=True)
    processor._apply_chunk_template = (
        lambda content_type, original_item, description: f"{content_type}:{description}"
    )
    file_path = "/docs/line-a/manual.pdf"

    await processor._store_multimodal_main_entities(
        [
            {
                "entity_info": {"entity_name": "Bolt"},
                "description": "long bolt caption",
                "content_type": "image",
                "original_item": {"type": "image"},
            },
            {
                "entity_info": {
                    "entity_name": "Torque Table",
                    "entity_type": "spec_table",
                    "summary": "20 Nm",
                },
                "description": "longer table analysis",
                "content_type": "table",
                "original_item": {"type": "table"},
            },
        ],
        lightrag_chunks={},
        file_path=file_path,
        doc_id="doc-1",
    )

    bolt_id = compute_mdhash_id("Bolt", prefix="ent-")
    table_id = compute_mdhash_id("Torque Table", prefix="ent-")
    bolt_source = compute_mdhash_id("image:long bolt caption", prefix="chunk-")
    table_source = compute_mdhash_id("table:longer table analysis", prefix="chunk-")

    assert [name for name, _node in graph.nodes] == ["Bolt", "Torque Table"]
    bolt_node = graph.nodes[0][1]
    table_node = graph.nodes[1][1]
    assert bolt_node["entity_type"] == "image"
    assert bolt_node["description"] == "long bolt caption"
    assert bolt_node["source_id"] == bolt_source
    assert bolt_node["file_path"] == file_path
    assert bolt_node["created_at"] == 1700000003
    assert table_node["entity_type"] == "spec_table"
    assert table_node["description"] == "20 Nm"
    assert table_node["source_id"] == table_source
    assert table_node["file_path"] == file_path

    stored_entities = vdb.upserts[0]
    assert stored_entities[bolt_id]["content"] == "long bolt caption"
    assert stored_entities[bolt_id]["entity_type"] == "image"
    assert stored_entities[bolt_id]["source_id"] == bolt_source
    assert stored_entities[bolt_id]["file_path"] == file_path
    assert stored_entities[table_id]["content"] == "20 Nm"
    assert stored_entities[table_id]["entity_type"] == "spec_table"
    assert vdb.index_done_calls == 1
    assert storage.records["doc-1"]["entity_names"] == ["Bolt", "Torque Table"]
    assert storage.records["doc-1"]["count"] == 2


@pytest.mark.asyncio
async def test_graph_failure_does_not_write_entities_vdb():
    graph = FakeGraph(error=RuntimeError("graph down"))
    vdb = FakeEntitiesVdb()
    storage = FakeFullEntities()
    processor = _processor(storage, graph=graph, entities_vdb=vdb)
    processor._apply_chunk_template = lambda *_args, **_kwargs: "formatted"

    with pytest.raises(RuntimeError, match="graph down"):
        await processor._store_multimodal_main_entities(
            [
                {
                    "entity_info": {
                        "entity_name": "Bolt",
                        "entity_type": "image",
                        "summary": "caption",
                    },
                    "description": "caption",
                    "content_type": "image",
                    "original_item": {"type": "image"},
                }
            ],
            lightrag_chunks={},
            file_path="manual.pdf",
            doc_id="doc-1",
        )

    assert vdb.upserts == []
    assert vdb.index_done_calls == 0
    assert storage.upserts == []
    assert storage.index_done_calls == 0
