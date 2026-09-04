"""Multimodal entity upsert when full_entities storage is disabled.

Graph and entities_vdb still receive the main multimodal entity; the
full_entities document index is skipped when that store is missing.
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


class FakeGraph:
    def __init__(self):
        self.nodes = []

    async def upsert_node(self, entity_name, node_data):
        self.nodes.append((entity_name, node_data))


class FakeEntitiesVdb:
    def __init__(self):
        self.upserts = []
        self.index_done_calls = 0

    async def upsert(self, data):
        self.upserts.append(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


def _multimodal_payload():
    return [
        {
            "entity_info": {
                "entity_name": "Figure 1 (image)",
                "entity_type": "image",
                "summary": "Pump schematic",
            },
            "description": "Pump schematic",
            "content_type": "image",
            "original_item": {"type": "image", "img_path": "/tmp/fig.png"},
        }
    ]


def _make_processor(full_entities):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type("Config", (), {"use_full_path": False})()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "chunk_entity_relation_graph": FakeGraph(),
            "entities_vdb": FakeEntitiesVdb(),
            "full_entities": full_entities,
        },
    )()
    processor._apply_chunk_template = (
        lambda content_type, original_item, description: description
    )
    return processor


@pytest.mark.asyncio
async def test_skips_full_entities_when_storage_is_none():
    processor = _make_processor(full_entities=None)
    full_entity_calls = []

    async def fake_store_full_entities(*args, **kwargs):
        full_entity_calls.append((args, kwargs))

    processor._store_multimodal_entities_to_full_entities = fake_store_full_entities

    await processor._store_multimodal_main_entities(
        _multimodal_payload(),
        lightrag_chunks={},
        file_path="/docs/manual.pdf",
        doc_id="doc-1",
    )

    graph = processor.lightrag.chunk_entity_relation_graph
    vdb = processor.lightrag.entities_vdb
    assert len(graph.nodes) == 1
    assert graph.nodes[0][0] == "Figure 1 (image)"
    assert graph.nodes[0][1]["entity_type"] == "image"
    assert graph.nodes[0][1]["file_path"] == "manual.pdf"
    assert len(vdb.upserts) == 1
    stored = next(iter(vdb.upserts[0].values()))
    assert stored["entity_name"] == "Figure 1 (image)"
    assert stored["file_path"] == "manual.pdf"
    assert vdb.index_done_calls == 1
    assert full_entity_calls == []


@pytest.mark.asyncio
async def test_skips_full_entities_when_doc_id_is_missing():
    processor = _make_processor(full_entities=object())
    full_entity_calls = []

    async def fake_store_full_entities(*args, **kwargs):
        full_entity_calls.append((args, kwargs))

    processor._store_multimodal_entities_to_full_entities = fake_store_full_entities

    await processor._store_multimodal_main_entities(
        _multimodal_payload(),
        lightrag_chunks={},
        file_path="manual.pdf",
        doc_id=None,
    )

    assert processor.lightrag.chunk_entity_relation_graph.nodes
    assert processor.lightrag.entities_vdb.upserts
    assert full_entity_calls == []
