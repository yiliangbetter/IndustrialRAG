"""Regression tests for multimodal main-entity and chunk storage.

Main entities must land in the graph, entities_vdb, and full_entities with
the same chunk_id as the templated retrieval chunk. Missing any of those
stores makes figures/tables unfindable after ingest.
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


class FakeKV:
    def __init__(self):
        self.store = {}
        self.index_done_calls = 0

    async def upsert(self, data):
        self.store.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1

    async def get_by_id(self, key):
        return self.store.get(key)


class FakeGraph:
    def __init__(self):
        self.nodes = {}

    async def upsert_node(self, name, data):
        self.nodes[name] = data


def _processor(use_full_path=False):
    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.logger = FakeLogger()
    dummy.config = type("Config", (), {"use_full_path": use_full_path})()
    dummy.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "tokenizer": type(
                "Tok", (), {"encode": staticmethod(lambda text: text.split())}
            )(),
            "text_chunks": FakeKV(),
            "chunks_vdb": FakeKV(),
            "entities_vdb": FakeKV(),
            "full_entities": FakeKV(),
            "chunk_entity_relation_graph": FakeGraph(),
        },
    )()
    dummy.full_entities_calls = []

    async def tracking_full_entities(entities_to_store, doc_id):
        dummy.full_entities_calls.append((entities_to_store, doc_id))

    dummy._store_multimodal_entities_to_full_entities = tracking_full_entities
    return dummy


def _image_data():
    return {
        "description": "Pump housing photo",
        "content_type": "image",
        "original_item": {
            "img_path": "/abs/fig1.png",
            "image_caption": ["Housing"],
            "image_footnote": [],
        },
        "entity_info": {
            "entity_name": "Housing (image)",
            "entity_type": "image",
            "summary": "Pump housing photo",
        },
        "item_info": {"page_idx": 2},
        "chunk_order_index": 4,
    }


@pytest.mark.asyncio
async def test_store_chunks_writes_text_and_vector_stores():
    dummy = _processor()
    chunks = {
        "chunk-1": {
            "content": "hello",
            "tokens": 1,
            "full_doc_id": "doc-1",
            "chunk_order_index": 0,
            "file_path": "a.pdf",
        }
    }

    await dummy._store_chunks_to_lightrag_storage_type_aware(chunks)

    assert dummy.lightrag.text_chunks.store == chunks
    assert dummy.lightrag.chunks_vdb.store == chunks


@pytest.mark.asyncio
async def test_store_chunks_raises_on_upsert_failure():
    dummy = _processor()

    async def boom(data):
        raise RuntimeError("vdb down")

    dummy.lightrag.chunks_vdb.upsert = boom

    with pytest.raises(RuntimeError, match="vdb down"):
        await dummy._store_chunks_to_lightrag_storage_type_aware({"chunk-1": {}})


@pytest.mark.asyncio
async def test_store_main_entities_empty_list_is_noop():
    dummy = _processor()
    await dummy._store_multimodal_main_entities([], {}, "doc.pdf", "doc-1")
    assert dummy.lightrag.chunk_entity_relation_graph.nodes == {}
    assert dummy.lightrag.entities_vdb.store == {}
    assert dummy.full_entities_calls == []


@pytest.mark.asyncio
async def test_store_main_entities_indexes_graph_vdb_and_full_entities():
    dummy = _processor(use_full_path=False)
    data = _image_data()
    formatted = dummy._apply_chunk_template(
        data["content_type"], data["original_item"], data["description"]
    )
    chunk_id = compute_mdhash_id(formatted, prefix="chunk-")
    entity_id = compute_mdhash_id(data["entity_info"]["entity_name"], prefix="ent-")

    await dummy._store_multimodal_main_entities(
        [data], {chunk_id: {"content": formatted}}, "/docs/spec.pdf", "doc-1"
    )

    node = dummy.lightrag.chunk_entity_relation_graph.nodes["Housing (image)"]
    assert node["entity_type"] == "image"
    assert node["description"] == "Pump housing photo"
    assert node["source_id"] == chunk_id
    assert node["file_path"] == "spec.pdf"

    stored = dummy.lightrag.entities_vdb.store[entity_id]
    assert stored["entity_name"] == "Housing (image)"
    assert stored["source_id"] == chunk_id
    assert stored["file_path"] == "spec.pdf"
    assert dummy.lightrag.entities_vdb.index_done_calls == 1
    assert dummy.full_entities_calls[0][1] == "doc-1"
    assert entity_id in dummy.full_entities_calls[0][0]


@pytest.mark.asyncio
async def test_store_main_entities_skips_full_entities_without_doc_id():
    dummy = _processor()
    data = _image_data()

    await dummy._store_multimodal_main_entities(
        [data], {}, "spec.pdf", doc_id=None
    )

    assert dummy.lightrag.entities_vdb.store
    assert dummy.full_entities_calls == []


@pytest.mark.asyncio
async def test_store_main_entities_uses_full_path_when_configured():
    dummy = _processor(use_full_path=True)
    data = _image_data()

    await dummy._store_multimodal_main_entities(
        [data], {}, "/docs/spec.pdf", "doc-1"
    )

    node = dummy.lightrag.chunk_entity_relation_graph.nodes["Housing (image)"]
    assert node["file_path"] == "/docs/spec.pdf"
