"""Regression tests for modal entity/chunk writes and generic fallback.

_create_entity_and_chunk is the shared write path for image/table/equation/
generic processors. A missed upsert silently drops KG nodes or retrieval
chunks for every multimodal item.
"""

import pytest
from lightrag.utils import compute_mdhash_id

from raganything.modalprocessors import (
    BaseModalProcessor,
    GenericModalProcessor,
)


class FakeKV:
    def __init__(self):
        self.store = {}

    async def upsert(self, data):
        self.store.update(data)

    async def get_by_id(self, key):
        return self.store.get(key)


class FakeGraph:
    def __init__(self):
        self.nodes = {}

    async def upsert_node(self, name, data):
        self.nodes[name] = data


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class ConcreteProcessor(BaseModalProcessor):
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.text_chunks_db = FakeKV()
        self.chunks_vdb = FakeKV()
        self.entities_vdb = FakeKV()
        self.knowledge_graph_inst = FakeGraph()
        self.extraction_calls = []

    async def process_multimodal_content(self, *args, **kwargs):
        pass

    async def _process_chunk_for_extraction(
        self, chunk_id, modal_entity_name, batch_mode=False
    ):
        self.extraction_calls.append((chunk_id, modal_entity_name, batch_mode))
        return [("nodes", "edges")]


def _entity_info():
    return {
        "entity_name": "Figure 1 (image)",
        "entity_type": "image",
        "summary": "A labeled schematic of the pump.",
    }


@pytest.mark.asyncio
async def test_create_entity_and_chunk_writes_chunk_node_and_entity():
    proc = ConcreteProcessor()
    modal_chunk = "Image Path: /tmp/fig.png\nEnhanced caption: pump schematic"
    entity_info = _entity_info()

    summary, meta, chunk_results = await proc._create_entity_and_chunk(
        modal_chunk,
        entity_info,
        file_path="manual.pdf",
        batch_mode=True,
        doc_id="doc-abc",
        chunk_order_index=3,
    )

    chunk_id = compute_mdhash_id(str(modal_chunk), prefix="chunk-")
    entity_id = compute_mdhash_id(entity_info["entity_name"], prefix="ent-")

    assert summary == entity_info["summary"]
    assert meta["chunk_id"] == chunk_id
    assert meta["entity_name"] == entity_info["entity_name"]
    assert chunk_results == [("nodes", "edges")]

    stored_chunk = proc.text_chunks_db.store[chunk_id]
    assert stored_chunk["full_doc_id"] == "doc-abc"
    assert stored_chunk["file_path"] == "manual.pdf"
    assert stored_chunk["chunk_order_index"] == 3
    assert stored_chunk["content"] == modal_chunk
    assert stored_chunk["tokens"] == len(modal_chunk.split())

    assert proc.chunks_vdb.store[chunk_id]["full_doc_id"] == "doc-abc"
    assert proc.knowledge_graph_inst.nodes[entity_info["entity_name"]]["source_id"] == (
        chunk_id
    )
    assert proc.entities_vdb.store[entity_id]["entity_name"] == entity_info["entity_name"]
    assert proc.entities_vdb.store[entity_id]["source_id"] == chunk_id
    assert proc.extraction_calls == [(chunk_id, entity_info["entity_name"], True)]


@pytest.mark.asyncio
async def test_create_entity_and_chunk_falls_back_to_chunk_id_as_doc_id():
    proc = ConcreteProcessor()
    modal_chunk = "generic chunk body"
    entity_info = _entity_info()

    _, meta, _ = await proc._create_entity_and_chunk(
        modal_chunk,
        entity_info,
        file_path="orphan.md",
    )

    chunk_id = compute_mdhash_id(str(modal_chunk), prefix="chunk-")
    assert meta["chunk_id"] == chunk_id
    assert proc.text_chunks_db.store[chunk_id]["full_doc_id"] == chunk_id
    assert proc.extraction_calls[0][2] is False


@pytest.mark.asyncio
async def test_generic_description_fallback_on_caption_error():
    proc = GenericModalProcessor.__new__(GenericModalProcessor)

    async def boom(*args, **kwargs):
        raise RuntimeError("llm down")

    proc.modal_caption_func = boom
    proc.content_source = None
    proc.content_format = "auto"
    proc.context_extractor = None

    caption, entity = await GenericModalProcessor.generate_description_only(
        proc,
        modal_content={"type": "chart", "content": "series A"},
        content_type="chart",
        entity_name="Chart A",
    )

    assert caption == str({"type": "chart", "content": "series A"})
    assert entity["entity_name"] == "Chart A"
    assert entity["entity_type"] == "chart"
    assert "series A" in entity["summary"]
