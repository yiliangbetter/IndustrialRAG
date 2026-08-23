"""Storage contracts for BaseModalProcessor._create_entity_and_chunk.

Wrong doc_id / chunk_id association here silently breaks retrieval and
belongs_to graph edges for every multimodal item.
"""

import pytest

from raganything.modalprocessors import BaseModalProcessor


class FakeKV:
    def __init__(self):
        self.records = {}

    async def upsert(self, data):
        self.records.update(data)


class FakeGraph:
    def __init__(self):
        self.nodes = {}

    async def upsert_node(self, name, data):
        self.nodes[name] = data


class FakeTokenizer:
    def encode(self, text):
        return list(range(max(len(text) // 4, 1)))


def _processor():
    proc = BaseModalProcessor.__new__(BaseModalProcessor)
    proc.text_chunks_db = FakeKV()
    proc.chunks_vdb = FakeKV()
    proc.entities_vdb = FakeKV()
    proc.knowledge_graph_inst = FakeGraph()
    proc.tokenizer = FakeTokenizer()
    proc.lightrag = object()
    return proc


@pytest.mark.asyncio
async def test_uses_provided_doc_id_and_returns_chunk_id():
    proc = _processor()
    extraction_calls = []

    async def fake_extract(chunk_id, entity_name, batch_mode=False):
        extraction_calls.append((chunk_id, entity_name, batch_mode))
        return [("nodes", {})]

    proc._process_chunk_for_extraction = fake_extract

    entity_info = {
        "entity_name": "PumpCurve (image)",
        "entity_type": "image",
        "summary": "A pump performance curve",
    }
    summary, stored, chunk_results = await proc._create_entity_and_chunk(
        modal_chunk="image chunk body",
        entity_info=entity_info,
        file_path="manual.pdf",
        batch_mode=True,
        doc_id="doc-abc",
        chunk_order_index=7,
    )

    assert summary == "A pump performance curve"
    assert stored["chunk_id"].startswith("chunk-")
    assert stored["entity_name"] == "PumpCurve (image)"
    assert stored["entity_type"] == "image"
    assert chunk_results == [("nodes", {})]

    chunk_id = stored["chunk_id"]
    chunk = proc.text_chunks_db.records[chunk_id]
    assert chunk["full_doc_id"] == "doc-abc"
    assert chunk["file_path"] == "manual.pdf"
    assert chunk["chunk_order_index"] == 7
    assert chunk["content"] == "image chunk body"
    assert chunk["tokens"] == len(FakeTokenizer().encode("image chunk body"))

    vdb_chunk = proc.chunks_vdb.records[chunk_id]
    assert vdb_chunk["full_doc_id"] == "doc-abc"
    assert vdb_chunk["file_path"] == "manual.pdf"

    node = proc.knowledge_graph_inst.nodes["PumpCurve (image)"]
    assert node["entity_type"] == "image"
    assert node["description"] == "A pump performance curve"
    assert node["source_id"] == chunk_id
    assert node["file_path"] == "manual.pdf"

    assert len(proc.entities_vdb.records) == 1
    entity_row = next(iter(proc.entities_vdb.records.values()))
    assert entity_row["entity_name"] == "PumpCurve (image)"
    assert entity_row["source_id"] == chunk_id
    assert "PumpCurve (image)" in entity_row["content"]
    assert "A pump performance curve" in entity_row["content"]

    assert extraction_calls == [(chunk_id, "PumpCurve (image)", True)]


@pytest.mark.asyncio
async def test_missing_doc_id_falls_back_to_chunk_id():
    proc = _processor()

    async def fake_extract(chunk_id, entity_name, batch_mode=False):
        return []

    proc._process_chunk_for_extraction = fake_extract

    _, stored, _ = await proc._create_entity_and_chunk(
        modal_chunk="table chunk",
        entity_info={
            "entity_name": "Spec (table)",
            "entity_type": "table",
            "summary": "specs",
        },
        file_path="doc.pdf",
        doc_id=None,
    )

    chunk_id = stored["chunk_id"]
    assert proc.text_chunks_db.records[chunk_id]["full_doc_id"] == chunk_id
    assert proc.chunks_vdb.records[chunk_id]["full_doc_id"] == chunk_id
