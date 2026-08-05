"""Regression tests for BaseModalProcessor._create_entity_and_chunk.

Batch multimodal storage paths are covered by open PRs (#80/#83/#85). The
individual (non-batch) write path still owns chunk ID generation, dual
text_chunks/chunks_vdb upserts, KG node + entities_vdb writes, and
doc_id/file_path/chunk_order_index contracts. Regressions here corrupt
retrieval and graph provenance for single-item multimodal ingest.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lightrag")

from lightrag.utils import compute_mdhash_id

from raganything.modalprocessors import BaseModalProcessor


class FakeStorage:
    def __init__(self):
        self.upserts = []
        self.nodes = []
        self.edges = []

    async def upsert(self, data):
        self.upserts.append(data)

    async def upsert_node(self, entity_name, node_data):
        self.nodes.append((entity_name, node_data))

    async def upsert_edge(self, src, tgt, data):
        self.edges.append((src, tgt, data))


class FakeTokenizer:
    def encode(self, text):
        # Token count used in stored chunk metadata.
        return list(range(max(1, len(text) // 4)))


def _make_processor():
    proc = object.__new__(BaseModalProcessor)
    proc.text_chunks_db = FakeStorage()
    proc.chunks_vdb = FakeStorage()
    proc.entities_vdb = FakeStorage()
    proc.relationships_vdb = FakeStorage()
    proc.knowledge_graph_inst = FakeStorage()
    proc.tokenizer = FakeTokenizer()
    proc.global_config = {}
    proc.hashing_kv = None
    return proc


@pytest.mark.asyncio
async def test_create_entity_and_chunk_writes_storages_with_doc_metadata():
    proc = _make_processor()
    extraction_calls = []

    async def fake_extract(chunk_id, modal_entity_name, batch_mode=False):
        extraction_calls.append((chunk_id, modal_entity_name, batch_mode))
        return [("nodes", "edges")]

    proc._process_chunk_for_extraction = fake_extract

    modal_chunk = "Image entity description for pump diagram"
    entity_info = {
        "entity_name": "PumpDiagram",
        "entity_type": "image",
        "summary": "Centrifugal pump overview",
    }

    summary, entity_meta, chunk_results = await proc._create_entity_and_chunk(
        modal_chunk,
        entity_info,
        file_path="/docs/manual.pdf",
        batch_mode=False,
        doc_id="doc-manual-1",
        chunk_order_index=3,
    )

    expected_chunk_id = compute_mdhash_id(str(modal_chunk), prefix="chunk-")

    assert summary == "Centrifugal pump overview"
    assert entity_meta == {
        "entity_name": "PumpDiagram",
        "entity_type": "image",
        "description": "Centrifugal pump overview",
        "chunk_id": expected_chunk_id,
    }
    assert chunk_results == [("nodes", "edges")]
    assert extraction_calls == [(expected_chunk_id, "PumpDiagram", False)]

    assert len(proc.text_chunks_db.upserts) == 1
    text_chunk = proc.text_chunks_db.upserts[0][expected_chunk_id]
    assert text_chunk["content"] == modal_chunk
    assert text_chunk["full_doc_id"] == "doc-manual-1"
    assert text_chunk["file_path"] == "/docs/manual.pdf"
    assert text_chunk["chunk_order_index"] == 3
    assert text_chunk["tokens"] == len(FakeTokenizer().encode(modal_chunk))

    assert len(proc.chunks_vdb.upserts) == 1
    vdb_chunk = proc.chunks_vdb.upserts[0][expected_chunk_id]
    assert vdb_chunk["full_doc_id"] == "doc-manual-1"
    assert vdb_chunk["file_path"] == "/docs/manual.pdf"
    assert vdb_chunk["chunk_order_index"] == 3

    assert proc.knowledge_graph_inst.nodes == [
        (
            "PumpDiagram",
            {
                "entity_id": "PumpDiagram",
                "entity_type": "image",
                "description": "Centrifugal pump overview",
                "source_id": expected_chunk_id,
                "file_path": "/docs/manual.pdf",
                "created_at": proc.knowledge_graph_inst.nodes[0][1]["created_at"],
            },
        )
    ]

    assert len(proc.entities_vdb.upserts) == 1
    entity_key = compute_mdhash_id("PumpDiagram", prefix="ent-")
    entity_row = proc.entities_vdb.upserts[0][entity_key]
    assert entity_row["entity_name"] == "PumpDiagram"
    assert entity_row["source_id"] == expected_chunk_id
    assert entity_row["file_path"] == "/docs/manual.pdf"
    assert "Centrifugal pump overview" in entity_row["content"]


@pytest.mark.asyncio
async def test_create_entity_and_chunk_defaults_doc_id_to_chunk_id():
    proc = _make_processor()

    async def fake_extract(chunk_id, modal_entity_name, batch_mode=False):
        return []

    proc._process_chunk_for_extraction = fake_extract

    modal_chunk = "Equation chunk"
    entity_info = {
        "entity_name": "Eq1",
        "entity_type": "equation",
        "summary": "Energy balance",
    }

    _, entity_meta, _ = await proc._create_entity_and_chunk(
        modal_chunk,
        entity_info,
        file_path="notes.md",
        batch_mode=True,
    )

    expected_chunk_id = compute_mdhash_id(str(modal_chunk), prefix="chunk-")
    assert entity_meta["chunk_id"] == expected_chunk_id
    text_chunk = proc.text_chunks_db.upserts[0][expected_chunk_id]
    assert text_chunk["full_doc_id"] == expected_chunk_id
    assert text_chunk["chunk_order_index"] == 0
