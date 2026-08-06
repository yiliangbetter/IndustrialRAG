"""Regression tests for BaseModalProcessor._process_chunk_for_extraction.

Open PR #95 covers _create_entity_and_chunk but mocks this helper. The
individual extraction path still owns chunks_vdb writes, belongs_to edge
creation (including self-skip), relationships_vdb upserts, and the
non-batch merge_nodes_and_edges / _insert_done contract. Regressions here
orphan multimodal entities from the graph or skip merge entirely.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("lightrag")

from lightrag.utils import compute_mdhash_id

from raganything.modalprocessors import BaseModalProcessor


class FakeKV:
    def __init__(self, rows=None):
        self.rows = dict(rows or {})
        self.upserts = []

    async def get_by_id(self, key):
        return self.rows.get(key)

    async def upsert(self, data):
        self.upserts.append(data)
        self.rows.update(data)


class FakeGraph:
    def __init__(self):
        self.edges = []

    async def upsert_edge(self, src, tgt, data):
        self.edges.append((src, tgt, data))


class FakeLightRAG:
    def __init__(self):
        self.full_entities = object()
        self.full_relations = object()
        self.entity_chunks = object()
        self.relation_chunks = object()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


def _make_processor(chunk_id: str, chunk_data: dict):
    proc = object.__new__(BaseModalProcessor)
    proc.text_chunks_db = FakeKV({chunk_id: chunk_data})
    proc.chunks_vdb = FakeKV()
    proc.relationships_vdb = FakeKV()
    proc.knowledge_graph_inst = FakeGraph()
    proc.entities_vdb = FakeKV()
    proc.global_config = {"working_dir": "/tmp"}
    proc.hashing_kv = object()
    proc.lightrag = FakeLightRAG()
    return proc


@pytest.mark.asyncio
async def test_process_chunk_for_extraction_missing_chunk_returns_none(caplog):
    proc = _make_processor("missing", {})
    proc.text_chunks_db = FakeKV()

    with caplog.at_level(logging.ERROR):
        result = await proc._process_chunk_for_extraction("missing", "ModalEntity")

    assert result is None
    assert proc.chunks_vdb.upserts == []
    assert "Chunk missing not found" in caplog.text


@pytest.mark.asyncio
async def test_process_chunk_for_extraction_adds_belongs_to_and_merges(
    monkeypatch,
):
    import raganything.modalprocessors as mp

    chunk_id = "chunk-modal-1"
    chunk_data = {
        "content": "Pump diagram description",
        "tokens": 8,
        "chunk_order_index": 2,
        "file_path": "/docs/manual.pdf",
        "full_doc_id": "doc-manual-1",
    }
    proc = _make_processor(chunk_id, chunk_data)

    extract_calls = []
    merge_calls = []

    async def fake_extract_entities(**kwargs):
        extract_calls.append(kwargs)
        return [
            (
                {"PumpDiagram": [{"entity_name": "PumpDiagram"}], "Bearing": [{}]},
                {},
            )
        ]

    async def fake_merge(**kwargs):
        merge_calls.append(kwargs)

    pipeline_status = {"busy": False}
    pipeline_lock = object()

    async def fake_get_namespace_data(name):
        assert name == "pipeline_status"
        return pipeline_status

    monkeypatch.setattr(mp, "extract_entities", fake_extract_entities)
    monkeypatch.setattr(mp, "merge_nodes_and_edges", fake_merge)
    monkeypatch.setattr(mp, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(mp, "get_pipeline_status_lock", lambda: pipeline_lock)

    result = await proc._process_chunk_for_extraction(
        chunk_id, "PumpDiagram", batch_mode=False
    )

    assert len(proc.chunks_vdb.upserts) == 1
    vdb_row = proc.chunks_vdb.upserts[0][chunk_id]
    assert vdb_row["content"] == chunk_data["content"]
    assert vdb_row["full_doc_id"] == chunk_id  # historical contract in this helper
    assert vdb_row["file_path"] == "/docs/manual.pdf"
    assert vdb_row["chunk_order_index"] == 2

    assert len(extract_calls) == 1
    assert extract_calls[0]["chunks"] == {chunk_id: chunk_data}
    assert extract_calls[0]["global_config"] is proc.global_config
    assert extract_calls[0]["pipeline_status"] is pipeline_status
    assert extract_calls[0]["pipeline_status_lock"] is pipeline_lock
    assert extract_calls[0]["llm_response_cache"] is proc.hashing_kv

    # Self-relationship skipped; only Bearing -> PumpDiagram
    assert len(proc.knowledge_graph_inst.edges) == 1
    src, tgt, relation = proc.knowledge_graph_inst.edges[0]
    assert src == "Bearing"
    assert tgt == "PumpDiagram"
    assert relation["source_id"] == chunk_id
    assert relation["weight"] == 10.0
    assert relation["file_path"] == "/docs/manual.pdf"
    assert "belongs_to" in relation["keywords"]

    relation_id = compute_mdhash_id("Bearing" + "PumpDiagram", prefix="rel-")
    assert len(proc.relationships_vdb.upserts) == 1
    rel_row = proc.relationships_vdb.upserts[0][relation_id]
    assert rel_row["src_id"] == "Bearing"
    assert rel_row["tgt_id"] == "PumpDiagram"
    assert rel_row["source_id"] == chunk_id
    assert rel_row["file_path"] == "/docs/manual.pdf"

    assert len(result) == 1
    maybe_nodes, maybe_edges = result[0]
    assert "Bearing" in maybe_nodes
    assert ("Bearing", "PumpDiagram") in maybe_edges

    assert len(merge_calls) == 1
    merge_kwargs = merge_calls[0]
    assert merge_kwargs["doc_id"] == "doc-manual-1"
    assert merge_kwargs["file_path"] == "/docs/manual.pdf"
    assert merge_kwargs["knowledge_graph_inst"] is proc.knowledge_graph_inst
    assert merge_kwargs["entity_vdb"] is proc.entities_vdb
    assert merge_kwargs["relationships_vdb"] is proc.relationships_vdb
    assert merge_kwargs["full_entities_storage"] is proc.lightrag.full_entities
    assert merge_kwargs["full_relations_storage"] is proc.lightrag.full_relations
    assert merge_kwargs["entity_chunks_storage"] is proc.lightrag.entity_chunks
    assert merge_kwargs["relation_chunks_storage"] is proc.lightrag.relation_chunks
    assert merge_kwargs["current_file_number"] == 1
    assert merge_kwargs["total_files"] == 1
    assert proc.lightrag.insert_done_calls == 1


@pytest.mark.asyncio
async def test_process_chunk_for_extraction_batch_mode_skips_merge(monkeypatch):
    import raganything.modalprocessors as mp

    chunk_id = "chunk-batch-1"
    chunk_data = {
        "content": "Table body",
        "tokens": 4,
        "chunk_order_index": 0,
        "file_path": "manual_creation",
        # missing full_doc_id exercises default file_path + None doc_id merge skip
    }
    proc = _make_processor(chunk_id, chunk_data)

    merge_calls = []

    async def fake_extract_entities(**kwargs):
        return [({"ColA": [{}]}, {})]

    async def fake_merge(**kwargs):
        merge_calls.append(kwargs)

    async def fake_get_namespace_data(name):
        return {"busy": False}

    monkeypatch.setattr(mp, "extract_entities", fake_extract_entities)
    monkeypatch.setattr(mp, "merge_nodes_and_edges", fake_merge)
    monkeypatch.setattr(mp, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(mp, "get_pipeline_status_lock", lambda: object())

    result = await proc._process_chunk_for_extraction(
        chunk_id, "TableEntity", batch_mode=True
    )

    assert len(result) == 1
    _, maybe_edges = result[0]
    assert ("ColA", "TableEntity") in maybe_edges
    assert merge_calls == []
    assert proc.lightrag.insert_done_calls == 0

    assert proc.knowledge_graph_inst.edges[0][2]["file_path"] == "manual_creation"
