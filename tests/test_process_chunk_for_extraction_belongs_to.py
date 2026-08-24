"""Regression tests for modalprocessor belongs_to writes during extraction.

`_process_chunk_for_extraction` is the non-batch path that attaches extracted
entities to the parent figure/table. Missing chunk lookups must fail closed,
self-loops must not be stored, and batch_mode must skip merge.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raganything.modalprocessors import GenericModalProcessor


class FakeKV:
    def __init__(self, records=None):
        self.store = dict(records or {})
        self.upserts = []

    async def get_by_id(self, key):
        return self.store.get(key)

    async def upsert(self, data):
        self.upserts.append(data)
        self.store.update(data)


class FakeGraph:
    def __init__(self):
        self.edges = []

    async def upsert_edge(self, src, tgt, data):
        self.edges.append((src, tgt, data))


def _processor(chunk_id="chunk-1"):
    processor = GenericModalProcessor.__new__(GenericModalProcessor)
    processor.text_chunks_db = FakeKV(
        {
            chunk_id: {
                "content": "Image caption: pump schematic",
                "tokens": 4,
                "chunk_order_index": 0,
                "file_path": "manual.pdf",
                "full_doc_id": "doc-1",
            }
        }
    )
    processor.chunks_vdb = FakeKV()
    processor.knowledge_graph_inst = FakeGraph()
    processor.entities_vdb = FakeKV()
    processor.relationships_vdb = FakeKV()
    processor.hashing_kv = object()
    processor.global_config = {"working_dir": "/tmp"}
    processor.lightrag = MagicMock()
    processor.lightrag.full_entities = object()
    processor.lightrag.full_relations = object()
    processor.lightrag.entity_chunks = object()
    processor.lightrag.relation_chunks = object()
    processor.lightrag._insert_done = AsyncMock()
    return processor


def _patch_extraction(extract_result, merge=None):
    import raganything.modalprocessors as mp

    async def fake_extract(**kwargs):
        return extract_result

    return (
        patch.object(mp, "extract_entities", new=fake_extract),
        patch.object(mp, "get_namespace_data", new=AsyncMock(return_value={})),
        patch.object(mp, "get_pipeline_status_lock", return_value=AsyncMock()),
        patch.object(mp, "merge_nodes_and_edges", new=merge or AsyncMock(name="merge")),
    )


@pytest.mark.asyncio
async def test_missing_chunk_returns_without_extraction():
    processor = _processor()
    processor.text_chunks_db.store.clear()
    extract = AsyncMock(return_value=[])

    import raganything.modalprocessors as mp

    with patch.object(mp, "extract_entities", new=extract):
        result = await processor._process_chunk_for_extraction(
            "chunk-missing", "Figure 1 (image)", batch_mode=True
        )

    assert result is None
    extract.assert_not_awaited()
    assert processor.knowledge_graph_inst.edges == []


@pytest.mark.asyncio
async def test_belongs_to_writes_graph_and_vdb_skipping_self():
    processor = _processor()
    modal_name = "Figure 1 (image)"
    extract_result = [
        (
            {
                "Pump": [{"source_id": "chunk-1"}],
                modal_name: [{"source_id": "chunk-1"}],
            },
            {("Pump", "other"): [{"description": "pre-existing"}]},
        )
    ]

    patches = _patch_extraction(extract_result)
    with patches[0], patches[1], patches[2], patches[3]:
        results = await processor._process_chunk_for_extraction(
            "chunk-1", modal_name, batch_mode=True
        )

    assert processor.knowledge_graph_inst.edges == [
        (
            "Pump",
            modal_name,
            {
                "description": f"Entity Pump belongs to {modal_name}",
                "keywords": "belongs_to,part_of,contained_in",
                "source_id": "chunk-1",
                "weight": 10.0,
                "file_path": "manual.pdf",
            },
        )
    ]
    assert len(processor.relationships_vdb.upserts) == 1
    rel_payload = next(iter(processor.relationships_vdb.upserts[0].values()))
    assert rel_payload["src_id"] == "Pump"
    assert rel_payload["tgt_id"] == modal_name
    assert rel_payload["source_id"] == "chunk-1"

    _, maybe_edges = results[0]
    assert (modal_name, modal_name) not in maybe_edges
    assert maybe_edges[("Pump", modal_name)][0]["keywords"].startswith("belongs_to")
    # Pre-existing unrelated edges are preserved.
    assert maybe_edges[("Pump", "other")][0]["description"] == "pre-existing"
    processor.lightrag._insert_done.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_modal_parent_entity_adds_no_belongs_to_edges():
    processor = _processor()
    modal_name = "Figure 1 (image)"
    extract_result = [({modal_name: [{}]}, {})]
    patches = _patch_extraction(extract_result)
    with patches[0], patches[1], patches[2], patches[3]:
        results = await processor._process_chunk_for_extraction(
            "chunk-1", modal_name, batch_mode=True
        )

    assert processor.knowledge_graph_inst.edges == []
    assert processor.relationships_vdb.upserts == []
    assert results[0][1] == {}


@pytest.mark.asyncio
async def test_non_batch_mode_merges_then_insert_done():
    processor = _processor()
    merge = AsyncMock()
    extract_result = [({"Pump": [{}]}, {})]
    patches = _patch_extraction(extract_result, merge=merge)
    with patches[0], patches[1], patches[2], patches[3]:
        await processor._process_chunk_for_extraction(
            "chunk-1", "Figure 1 (image)", batch_mode=False
        )

    merge.assert_awaited_once()
    processor.lightrag._insert_done.assert_awaited_once()
    kwargs = merge.await_args.kwargs
    assert kwargs["doc_id"] == "doc-1"
    assert kwargs["file_path"] == "manual.pdf"
    # belongs_to still recorded before merge
    assert processor.knowledge_graph_inst.edges[0][0] == "Pump"
