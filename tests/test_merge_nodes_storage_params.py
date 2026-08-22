"""Regression tests for merge_nodes_and_edges storage kwargs (issue #241).

Omitting entity_chunks_storage / relation_chunks_storage silently drops chunk
provenance during multimodal graph merges. Assert both the processor batch-merge
path and the modalprocessor non-batch path forward those storages.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raganything.modalprocessors import GenericModalProcessor
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


@pytest.mark.asyncio
async def test_processor_batch_merge_forwards_chunk_storages():
    class DummyProcessor(ProcessorMixin):
        pass

    entity_chunks = object()
    relation_chunks = object()
    full_entities = object()
    full_relations = object()
    llm_cache = object()
    graph = object()
    entities_vdb = object()
    relationships_vdb = object()

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type("Config", (), {"use_full_path": False})()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "chunk_entity_relation_graph": graph,
            "entities_vdb": entities_vdb,
            "relationships_vdb": relationships_vdb,
            "full_entities": full_entities,
            "full_relations": full_relations,
            "llm_response_cache": llm_cache,
            "entity_chunks": entity_chunks,
            "relation_chunks": relation_chunks,
            "_insert_done": AsyncMock(),
        },
    )()

    captured = {}

    async def fake_merge_nodes_and_edges(**kwargs):
        captured.update(kwargs)

    with (
        patch(
            "lightrag.kg.shared_storage.get_namespace_data",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "lightrag.kg.shared_storage.get_pipeline_status_lock",
            return_value=AsyncMock(),
        ),
        patch(
            "lightrag.operate.merge_nodes_and_edges",
            new=fake_merge_nodes_and_edges,
        ),
    ):
        await processor._batch_merge_lightrag_style_type_aware(
            enhanced_chunk_results=[({}, {})],
            file_path="/docs/manual.pdf",
            doc_id="doc-1",
        )

    assert captured["entity_chunks_storage"] is entity_chunks
    assert captured["relation_chunks_storage"] is relation_chunks
    assert captured["full_entities_storage"] is full_entities
    assert captured["full_relations_storage"] is full_relations
    assert captured["doc_id"] == "doc-1"
    assert captured["file_path"] == "manual.pdf"
    processor.lightrag._insert_done.assert_awaited_once()


@pytest.mark.asyncio
async def test_modalprocessor_non_batch_merge_forwards_chunk_storages():
    entity_chunks = object()
    relation_chunks = object()

    processor = GenericModalProcessor.__new__(GenericModalProcessor)
    processor.text_chunks_db = AsyncMock()
    processor.chunks_vdb = AsyncMock()
    processor.knowledge_graph_inst = AsyncMock()
    processor.entities_vdb = AsyncMock()
    processor.relationships_vdb = AsyncMock()
    processor.hashing_kv = AsyncMock()
    processor.global_config = {}
    processor.lightrag = MagicMock()
    processor.lightrag.full_entities = AsyncMock()
    processor.lightrag.full_relations = AsyncMock()
    processor.lightrag.entity_chunks = entity_chunks
    processor.lightrag.relation_chunks = relation_chunks
    processor.lightrag._insert_done = AsyncMock()

    processor.text_chunks_db.get_by_id.return_value = {
        "content": "figure caption",
        "full_doc_id": "doc-modal-1",
        "tokens": 2,
        "chunk_order_index": 0,
        "file_path": "manual.pdf",
    }

    async def fake_extract_entities(**kwargs):
        return [({"Pump": {"entity_name": "Pump"}}, {})]

    captured = {}

    async def fake_merge_nodes_and_edges(**kwargs):
        captured.update(kwargs)

    import raganything.modalprocessors as mp

    with (
        patch.object(mp, "extract_entities", new=fake_extract_entities),
        patch.object(mp, "get_namespace_data", new=AsyncMock(return_value={})),
        patch.object(mp, "get_pipeline_status_lock", return_value=AsyncMock()),
        patch.object(mp, "merge_nodes_and_edges", new=fake_merge_nodes_and_edges),
        patch.object(mp, "compute_mdhash_id", return_value="rel-1"),
    ):
        await processor._process_chunk_for_extraction(
            "chunk-1", "Figure 1 (image)", batch_mode=False
        )

    assert captured["entity_chunks_storage"] is entity_chunks
    assert captured["relation_chunks_storage"] is relation_chunks
    assert captured["doc_id"] == "doc-modal-1"
    assert captured["file_path"] == "manual.pdf"
    processor.lightrag._insert_done.assert_awaited_once()
