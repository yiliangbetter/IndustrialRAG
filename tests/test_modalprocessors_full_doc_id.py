"""Multimodal VDB must keep document id, not overwrite with chunk id."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from raganything.modalprocessors import GenericModalProcessor, ImageModalProcessor


@pytest.mark.asyncio
async def test_process_chunk_for_extraction_preserves_full_doc_id():
    text_chunks_db = AsyncMock()
    chunks_vdb = AsyncMock()

    processor = GenericModalProcessor.__new__(GenericModalProcessor)
    processor.text_chunks_db = text_chunks_db
    processor.chunks_vdb = chunks_vdb
    processor.knowledge_graph_inst = AsyncMock()
    processor.entities_vdb = AsyncMock()
    processor.relationships_vdb = AsyncMock()
    processor.hashing_kv = AsyncMock()
    processor.global_config = {}
    processor.lightrag = MagicMock()
    processor.lightrag.full_entities = AsyncMock()
    processor.lightrag.full_relations = AsyncMock()
    processor.lightrag.entity_chunks = AsyncMock()
    processor.lightrag.relation_chunks = AsyncMock()

    chunk_id = "chunk-abc"
    doc_id = "doc-real-document"
    text_chunks_db.get_by_id.return_value = {
        "content": "caption text",
        "full_doc_id": doc_id,
        "tokens": 3,
        "chunk_order_index": 0,
        "file_path": "manual.pdf",
    }

    async def _fake_extract(*args, **kwargs):
        return []

    import raganything.modalprocessors as mp

    original_extract = mp.extract_entities
    original_get_ns = mp.get_namespace_data
    original_get_lock = mp.get_pipeline_status_lock
    mp.extract_entities = _fake_extract
    mp.get_namespace_data = AsyncMock(return_value={})
    mp.get_pipeline_status_lock = MagicMock(return_value=AsyncMock())

    try:
        await processor._process_chunk_for_extraction(
            chunk_id, "ImageEntity", batch_mode=True
        )
    finally:
        mp.extract_entities = original_extract
        mp.get_namespace_data = original_get_ns
        mp.get_pipeline_status_lock = original_get_lock

    chunks_vdb.upsert.assert_awaited()
    upserted = chunks_vdb.upsert.await_args.args[0]
    assert chunk_id in upserted
    assert upserted[chunk_id]["full_doc_id"] == doc_id
    assert upserted[chunk_id]["full_doc_id"] != chunk_id


@pytest.mark.asyncio
async def test_process_multimodal_content_reraises_instead_of_two_tuple():
    img = ImageModalProcessor.__new__(ImageModalProcessor)
    img.generate_description_only = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("vision failed")
    )
    # Previously the except block returned a 2-tuple and callers unpacking
    # three values crashed with ValueError, which was then swallowed.
    with pytest.raises(RuntimeError, match="vision failed"):
        await img.process_multimodal_content(
            {"img_path": "/tmp/x.png"},
            content_type="image",
            file_path="manual.pdf",
            doc_id="doc-1",
        )
