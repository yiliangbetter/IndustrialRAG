"""Concurrent folder ingest must not cross-wire multimodal content_source."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from raganything.modalprocessors import BaseModalProcessor, ContextExtractor


def _make_processor() -> BaseModalProcessor:
    lightrag = MagicMock()
    lightrag.text_chunks = MagicMock()
    lightrag.chunks_vdb = MagicMock()
    lightrag.entities_vdb = MagicMock()
    lightrag.relationships_vdb = MagicMock()
    lightrag.chunk_entity_relation_graph = MagicMock()
    lightrag.embedding_func = MagicMock()
    lightrag.llm_model_func = MagicMock()
    lightrag.llm_response_cache = MagicMock()
    lightrag.tokenizer = MagicMock()
    with patch("raganything.modalprocessors.asdict", return_value={}):
        return BaseModalProcessor(
            lightrag=lightrag,
            modal_caption_func=MagicMock(),
            context_extractor=ContextExtractor(tokenizer=None),
        )


@pytest.mark.asyncio
async def test_concurrent_set_content_source_does_not_cross_wire():
    """README process_folder_complete(max_workers=4) shares modal processors.

    Doc A must keep reading CONTENT_A even if Doc B overwrites the shared
    processor instance attribute before A's multimodal context lookup.
    """
    processor = _make_processor()

    async def process_doc(label: str, delay_before_read: float) -> tuple[str, str]:
        content = [
            {"type": "text", "text": label, "page_idx": 0},
            {"type": "image", "img_path": f"{label}.png", "page_idx": 0},
        ]
        processor.set_content_source(content, "minerU")
        await asyncio.sleep(delay_before_read)
        source, _fmt = processor._resolve_content_source()
        ctx = processor._get_context_for_item({"page_idx": 0, "index": 1})
        return source[0]["text"], ctx

    task_a = asyncio.create_task(process_doc("CONTENT_A", delay_before_read=0.05))
    await asyncio.sleep(0.01)
    task_b = asyncio.create_task(process_doc("CONTENT_B", delay_before_read=0.0))
    (source_a, ctx_a), (source_b, ctx_b) = await asyncio.gather(task_a, task_b)

    assert source_a == "CONTENT_A"
    assert source_b == "CONTENT_B"
    assert "CONTENT_A" in ctx_a
    assert "CONTENT_B" in ctx_b
    # Instance attribute reflects the last writer; task-local state must not.
    assert processor.content_source[0]["text"] == "CONTENT_B"


@pytest.mark.asyncio
async def test_child_tasks_inherit_task_local_content_source():
    """Multimodal batch create_task children must see the parent document source."""
    processor = _make_processor()
    content_a = [{"type": "text", "text": "PARENT_A", "page_idx": 0}]
    content_b = [{"type": "text", "text": "PARENT_B", "page_idx": 0}]

    async def parent(content, delay_before_spawn: float):
        processor.set_content_source(content, "minerU")
        await asyncio.sleep(delay_before_spawn)

        async def child():
            source, _ = processor._resolve_content_source()
            return source[0]["text"]

        return await asyncio.create_task(child())

    task_a = asyncio.create_task(parent(content_a, delay_before_spawn=0.05))
    await asyncio.sleep(0.01)
    task_b = asyncio.create_task(parent(content_b, delay_before_spawn=0.0))
    got_a, got_b = await asyncio.gather(task_a, task_b)

    assert got_a == "PARENT_A"
    assert got_b == "PARENT_B"


@pytest.mark.asyncio
async def test_without_contextvar_falls_back_to_instance_attribute():
    """Single-threaded callers that only set instance attrs keep working."""
    processor = _make_processor()
    processor.content_source = [{"type": "text", "text": "LEGACY", "page_idx": 0}]
    processor.content_format = "minerU"
    source, fmt = processor._resolve_content_source()
    assert source[0]["text"] == "LEGACY"
    assert fmt == "minerU"
    ctx = processor._get_context_for_item({"page_idx": 0, "index": 0})
    assert "LEGACY" in ctx
