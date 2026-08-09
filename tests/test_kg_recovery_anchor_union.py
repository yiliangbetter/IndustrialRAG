"""Regression: multimodal merge must not wipe KG recovery anchors (D16)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raganything.processor import ProcessorMixin
from raganything.utils import snapshot_kg_recovery_anchors, union_kg_recovery_anchors


class FakeKV:
    def __init__(self, initial: Optional[Dict[str, Any]] = None):
        self.data: Dict[str, Any] = dict(initial or {})
        self.flushed = 0

    async def get_by_id(self, key: str):
        return self.data.get(key)

    async def upsert(self, payload: Dict[str, Any]):
        self.data.update(payload)

    async def index_done_callback(self):
        self.flushed += 1


@pytest.mark.asyncio
async def test_union_kg_recovery_anchors_preserves_prior_entities_and_relations():
    full_entities = FakeKV(
        {
            "doc-1": {
                "entity_names": ["Pump", "Valve"],
                "count": 2,
            }
        }
    )
    full_relations = FakeKV(
        {
            "doc-1": {
                "relation_pairs": [["Pump", "Valve"]],
                "count": 1,
            }
        }
    )
    prior_entities = {
        "entity_names": ["Alice", "Table1 (table)"],
        "count": 2,
        "source": "text_pipeline",
    }
    prior_relations = {
        "relation_pairs": [["Alice", "Bob"], ["Sensor", "Table1 (table)"]],
        "count": 2,
        "source": "text_pipeline",
    }

    await union_kg_recovery_anchors(
        full_entities,
        full_relations,
        "doc-1",
        prior_entities,
        prior_relations,
    )

    entities = await full_entities.get_by_id("doc-1")
    relations = await full_relations.get_by_id("doc-1")

    assert entities["entity_names"] == ["Alice", "Pump", "Table1 (table)", "Valve"]
    assert entities["count"] == 4
    assert entities["source"] == "text_pipeline"
    assert relations["relation_pairs"] == [
        ["Alice", "Bob"],
        ["Pump", "Valve"],
        ["Sensor", "Table1 (table)"],
    ]
    assert relations["count"] == 3
    assert full_entities.flushed == 1
    assert full_relations.flushed == 1


@pytest.mark.asyncio
async def test_snapshot_kg_recovery_anchors_is_deepcopy():
    full_entities = FakeKV(
        {"doc-1": {"entity_names": ["Alice"], "count": 1, "nested": {"a": 1}}}
    )
    full_relations = FakeKV({"doc-1": {"relation_pairs": [["A", "B"]], "count": 1}})

    prior_entities, prior_relations = await snapshot_kg_recovery_anchors(
        full_entities, full_relations, "doc-1"
    )
    prior_entities["entity_names"].append("MUTATED")
    prior_entities["nested"]["a"] = 99
    prior_relations["relation_pairs"].append(["X", "Y"])

    assert (await full_entities.get_by_id("doc-1"))["entity_names"] == ["Alice"]
    assert (await full_entities.get_by_id("doc-1"))["nested"]["a"] == 1
    assert (await full_relations.get_by_id("doc-1"))["relation_pairs"] == [["A", "B"]]


@pytest.mark.asyncio
async def test_batch_merge_restores_anchors_wiped_by_phase0():
    """Hot-path multimodal merge: Phase 0 overwrite must not drop prior anchors."""

    proc = object.__new__(ProcessorMixin)
    proc.logger = MagicMock()
    proc._get_file_reference = lambda p: p

    full_entities = FakeKV(
        {
            "doc-1": {
                "entity_names": ["Alice", "Table1 (table)"],
                "count": 2,
                "source": "text_pipeline",
            }
        }
    )
    full_relations = FakeKV(
        {
            "doc-1": {
                "relation_pairs": [["Alice", "Bob"]],
                "count": 1,
                "source": "text_pipeline",
            }
        }
    )

    async def fake_merge(**kwargs):
        # Reproduce LightRAG Phase 0 replacement semantics.
        doc_id = kwargs["doc_id"]
        await kwargs["full_entities_storage"].upsert(
            {
                doc_id: {
                    "entity_names": ["Pump"],
                    "count": 1,
                }
            }
        )
        await kwargs["full_relations_storage"].upsert(
            {
                doc_id: {
                    "relation_pairs": [["Pump", "Valve"]],
                    "count": 1,
                }
            }
        )

    class _LightRAGStub:
        pass

    lightrag = _LightRAGStub()
    lightrag.full_entities = full_entities
    lightrag.full_relations = full_relations
    lightrag.chunk_entity_relation_graph = MagicMock()
    lightrag.entities_vdb = MagicMock()
    lightrag.relationships_vdb = MagicMock()
    lightrag.llm_response_cache = MagicMock()
    lightrag.entity_chunks = MagicMock()
    lightrag.relation_chunks = MagicMock()
    lightrag._insert_done = AsyncMock()
    proc.lightrag = lightrag

    with (
        patch(
            "lightrag.operate.merge_nodes_and_edges",
            new=AsyncMock(side_effect=fake_merge),
        ),
        patch(
            "lightrag.kg.shared_storage.get_namespace_data",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "lightrag.kg.shared_storage.get_pipeline_status_lock",
            return_value=asyncio.Lock(),
        ),
    ):
        await proc._batch_merge_lightrag_style_type_aware(
            enhanced_chunk_results=[({}, {})],
            file_path="doc.pdf",
            doc_id="doc-1",
        )

    entities = await full_entities.get_by_id("doc-1")
    relations = await full_relations.get_by_id("doc-1")

    assert entities["entity_names"] == ["Alice", "Pump", "Table1 (table)"]
    assert entities["source"] == "text_pipeline"
    assert relations["relation_pairs"] == [["Alice", "Bob"], ["Pump", "Valve"]]
    assert relations["source"] == "text_pipeline"
    proc.lightrag._insert_done.assert_awaited_once()
