from raganything.clarify_context import filter_kg_by_chunk_ids, scope_kg_to_llm_chunks


def test_filter_kg_empty_when_no_chunks():
    entities = [
        {"entity_name": "NB6J", "source_id": "chunk-cd0f44fe7ace4cd683ec7a81e53b20c7"}
    ]
    relations = [
        {
            "src_id": "NB6J",
            "tgt_id": "自动封边机",
            "source_id": "chunk-cd0f44fe7ace4cd683ec7a81e53b20c7",
        }
    ]
    out_e, out_r = filter_kg_by_chunk_ids(entities, relations, set())
    assert out_e == []
    assert out_r == []


def test_scope_kg_keeps_matching_entity_only():
    raw = {
        "data": {
            "entities": [
                {"entity_name": "NB6J", "source_id": "chunk-a"},
                {"entity_name": "NB6S", "source_id": "chunk-b"},
            ],
            "relationships": [
                {"src_id": "NB6J", "tgt_id": "x", "source_id": "chunk-a"},
                {"src_id": "NB6S", "tgt_id": "y", "source_id": "chunk-b"},
            ],
            "chunks": [
                {"chunk_id": "chunk-a", "content": "auto models", "reference_id": "1"}
            ],
            "references": [{"reference_id": "1", "file_path": "a.pdf"}],
        }
    }
    context, scoped = scope_kg_to_llm_chunks(raw, "old context")
    assert scoped is not None
    data = scoped["data"]
    assert len(data["entities"]) == 1
    assert data["entities"][0]["entity_name"] == "NB6J"
    assert len(data["relationships"]) == 1
    assert data["relationships"][0]["src_id"] == "NB6J"
    assert "NB6S" not in (context or "")


def test_scope_kg_clears_kg_when_zero_chunks():
    raw = {
        "data": {
            "entities": [{"entity_name": "NB6J", "source_id": "chunk-a"}],
            "relationships": [],
            "chunks": [],
            "references": [],
        }
    }
    context, scoped = scope_kg_to_llm_chunks(raw, "stale")
    assert scoped is not None
    assert scoped["data"]["entities"] == []
    assert scoped["data"]["relationships"] == []
    assert "NB6J" not in (context or "")
