"""Regression tests for LightRAG batch entity extract/merge wiring."""

import pytest

from raganything.processor import ProcessorMixin


class FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []
        self.debugs = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg))

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))

    def debug(self, msg, *args, **kwargs):
        self.debugs.append(str(msg))


def _make_processor(*, use_full_path=False):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type("Config", (), {"use_full_path": use_full_path})()

    async def fake_insert_done():
        processor.insert_done_calls += 1

    processor.insert_done_calls = 0
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "llm_response_cache": object(),
            "text_chunks": object(),
            "chunk_entity_relation_graph": object(),
            "entities_vdb": object(),
            "relationships_vdb": object(),
            "full_entities": object(),
            "full_relations": object(),
            "entity_chunks": object(),
            "relation_chunks": object(),
        },
    )()
    # Assign on the instance so Python does not bind an implicit self.
    processor.lightrag._insert_done = fake_insert_done
    return processor


@pytest.mark.asyncio
async def test_batch_extract_entities_forwards_lightrag_context(monkeypatch):
    processor = _make_processor()
    chunks = {"chunk-1": {"content": "table text", "full_doc_id": "doc-1"}}
    pipeline_status = {"latest_message": "", "history_messages": []}
    pipeline_lock = object()
    extract_calls = []

    async def fake_get_namespace_data(name):
        assert name == "pipeline_status"
        return pipeline_status

    def fake_get_pipeline_status_lock():
        return pipeline_lock

    async def fake_extract_entities(**kwargs):
        extract_calls.append(kwargs)
        return [({"EntityA": [{"source_id": "chunk-1"}]}, {})]

    import lightrag.kg.shared_storage as shared_mod
    import lightrag.operate as operate_mod

    monkeypatch.setattr(shared_mod, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(
        shared_mod, "get_pipeline_status_lock", fake_get_pipeline_status_lock
    )
    monkeypatch.setattr(operate_mod, "extract_entities", fake_extract_entities)

    results = await processor._batch_extract_entities_lightrag_style_type_aware(chunks)

    assert results == [({"EntityA": [{"source_id": "chunk-1"}]}, {})]
    assert len(extract_calls) == 1
    call = extract_calls[0]
    assert call["chunks"] is chunks
    assert call["global_config"] is processor.lightrag.__dict__
    assert call["pipeline_status"] is pipeline_status
    assert call["pipeline_status_lock"] is pipeline_lock
    assert call["llm_response_cache"] is processor.lightrag.llm_response_cache
    assert call["text_chunks_storage"] is processor.lightrag.text_chunks
    assert any(
        "Extracted entities from 1 multimodal chunks" in msg
        for msg in processor.logger.infos
    )


@pytest.mark.asyncio
async def test_batch_merge_uses_basename_and_calls_insert_done(monkeypatch):
    processor = _make_processor(use_full_path=False)
    enhanced = [({"EntityA": [{"source_id": "chunk-1"}]}, {})]
    pipeline_status = {"latest_message": ""}
    pipeline_lock = object()
    merge_calls = []

    async def fake_get_namespace_data(name):
        assert name == "pipeline_status"
        return pipeline_status

    def fake_get_pipeline_status_lock():
        return pipeline_lock

    async def fake_merge_nodes_and_edges(**kwargs):
        merge_calls.append(kwargs)

    import lightrag.kg.shared_storage as shared_mod
    import lightrag.operate as operate_mod

    monkeypatch.setattr(shared_mod, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(
        shared_mod, "get_pipeline_status_lock", fake_get_pipeline_status_lock
    )
    monkeypatch.setattr(operate_mod, "merge_nodes_and_edges", fake_merge_nodes_and_edges)

    await processor._batch_merge_lightrag_style_type_aware(
        enhanced, "/docs/manuals/spec.pdf", "doc-1"
    )

    assert len(merge_calls) == 1
    call = merge_calls[0]
    assert call["chunk_results"] is enhanced
    assert call["doc_id"] == "doc-1"
    assert call["file_path"] == "spec.pdf"
    assert call["knowledge_graph_inst"] is processor.lightrag.chunk_entity_relation_graph
    assert call["entity_vdb"] is processor.lightrag.entities_vdb
    assert call["relationships_vdb"] is processor.lightrag.relationships_vdb
    assert call["full_entities_storage"] is processor.lightrag.full_entities
    assert call["full_relations_storage"] is processor.lightrag.full_relations
    assert call["entity_chunks_storage"] is processor.lightrag.entity_chunks
    assert call["relation_chunks_storage"] is processor.lightrag.relation_chunks
    assert call["pipeline_status"] is pipeline_status
    assert call["pipeline_status_lock"] is pipeline_lock
    assert call["current_file_number"] == 1
    assert call["total_files"] == 1
    assert processor.insert_done_calls == 1


@pytest.mark.asyncio
async def test_batch_merge_preserves_full_path_when_configured(monkeypatch):
    processor = _make_processor(use_full_path=True)
    merge_calls = []

    async def fake_get_namespace_data(name):
        return {}

    def fake_get_pipeline_status_lock():
        return object()

    async def fake_merge_nodes_and_edges(**kwargs):
        merge_calls.append(kwargs)

    import lightrag.kg.shared_storage as shared_mod
    import lightrag.operate as operate_mod

    monkeypatch.setattr(shared_mod, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(
        shared_mod, "get_pipeline_status_lock", fake_get_pipeline_status_lock
    )
    monkeypatch.setattr(operate_mod, "merge_nodes_and_edges", fake_merge_nodes_and_edges)

    await processor._batch_merge_lightrag_style_type_aware(
        [], "/docs/manuals/spec.pdf", "doc-9"
    )

    assert merge_calls[0]["file_path"] == "/docs/manuals/spec.pdf"
    assert merge_calls[0]["doc_id"] == "doc-9"
    assert processor.insert_done_calls == 1
