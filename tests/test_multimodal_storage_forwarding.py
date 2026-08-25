"""Storage contracts for multimodal chunk write and entity extraction.

If only one of text_chunks / chunks_vdb is written, retrieval and KG
extraction diverge. extract_entities must receive LightRAG caches/storages
or multimodal entities are extracted without the insert pipeline's state.
"""

import pytest

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


class RecordingStorage:
    def __init__(self, fail_with=None):
        self.upserts = []
        self.fail_with = fail_with

    async def upsert(self, data):
        if self.fail_with is not None:
            raise self.fail_with
        self.upserts.append(data)


def _processor():
    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.logger = FakeLogger()
    dummy.config = type("Config", (), {})()
    return dummy


@pytest.mark.asyncio
async def test_store_chunks_writes_text_and_vector_storages():
    dummy = _processor()
    text_chunks = RecordingStorage()
    chunks_vdb = RecordingStorage()
    dummy.lightrag = type(
        "FakeLightRAG", (), {"text_chunks": text_chunks, "chunks_vdb": chunks_vdb}
    )()

    chunks = {"chunk-1": {"content": "Figure 1", "tokens": 3}}
    await dummy._store_chunks_to_lightrag_storage_type_aware(chunks)

    assert text_chunks.upserts == [chunks]
    assert chunks_vdb.upserts == [chunks]


@pytest.mark.asyncio
async def test_store_chunks_does_not_write_vdb_when_text_chunks_fails():
    dummy = _processor()
    text_chunks = RecordingStorage(fail_with=RuntimeError("text_chunks down"))
    chunks_vdb = RecordingStorage()
    dummy.lightrag = type(
        "FakeLightRAG", (), {"text_chunks": text_chunks, "chunks_vdb": chunks_vdb}
    )()

    with pytest.raises(RuntimeError, match="text_chunks down"):
        await dummy._store_chunks_to_lightrag_storage_type_aware({"chunk-1": {}})

    assert chunks_vdb.upserts == []


@pytest.mark.asyncio
async def test_store_chunks_reraises_vector_storage_failure():
    dummy = _processor()
    text_chunks = RecordingStorage()
    chunks_vdb = RecordingStorage(fail_with=RuntimeError("chunks_vdb down"))
    dummy.lightrag = type(
        "FakeLightRAG", (), {"text_chunks": text_chunks, "chunks_vdb": chunks_vdb}
    )()

    with pytest.raises(RuntimeError, match="chunks_vdb down"):
        await dummy._store_chunks_to_lightrag_storage_type_aware({"chunk-1": {}})

    assert text_chunks.upserts == [{"chunk-1": {}}]


@pytest.mark.asyncio
async def test_batch_extract_entities_forwards_lightrag_storages(monkeypatch):
    pytest.importorskip("lightrag")

    captured = {}

    async def fake_get_namespace_data(name):
        return {"namespace": name}

    def fake_get_pipeline_status_lock():
        return "pipeline-lock"

    async def fake_extract_entities(**kwargs):
        captured.update(kwargs)
        return [("nodes", "edges")]

    monkeypatch.setattr(
        "lightrag.kg.shared_storage.get_namespace_data", fake_get_namespace_data
    )
    monkeypatch.setattr(
        "lightrag.kg.shared_storage.get_pipeline_status_lock",
        fake_get_pipeline_status_lock,
    )
    monkeypatch.setattr("lightrag.operate.extract_entities", fake_extract_entities)

    dummy = _processor()
    dummy.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "llm_response_cache": "llm-cache",
            "text_chunks": "text-chunks-storage",
        },
    )()
    chunks = {"chunk-1": {"content": "Table 2"}}

    result = await dummy._batch_extract_entities_lightrag_style_type_aware(chunks)

    assert result == [("nodes", "edges")]
    assert captured["chunks"] is chunks
    assert captured["global_config"] == dummy.lightrag.__dict__
    assert captured["pipeline_status"] == {"namespace": "pipeline_status"}
    assert captured["pipeline_status_lock"] == "pipeline-lock"
    assert captured["llm_response_cache"] == "llm-cache"
    assert captured["text_chunks_storage"] == "text-chunks-storage"
