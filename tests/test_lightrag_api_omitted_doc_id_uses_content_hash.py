"""LightRAG API ingest must use parse-time content ids when the caller omits doc_id.

Content-based ids are how duplicate manuals collapse. If omitted doc_id is
replaced with a path-derived id, the same PDF ingested twice becomes two
graphs. split_by_character must also reach ainsert so plant-manual chunking
is not silently dropped on this API path.
"""

from types import SimpleNamespace

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


class FakeDocStatusStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class DummyLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_omitted_doc_id_uses_content_based_id_and_forwards_split(
    monkeypatch,
):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(
        use_full_path=False,
        parser="mineru",
        parse_method="auto",
        parser_output_dir="./output",
        display_content_stats=False,
        content_format="minerU",
    )
    processor.lightrag = SimpleNamespace(doc_status=FakeDocStatusStorage())

    async def fake_ensure():
        return {"success": True}

    async def fake_parse(file_path, output_dir, parse_method, display_stats, **kwargs):
        return ([{"type": "text", "text": "intro", "page_idx": 0}], "doc-from-content")

    captured = {}

    async def fake_insert(lightrag, input=None, **kwargs):
        captured["input"] = input
        captured.update(kwargs)

    processor._ensure_lightrag_initialized = fake_ensure
    processor.parse_document = fake_parse
    monkeypatch.setattr(
        "raganything.processor.insert_text_content_with_multimodal_content",
        fake_insert,
    )

    import lightrag.kg.shared_storage as shared_storage

    pipeline_status = {"history_messages": []}

    async def fake_get_namespace_data(_name):
        return pipeline_status

    monkeypatch.setattr(shared_storage, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(shared_storage, "get_pipeline_status_lock", lambda: DummyLock())

    result = await processor.process_document_complete_lightrag_api(
        "sample.pdf",
        split_by_character="\n",
        split_by_character_only=True,
    )

    assert result is True
    assert captured["input"] == "intro"
    assert captured["ids"] == "doc-from-content"
    assert captured["split_by_character"] == "\n"
    assert captured["split_by_character_only"] is True
    assert captured["file_paths"] == "sample.pdf"
