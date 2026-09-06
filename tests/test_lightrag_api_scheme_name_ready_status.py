"""LightRAG API must persist scheme_name on first-seen documents.

#142 forwards scheme_name through the insert helper. #143 preserves an
already-stored scheme_name when init fails. New documents still need the
scheme written on the initial READY upsert and passed into insert, or
plant/scheme scanners lose grouping for first-time ingest.
"""

from unittest.mock import AsyncMock

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
        self.upserts = []

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.upserts.append(data)
        self.records.update(data)

    async def index_done_callback(self):
        return None


class DummyLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_new_document_ready_status_and_insert_keep_scheme_name(
    monkeypatch, tmp_path
):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "use_full_path": False,
            "parser": "mineru",
            "parse_method": "auto",
            "parser_output_dir": "./output",
            "display_content_stats": False,
            "content_format": "minerU",
        },
    )()
    storage = FakeDocStatusStorage()
    processor.lightrag = type("FakeLightRAG", (), {"doc_status": storage})()

    async def fake_ensure():
        return {"success": True}

    async def fake_parse(file_path, output_dir, parse_method, display_stats, **kwargs):
        return ([{"type": "text", "text": "torque table"}], "doc-scheme")

    insert_mock = AsyncMock()

    processor._ensure_lightrag_initialized = fake_ensure
    processor.parse_document = fake_parse
    monkeypatch.setattr(
        "raganything.processor.insert_text_content_with_multimodal_content",
        insert_mock,
    )

    import lightrag.kg.shared_storage as shared_storage

    pipeline_status = {"history_messages": []}

    async def fake_get_namespace_data(_name):
        return pipeline_status

    monkeypatch.setattr(shared_storage, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(shared_storage, "get_pipeline_status_lock", lambda: DummyLock())

    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    result = await processor.process_document_complete_lightrag_api(
        str(pdf_path), scheme_name="plant-a"
    )

    assert result is True
    assert storage.upserts, "expected an initial doc_status write"
    first_record = next(iter(storage.upserts[0].values()))
    assert first_record["scheme_name"] == "plant-a"
    assert first_record["file_path"] == "manual.pdf"

    persisted = storage.records["doc-pre-manual.pdf"]
    assert persisted["scheme_name"] == "plant-a"

    insert_mock.assert_awaited_once()
    assert insert_mock.await_args.kwargs["scheme_name"] == "plant-a"
    assert insert_mock.await_args.kwargs["ids"] == "doc-scheme"
