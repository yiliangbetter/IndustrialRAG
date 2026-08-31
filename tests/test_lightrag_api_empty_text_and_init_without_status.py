"""LightRAG API ingest edges still unclaimed by open coverage PRs.

Locks current production contracts:
- Image-only (no harvestable text) parses still return True but skip
  ``insert_text_content_with_multimodal_content``, so figures are not
  forwarded on this API path.
- Init failure with no ``doc_status`` storage returns False without raising.
- ``parser=`` mutates ``config.parser`` before init is attempted.
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


def _processor_with_doc_status():
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
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatusStorage()},
    )()
    return processor


@pytest.mark.asyncio
async def test_empty_text_skips_multimodal_ainsert_but_returns_true(
    monkeypatch, tmp_path
):
    """Image-only extracts currently skip the multimodal ainsert helper."""
    processor = _processor_with_doc_status()

    async def fake_ensure():
        return {"success": True}

    async def fake_parse(file_path, output_dir, parse_method, display_stats, **kwargs):
        return (
            [
                {
                    "type": "image",
                    "img_path": "/abs/fig1.png",
                    "image_caption": ["pump diagram"],
                    "page_idx": 0,
                }
            ],
            "doc-image-only",
        )

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

    result = await processor.process_document_complete_lightrag_api(str(pdf_path))

    assert result is True
    insert_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_init_failure_without_doc_status_returns_false_without_raising():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {"use_full_path": False, "parser": "mineru"},
    )()
    processor.lightrag = object()

    async def fake_ensure():
        return {"success": False, "error": "missing llm_model_func"}

    processor._ensure_lightrag_initialized = fake_ensure

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False


@pytest.mark.asyncio
async def test_parser_kwarg_mutates_config_before_init():
    processor = _processor_with_doc_status()
    processor.config.parser = "mineru"

    async def fake_ensure():
        return {"success": False, "error": "forced"}

    processor._ensure_lightrag_initialized = fake_ensure

    result = await processor.process_document_complete_lightrag_api(
        "sample.pdf", parser="paddleocr"
    )

    assert result is False
    assert processor.config.parser == "paddleocr"
