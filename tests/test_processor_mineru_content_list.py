from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import raganything.processor as processor_module
from raganything.callbacks import CallbackManager, ProcessingCallback
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


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_document_complete(self, file_path, **kwargs):
        self.events.append(("document_complete", file_path, kwargs.get("doc_id")))


class DummyProcessor(ProcessorMixin):
    pass


def make_processor(allow_embedding_only_ingestion=False):
    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.lightrag = SimpleNamespace()
    processor.callback_manager = CallbackManager()
    processor.config = SimpleNamespace(
        allow_embedding_only_ingestion=allow_embedding_only_ingestion,
        content_format="minerU",
        display_content_stats=False,
        use_full_path=False,
    )

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    return processor


def test_normalize_nested_content_list_flattens_wrappers():
    processor = make_processor()
    nested = [
        [{"type": "paragraph", "content": {"paragraph_content": []}}],
        {"type": "text", "text": "plain text"},
    ]

    normalized = processor._normalize_nested_content_list(nested)

    assert normalized == [
        {"type": "paragraph", "content": {"paragraph_content": []}},
        {"type": "text", "text": "plain text"},
    ]


def test_plaintext_from_mineru_blocks_extracts_paragraph_title_list():
    processor = make_processor()
    blocks = [
        {
            "type": "title",
            "content": {
                "title_content": [{"type": "text", "content": "Safety Checklist"}]
            },
        },
        {
            "type": "paragraph",
            "content": {
                "paragraph_content": [
                    {"type": "text", "content": "Inspect valve"},
                    {"type": "text", "content": "before startup"},
                ]
            },
        },
        {
            "type": "list",
            "content": {
                "list_items": [
                    {
                        "prefix": "1.",
                        "item_content": [{"type": "text", "content": "Lock out"}],
                    },
                    {
                        "prefix": "2.",
                        "item_content": [{"type": "text", "content": "Tag out"}],
                    },
                ]
            },
        },
    ]

    plaintext = processor._plaintext_from_mineru_blocks(blocks)

    assert plaintext == (
        "Safety Checklist\n\n"
        "Inspect valve before startup\n\n"
        "1. Lock out\n2. Tag out"
    )


def test_generate_content_based_doc_id_uses_normalized_blocks():
    processor = make_processor()
    wrapped_content = [[{"type": "text", "text": "wrapped content"}]]
    normalized = processor._normalize_nested_content_list(wrapped_content)

    normalized_doc_id = processor._generate_content_based_doc_id(normalized)
    wrapper_doc_id = processor._generate_content_based_doc_id(wrapped_content)

    assert normalized_doc_id != wrapper_doc_id


@pytest.mark.asyncio
async def test_insert_content_list_skip_multimodal_marks_complete_without_multimodal_batch(
    monkeypatch,
):
    processor = make_processor()
    processor._mark_multimodal_processing_complete = AsyncMock()
    processor._process_multimodal_content = AsyncMock()
    insert_text_content = AsyncMock()
    monkeypatch.setattr(processor_module, "insert_text_content", insert_text_content)
    callback = RecordingCallback()
    processor.callback_manager.register(callback)

    await processor.insert_content_list(
        [
            {"type": "text", "text": "pump status"},
            {"type": "image", "img_path": "/tmp/pump.png"},
        ],
        file_path="/docs/pump_content_list_v2.json",
        doc_id="doc-pump",
        skip_multimodal_processing=True,
    )

    insert_text_content.assert_awaited_once()
    assert insert_text_content.await_args.kwargs["input"] == "pump status"
    assert insert_text_content.await_args.kwargs["file_paths"] == (
        "pump_content_list_v2.json"
    )
    assert insert_text_content.await_args.kwargs["ids"] == "doc-pump"
    processor._mark_multimodal_processing_complete.assert_awaited_once_with("doc-pump")
    processor._process_multimodal_content.assert_not_called()
    assert callback.events == [
        ("document_complete", "/docs/pump_content_list_v2.json", "doc-pump")
    ]
