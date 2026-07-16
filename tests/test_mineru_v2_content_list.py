from types import SimpleNamespace

import pytest

import raganything.processor as processor_module


class DummyProcessor(processor_module.ProcessorMixin):
    pass


class DummyLogger:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def test_normalize_nested_content_list_flattens_mineru_wrapper():
    processor = DummyProcessor()
    paragraph = {"type": "paragraph", "content": {"paragraph_content": []}}
    text = {"type": "text", "text": "plain text"}

    normalized = processor._normalize_nested_content_list(
        [[paragraph, "ignored"], text, None]
    )

    assert normalized == [paragraph, text]


@pytest.mark.asyncio
async def test_insert_content_list_recovers_text_from_mineru_v2_blocks(monkeypatch):
    processor = DummyProcessor()
    processor.config = SimpleNamespace(
        allow_embedding_only_ingestion=False,
        display_content_stats=False,
        use_full_path=False,
    )
    processor.logger = DummyLogger()

    async def ensure_initialized():
        return {"success": True}

    completed_doc_ids = []

    async def mark_complete(doc_id):
        completed_doc_ids.append(doc_id)

    inserted = {}

    async def insert_text_content(_lightrag, **kwargs):
        inserted.update(kwargs)

    processor._ensure_lightrag_initialized = ensure_initialized
    processor._mark_multimodal_processing_complete = mark_complete
    processor.lightrag = object()
    monkeypatch.setattr(processor_module, "insert_text_content", insert_text_content)

    content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [
                        {"type": "text", "content": "Safety"},
                        {"content": [{"type": "text", "content": "Protocol"}]},
                    ]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Wear eye protection."}
                    ]
                },
            },
            {
                "type": "list",
                "content": {
                    "list_items": [
                        {
                            "prefix": "1.",
                            "item_content": [
                                {"type": "text", "content": "Inspect equipment"}
                            ],
                        },
                        {
                            "prefix": "2.",
                            "item_content": [
                                {"type": "text", "content": "Record results"}
                            ],
                        },
                    ]
                },
            },
            {
                "type": "table",
                "content": {"html": "<table><tr><td>OK</td></tr></table>"},
            },
            {
                "type": "image",
                "content": {"image_caption": ["Emergency shutoff location"]},
            },
        ]
    ]

    await processor.insert_content_list(
        content_list,
        file_path="/data/manual.json",
        doc_id="doc-manual",
        skip_multimodal_processing=True,
    )

    assert inserted["input"] == (
        "Safety Protocol\n\n"
        "Wear eye protection.\n\n"
        "1. Inspect equipment\n"
        "2. Record results\n\n"
        "<table><tr><td>OK</td></tr></table>\n\n"
        "Emergency shutoff location"
    )
    assert inserted["file_paths"] == "manual.json"
    assert inserted["ids"] == "doc-manual"
    assert completed_doc_ids == ["doc-manual"]
