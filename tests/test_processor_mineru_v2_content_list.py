"""Regression tests for MinerU v2 content-list graph ingestion.

Covers nested *_content_list_v2.json normalization, plaintext recovery when
there are no flat type=text blocks, doc_id hashing after flatten, and the
skip_multimodal_processing text-first KG path used by batch graph ingest.
"""

import pytest

import raganything.processor as processor_module
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


class DummyProcessor(ProcessorMixin):
    def __init__(self):
        self.logger = FakeLogger()
        self.callback_manager = None
        self.config = type(
            "Config",
            (),
            {
                "display_content_stats": False,
                "use_full_path": False,
                "content_format": "minerU",
                "allow_embedding_only_ingestion": False,
            },
        )()
        self.lightrag = object()
        self.marked_complete_doc_ids = []

    async def _ensure_lightrag_initialized(self):
        return {"success": True}

    async def _mark_multimodal_processing_complete(self, doc_id):
        self.marked_complete_doc_ids.append(doc_id)

    async def _process_multimodal_content(self, items, file_ref, doc_id):
        raise AssertionError("skip_multimodal_processing should bypass multimodal work")


@pytest.mark.asyncio
async def test_insert_content_list_recovers_text_and_doc_id_from_mineru_v2(
    monkeypatch,
):
    """Nested MinerU v2 blocks become searchable text with a stable doc_id."""
    processor = DummyProcessor()
    nested_content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "Storage Report"}]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "MinerU v2"},
                        {"type": "text", "content": "preserves text"},
                    ]
                },
            },
        ],
        [
            {
                "type": "list",
                "content": {
                    "list_items": [
                        {
                            "prefix": "1.",
                            "item_content": [
                                {"type": "text", "content": "Normalize wrappers"}
                            ],
                        },
                        {
                            "prefix": "2.",
                            "item_content": [
                                {"type": "text", "content": "Recover list text"}
                            ],
                        },
                    ]
                },
            }
        ],
    ]
    captured_insert = {}

    async def fake_insert_text_content(lightrag, **kwargs):
        captured_insert["lightrag"] = lightrag
        captured_insert.update(kwargs)

    monkeypatch.setattr(
        processor_module, "insert_text_content", fake_insert_text_content
    )

    await processor.insert_content_list(
        nested_content_list,
        file_path="/tmp/mineru_v2_content_list.json",
        skip_multimodal_processing=True,
    )

    normalized = processor._normalize_nested_content_list(nested_content_list)
    expected_doc_id = processor._generate_content_based_doc_id(normalized)
    empty_wrapper_doc_id = processor._generate_content_based_doc_id(nested_content_list)

    assert captured_insert["lightrag"] is processor.lightrag
    assert captured_insert["file_paths"] == "mineru_v2_content_list.json"
    assert captured_insert["ids"] == expected_doc_id
    assert captured_insert["ids"] != empty_wrapper_doc_id
    assert captured_insert["input"] == (
        "Storage Report\n\n"
        "MinerU v2 preserves text\n\n"
        "1. Normalize wrappers\n"
        "2. Recover list text"
    )
    assert processor.marked_complete_doc_ids == [expected_doc_id]


def test_normalize_nested_content_list_flattens_top_level_lists_only():
    processor = DummyProcessor()
    text_block = {"type": "text", "text": "kept"}
    image_block = {"type": "image", "img_path": "/tmp/img.png"}

    assert processor._normalize_nested_content_list(
        [
            [text_block, "drop-me"],
            image_block,
            "drop-me-too",
            [[{"type": "text", "text": "still nested"}]],
        ]
    ) == [text_block, image_block]


def test_plaintext_from_mineru_v2_blocks_recovers_searchable_text():
    processor = DummyProcessor()

    text = processor._plaintext_from_mineru_blocks(
        [
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "Quarterly report"}]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Revenue"},
                        {"type": "text", "content": "grew"},
                    ]
                },
            },
            {
                "type": "list",
                "content": {
                    "list_items": [
                        {
                            "prefix": "1.",
                            "item_content": [{"type": "text", "content": "Validate"}],
                        },
                        {"prefix": "-", "item_content": {"text": "Ship"}},
                    ]
                },
            },
            {
                "type": "table",
                "content": {"html": "<table><tr><td>42</td></tr></table>"},
            },
            {"type": "image", "content": {"image_caption": ["Factory floor"]}},
            {"type": "unknown", "content": {"text": "ignored"}},
            "not-a-block",
        ]
    )

    assert text == (
        "Quarterly report\n\n"
        "Revenue grew\n\n"
        "1. Validate\n"
        "- Ship\n\n"
        "<table><tr><td>42</td></tr></table>\n\n"
        "Factory floor"
    )
