import pytest

from raganything.base import DocStatus
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


class FakeStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeTokenizer:
    def encode(self, text):
        if "fallback" in text:
            raise RuntimeError("tokenizer unavailable")
        return text.split()


class FakeLightRAG:
    def __init__(self):
        self.text_chunks = FakeStorage()
        self.chunks_vdb = FakeStorage()
        self.doc_status = FakeStorage()
        self.tokenizer = FakeTokenizer()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


class DummyProcessor(ProcessorMixin):
    pass


def make_processor():
    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.callback_manager = None
    processor.lightrag = FakeLightRAG()
    processor.config = type(
        "Config",
        (),
        {
            "allow_embedding_only_ingestion": True,
            "content_format": "minerU",
            "display_content_stats": False,
            "use_full_path": False,
        },
    )()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fail_multimodal(*args, **kwargs):
        raise AssertionError("embedding-only ingestion must not process multimodal items")

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor._process_multimodal_content = fail_multimodal
    return processor


def test_normalize_nested_content_list_flattens_top_level_lists_only():
    processor = make_processor()
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
    processor = make_processor()

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
            {"type": "table", "content": {"html": "<table><tr><td>42</td></tr></table>"}},
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


@pytest.mark.asyncio
async def test_insert_content_list_embedding_only_indexes_mineru_v2_text_blocks():
    processor = make_processor()

    await processor.insert_content_list(
        [
            [
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "First fallback paragraph"}
                        ]
                    },
                },
                {
                    "type": "list",
                    "content": {
                        "list_items": [
                            {
                                "prefix": "-",
                                "item_content": [
                                    {"type": "text", "content": "second chunk"}
                                ],
                            }
                        ]
                    },
                },
                "drop-me",
            ]
        ],
        file_path="/tmp/report_content_list_v2.json",
    )

    assert processor.lightrag.insert_done_calls == 1
    assert processor.lightrag.text_chunks.index_done_calls == 1
    assert processor.lightrag.chunks_vdb.index_done_calls == 1

    chunk_records = processor.lightrag.text_chunks.records
    assert processor.lightrag.chunks_vdb.records == chunk_records
    assert [chunk["content"] for chunk in chunk_records.values()] == [
        "First fallback paragraph",
        "- second chunk",
    ]
    assert [chunk["tokens"] for chunk in chunk_records.values()] == [3, 3]
    assert {chunk["file_path"] for chunk in chunk_records.values()} == {
        "report_content_list_v2.json"
    }

    doc_id = next(iter(processor.lightrag.doc_status.records))
    doc_status = processor.lightrag.doc_status.records[doc_id]
    assert doc_id.startswith("doc-")
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["multimodal_processed"] is True
    assert doc_status["chunks_count"] == 2
    assert doc_status["chunks_list"] == list(chunk_records)
    assert doc_status["file_path"] == "report_content_list_v2.json"
