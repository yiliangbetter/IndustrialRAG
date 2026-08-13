"""Embedding-only ingestion edges that would silently drop or skip index writes.

These paths persist chunks without LLM extraction. Empty input still reports
document-complete; tokenizer failures must not abort upsert; multimodal items
must not be processed when embedding-only is on.
"""

import pytest

from raganything.base import DocStatus
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


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class RaisingTokenizer:
    def encode(self, text):
        raise RuntimeError("tokenizer unavailable")


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


class FakeLightRAG:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or FakeTokenizer()
        self.text_chunks = FakeStorage()
        self.chunks_vdb = FakeStorage()
        self.doc_status = FakeStorage()
        self.insert_done_calls = 0

        async def _insert_done():
            self.insert_done_calls += 1

        # Bind on the instance: a class-dict coroutine would consume `self`.
        self._insert_done = _insert_done


class RecordingDocumentCallback(ProcessingCallback):
    def __init__(self):
        self.document_completes = []

    def on_document_complete(self, file_path, doc_id=None, **kwargs):
        self.document_completes.append((file_path, doc_id))


class DummyProcessor(ProcessorMixin):
    pass


def make_embedding_only_processor(tmp_path, tokenizer=None):
    processor = DummyProcessor()
    processor.config = type(
        "Config",
        (),
        {
            "allow_embedding_only_ingestion": True,
            "content_format": "minerU",
            "display_content_stats": False,
            "parse_method": "auto",
            "parser": "mineru",
            "parser_output_dir": str(tmp_path / "output"),
            "use_full_path": False,
        },
    )()
    processor.logger = FakeLogger()
    processor.lightrag = FakeLightRAG(tokenizer=tokenizer)
    processor.callback_manager = CallbackManager()
    processor.parse_cache = None

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    return processor


@pytest.mark.asyncio
async def test_embedding_only_empty_text_completes_without_chunks(tmp_path):
    """Whitespace-only content still emits complete and writes no chunks."""
    processor = make_embedding_only_processor(tmp_path)
    callback = RecordingDocumentCallback()
    processor.callback_manager.register(callback)

    mm_calls = []

    async def fake_mm(*args, **kwargs):
        mm_calls.append((args, kwargs))

    processor._process_multimodal_content = fake_mm

    await processor.insert_content_list(
        [{"type": "text", "text": "   \n\t  "}],
        file_path="empty.pdf",
        doc_id="doc-empty",
    )

    assert callback.document_completes == [("empty.pdf", "doc-empty")]
    assert processor.lightrag.text_chunks.records == {}
    assert processor.lightrag.chunks_vdb.records == {}
    assert processor.lightrag.doc_status.records == {}
    assert processor.lightrag.insert_done_calls == 0
    assert mm_calls == []


@pytest.mark.asyncio
async def test_embedding_only_tokenizer_failure_falls_back_to_word_count(tmp_path):
    """Chunk upsert must survive tokenizer.encode failures via word-count tokens."""
    processor = make_embedding_only_processor(tmp_path, tokenizer=RaisingTokenizer())

    await processor.insert_content_list(
        [{"type": "text", "text": "alpha beta gamma"}],
        file_path="manual.pdf",
        doc_id="doc-tokens",
    )

    chunks = processor.lightrag.text_chunks.records
    assert len(chunks) == 1
    chunk = next(iter(chunks.values()))
    assert chunk["content"] == "alpha beta gamma"
    assert chunk["tokens"] == 3
    assert chunk["full_doc_id"] == "doc-tokens"
    assert processor.lightrag.insert_done_calls == 1

    doc_status = processor.lightrag.doc_status.records["doc-tokens"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["multimodal_processed"] is True
    assert doc_status["chunks_count"] == 1


@pytest.mark.asyncio
async def test_embedding_only_skips_multimodal_and_llm_insert(tmp_path, monkeypatch):
    """Images in the list must not run multimodal processors or LightRAG ainsert."""
    import raganything.processor as processor_module

    processor = make_embedding_only_processor(tmp_path)
    mm_calls = []
    insert_calls = []

    async def fake_mm(*args, **kwargs):
        mm_calls.append((args, kwargs))

    async def fake_insert(*args, **kwargs):
        insert_calls.append(kwargs)

    processor._process_multimodal_content = fake_mm
    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await processor.insert_content_list(
        [
            {"type": "text", "text": "Keep this prose"},
            {"type": "image", "img_path": "/abs/figure.png"},
            {"type": "table", "table_body": "a|b"},
        ],
        file_path="mixed.pdf",
        doc_id="doc-mixed",
    )

    assert insert_calls == []
    assert mm_calls == []
    chunk_text = "\n".join(
        chunk["content"] for chunk in processor.lightrag.text_chunks.records.values()
    )
    assert "Keep this prose" in chunk_text
    assert processor.lightrag.doc_status.records["doc-mixed"]["chunks_count"] == 1


def test_plaintext_from_mineru_blocks_recovers_nested_v2_structure():
    """MinerU v2 nested blocks must yield searchable plaintext when type=text is absent."""
    processor = DummyProcessor()
    text = processor._plaintext_from_mineru_blocks(
        [
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "Executive Summary"}]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Alpha"},
                        {"type": "text", "content": "Beta"},
                    ]
                },
            },
            {
                "type": "list",
                "content": {
                    "list_items": [
                        {
                            "prefix": "1.",
                            "item_content": [{"type": "text", "content": "First"}],
                        },
                        {"prefix": "2.", "item_content": "Second"},
                    ]
                },
            },
            {
                "type": "table",
                "content": {"html": "<table><tr><td>Total</td></tr></table>"},
            },
            {"type": "image", "content": {"image_caption": ["Pump diagram"]}},
        ]
    )

    assert "Executive Summary" in text
    assert "Alpha Beta" in text
    assert "1. First" in text
    assert "2. Second" in text
    assert "<table><tr><td>Total</td></tr></table>" in text
    assert "Pump diagram" in text


@pytest.mark.asyncio
async def test_insert_content_list_embedding_only_indexes_mineru_v2_plaintext(
    tmp_path,
):
    """Nested MinerU wrappers with no flat text still become embedding chunks."""
    processor = make_embedding_only_processor(tmp_path)

    await processor.insert_content_list(
        [
            [
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "Recovered body"}
                        ]
                    },
                }
            ]
        ],
        file_path="doc_content_list_v2.json",
        doc_id="doc-v2",
    )

    chunk_text = "\n".join(
        chunk["content"] for chunk in processor.lightrag.text_chunks.records.values()
    )
    assert "Recovered body" in chunk_text
    assert (
        processor.lightrag.doc_status.records["doc-v2"]["status"] == DocStatus.PROCESSED
    )
    assert processor.lightrag.insert_done_calls == 1
