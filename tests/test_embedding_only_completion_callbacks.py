"""Embedding-only ingestion must still report document completion.

The early return that skips LLM extraction used to leave progress hooks
silent. These tests lock the callback, the citation path, and the fact that
images on that path are not sent to multimodal processors.
"""

import pytest

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
        self.completed = []
        self.errors = []
        self.text_inserts = []

    def on_document_complete(
        self, file_path, doc_id="", duration_seconds=0.0, **kwargs
    ):
        self.completed.append(
            {
                "file_path": file_path,
                "doc_id": doc_id,
                "duration_seconds": duration_seconds,
            }
        )

    def on_document_error(self, file_path, error="", stage="", **kwargs):
        self.errors.append(
            {
                "file_path": file_path,
                "error": error,
                "stage": stage,
                "doc_id": kwargs.get("doc_id"),
            }
        )

    def on_text_insert_start(self, file_path, **kwargs):
        self.text_inserts.append(file_path)


class EmbeddingOnlyProcessor(ProcessorMixin):
    def __init__(self):
        self.logger = FakeLogger()
        self.config = type(
            "Config",
            (),
            {
                "allow_embedding_only_ingestion": True,
                "parser_output_dir": "/tmp/unused-output",
                "parse_method": "auto",
                "display_content_stats": False,
                "use_full_path": False,
                "content_format": "minerU",
            },
        )()
        self.callback_manager = CallbackManager()
        self.callback = RecordingCallback()
        self.callback_manager.register(self.callback)
        self.inserts = []
        self.context_sources = []
        self.multimodal_calls = []
        self.marked_complete = []
        self.insert_error = None

        async def ensure_ready():
            return {"success": True}

        self._ensure_lightrag_initialized = ensure_ready

    def set_content_source_for_context(self, content_list, content_format):
        self.context_sources.append((content_list, content_format))

    async def _insert_text_content_embedding_only(self, text_content, file_ref, doc_id):
        self.inserts.append(
            {
                "text_content": text_content,
                "file_ref": file_ref,
                "doc_id": doc_id,
            }
        )
        if self.insert_error is not None:
            raise self.insert_error

    async def _process_multimodal_content(self, items, file_path, doc_id):
        self.multimodal_calls.append((items, file_path, doc_id))

    async def _mark_multimodal_processing_complete(self, doc_id):
        self.marked_complete.append(doc_id)


def _manual_blocks():
    return [
        {"type": "text", "text": "Lock out the breaker before service.", "page_idx": 0},
        {
            "type": "image",
            "img_path": "/manuals/figures/breaker.png",
            "image_caption": ["Breaker lockout"],
            "page_idx": 1,
        },
    ]


@pytest.mark.asyncio
async def test_document_embedding_only_notifies_completion_without_multimodal(
    monkeypatch,
):
    import raganything.processor as processor_module

    processor = EmbeddingOnlyProcessor()
    blocks = _manual_blocks()

    async def fake_parse(file_path, output_dir, parse_method, display_stats, **kwargs):
        return blocks, "doc-from-bytes"

    async def unexpected_lightrag_insert(*args, **kwargs):
        raise AssertionError("embedding-only path used LightRAG text insert")

    processor.parse_document = fake_parse
    monkeypatch.setattr(
        processor_module, "insert_text_content", unexpected_lightrag_insert
    )

    await processor.process_document_complete(
        "manuals/press.pdf",
        doc_id="doc-press",
    )

    assert processor.inserts == [
        {
            "text_content": "Lock out the breaker before service.",
            "file_ref": "press.pdf",
            "doc_id": "doc-press",
        }
    ]
    assert processor.context_sources == []
    assert processor.multimodal_calls == []
    assert processor.marked_complete == []
    assert processor.callback.text_inserts == []
    assert processor.callback.errors == []
    assert len(processor.callback.completed) == 1
    completed = processor.callback.completed[0]
    assert completed["file_path"] == "manuals/press.pdf"
    assert completed["doc_id"] == "doc-press"
    assert completed["duration_seconds"] >= 0


@pytest.mark.asyncio
async def test_document_embedding_only_failure_reports_error_not_completion():
    processor = EmbeddingOnlyProcessor()
    processor.insert_error = RuntimeError("vector store unavailable")

    async def fake_parse(file_path, output_dir, parse_method, display_stats, **kwargs):
        return _manual_blocks(), "doc-from-bytes"

    processor.parse_document = fake_parse

    with pytest.raises(RuntimeError, match="vector store unavailable"):
        await processor.process_document_complete(
            "manuals/press.pdf",
            doc_id="doc-press",
        )

    assert processor.callback.completed == []
    assert processor.multimodal_calls == []
    assert processor.callback.errors == [
        {
            "file_path": "manuals/press.pdf",
            "error": processor.insert_error,
            "stage": "parse",
            "doc_id": "doc-press",
        }
    ]


@pytest.mark.asyncio
async def test_content_list_embedding_only_notifies_completion_without_multimodal(
    monkeypatch,
):
    import raganything.processor as processor_module

    processor = EmbeddingOnlyProcessor()
    blocks = _manual_blocks()

    async def unexpected_lightrag_insert(*args, **kwargs):
        raise AssertionError("embedding-only path used LightRAG text insert")

    monkeypatch.setattr(
        processor_module, "insert_text_content", unexpected_lightrag_insert
    )

    await processor.insert_content_list(
        blocks,
        file_path="manuals/press.pdf",
        doc_id="doc-press",
        display_stats=False,
    )

    assert processor.context_sources == [(blocks, "minerU")]
    assert processor.inserts == [
        {
            "text_content": "Lock out the breaker before service.",
            "file_ref": "press.pdf",
            "doc_id": "doc-press",
        }
    ]
    assert processor.multimodal_calls == []
    assert processor.marked_complete == []
    assert processor.callback.text_inserts == []
    assert processor.callback.errors == []
    assert len(processor.callback.completed) == 1
    completed = processor.callback.completed[0]
    assert completed["file_path"] == "manuals/press.pdf"
    assert completed["doc_id"] == "doc-press"
    assert completed["duration_seconds"] >= 0


@pytest.mark.asyncio
async def test_image_only_content_list_still_notifies_completion():
    processor = EmbeddingOnlyProcessor()
    image_only = [
        {
            "type": "image",
            "img_path": "/manuals/figures/breaker.png",
            "image_caption": ["Breaker lockout"],
            "page_idx": 0,
        }
    ]

    await processor.insert_content_list(
        image_only,
        file_path="manuals/figure-only.pdf",
        doc_id="doc-figure",
        display_stats=False,
    )

    assert processor.inserts == [
        {
            "text_content": "",
            "file_ref": "figure-only.pdf",
            "doc_id": "doc-figure",
        }
    ]
    assert processor.multimodal_calls == []
    assert processor.callback.errors == []
    assert [event["file_path"] for event in processor.callback.completed] == [
        "manuals/figure-only.pdf"
    ]
    assert processor.callback.completed[0]["doc_id"] == "doc-figure"


@pytest.mark.asyncio
async def test_content_list_embedding_failure_does_not_report_completion():
    processor = EmbeddingOnlyProcessor()
    processor.insert_error = RuntimeError("vector store unavailable")

    with pytest.raises(RuntimeError, match="vector store unavailable"):
        await processor.insert_content_list(
            _manual_blocks(),
            file_path="manuals/press.pdf",
            doc_id="doc-press",
            display_stats=False,
        )

    assert processor.context_sources == [(_manual_blocks(), "minerU")]
    assert processor.callback.completed == []
    assert processor.callback.errors == []
    assert processor.multimodal_calls == []
