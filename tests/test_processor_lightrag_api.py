import asyncio
from types import SimpleNamespace

import pytest

import raganything.utils as utils_module
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


def _install_pipeline_status(monkeypatch):
    pipeline_status = {"history_messages": []}
    pipeline_status_lock = asyncio.Lock()

    async def fake_get_namespace_data(namespace):
        assert namespace == "pipeline_status"
        return pipeline_status

    monkeypatch.setattr(
        "lightrag.kg.shared_storage.get_namespace_data",
        fake_get_namespace_data,
    )
    monkeypatch.setattr(
        "lightrag.kg.shared_storage.get_pipeline_status_lock",
        lambda: pipeline_status_lock,
    )


def _make_lightrag_api_processor(monkeypatch, content_list, ainsert_side_effect=None):
    class DummyProcessor(ProcessorMixin):
        pass

    class FakeLightRAG:
        def __init__(self):
            self.doc_status = FakeDocStatusStorage()
            self.ainsert_calls = []

        async def ainsert(self, **kwargs):
            self.ainsert_calls.append(kwargs)
            if ainsert_side_effect is not None:
                raise ainsert_side_effect

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(
        use_full_path=False,
        parser="mineru",
        parser_output_dir="output",
        parse_method="auto",
        display_content_stats=False,
        content_format="minerU",
    )
    processor.lightrag = FakeLightRAG()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*args, **kwargs):
        return content_list, "doc-content"

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor.parse_document = fake_parse_document
    _install_pipeline_status(monkeypatch)
    return processor


@pytest.mark.asyncio
async def test_lightrag_api_init_failure_persists_failed_doc_status():
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
        },
    )()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatusStorage()},
    )()

    async def fake_ensure_lightrag_initialized():
        return {"success": False, "error": "missing llm_model_func"}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert doc_status["error_msg"] == "missing llm_model_func"
    assert doc_status["file_path"] == "sample.pdf"
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_lightrag_api_parse_failure_persists_failed_doc_status(monkeypatch):
    processor = _make_lightrag_api_processor(
        monkeypatch, [{"type": "text", "text": "x"}]
    )

    async def boom(*args, **kwargs):
        raise RuntimeError("parse exploded")

    processor.parse_document = boom

    result = await processor.process_document_complete_lightrag_api("broken.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-broken.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert "parse exploded" in doc_status["error_msg"]
    assert processor.lightrag.doc_status.index_done_calls >= 1


@pytest.mark.asyncio
async def test_lightrag_api_success_marks_processed_and_flushes(monkeypatch):
    processor = _make_lightrag_api_processor(
        monkeypatch, [{"type": "text", "text": "indexed body", "page_idx": 0}]
    )

    result = await processor.process_document_complete_lightrag_api("ok.pdf")

    assert result is True
    doc_status = processor.lightrag.doc_status.records["doc-pre-ok.pdf"]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert processor.lightrag.doc_status.index_done_calls >= 1
    assert processor.lightrag.ainsert_calls


@pytest.mark.asyncio
async def test_lightrag_api_empty_extraction_returns_false(monkeypatch):
    processor = _make_lightrag_api_processor(monkeypatch, [])

    # parse_document normally raises on empty; simulate a non-empty list of
    # non-text blocks that separate_content cannot turn into insertable text.
    async def fake_parse(*args, **kwargs):
        return (
            [{"type": "image", "img_path": "/tmp/x.png", "page_idx": 0}],
            "doc-img-only",
        )

    processor.parse_document = fake_parse

    result = await processor.process_document_complete_lightrag_api("image-only.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-image-only.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert "No text content extracted" in doc_status["error_msg"]
    assert processor.lightrag.ainsert_calls == []


@pytest.mark.asyncio
async def test_lightrag_api_insert_failure_persists_failed_doc_status(monkeypatch):
    processor = _make_lightrag_api_processor(
        monkeypatch,
        [{"type": "text", "text": "body", "page_idx": 0}],
        ainsert_side_effect=RuntimeError("storage down"),
    )

    result = await processor.process_document_complete_lightrag_api("fail.pdf")

    assert result is False
    doc_status = processor.lightrag.doc_status.records["doc-pre-fail.pdf"]
    assert doc_status["status"] == DocStatus.FAILED
    assert "storage down" in doc_status["error_msg"]


@pytest.mark.asyncio
async def test_insert_text_content_with_multimodal_content_reraises():
    class BoomRAG:
        async def ainsert(self, **kwargs):
            raise RuntimeError("ainsert failed")

    with pytest.raises(RuntimeError, match="ainsert failed"):
        await utils_module.insert_text_content_with_multimodal_content(
            BoomRAG(),
            input="hello",
            multimodal_content=[],
            file_paths="a.pdf",
            ids="doc-1",
        )


@pytest.mark.asyncio
async def test_lightrag_api_recovers_mineru_v2_plaintext_when_no_type_text(
    monkeypatch,
):
    processor = _make_lightrag_api_processor(monkeypatch, [])

    async def fake_parse_document(*args, **kwargs):
        # MinerU v2: prose lives in paragraph/title, not type=text.
        return (
            [
                {
                    "type": "title",
                    "content": {
                        "title_content": [{"type": "text", "content": "Safety"}]
                    },
                },
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "Wear protective gloves."}
                        ]
                    },
                },
            ],
            "doc-content-v2",
        )

    processor.parse_document = fake_parse_document

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is True
    assert len(processor.lightrag.ainsert_calls) == 1
    inserted = processor.lightrag.ainsert_calls[0].get("input")
    assert "Safety" in inserted
    assert "Wear protective gloves." in inserted
    doc_status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert doc_status["status"] == DocStatus.PROCESSED
