import pytest

from raganything.callbacks import CallbackManager
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
    pass


def make_embedding_only_processor(tmp_path):
    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.callback_manager = CallbackManager()
    processor.callback_manager.enable_event_log(True)
    processor.config = type(
        "Config",
        (),
        {
            "allow_embedding_only_ingestion": True,
            "content_format": "minerU",
            "display_content_stats": False,
            "parse_method": "auto",
            "parser_output_dir": str(tmp_path / "output"),
            "use_full_path": False,
        },
    )()

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fail_multimodal_processing(*args, **kwargs):
        raise AssertionError("embedding-only ingestion must skip multimodal processing")

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    processor._process_multimodal_content = fail_multimodal_processing
    return processor


@pytest.mark.asyncio
async def test_process_document_complete_embedding_only_emits_document_complete(
    tmp_path,
):
    processor = make_embedding_only_processor(tmp_path)
    inserted = []

    async def fake_parse_document(
        file_path, output_dir, parse_method, display_stats, **kw
    ):
        return [{"type": "text", "text": "hello world", "page_idx": 0}], "doc-from-parse"

    async def fake_insert_text_content_embedding_only(text_content, file_ref, doc_id):
        inserted.append(
            {"text_content": text_content, "file_ref": file_ref, "doc_id": doc_id}
        )

    processor.parse_document = fake_parse_document
    processor._insert_text_content_embedding_only = (
        fake_insert_text_content_embedding_only
    )

    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4\n")

    await processor.process_document_complete(str(source), file_name="custom-ref.pdf")

    assert inserted == [
        {
            "text_content": "hello world",
            "file_ref": "custom-ref.pdf",
            "doc_id": "doc-from-parse",
        }
    ]
    document_events = [
        event
        for event in processor.callback_manager.event_log
        if event.event_type == "on_document_complete"
    ]
    assert len(document_events) == 1
    assert document_events[0].file_path == str(source)
    assert document_events[0].doc_id == "doc-from-parse"


@pytest.mark.asyncio
async def test_insert_content_list_embedding_only_emits_document_complete(tmp_path):
    processor = make_embedding_only_processor(tmp_path)
    inserted = []

    async def fake_insert_text_content_embedding_only(text_content, file_ref, doc_id):
        inserted.append(
            {"text_content": text_content, "file_ref": file_ref, "doc_id": doc_id}
        )

    processor._insert_text_content_embedding_only = (
        fake_insert_text_content_embedding_only
    )
    processor._generate_content_based_doc_id = (
        lambda content_list: "doc-from-content-list"
    )

    await processor.insert_content_list(
        [{"type": "text", "text": "content list text", "page_idx": 0}],
        file_path=str(tmp_path / "nested" / "doc_content_list_v2.json"),
    )

    assert inserted == [
        {
            "text_content": "content list text",
            "file_ref": "doc_content_list_v2.json",
            "doc_id": "doc-from-content-list",
        }
    ]
    document_events = [
        event
        for event in processor.callback_manager.event_log
        if event.event_type == "on_document_complete"
    ]
    assert len(document_events) == 1
    assert document_events[0].file_path.endswith("doc_content_list_v2.json")
    assert document_events[0].doc_id == "doc-from-content-list"
