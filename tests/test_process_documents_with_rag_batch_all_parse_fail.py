"""Batch RAG must not invent successes when every parse failed.

process_documents_with_rag_batch still initializes LightRAG after parse,
then only walks parse_result.successful_files. An all-parse-fail folder
job should return empty rag_results and never call process_document_complete.
"""

import pytest

from raganything.batch import BatchMixin
from raganything.batch_parser import BatchProcessingResult
from raganything.callbacks import CallbackManager, ProcessingCallback


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
        self.starts = []
        self.completes = []

    def on_batch_start(self, file_count=0, **kw):
        self.starts.append(file_count)

    def on_batch_complete(self, total_files=0, successful=0, failed=0, **kw):
        self.completes.append(
            {
                "total_files": total_files,
                "successful": successful,
                "failed": failed,
            }
        )


@pytest.mark.asyncio
async def test_all_parse_failures_skip_rag_processing():
    class DummyBatch(BatchMixin):
        pass

    batch = DummyBatch()
    batch.logger = FakeLogger()
    batch.config = type(
        "Config",
        (),
        {
            "parser_output_dir": "./output",
            "parse_method": "auto",
            "max_concurrent_files": 1,
            "recursive_folder_processing": True,
        },
    )()

    parse_result = BatchProcessingResult(
        successful_files=[],
        failed_files=["a.pdf", "b.pdf"],
        total_files=2,
        processing_time=0.1,
        errors={"a.pdf": "parse error", "b.pdf": "timeout"},
        output_dir="./output",
    )

    def fake_process_documents_batch(**kwargs):
        return parse_result

    init_calls = []
    rag_calls = []

    async def fake_ensure_lightrag_initialized():
        init_calls.append(True)
        return {"success": True}

    async def fake_process_document_complete(file_path, **kwargs):
        rag_calls.append(file_path)

    batch.process_documents_batch = fake_process_documents_batch
    batch._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    batch.process_document_complete = fake_process_document_complete

    callback = RecordingCallback()
    batch.callback_manager = CallbackManager()
    batch.callback_manager.register(callback)

    result = await batch.process_documents_with_rag_batch(
        file_paths=["a.pdf", "b.pdf"],
        show_progress=False,
    )

    assert result["parse_result"] is parse_result
    assert result["rag_results"] == {}
    assert result["successful_rag_files"] == 0
    assert result["failed_rag_files"] == 0
    assert rag_calls == []
    assert init_calls == [True]
    assert callback.starts == [2]
    assert callback.completes == [
        {"total_files": 2, "successful": 0, "failed": 0},
    ]
