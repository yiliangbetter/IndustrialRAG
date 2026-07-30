"""Regression tests for BatchMixin.process_documents_with_rag_batch."""

import pytest

from raganything.batch import BatchMixin
from raganything.batch_parser import BatchProcessingResult
from raganything.callbacks import CallbackManager, ProcessingCallback


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.infos = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg))

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))

    def debug(self, *args, **kwargs):
        pass


class RecordingBatchCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_batch_start(self, file_count=0, **kw):
        self.events.append(("batch_start", file_count))

    def on_batch_complete(
        self, total_files=0, successful=0, failed=0, duration_seconds=0.0, **kw
    ):
        self.events.append(
            ("batch_complete", total_files, successful, failed, duration_seconds)
        )


def _make_batch(tmp_path, parse_result=None, fail_rag_paths=None):
    class DummyBatch(BatchMixin):
        pass

    batch = DummyBatch()
    batch.logger = FakeLogger()
    batch.config = type(
        "Config",
        (),
        {
            "parser": "mineru",
            "parser_output_dir": str(tmp_path / "out"),
            "parse_method": "auto",
            "recursive_folder_processing": True,
            "max_concurrent_files": 2,
        },
    )()

    manager = CallbackManager()
    callback = RecordingBatchCallback()
    manager.register(callback)
    batch.callback_manager = manager

    if parse_result is None:
        parse_result = BatchProcessingResult(
            successful_files=[],
            failed_files=[],
            total_files=0,
            processing_time=0.0,
            errors={},
            output_dir=str(tmp_path / "out"),
        )

    fail_rag_paths = set(fail_rag_paths or [])
    rag_calls = []

    def fake_process_documents_batch(**kwargs):
        return parse_result

    async def fake_ensure():
        return {"success": True}

    async def fake_process_document_complete(
        file_path, output_dir=None, parse_method=None, **kwargs
    ):
        rag_calls.append(
            {
                "file_path": file_path,
                "output_dir": output_dir,
                "parse_method": parse_method,
            }
        )
        if file_path in fail_rag_paths:
            raise RuntimeError(f"rag fail:{file_path}")

    batch.process_documents_batch = fake_process_documents_batch
    batch._ensure_lightrag_initialized = fake_ensure
    batch.process_document_complete = fake_process_document_complete
    return batch, callback, rag_calls


@pytest.mark.asyncio
async def test_process_documents_with_rag_batch_emits_callbacks(tmp_path):
    ok = str(tmp_path / "ok.pdf")
    parse_result = BatchProcessingResult(
        successful_files=[ok],
        failed_files=[],
        total_files=1,
        processing_time=0.1,
        errors={},
        output_dir=str(tmp_path / "out"),
    )
    batch, callback, rag_calls = _make_batch(tmp_path, parse_result=parse_result)

    result = await batch.process_documents_with_rag_batch(
        [ok],
        output_dir=str(tmp_path / "out"),
        parse_method="ocr",
        show_progress=False,
    )

    assert rag_calls == [
        {
            "file_path": ok,
            "output_dir": str(tmp_path / "out"),
            "parse_method": "ocr",
        }
    ]
    assert result["successful_rag_files"] == 1
    assert result["failed_rag_files"] == 0
    assert result["rag_results"][ok]["status"] == "success"
    assert result["parse_result"] is parse_result
    assert result["total_processing_time"] >= 0

    assert callback.events[0] == ("batch_start", 1)
    assert callback.events[1][0] == "batch_complete"
    assert callback.events[1][1:] == (1, 1, 0, callback.events[1][4])
    assert callback.events[1][4] >= 0


@pytest.mark.asyncio
async def test_process_documents_with_rag_batch_isolates_rag_failures(tmp_path):
    ok = str(tmp_path / "ok.pdf")
    bad = str(tmp_path / "bad.pdf")
    parse_result = BatchProcessingResult(
        successful_files=[ok, bad],
        failed_files=[],
        total_files=2,
        processing_time=0.2,
        errors={},
        output_dir=str(tmp_path / "out"),
    )
    batch, callback, rag_calls = _make_batch(
        tmp_path, parse_result=parse_result, fail_rag_paths={bad}
    )

    result = await batch.process_documents_with_rag_batch(
        [ok, bad],
        output_dir=str(tmp_path / "out"),
        show_progress=False,
    )

    assert [c["file_path"] for c in rag_calls] == [ok, bad]
    assert result["successful_rag_files"] == 1
    assert result["failed_rag_files"] == 1
    assert result["rag_results"][ok] == {"status": "success", "processed": True}
    assert result["rag_results"][bad]["status"] == "failed"
    assert result["rag_results"][bad]["processed"] is False
    assert "rag fail" in result["rag_results"][bad]["error"]
    assert any(bad in err for err in batch.logger.errors)

    assert callback.events[0] == ("batch_start", 2)
    assert callback.events[1][0] == "batch_complete"
    assert callback.events[1][1:4] == (2, 1, 1)


@pytest.mark.asyncio
async def test_process_documents_with_rag_batch_skips_rag_when_parse_empty(tmp_path):
    skipped = str(tmp_path / "skip.pdf")
    parse_result = BatchProcessingResult(
        successful_files=[],
        failed_files=[skipped],
        total_files=1,
        processing_time=0.05,
        errors={skipped: "parse failed"},
        output_dir=str(tmp_path / "out"),
    )
    batch, callback, rag_calls = _make_batch(tmp_path, parse_result=parse_result)

    result = await batch.process_documents_with_rag_batch(
        [skipped],
        show_progress=False,
    )

    assert rag_calls == []
    assert result["rag_results"] == {}
    assert result["successful_rag_files"] == 0
    assert result["failed_rag_files"] == 0
    assert callback.events == [
        ("batch_start", 1),
        ("batch_complete", 1, 0, 0, callback.events[1][4]),
    ]


@pytest.mark.asyncio
async def test_process_documents_with_rag_batch_init_failure_raises(tmp_path):
    ok = str(tmp_path / "ok.pdf")
    parse_result = BatchProcessingResult(
        successful_files=[ok],
        failed_files=[],
        total_files=1,
        processing_time=0.01,
        errors={},
        output_dir=str(tmp_path / "out"),
    )
    batch, callback, rag_calls = _make_batch(tmp_path, parse_result=parse_result)

    async def fail_init():
        return {"success": False, "error": "no llm"}

    batch._ensure_lightrag_initialized = fail_init

    with pytest.raises(RuntimeError, match="LightRAG initialization failed"):
        await batch.process_documents_with_rag_batch([ok], show_progress=False)

    assert rag_calls == []
    # start fires before parse/init; complete must not fire on hard init failure
    assert callback.events == [("batch_start", 1)]
