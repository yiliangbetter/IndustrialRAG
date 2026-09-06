"""LightRAG API ``parser=`` must switch the shared config before init/parse.

Scanner jobs pass parser per document. A no-op override leaves later parse
routing on the previous parser; a blank string must not wipe a valid default.
"""

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


def _make_processor():
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
            "parser_output_dir": "./output",
            "parse_method": "auto",
            "display_content_stats": False,
        },
    )()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatusStorage()},
    )()
    return processor


@pytest.mark.asyncio
async def test_parser_kwarg_overrides_config_before_init_failure_is_recorded():
    processor = _make_processor()

    async def fake_ensure():
        return {"success": False, "error": "docling missing"}

    processor._ensure_lightrag_initialized = fake_ensure

    result = await processor.process_document_complete_lightrag_api(
        "sample.pdf", parser="docling"
    )

    assert result is False
    assert processor.config.parser == "docling"
    status = processor.lightrag.doc_status.records["doc-pre-sample.pdf"]
    assert status["status"] == DocStatus.FAILED
    assert status["error_msg"] == "docling missing"


@pytest.mark.asyncio
async def test_blank_or_omitted_parser_leaves_configured_parser():
    processor = _make_processor()

    async def fake_ensure():
        return {"success": False, "error": "stop"}

    processor._ensure_lightrag_initialized = fake_ensure

    await processor.process_document_complete_lightrag_api("sample.pdf")
    assert processor.config.parser == "mineru"

    await processor.process_document_complete_lightrag_api("sample.pdf", parser="")
    assert processor.config.parser == "mineru"

    await processor.process_document_complete_lightrag_api("sample.pdf", parser=None)
    assert processor.config.parser == "mineru"
