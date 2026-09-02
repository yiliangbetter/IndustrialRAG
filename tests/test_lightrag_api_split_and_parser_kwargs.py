"""LightRAG API ingest must keep parser kwargs and split options on the right calls.

API ingest is the LightRAG scanner path. Dropping lang/page-range into the
wrong call, or omitting split/scheme on ainsert, silently indexes the wrong
pages or chunks.
"""

import pytest

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


class FakeLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _patch_pipeline_status(monkeypatch):
    from lightrag.kg import shared_storage

    pipeline = {"history_messages": []}

    async def fake_get_namespace_data(name):
        return pipeline

    monkeypatch.setattr(shared_storage, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(shared_storage, "get_pipeline_status_lock", lambda: FakeLock())
    return pipeline


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
            "parser_output_dir": "./api-output",
            "parse_method": "txt",
            "display_content_stats": False,
        },
    )()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatusStorage()},
    )()
    processor.parse_calls = []
    return processor


@pytest.mark.asyncio
async def test_lightrag_api_forwards_split_kwargs_to_insert_not_parse(monkeypatch):
    import raganything.processor as processor_module

    _patch_pipeline_status(monkeypatch)
    processor = _make_processor()
    captured = {}

    async def fake_ensure():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        processor.parse_calls.append({"args": args, "kwargs": kwargs})
        return (
            [{"type": "text", "text": "intro", "page_idx": 0}],
            "doc-content",
        )

    async def fake_insert(lightrag, input=None, **kwargs):
        captured["input"] = input
        captured.update(kwargs)

    processor._ensure_lightrag_initialized = fake_ensure
    processor.parse_document = fake_parse
    monkeypatch.setattr(
        processor_module,
        "insert_text_content_with_multimodal_content",
        fake_insert,
    )

    result = await processor.process_document_complete_lightrag_api(
        "sample.pdf",
        doc_id="doc-content",
        scheme_name="plant-a",
        split_by_character="\n",
        split_by_character_only=True,
        lang="ch",
        start_page=1,
        end_page=3,
    )

    assert result is True
    parse_kwargs = processor.parse_calls[0]["kwargs"]
    assert parse_kwargs["lang"] == "ch"
    assert parse_kwargs["start_page"] == 1
    assert parse_kwargs["end_page"] == 3
    assert "split_by_character" not in parse_kwargs
    assert "scheme_name" not in parse_kwargs

    assert captured["input"] == "intro"
    assert captured["ids"] == "doc-content"
    assert captured["file_paths"] == "sample.pdf"
    assert captured["scheme_name"] == "plant-a"
    assert captured["split_by_character"] == "\n"
    assert captured["split_by_character_only"] is True


@pytest.mark.asyncio
async def test_lightrag_api_inherits_parse_defaults_when_args_omitted(monkeypatch):
    import raganything.processor as processor_module

    _patch_pipeline_status(monkeypatch)
    processor = _make_processor()

    async def fake_ensure():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        processor.parse_calls.append({"args": args, "kwargs": kwargs})
        return ([{"type": "text", "text": "body", "page_idx": 0}], "doc-1")

    async def fake_insert(lightrag, input=None, **kwargs):
        return None

    processor._ensure_lightrag_initialized = fake_ensure
    processor.parse_document = fake_parse
    monkeypatch.setattr(
        processor_module,
        "insert_text_content_with_multimodal_content",
        fake_insert,
    )

    result = await processor.process_document_complete_lightrag_api("sample.pdf")

    assert result is True
    args = processor.parse_calls[0]["args"]
    assert args[1] == "./api-output"
    assert args[2] == "txt"
    assert args[3] is False
