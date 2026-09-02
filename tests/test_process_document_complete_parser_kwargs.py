"""process_document_complete must forward parser kwargs and config defaults.

OCR language, page range, device, and backend change what is extracted. If
those kwargs stay on the ingest wrapper, parse_document silently uses the
wrong method while split kwargs still look correct on insert.
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


def _dummy():
    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.logger = FakeLogger()
    dummy.config = type(
        "Config",
        (),
        {
            "parser_output_dir": "./parsed-default",
            "parse_method": "ocr",
            "display_content_stats": False,
            "use_full_path": False,
            "allow_embedding_only_ingestion": False,
            "content_format": "minerU",
        },
    )()
    dummy.lightrag = object()
    dummy.parse_calls = []
    dummy.insert_kwargs = []

    async def fake_init():
        return {"success": True}

    async def fake_parse(*args, **kwargs):
        dummy.parse_calls.append({"args": args, "kwargs": kwargs})
        return ([{"type": "text", "text": "hello world"}], "doc-from-content")

    async def fake_mark(doc_id):
        return None

    dummy._ensure_lightrag_initialized = fake_init
    dummy.parse_document = fake_parse
    dummy._mark_multimodal_processing_complete = fake_mark
    return dummy


@pytest.mark.asyncio
async def test_process_document_complete_forwards_parser_kwargs_not_split(
    monkeypatch,
):
    dummy = _dummy()
    import raganything.processor as processor_module

    async def fake_insert(lightrag, **kwargs):
        dummy.insert_kwargs.append(kwargs)

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await dummy.process_document_complete(
        "/tmp/manual.pdf",
        split_by_character="\n",
        split_by_character_only=True,
        lang="ch",
        start_page=2,
        end_page=5,
        device="cpu",
        backend="pipeline",
        formula=True,
        table=False,
        source="local",
    )

    assert len(dummy.parse_calls) == 1
    parse_kwargs = dummy.parse_calls[0]["kwargs"]
    assert parse_kwargs["lang"] == "ch"
    assert parse_kwargs["start_page"] == 2
    assert parse_kwargs["end_page"] == 5
    assert parse_kwargs["device"] == "cpu"
    assert parse_kwargs["backend"] == "pipeline"
    assert parse_kwargs["formula"] is True
    assert parse_kwargs["table"] is False
    assert parse_kwargs["source"] == "local"
    assert "split_by_character" not in parse_kwargs
    assert "split_by_character_only" not in parse_kwargs

    assert dummy.insert_kwargs[0]["split_by_character"] == "\n"
    assert dummy.insert_kwargs[0]["split_by_character_only"] is True


@pytest.mark.asyncio
async def test_process_document_complete_inherits_parse_defaults(monkeypatch):
    dummy = _dummy()
    import raganything.processor as processor_module

    async def fake_insert(lightrag, **kwargs):
        dummy.insert_kwargs.append(kwargs)

    monkeypatch.setattr(processor_module, "insert_text_content", fake_insert)

    await dummy.process_document_complete("/tmp/manual.pdf")

    args = dummy.parse_calls[0]["args"]
    # file_path, output_dir, parse_method, display_stats
    assert args[1] == "./parsed-default"
    assert args[2] == "ocr"
    assert args[3] is False
