from pathlib import Path
from types import SimpleNamespace

import pytest
from lightrag.utils import compute_mdhash_id


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _config(tmp_path, **overrides):
    values = {
        "parser": "mineru",
        "parser_output_dir": str(tmp_path / "output"),
        "parse_method": "auto",
        "display_content_stats": False,
        "use_full_path": False,
        "allow_embedding_only_ingestion": False,
        "content_format": "minerU",
    }
    values.update(overrides)
    return type("Config", (), values)()


@pytest.mark.asyncio
async def test_parse_document_hashes_normalized_mineru_v2_blocks(monkeypatch, tmp_path):
    import raganything.processor as processor_module

    nested_content = [[{"type": "text", "text": "nested text", "page_idx": 0}]]

    class FakeParser:
        def parse_pdf(self, **kwargs):
            return nested_content

    class DummyProcessor(processor_module.ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.config = _config(tmp_path)
    dummy.logger = FakeLogger()
    dummy.parse_cache = None

    async def fake_store_cached_result(*args, **kwargs):
        return None

    monkeypatch.setattr(
        processor_module, "get_parser", lambda parser_name: FakeParser()
    )
    monkeypatch.setattr(
        DummyProcessor,
        "_store_cached_result",
        fake_store_cached_result,
        raising=False,
    )

    pdf_path = tmp_path / "nested.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    content_list, doc_id = await dummy.parse_document(str(pdf_path))

    assert content_list == nested_content[0]
    assert doc_id == dummy._generate_content_based_doc_id(nested_content[0])
    assert doc_id != compute_mdhash_id("", prefix="doc-")


@pytest.mark.asyncio
async def test_parse_document_routes_docling_html_to_parse_html(monkeypatch, tmp_path):
    import raganything.processor as processor_module

    captured = {}

    class FakeParser:
        def parse_html(self, **kwargs):
            captured.update(kwargs)
            return [{"type": "text", "text": "html parsed", "page_idx": 0}]

        def parse_office_doc(self, **kwargs):
            raise AssertionError("HTML must not be routed through parse_office_doc")

    class DummyProcessor(processor_module.ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.config = _config(tmp_path, parser="docling")
    dummy.logger = FakeLogger()
    dummy.parse_cache = None

    async def fake_store_cached_result(*args, **kwargs):
        return None

    monkeypatch.setattr(
        processor_module, "get_parser", lambda parser_name: FakeParser()
    )
    monkeypatch.setattr(
        DummyProcessor,
        "_store_cached_result",
        fake_store_cached_result,
        raising=False,
    )

    html_path = tmp_path / "page.html"
    html_path.write_text("<html><body>hi</body></html>", encoding="utf-8")

    content_list, _doc_id = await dummy.parse_document(str(html_path))

    assert content_list[0]["text"] == "html parsed"
    assert Path(captured["html_path"]) == html_path.resolve()


@pytest.mark.asyncio
async def test_embedding_only_empty_text_raises(tmp_path):
    from raganything.processor import ProcessorMixin

    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.config = _config(tmp_path, allow_embedding_only_ingestion=True)
    dummy.logger = FakeLogger()
    dummy.lightrag = SimpleNamespace()

    with pytest.raises(ValueError, match="No text content extracted"):
        await dummy._insert_text_content_embedding_only("", "empty.pdf", "doc-empty")


@pytest.mark.asyncio
async def test_insert_content_list_empty_raises(tmp_path):
    from raganything.processor import ProcessorMixin

    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.config = _config(tmp_path)
    dummy.logger = FakeLogger()

    async def fake_ensure():
        return {"success": True}

    dummy._ensure_lightrag_initialized = fake_ensure

    with pytest.raises(ValueError, match="No text or multimodal content"):
        await dummy.insert_content_list([])


def test_conversion_output_dir_is_unique_per_source_path(tmp_path):
    from raganything.parser import Parser

    a = tmp_path / "dir_a" / "report.docx"
    b = tmp_path / "dir_b" / "report.docx"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_bytes(b"a")
    b.write_bytes(b"b")

    out = tmp_path / "shared_out"
    dir_a = Parser._conversion_output_dir(a, str(out), "libreoffice_output")
    dir_b = Parser._conversion_output_dir(b, str(out), "libreoffice_output")

    assert dir_a != dir_b
    assert dir_a.parent == out
    assert dir_b.parent == out
