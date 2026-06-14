from types import SimpleNamespace
from pathlib import Path

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

    monkeypatch.setattr(processor_module, "get_parser", lambda parser_name: FakeParser())
    monkeypatch.setattr(
        DummyProcessor,
        "_store_cached_result",
        fake_store_cached_result,
        raising=False,
    )

    pdf_path = tmp_path / "nested.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    content_list, doc_id = await dummy.parse_document(str(pdf_path))

    assert content_list == nested_content
    assert doc_id == dummy._generate_content_based_doc_id(nested_content[0])
    assert doc_id != compute_mdhash_id("", prefix="doc-")


@pytest.mark.asyncio
async def test_embedding_only_process_document_recovers_mineru_v2_plaintext(tmp_path):
    from raganything.processor import ProcessorMixin

    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.config = _config(tmp_path, allow_embedding_only_ingestion=True)
    dummy.logger = FakeLogger()
    captured = {}

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    async def fake_parse_document(*args, **kwargs):
        return (
            [
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "Recovered paragraph"}
                        ]
                    },
                }
            ],
            "doc-paragraph",
        )

    async def fake_insert_text_content_embedding_only(text_content, file_ref, doc_id):
        captured["text_content"] = text_content
        captured["file_ref"] = file_ref
        captured["doc_id"] = doc_id

    dummy._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    dummy.parse_document = fake_parse_document
    dummy._insert_text_content_embedding_only = fake_insert_text_content_embedding_only

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    await dummy.process_document_complete(str(pdf_path), display_stats=False)

    assert captured == {
        "text_content": "Recovered paragraph",
        "file_ref": "doc.pdf",
        "doc_id": "doc-paragraph",
    }


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

    monkeypatch.setattr(processor_module, "get_parser", lambda parser_name: FakeParser())
    monkeypatch.setattr(
        DummyProcessor,
        "_store_cached_result",
        fake_store_cached_result,
        raising=False,
    )

    html_path = tmp_path / "page.html"
    html_path.write_text("<html><body>hello</body></html>", encoding="utf-8")

    content_list, _ = await dummy.parse_document(str(html_path))

    assert content_list == [{"type": "text", "text": "html parsed", "page_idx": 0}]
    assert captured["html_path"] == html_path
    assert captured["output_dir"] == str(tmp_path / "output")


def test_office_conversion_uses_unique_output_dirs_for_same_basename(
    monkeypatch, tmp_path
):
    import raganything.parser as parser_module
    from raganything.parser import Parser

    first_doc = tmp_path / "a" / "report.docx"
    second_doc = tmp_path / "b" / "report.docx"
    first_doc.parent.mkdir()
    second_doc.parent.mkdir()
    first_doc.write_bytes(b"first")
    second_doc.write_bytes(b"second")

    def fake_run(convert_cmd, **kwargs):
        outdir = Path(convert_cmd[convert_cmd.index("--outdir") + 1])
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "report.pdf").write_bytes(b"%PDF-1.4\n" + b"x" * 200)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(parser_module.subprocess, "run", fake_run)

    output_dir = tmp_path / "out"
    first_pdf = Parser.convert_office_to_pdf(first_doc, output_dir)
    second_pdf = Parser.convert_office_to_pdf(second_doc, output_dir)

    assert first_pdf != second_pdf
    assert first_pdf.parent != second_pdf.parent
    assert first_pdf.name == "report.pdf"
    assert second_pdf.name == "report.pdf"
