"""Tests for PaddleOCRParser.parse_document extension routing.

Distinct from OCR line-extraction helpers: this covers the top-level
dispatcher that chooses PDF/image/office/text entrypoints or rejects
unsupported formats.
"""

from pathlib import Path

import pytest

from raganything.parser import PaddleOCRParser


@pytest.fixture
def parser():
    return PaddleOCRParser()


class TestPaddleOCRParseDocumentRouting:
    def test_routes_pdf(self, parser, tmp_path, monkeypatch):
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        called = {}

        def fake_parse_pdf(file_path, output_dir, lang=None, **kwargs):
            called["path"] = Path(file_path)
            called["output_dir"] = output_dir
            called["lang"] = lang
            called["kwargs"] = kwargs
            return [{"type": "text", "text": "from-pdf", "page_idx": 0}]

        monkeypatch.setattr(parser, "parse_pdf", fake_parse_pdf)

        result = parser.parse_document(
            pdf, method="auto", output_dir=str(tmp_path), lang="en", cls=False
        )

        assert result[0]["text"] == "from-pdf"
        assert called["path"] == pdf
        assert called["output_dir"] == str(tmp_path)
        assert called["lang"] == "en"
        assert called["kwargs"]["cls"] is False

    def test_routes_image(self, parser, tmp_path, monkeypatch):
        image = tmp_path / "photo.jpg"
        image.write_bytes(b"jpeg-bytes")
        called = {}

        def fake_parse_image(file_path, output_dir, lang=None, **kwargs):
            called["path"] = Path(file_path)
            called["lang"] = lang
            return [{"type": "text", "text": "from-image", "page_idx": 0}]

        monkeypatch.setattr(parser, "parse_image", fake_parse_image)

        result = parser.parse_document(image, lang="ch")

        assert result[0]["text"] == "from-image"
        assert called["path"] == image
        assert called["lang"] == "ch"

    def test_routes_office_via_conversion(self, parser, tmp_path, monkeypatch):
        docx = tmp_path / "report.docx"
        docx.write_bytes(b"PK")
        called = {}

        def fake_parse_office(file_path, output_dir, lang=None, **kwargs):
            called["path"] = Path(file_path)
            called["output_dir"] = output_dir
            called["lang"] = lang
            return [{"type": "text", "text": "from-office", "page_idx": 0}]

        monkeypatch.setattr(parser, "parse_office_doc", fake_parse_office)

        result = parser.parse_document(docx, output_dir=str(tmp_path), lang="en")

        assert result[0]["text"] == "from-office"
        assert called["path"] == docx
        assert called["output_dir"] == str(tmp_path)
        assert called["lang"] == "en"

    def test_routes_text_via_conversion(self, parser, tmp_path, monkeypatch):
        text = tmp_path / "notes.txt"
        text.write_text("plain text", encoding="utf-8")
        called = {}

        def fake_parse_text(file_path, output_dir, lang=None, **kwargs):
            called["path"] = Path(file_path)
            return [{"type": "text", "text": "from-text", "page_idx": 0}]

        monkeypatch.setattr(parser, "parse_text_file", fake_parse_text)

        result = parser.parse_document(text)

        assert result[0]["text"] == "from-text"
        assert called["path"] == text

    def test_unsupported_extension_raises(self, parser, tmp_path):
        weird = tmp_path / "archive.zip"
        weird.write_bytes(b"PK")

        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse_document(weird)

    def test_missing_file_raises(self, parser, tmp_path):
        missing = tmp_path / "missing.png"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            parser.parse_document(missing)

    def test_method_argument_is_ignored(self, parser, tmp_path, monkeypatch):
        """PaddleOCR parse_document intentionally discards method."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        seen = {}

        def fake_parse_pdf(file_path, output_dir, lang=None, **kwargs):
            seen["kwargs"] = kwargs
            return [{"type": "text", "text": "ok", "page_idx": 0}]

        monkeypatch.setattr(parser, "parse_pdf", fake_parse_pdf)

        parser.parse_document(pdf, method="ocr")

        assert "method" not in seen["kwargs"]
