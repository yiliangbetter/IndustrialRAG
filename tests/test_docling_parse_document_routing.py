"""Tests for DoclingParser.parse_document extension routing and URL cleanup.

Routing mistakes send office/HTML through the wrong parser path or leave
downloaded temp files behind after URL ingestion.
"""

from pathlib import Path

import pytest

from raganything.parser import DoclingParser


@pytest.fixture
def parser():
    return DoclingParser()


class TestDoclingParseDocumentRouting:
    def test_routes_pdf(self, parser, tmp_path, monkeypatch):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        called = {}

        def fake_parse_pdf(file_path, output_dir, method, lang, **kwargs):
            called["args"] = (Path(file_path), output_dir, method, lang, kwargs)
            return [{"type": "text", "text": "from-pdf"}]

        monkeypatch.setattr(parser, "parse_pdf", fake_parse_pdf)

        result = parser.parse_document(pdf, method="ocr", output_dir=str(tmp_path))

        assert result == [{"type": "text", "text": "from-pdf"}]
        assert called["args"][0] == pdf
        assert called["args"][2] == "ocr"

    def test_routes_office(self, parser, tmp_path, monkeypatch):
        docx = tmp_path / "sheet.docx"
        docx.write_bytes(b"PK")
        called = {}

        def fake_parse_office(file_path, output_dir, lang, **kwargs):
            called["path"] = Path(file_path)
            called["lang"] = lang
            return [{"type": "text", "text": "from-office"}]

        monkeypatch.setattr(parser, "parse_office_doc", fake_parse_office)

        result = parser.parse_document(docx, lang="en")

        assert result[0]["text"] == "from-office"
        assert called["path"] == docx
        assert called["lang"] == "en"

    def test_routes_html(self, parser, tmp_path, monkeypatch):
        html = tmp_path / "page.html"
        html.write_text("<html></html>", encoding="utf-8")
        called = {}

        def fake_parse_html(file_path, output_dir, lang, **kwargs):
            called["path"] = Path(file_path)
            return [{"type": "text", "text": "from-html"}]

        monkeypatch.setattr(parser, "parse_html", fake_parse_html)

        result = parser.parse_document(html)

        assert result[0]["text"] == "from-html"
        assert called["path"] == html

    def test_unsupported_extension_raises(self, parser, tmp_path):
        png = tmp_path / "img.png"
        png.write_bytes(b"\x89PNG")

        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse_document(png)

    def test_missing_file_raises(self, parser, tmp_path):
        missing = tmp_path / "missing.pdf"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            parser.parse_document(missing)

    def test_url_temp_file_cleaned_up_after_success(self, parser, tmp_path, monkeypatch):
        downloaded = tmp_path / "remote.pdf"
        downloaded.write_bytes(b"%PDF-1.4\n")

        monkeypatch.setattr(parser, "_is_url", lambda path: True)
        monkeypatch.setattr(parser, "_download_file", lambda url: downloaded)
        monkeypatch.setattr(
            parser,
            "parse_pdf",
            lambda *a, **k: [{"type": "text", "text": "url-pdf"}],
        )

        result = parser.parse_document("https://example.com/remote.pdf")

        assert result[0]["text"] == "url-pdf"
        assert not downloaded.exists()

    def test_url_temp_file_cleaned_up_after_parse_error(
        self, parser, tmp_path, monkeypatch
    ):
        downloaded = tmp_path / "broken.pdf"
        downloaded.write_bytes(b"%PDF-1.4\n")

        monkeypatch.setattr(parser, "_is_url", lambda path: True)
        monkeypatch.setattr(parser, "_download_file", lambda url: downloaded)

        def boom(*a, **k):
            raise RuntimeError("parse failed")

        monkeypatch.setattr(parser, "parse_pdf", boom)

        with pytest.raises(RuntimeError, match="parse failed"):
            parser.parse_document("https://example.com/broken.pdf")

        assert not downloaded.exists()
