"""Tests for MineruParser.parse_document extension routing.

Routing mistakes send office/text/unsupported files through the wrong
MinerU entrypoint. Unsupported extensions currently fall through to PDF
parsing with a warning — regressions here cause silent mis-ingest.
"""

from pathlib import Path

import pytest

from raganything.parser import MineruParser


@pytest.fixture
def parser():
    return MineruParser()


class TestMineruParseDocumentRouting:
    def test_routes_pdf(self, parser, tmp_path, monkeypatch):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        called = {}

        def fake_parse_pdf(file_path, output_dir, method, lang, **kwargs):
            called["args"] = (Path(file_path), output_dir, method, lang, kwargs)
            return [{"type": "text", "text": "from-pdf"}]

        monkeypatch.setattr(parser, "parse_pdf", fake_parse_pdf)

        result = parser.parse_document(
            pdf, method="ocr", output_dir=str(tmp_path), lang="en", backend="pipeline"
        )

        assert result == [{"type": "text", "text": "from-pdf"}]
        assert called["args"][0] == pdf
        assert called["args"][2] == "ocr"
        assert called["args"][3] == "en"
        assert called["args"][4]["backend"] == "pipeline"

    def test_routes_image(self, parser, tmp_path, monkeypatch):
        image = tmp_path / "scan.png"
        image.write_bytes(b"\x89PNG\r\n")
        called = {}

        def fake_parse_image(file_path, output_dir, lang, **kwargs):
            called["path"] = Path(file_path)
            called["output_dir"] = output_dir
            called["lang"] = lang
            called["kwargs"] = kwargs
            return [{"type": "text", "text": "from-image"}]

        monkeypatch.setattr(parser, "parse_image", fake_parse_image)

        result = parser.parse_document(
            image, output_dir=str(tmp_path), lang="ch", formula=False
        )

        assert result[0]["text"] == "from-image"
        assert called["path"] == image
        assert called["output_dir"] == str(tmp_path)
        assert called["lang"] == "ch"
        assert called["kwargs"]["formula"] is False

    def test_routes_office_with_warning(self, parser, tmp_path, monkeypatch, caplog):
        docx = tmp_path / "manual.docx"
        docx.write_bytes(b"PK")
        called = {}

        def fake_parse_office(file_path, output_dir, lang, method="auto", **kwargs):
            called["path"] = Path(file_path)
            called["method"] = method
            called["lang"] = lang
            return [{"type": "text", "text": "from-office"}]

        monkeypatch.setattr(parser, "parse_office_doc", fake_parse_office)

        with caplog.at_level("WARNING"):
            result = parser.parse_document(docx, method="txt", lang="en")

        assert result[0]["text"] == "from-office"
        assert called["path"] == docx
        assert called["method"] == "txt"
        assert called["lang"] == "en"
        assert any(
            "Office document detected" in message and ".docx" in message
            for message in caplog.messages
        )

    def test_routes_text(self, parser, tmp_path, monkeypatch):
        text = tmp_path / "notes.md"
        text.write_text("# hello", encoding="utf-8")
        called = {}

        def fake_parse_text(file_path, output_dir, lang, method="auto", **kwargs):
            called["path"] = Path(file_path)
            called["method"] = method
            return [{"type": "text", "text": "from-text"}]

        monkeypatch.setattr(parser, "parse_text_file", fake_parse_text)

        result = parser.parse_document(text, method="ocr")

        assert result[0]["text"] == "from-text"
        assert called["path"] == text
        assert called["method"] == "ocr"

    def test_unsupported_extension_falls_through_to_pdf(
        self, parser, tmp_path, monkeypatch, caplog
    ):
        weird = tmp_path / "payload.xyz"
        weird.write_bytes(b"not-a-pdf")
        called = {}

        def fake_parse_pdf(file_path, output_dir, method, lang, **kwargs):
            called["path"] = Path(file_path)
            called["method"] = method
            return [{"type": "text", "text": "treated-as-pdf"}]

        monkeypatch.setattr(parser, "parse_pdf", fake_parse_pdf)

        with caplog.at_level("WARNING"):
            result = parser.parse_document(weird, method="auto")

        assert result[0]["text"] == "treated-as-pdf"
        assert called["path"] == weird
        assert called["method"] == "auto"
        assert any(
            "Unsupported file extension" in message
            and ".xyz" in message
            and "attempting to parse as PDF" in message
            for message in caplog.messages
        )

    def test_missing_file_raises(self, parser, tmp_path):
        missing = tmp_path / "missing.pdf"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            parser.parse_document(missing)
