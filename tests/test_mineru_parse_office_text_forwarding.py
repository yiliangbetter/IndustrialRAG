"""MinerU Office/text parsers must convert then forward method/lang/kwargs.

Office and markdown files are converted to PDF before MinerU runs. Dropping
`method`, `lang`, or mineru kwargs on that second hop silently falls back to
auto/txt and can skip OCR for scans or ignore language hints.
"""

from pathlib import Path

import pytest

from raganything.parser import MineruParser


class TestParseOfficeDocForwarding:
    def test_converts_then_forwards_parse_pdf_kwargs(self, monkeypatch, tmp_path):
        parser = MineruParser()
        converted = tmp_path / "memo.pdf"
        converted.write_bytes(b"%PDF-1.4\n")
        convert_calls = []
        parse_calls = []

        def fake_convert(cls, doc_path, output_dir=None):
            convert_calls.append({"doc_path": Path(doc_path), "output_dir": output_dir})
            return converted

        def fake_parse_pdf(
            self, pdf_path, output_dir=None, method="auto", lang=None, **kwargs
        ):
            parse_calls.append(
                {
                    "pdf_path": Path(pdf_path),
                    "output_dir": output_dir,
                    "method": method,
                    "lang": lang,
                    "kwargs": kwargs,
                }
            )
            return [{"type": "text", "text": "office body", "page_idx": 0}]

        monkeypatch.setattr(
            MineruParser, "convert_office_to_pdf", classmethod(fake_convert)
        )
        monkeypatch.setattr(MineruParser, "parse_pdf", fake_parse_pdf)

        docx = tmp_path / "memo.docx"
        docx.write_bytes(b"PK\x03\x04")
        content = parser.parse_office_doc(
            docx,
            output_dir=str(tmp_path / "out"),
            method="ocr",
            lang="ch",
            backend="pipeline",
        )

        assert content[0]["text"] == "office body"
        assert convert_calls[0]["doc_path"].resolve() == docx.resolve()
        assert convert_calls[0]["output_dir"] == str(tmp_path / "out")
        assert parse_calls[0]["pdf_path"].resolve() == converted.resolve()
        assert parse_calls[0]["method"] == "ocr"
        assert parse_calls[0]["lang"] == "ch"
        assert parse_calls[0]["kwargs"]["backend"] == "pipeline"
        assert parse_calls[0]["output_dir"] == str(tmp_path / "out")

    def test_conversion_failure_does_not_call_parse_pdf(self, monkeypatch, tmp_path):
        parser = MineruParser()
        parse_calls = []

        def fake_convert(cls, doc_path, output_dir=None):
            raise RuntimeError("LibreOffice conversion failed for memo.docx")

        def fake_parse_pdf(self, *args, **kwargs):
            parse_calls.append(True)
            return []

        monkeypatch.setattr(
            MineruParser, "convert_office_to_pdf", classmethod(fake_convert)
        )
        monkeypatch.setattr(MineruParser, "parse_pdf", fake_parse_pdf)

        with pytest.raises(RuntimeError, match="LibreOffice conversion failed"):
            parser.parse_office_doc(tmp_path / "memo.docx")

        assert parse_calls == []


class TestParseTextFileForwarding:
    def test_converts_then_forwards_parse_pdf_kwargs(self, monkeypatch, tmp_path):
        parser = MineruParser()
        converted = tmp_path / "notes.pdf"
        converted.write_bytes(b"%PDF-1.4\n")
        convert_calls = []
        parse_calls = []

        def fake_convert(cls, text_path, output_dir=None):
            convert_calls.append(
                {"text_path": Path(text_path), "output_dir": output_dir}
            )
            return converted

        def fake_parse_pdf(
            self, pdf_path, output_dir=None, method="auto", lang=None, **kwargs
        ):
            parse_calls.append(
                {
                    "pdf_path": Path(pdf_path),
                    "method": method,
                    "lang": lang,
                    "kwargs": kwargs,
                }
            )
            return [{"type": "text", "text": "md body", "page_idx": 0}]

        monkeypatch.setattr(
            MineruParser, "convert_text_to_pdf", classmethod(fake_convert)
        )
        monkeypatch.setattr(MineruParser, "parse_pdf", fake_parse_pdf)

        notes = tmp_path / "notes.md"
        notes.write_text("# title\n")
        content = parser.parse_text_file(
            notes,
            output_dir=str(tmp_path / "out"),
            method="txt",
            lang="en",
            formula=False,
        )

        assert content[0]["text"] == "md body"
        assert convert_calls[0]["text_path"].resolve() == notes.resolve()
        assert parse_calls[0]["method"] == "txt"
        assert parse_calls[0]["lang"] == "en"
        assert parse_calls[0]["kwargs"]["formula"] is False

    def test_conversion_failure_does_not_call_parse_pdf(self, monkeypatch, tmp_path):
        parser = MineruParser()
        parse_calls = []

        def fake_convert(cls, text_path, output_dir=None):
            raise FileNotFoundError("Text file does not exist")

        def fake_parse_pdf(self, *args, **kwargs):
            parse_calls.append(True)
            return []

        monkeypatch.setattr(
            MineruParser, "convert_text_to_pdf", classmethod(fake_convert)
        )
        monkeypatch.setattr(MineruParser, "parse_pdf", fake_parse_pdf)

        with pytest.raises(FileNotFoundError):
            parser.parse_text_file(tmp_path / "missing.md")

        assert parse_calls == []
