"""Regression tests for Parser.convert_text_to_pdf (reportlab path).

Txt/md ingest converts to PDF before MinerU. Encoding, empty files, and
unsupported extensions are easy to break and drop documents from the index.
"""

import pytest

from raganything.parser import Parser


class TestConvertTextToPdfValidation:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "gone.txt"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            Parser.convert_text_to_pdf(missing, output_dir=str(tmp_path))

    def test_unsupported_extension_raises(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported text format"):
            Parser.convert_text_to_pdf(csv_path, output_dir=str(tmp_path))


class TestConvertTextToPdfReportlab:
    def test_plain_text_writes_nonempty_pdf(self, tmp_path):
        src = tmp_path / "notes.txt"
        src.write_text("Line 1\nLine 2 with <angle> & ampersand\n", encoding="utf-8")

        pdf = Parser.convert_text_to_pdf(src, output_dir=str(tmp_path / "out"))

        assert pdf.exists()
        assert pdf.name == "notes.pdf"
        assert pdf.stat().st_size >= 100
        assert pdf.read_bytes().startswith(b"%PDF")

    def test_markdown_headers_write_nonempty_pdf(self, tmp_path):
        src = tmp_path / "spec.md"
        src.write_text(
            "# Title\n\nA paragraph.\n## Section\nMore text.\n", encoding="utf-8"
        )

        pdf = Parser.convert_text_to_pdf(src, output_dir=str(tmp_path / "out"))

        assert pdf.exists()
        assert pdf.stat().st_size >= 100
        assert pdf.read_bytes().startswith(b"%PDF")

    def test_empty_txt_still_produces_pdf(self, tmp_path):
        src = tmp_path / "empty.txt"
        src.write_text("", encoding="utf-8")

        pdf = Parser.convert_text_to_pdf(src, output_dir=str(tmp_path / "out"))

        assert pdf.exists()
        assert pdf.stat().st_size >= 100

    def test_latin1_fallback_when_not_utf8(self, tmp_path):
        src = tmp_path / "latin1.txt"
        src.write_bytes(b"caf\xe9 notes\n")

        pdf = Parser.convert_text_to_pdf(src, output_dir=str(tmp_path / "out"))

        assert pdf.exists()
        assert pdf.read_bytes().startswith(b"%PDF")
