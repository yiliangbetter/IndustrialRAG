"""Regression tests for Parser.convert_text_to_pdf fail-closed validation."""

from pathlib import Path

import pytest
import reportlab.platypus as platypus

from raganything.parser import Parser


class TestConvertTextToPdfValidation:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            Parser.convert_text_to_pdf(tmp_path / "missing.txt")

    def test_unsupported_suffix_raises(self, tmp_path):
        bad = tmp_path / "notes.rst"
        bad.write_text("hello", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported text format"):
            Parser.convert_text_to_pdf(bad)

    def test_encoding_fallback_reads_latin1(self, tmp_path):
        text_file = tmp_path / "latin.txt"
        # Byte 0xE9 is 'é' in latin-1 and invalid as UTF-8 lead.
        text_file.write_bytes(b"caf\xe9")
        out_dir = tmp_path / "out"
        pdf_path = Parser.convert_text_to_pdf(text_file, output_dir=str(out_dir))
        assert pdf_path.exists()
        assert pdf_path.stat().st_size >= 100
        assert pdf_path.name == "latin.pdf"

    def test_reportlab_import_error_is_runtime_error(self, tmp_path, monkeypatch):
        text_file = tmp_path / "plain.txt"
        text_file.write_text("hello", encoding="utf-8")

        import builtins

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "reportlab" or (
                isinstance(name, str) and name.startswith("reportlab.")
            ):
                raise ImportError("no reportlab")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(RuntimeError, match="reportlab is required"):
            Parser.convert_text_to_pdf(text_file, output_dir=str(tmp_path / "out"))

    def test_empty_or_tiny_pdf_rejected(self, tmp_path, monkeypatch):
        text_file = tmp_path / "plain.txt"
        text_file.write_text("hello", encoding="utf-8")
        out_dir = tmp_path / "out"

        class TinyDoc:
            def __init__(self, filename, **kwargs):
                self.filename = filename

            def build(self, story):
                Path(self.filename).write_bytes(b"%PDF-tiny")

        monkeypatch.setattr(platypus, "SimpleDocTemplate", TinyDoc)
        with pytest.raises(RuntimeError, match="empty or corrupted"):
            Parser.convert_text_to_pdf(text_file, output_dir=str(out_dir))

    def test_successful_txt_and_md_conversion(self, tmp_path):
        txt = tmp_path / "notes.txt"
        txt.write_text("Line one\n\nLine two & <tag>", encoding="utf-8")
        md = tmp_path / "readme.md"
        md.write_text("# Header\n\nParagraph", encoding="utf-8")

        txt_pdf = Parser.convert_text_to_pdf(txt, output_dir=str(tmp_path / "pdfs"))
        md_pdf = Parser.convert_text_to_pdf(md, output_dir=str(tmp_path / "pdfs"))
        assert txt_pdf.exists() and txt_pdf.stat().st_size >= 100
        assert md_pdf.exists() and md_pdf.stat().st_size >= 100
        assert txt_pdf.name == "notes.pdf"
        assert md_pdf.name == "readme.pdf"
