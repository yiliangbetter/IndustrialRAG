"""Regression tests for MinerU Office/text → PDF method forwarding.

After LibreOffice or text-to-PDF conversion, parse_office_doc / parse_text_file
must forward method, lang, and extra kwargs into parse_pdf. A regression
silently forces method=auto and drops OCR/txt selection for manuals.
"""

import pytest

from raganything.parser import MineruParser


@pytest.mark.parametrize(
    ("conversion_method", "parse_method_name", "source_name"),
    [
        ("convert_text_to_pdf", "parse_text_file", "source.md"),
        ("convert_office_to_pdf", "parse_office_doc", "source.docx"),
    ],
)
def test_mineru_converted_documents_forward_parse_method(
    monkeypatch, tmp_path, conversion_method, parse_method_name, source_name
):
    source = tmp_path / source_name
    source.write_bytes(b"input")
    converted_pdf = tmp_path / "converted.pdf"
    converted_pdf.write_bytes(b"%PDF-1.4\n")
    captured = {}

    def fake_convert(cls, file_path, output_dir=None):
        captured["converted_from"] = file_path
        captured["convert_output_dir"] = output_dir
        return converted_pdf

    def fake_parse_pdf(self, pdf_path, output_dir=None, method="auto", lang=None, **kw):
        captured["parse_pdf"] = {
            "pdf_path": pdf_path,
            "output_dir": output_dir,
            "method": method,
            "lang": lang,
            "kwargs": kw,
        }
        return [{"type": "text", "text": "parsed"}]

    monkeypatch.setattr(MineruParser, conversion_method, classmethod(fake_convert))
    monkeypatch.setattr(MineruParser, "parse_pdf", fake_parse_pdf)

    parser = MineruParser()
    result = getattr(parser, parse_method_name)(
        source,
        output_dir=str(tmp_path / "out"),
        lang="ch",
        method="ocr",
        backend="pipeline",
    )

    assert result == [{"type": "text", "text": "parsed"}]
    assert captured["converted_from"] == source
    assert captured["convert_output_dir"] == str(tmp_path / "out")
    assert captured["parse_pdf"] == {
        "pdf_path": converted_pdf,
        "output_dir": str(tmp_path / "out"),
        "method": "ocr",
        "lang": "ch",
        "kwargs": {"backend": "pipeline"},
    }
