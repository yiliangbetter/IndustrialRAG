"""Docling parse_pdf output-dir wiring and missing-file fail-closed.

HTML/Office unique-dir coverage lives in another PR. PDF is the default
ingest path for scanned manuals; a shared output_dir without a per-file
subdir collides same-basename PDFs and mixes harvested blocks.
"""

from pathlib import Path

import pytest

from raganything.parser import DoclingParser, Parser


def test_parse_pdf_missing_file_raises(tmp_path):
    parser = DoclingParser()
    with pytest.raises(FileNotFoundError, match="PDF file does not exist"):
        parser.parse_pdf(tmp_path / "missing.pdf")


def test_parse_pdf_uses_unique_subdir_and_forwards_kwargs(tmp_path, monkeypatch):
    parser = DoclingParser()
    pdf = tmp_path / "drawing.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    out = tmp_path / "shared_out"
    seen = {}

    def fake_run(input_path, output_dir, file_stem, **kwargs):
        seen["input_path"] = Path(input_path)
        seen["output_dir"] = Path(output_dir)
        seen["file_stem"] = file_stem
        seen["kwargs"] = kwargs
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def fake_read(output_dir, name_without_suff):
        seen["read_args"] = (Path(output_dir), name_without_suff)
        return [{"type": "text", "text": "docling pdf"}], "md"

    monkeypatch.setattr(parser, "_run_docling_command", fake_run)
    monkeypatch.setattr(parser, "_read_output_files", fake_read)

    result = parser.parse_pdf(
        pdf,
        output_dir=str(out),
        method="ocr",
        lang="en",
        image_export_mode="placeholder",
    )

    expected = Parser._unique_output_dir(out, pdf)
    assert seen["output_dir"] == expected
    assert expected.parent == out
    assert seen["file_stem"] == "drawing"
    assert seen["input_path"] == pdf
    assert seen["kwargs"]["image_export_mode"] == "placeholder"
    assert seen["read_args"] == (expected, "drawing")
    assert result == [{"type": "text", "text": "docling pdf"}]


def test_parse_pdf_default_output_is_sibling_docling_output(tmp_path, monkeypatch):
    parser = DoclingParser()
    pdf = tmp_path / "notes.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    seen = {}

    def fake_run(input_path, output_dir, file_stem, **kwargs):
        seen["output_dir"] = Path(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(parser, "_run_docling_command", fake_run)
    monkeypatch.setattr(
        parser, "_read_output_files", lambda *_a, **_k: ([{"type": "text"}], "md")
    )

    parser.parse_pdf(pdf)

    assert seen["output_dir"] == pdf.parent / "docling_output"
