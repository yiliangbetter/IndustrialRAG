"""Txt/md conversion without output_dir must land next to the source file.

Open PR #125 always passes output_dir. Open PR #153 locks the Office sibling
libreoffice_output/ default. Txt/md ingest uses ReportLab instead: omitting
output_dir must write <parent>/reportlab_output/<stem>.pdf, not cwd and not a
hashed unique dir, so the follow-on MinerU parse_pdf can find the PDF.
"""

from pathlib import Path

from raganything.parser import Parser


def test_default_output_dir_is_sibling_reportlab_output(tmp_path, monkeypatch):
    src = tmp_path / "notes.txt"
    src.write_text("Line 1\nLine 2\n", encoding="utf-8")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    pdf_path = Parser.convert_text_to_pdf(src)

    expected = tmp_path / "reportlab_output" / "notes.pdf"
    assert pdf_path == expected
    assert pdf_path.exists()
    assert pdf_path.stat().st_size >= 100
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert not (cwd / "notes.pdf").exists()
    assert Parser._unique_output_dir(tmp_path, src) != pdf_path.parent


def test_provided_output_dir_is_used_without_unique_subdir(tmp_path):
    src = tmp_path / "spec.md"
    src.write_text("# Title\n\nBody.\n", encoding="utf-8")
    out = tmp_path / "explicit_out"

    pdf_path = Parser.convert_text_to_pdf(src, output_dir=str(out))

    assert pdf_path == out / "spec.pdf"
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert pdf_path.parent == Path(out)
    assert pdf_path.parent != Parser._unique_output_dir(out, src)
