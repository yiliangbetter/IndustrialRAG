"""LibreOffice conversion without output_dir must land next to the source file.

parse_office_doc forwards output_dir through convert_office_to_pdf. When the
caller omits it, PDFs must go to <parent>/libreoffice_output/<stem>.pdf — not
the process cwd or a hashed unique dir — so subsequent parse_pdf can find them.
"""

from pathlib import Path
from unittest.mock import MagicMock

from raganything.parser import Parser


def test_default_output_dir_is_sibling_libreoffice_output(tmp_path, monkeypatch):
    src = tmp_path / "manual.docx"
    src.write_bytes(b"PK\x03\x04" + b"\x00" * 32)

    def fake_run(cmd, **kwargs):
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        pdf = outdir / "manual.pdf"
        pdf.write_bytes(b"%PDF-1.4\n" + b"\x00" * 200)
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        result.stdout = ""
        return result

    monkeypatch.setattr("raganything.parser.subprocess.run", fake_run)

    pdf_path = Parser.convert_office_to_pdf(src)

    expected = tmp_path / "libreoffice_output" / "manual.pdf"
    assert pdf_path == expected
    assert pdf_path.exists()
    assert pdf_path.stat().st_size >= 100
    assert pdf_path.read_bytes().startswith(b"%PDF")
