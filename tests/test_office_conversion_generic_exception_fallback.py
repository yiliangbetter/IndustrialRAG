"""LibreOffice conversion must try the next binary after a generic exception.

FileNotFoundError fallback is covered elsewhere. A RuntimeError/OSError from
the first command must not abort conversion before ``soffice`` is attempted.
"""

from pathlib import Path
from unittest.mock import MagicMock

from raganything.parser import Parser


def test_generic_exception_on_libreoffice_falls_through_to_soffice(
    tmp_path, monkeypatch
):
    src = tmp_path / "manual.docx"
    src.write_bytes(b"PK\x03\x04" + b"\x00" * 32)
    out_dir = tmp_path / "pdf-out"

    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd[0])
        if cmd[0] == "libreoffice":
            raise RuntimeError("unexpected soffice wrapper crash")
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        pdf = outdir / "manual.pdf"
        pdf.write_bytes(b"%PDF-1.4\n" + b"\x00" * 200)
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        result.stdout = ""
        return result

    monkeypatch.setattr("raganything.parser.subprocess.run", fake_run)

    pdf_path = Parser.convert_office_to_pdf(src, output_dir=str(out_dir))

    assert pdf_path.exists()
    assert pdf_path.name == "manual.pdf"
    assert pdf_path.stat().st_size >= 100
    assert commands == ["libreoffice", "soffice"]
