"""Regression tests for macOS LibreOffice discovery in Office→PDF conversion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from raganything.parser import MineruParser


def test_macos_libreoffice_app_is_preferred(monkeypatch, tmp_path):
    doc_path = tmp_path / "sample.docx"
    doc_path.write_text("office content")
    output_dir = tmp_path / "output"
    mac_soffice = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    original_is_file = Path.is_file

    monkeypatch.setattr("raganything.parser.sys.platform", "darwin")
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: str(path) == mac_soffice or original_is_file(path),
    )

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        generated_pdf = Path(command[command.index("--outdir") + 1]) / "sample.pdf"
        generated_pdf.write_bytes(b"%PDF-1.4\n" + b"0" * 128)
        return MagicMock(returncode=0, stderr="")

    monkeypatch.setattr("raganything.parser.subprocess.run", fake_run)

    result = MineruParser.convert_office_to_pdf(doc_path, str(output_dir))

    assert commands[0][0] == mac_soffice
    assert len(commands) == 1
    assert result == output_dir / "sample.pdf"
    assert result.stat().st_size > 100
