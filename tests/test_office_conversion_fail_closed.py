"""LibreOffice Office-to-PDF conversion must fail closed, not return empty output."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from raganything.parser import Parser


def _docx(tmp_path: Path) -> Path:
    doc = tmp_path / "memo.docx"
    doc.write_bytes(b"PK\x03\x04")
    return doc


def test_missing_office_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Parser.convert_office_to_pdf(tmp_path / "missing.docx")


def test_all_libreoffice_commands_missing_raises(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("no such command")

    monkeypatch.setattr("raganything.parser.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="LibreOffice conversion failed"):
        Parser.convert_office_to_pdf(_docx(tmp_path), output_dir=str(tmp_path / "out"))


def test_timeout_on_all_commands_raises(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="libreoffice", timeout=60)

    monkeypatch.setattr("raganything.parser.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="LibreOffice conversion failed"):
        Parser.convert_office_to_pdf(_docx(tmp_path), output_dir=str(tmp_path / "out"))


def test_nonzero_returncode_then_success_uses_fallback_command(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == "libreoffice":
            return SimpleNamespace(returncode=1, stdout="", stderr="busy")
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        (outdir / "memo.pdf").write_bytes(b"%PDF-1.4\n" + b"x" * 200)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("raganything.parser.subprocess.run", fake_run)

    result = Parser.convert_office_to_pdf(
        _docx(tmp_path), output_dir=str(tmp_path / "out")
    )

    assert calls[0] == "libreoffice"
    assert "soffice" in calls
    assert result.name == "memo.pdf"
    assert result.stat().st_size >= 100


def test_success_with_no_pdf_raises(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("raganything.parser.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="no PDF file generated"):
        Parser.convert_office_to_pdf(_docx(tmp_path), output_dir=str(tmp_path / "out"))


def test_tiny_pdf_is_rejected_as_corrupt(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        (outdir / "memo.pdf").write_bytes(b"%PDF")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("raganything.parser.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="empty or corrupted"):
        Parser.convert_office_to_pdf(_docx(tmp_path), output_dir=str(tmp_path / "out"))


def test_successful_conversion_copies_pdf_to_output_dir(monkeypatch, tmp_path):
    out = tmp_path / "converted"

    def fake_run(cmd, **kwargs):
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        (outdir / "memo.pdf").write_bytes(b"%PDF-1.4\n" + b"x" * 200)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("raganything.parser.subprocess.run", fake_run)

    result = Parser.convert_office_to_pdf(_docx(tmp_path), output_dir=str(out))

    assert result == out / "memo.pdf"
    assert result.read_bytes().startswith(b"%PDF-1.4")
