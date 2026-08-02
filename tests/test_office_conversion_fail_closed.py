"""Regression tests for Parser.convert_office_to_pdf fail-closed checks."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from raganything.parser import Parser


def _touch_docx(path: Path) -> Path:
    path.write_bytes(b"PK\x03\x04fake-docx")
    return path


class TestConvertOfficeToPdfFailClosed:
    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            Parser.convert_office_to_pdf(tmp_path / "missing.docx")

    def test_timeout_exhaustion_raises_runtime_error(self, tmp_path, monkeypatch):
        doc = _touch_docx(tmp_path / "report.docx")

        def boom(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=60)

        monkeypatch.setattr(subprocess, "run", boom)
        with pytest.raises(RuntimeError, match="LibreOffice conversion failed"):
            Parser.convert_office_to_pdf(doc, output_dir=str(tmp_path / "out"))

    def test_nonzero_returncode_raises(self, tmp_path, monkeypatch):
        doc = _touch_docx(tmp_path / "report.docx")

        def fail(*args, **kwargs):
            result = MagicMock()
            result.returncode = 1
            result.stderr = "convert error"
            result.stdout = ""
            return result

        monkeypatch.setattr(subprocess, "run", fail)
        with pytest.raises(RuntimeError, match="LibreOffice conversion failed"):
            Parser.convert_office_to_pdf(doc, output_dir=str(tmp_path / "out"))

    def test_success_but_no_pdf_raises(self, tmp_path, monkeypatch):
        doc = _touch_docx(tmp_path / "report.docx")

        def ok_no_pdf(*args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            result.stdout = ""
            return result

        monkeypatch.setattr(subprocess, "run", ok_no_pdf)
        with pytest.raises(RuntimeError, match="no PDF file generated"):
            Parser.convert_office_to_pdf(doc, output_dir=str(tmp_path / "out"))

    def test_tiny_pdf_rejected(self, tmp_path, monkeypatch):
        doc = _touch_docx(tmp_path / "report.docx")

        def ok_tiny_pdf(cmd, **kwargs):
            # LibreOffice --outdir is the temp dir argument in the command list.
            outdir = Path(cmd[cmd.index("--outdir") + 1])
            (outdir / "report.pdf").write_bytes(b"%PDF-tiny")
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            result.stdout = ""
            return result

        monkeypatch.setattr(subprocess, "run", ok_tiny_pdf)
        with pytest.raises(RuntimeError, match="empty or corrupted"):
            Parser.convert_office_to_pdf(doc, output_dir=str(tmp_path / "out"))

    def test_valid_pdf_is_copied_to_output(self, tmp_path, monkeypatch):
        doc = _touch_docx(tmp_path / "report.docx")
        out_dir = tmp_path / "out"

        def ok_pdf(cmd, **kwargs):
            outdir = Path(cmd[cmd.index("--outdir") + 1])
            (outdir / "report.pdf").write_bytes(b"%PDF-1.4\n" + b"0" * 200)
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            result.stdout = ""
            return result

        monkeypatch.setattr(subprocess, "run", ok_pdf)
        pdf_path = Parser.convert_office_to_pdf(doc, output_dir=str(out_dir))
        assert pdf_path == out_dir / "report.pdf"
        assert pdf_path.exists()
        assert pdf_path.stat().st_size >= 100
