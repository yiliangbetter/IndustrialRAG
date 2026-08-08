"""Tests for LibreOffice command-candidate fallback logging.

Missing `libreoffice` should try `soffice` without a spurious WARNING; exhausting
all candidates must still fail closed with a user-visible warning.
"""

import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from raganything.parser import Parser


def _make_docx(tmp_path: Path) -> Path:
    doc = tmp_path / "memo.docx"
    doc.write_bytes(b"PK\x03\x04fake")
    return doc


class TestLibreOfficeCommandFallback:
    def test_file_not_found_on_first_candidate_falls_back_to_soffice(
        self, monkeypatch, tmp_path, caplog
    ):
        doc = _make_docx(tmp_path)
        out = tmp_path / "pdf_out"
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd[0])
            if cmd[0] == "libreoffice":
                raise FileNotFoundError("libreoffice")
            # soffice success: create expected PDF in --outdir (>=100 bytes)
            outdir = Path(cmd[cmd.index("--outdir") + 1])
            (outdir / f"{doc.stem}.pdf").write_bytes(b"%PDF-1.4\n" + b"0" * 120)
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("raganything.parser.sys.platform", "linux")
        monkeypatch.setattr(subprocess, "run", fake_run)

        with caplog.at_level(logging.DEBUG, logger="raganything.parser"):
            pdf_path = Parser.convert_office_to_pdf(doc, output_dir=str(out))

        assert pdf_path.exists()
        assert pdf_path.suffix == ".pdf"
        assert calls == ["libreoffice", "soffice"]
        assert any(
            "trying next candidate" in r.message for r in caplog.records
        )
        assert not any(
            r.levelno >= logging.WARNING and "libreoffice" in r.message.lower()
            for r in caplog.records
            if "not found" in r.message.lower()
        )

    def test_all_candidates_missing_warns_and_raises(
        self, monkeypatch, tmp_path, caplog
    ):
        doc = _make_docx(tmp_path)

        def fake_run(cmd, **kwargs):
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr("raganything.parser.sys.platform", "linux")
        monkeypatch.setattr(subprocess, "run", fake_run)

        with caplog.at_level(logging.DEBUG, logger="raganything.parser"):
            with pytest.raises(RuntimeError, match="LibreOffice conversion failed"):
                Parser.convert_office_to_pdf(doc, output_dir=str(tmp_path / "out"))

        warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "not found" in r.message.lower()
        ]
        assert len(warnings) == 1
        assert "soffice" in warnings[0].message
