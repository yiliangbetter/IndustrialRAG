"""LibreOffice command-candidate fallback logging and discovery.

Missing `libreoffice` should try `soffice` without a spurious WARNING. Exhausting
all candidates must still fail closed. macOS should prefer the app-bundle binary
when it exists.
"""

import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from raganything.parser import Parser

MAC_SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"


def _make_docx(tmp_path: Path) -> Path:
    doc = tmp_path / "memo.docx"
    doc.write_bytes(b"PK\x03\x04fake")
    return doc


def _write_pdf(cmd, stem: str) -> SimpleNamespace:
    outdir = Path(cmd[cmd.index("--outdir") + 1])
    (outdir / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n" + b"0" * 120)
    return SimpleNamespace(returncode=0, stdout="", stderr="")


class TestLibreOfficeCommandFallback:
    def test_file_not_found_on_first_candidate_falls_back_to_soffice(
        self, monkeypatch, tmp_path, caplog
    ):
        doc = _make_docx(tmp_path)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd[0])
            if cmd[0] == "libreoffice":
                raise FileNotFoundError("libreoffice")
            return _write_pdf(cmd, doc.stem)

        monkeypatch.setattr("raganything.parser.sys.platform", "linux")
        monkeypatch.setattr("raganything.parser.subprocess.run", fake_run)

        with caplog.at_level(logging.DEBUG, logger="raganything.parser"):
            pdf_path = Parser.convert_office_to_pdf(
                doc, output_dir=str(tmp_path / "pdf_out")
            )

        assert pdf_path.exists()
        assert pdf_path.suffix == ".pdf"
        assert calls == ["libreoffice", "soffice"]
        assert any("trying next candidate" in r.message for r in caplog.records)
        assert not any(
            r.levelno >= logging.WARNING
            and "not found" in r.message.lower()
            and "libreoffice" in r.message.lower()
            for r in caplog.records
        )

    def test_all_candidates_missing_warns_and_raises(
        self, monkeypatch, tmp_path, caplog
    ):
        doc = _make_docx(tmp_path)

        def fake_run(cmd, **kwargs):
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr("raganything.parser.sys.platform", "linux")
        monkeypatch.setattr("raganything.parser.subprocess.run", fake_run)

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

    def test_timeout_on_first_candidate_still_tries_soffice(
        self, monkeypatch, tmp_path
    ):
        doc = _make_docx(tmp_path)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd[0])
            if cmd[0] == "libreoffice":
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=60)
            return _write_pdf(cmd, doc.stem)

        monkeypatch.setattr("raganything.parser.sys.platform", "linux")
        monkeypatch.setattr("raganything.parser.subprocess.run", fake_run)

        pdf_path = Parser.convert_office_to_pdf(
            doc, output_dir=str(tmp_path / "pdf_out")
        )

        assert calls == ["libreoffice", "soffice"]
        assert pdf_path.exists()
        assert pdf_path.stat().st_size >= 100

    def test_macos_app_bundle_is_tried_first(self, monkeypatch, tmp_path):
        doc = _make_docx(tmp_path)
        calls = []
        original_is_file = Path.is_file

        def fake_is_file(self):
            if str(self) == MAC_SOFFICE:
                return True
            return original_is_file(self)

        def fake_run(cmd, **kwargs):
            calls.append(cmd[0])
            return _write_pdf(cmd, doc.stem)

        monkeypatch.setattr("raganything.parser.sys.platform", "darwin")
        monkeypatch.setattr(Path, "is_file", fake_is_file)
        monkeypatch.setattr("raganything.parser.subprocess.run", fake_run)

        Parser.convert_office_to_pdf(doc, output_dir=str(tmp_path / "out"))

        assert calls[0] == MAC_SOFFICE
