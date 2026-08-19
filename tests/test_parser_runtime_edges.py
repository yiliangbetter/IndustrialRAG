"""Parser runtime edges: MinerU errors, loopback NO_PROXY, Office fallback.

These paths fail closed in production (parse FAILED status, proxy 502 to
mineru-api, missing soffice). Tests mock subprocess only — no LibreOffice
or mineru binary is required.
"""

import contextlib
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from raganything.parser import (
    MineruExecutionError,
    MineruParser,
    Parser,
    get_parser,
)


class TestMineruExecutionError:
    def test_stores_return_code_and_error_lines(self):
        err = MineruExecutionError(2, ["CUDA OOM", "retry failed"])
        assert err.return_code == 2
        assert err.error_msg == ["CUDA OOM", "retry failed"]
        message = str(err)
        assert "return code 2" in message
        assert "CUDA OOM" in message

    def test_string_error_msg_is_preserved(self):
        err = MineruExecutionError(1, "command not found")
        assert err.error_msg == "command not found"
        assert "command not found" in str(err)


class TestGetParserDefaults:
    def test_none_and_blank_default_to_mineru(self):
        assert isinstance(get_parser(None), MineruParser)
        assert isinstance(get_parser(""), MineruParser)
        assert isinstance(get_parser("  MINERU  "), MineruParser)


def _write_pdf_in_outdir(cmd, size=200, returncode=0):
    outdir = Path(cmd[cmd.index("--outdir") + 1])
    (outdir / "converted.pdf").write_bytes(b"%PDF" + b"x" * size)
    return MagicMock(returncode=returncode, stdout="", stderr="fail")


class TestConvertOfficeToPdf:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "gone.docx"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            Parser.convert_office_to_pdf(missing, output_dir=str(tmp_path))

    def test_falls_back_from_libreoffice_to_soffice(self, tmp_path):
        doc = tmp_path / "spec.docx"
        doc.write_bytes(b"fake-office")
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd[0])
            if cmd[0] == "libreoffice":
                raise FileNotFoundError("libreoffice")
            return _write_pdf_in_outdir(cmd)

        with patch("subprocess.run", side_effect=fake_run):
            pdf = Parser.convert_office_to_pdf(doc, output_dir=str(tmp_path / "out"))

        assert calls[:2] == ["libreoffice", "soffice"]
        assert pdf.exists()
        assert pdf.stat().st_size >= 100
        assert pdf.name == "spec.pdf"

    def test_timeout_then_success_on_next_binary(self, tmp_path):
        doc = tmp_path / "spec.docx"
        doc.write_bytes(b"fake-office")

        def fake_run(cmd, **kwargs):
            if cmd[0] == "libreoffice":
                raise subprocess.TimeoutExpired(cmd, 60)
            return _write_pdf_in_outdir(cmd)

        with patch("subprocess.run", side_effect=fake_run):
            pdf = Parser.convert_office_to_pdf(doc, output_dir=str(tmp_path / "out"))
        assert pdf.exists()

    def test_empty_pdf_is_rejected(self, tmp_path):
        doc = tmp_path / "spec.docx"
        doc.write_bytes(b"fake-office")

        def fake_run(cmd, **kwargs):
            return _write_pdf_in_outdir(cmd, size=10)

        with (
            patch("subprocess.run", side_effect=fake_run),
            pytest.raises(RuntimeError, match="empty or corrupted"),
        ):
            Parser.convert_office_to_pdf(doc, output_dir=str(tmp_path / "out"))

    def test_all_binaries_missing_raises(self, tmp_path):
        doc = tmp_path / "spec.docx"
        doc.write_bytes(b"fake-office")

        with (
            patch("subprocess.run", side_effect=FileNotFoundError("missing")),
            pytest.raises(RuntimeError, match="LibreOffice conversion failed"),
        ):
            Parser.convert_office_to_pdf(doc, output_dir=str(tmp_path / "out"))


@patch("subprocess.Popen")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.mkdir")
def test_mineru_no_proxy_merges_loopback_hosts(
    mock_mkdir, mock_exists, mock_popen, monkeypatch
):
    mock_exists.return_value = True
    mock_process = MagicMock()
    mock_process.poll.return_value = 0
    mock_process.wait.return_value = 0
    mock_process.stdout.readline.return_value = ""
    mock_process.stderr.readline.return_value = ""
    mock_popen.return_value = mock_process

    monkeypatch.setenv("NO_PROXY", "example.com,127.0.0.1")
    monkeypatch.setenv("no_proxy", "example.com")

    parser = MineruParser()
    # Subprocess mock may still raise after env is assembled; we only assert Popen env.
    with contextlib.suppress(Exception):
        parser._run_mineru_command("dummy.pdf", "out")

    env = mock_popen.call_args.kwargs["env"]
    no_proxy = env["NO_PROXY"]
    assert env["no_proxy"] == no_proxy
    parts = no_proxy.split(",")
    assert parts.count("127.0.0.1") == 1
    assert "localhost" in parts
    assert "::1" in parts
    assert "example.com" in parts
    # Parent PATH must still be present so mineru can be found.
    assert env["PATH"] == os.environ["PATH"]
