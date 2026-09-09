"""Docling CLI invocation: output layout, dual --to flags, missing binary.

`_run_docling_command` is mocked by HTML/PDF method tests. If the real command
drops `--to json` or `--to md`, harvest finds nothing. A missing binary must
surface as RuntimeError so ingest can fail closed instead of FileNotFoundError.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from raganything.parser import DoclingParser


def test_creates_stem_docling_dir_and_requests_json_and_md(tmp_path):
    parser = DoclingParser()
    output_dir = tmp_path / "out"
    input_path = tmp_path / "drawing.pdf"
    input_path.write_bytes(b"%PDF-1.4\n")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        parser._run_docling_command(input_path, output_dir, "drawing")

    expected_out = output_dir / "drawing" / "docling"
    assert expected_out.is_dir()

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "docling"
    assert cmd[1:3] == ["--output", str(expected_out)]
    # Both JSON (harvest) and markdown must be requested.
    to_flags = [cmd[i + 1] for i, token in enumerate(cmd) if token == "--to"]
    assert to_flags == ["json", "md"]
    assert str(input_path) in cmd


def test_missing_binary_raises_runtime_error(tmp_path):
    parser = DoclingParser()
    output_dir = tmp_path / "out"
    input_path = tmp_path / "doc.pdf"
    input_path.write_bytes(b"%PDF-1.4\n")

    with patch("subprocess.run", side_effect=FileNotFoundError("docling")):
        with pytest.raises(RuntimeError, match="docling command not found"):
            parser._run_docling_command(input_path, output_dir, "doc")


def test_nonzero_exit_reraises_called_process_error(tmp_path):
    parser = DoclingParser()
    output_dir = tmp_path / "out"
    input_path = tmp_path / "doc.pdf"
    input_path.write_bytes(b"%PDF-1.4\n")
    original = subprocess.CalledProcessError(
        returncode=2, cmd=["docling"], stderr="model missing"
    )

    with patch("subprocess.run", side_effect=original):
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            parser._run_docling_command(input_path, output_dir, "doc")

    assert excinfo.value is original
    assert excinfo.value.returncode == 2
