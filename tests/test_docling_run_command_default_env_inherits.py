"""Docling CLI must inherit the process environment when env is omitted.

Custom env merge and invalid-env TypeErrors are covered on main. If the
default path starts passing env={}, PATH is wiped and the docling binary
cannot be found even when it is installed. That fails closed as a missing
binary and looks like a packaging bug.
"""

from unittest.mock import MagicMock, patch

from raganything.parser import DoclingParser


def test_omitted_env_is_none_so_subprocess_inherits_parent(tmp_path):
    parser = DoclingParser()
    pdf = tmp_path / "drawing.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        parser._run_docling_command(pdf, tmp_path / "out", "drawing")

    kwargs = mock_run.call_args.kwargs
    assert kwargs.get("env") is None
