"""MinerU subprocess timeout, CLI flags, and fail-closed execution errors.

Merged timeout handling (#254) kills hung mineru processes. Command flags
decide OCR/lang/backend selection. These paths had no tests on main.
"""

from unittest.mock import MagicMock, patch

import pytest

from raganything.parser import MineruExecutionError, MineruParser


def _ready_process(return_code=0):
    process = MagicMock()
    process.poll.return_value = return_code
    process.wait.return_value = return_code
    process.stdout.readline.return_value = ""
    process.stderr.readline.return_value = ""
    return process


def _hung_process():
    process = MagicMock()
    process.poll.return_value = None
    process.wait.return_value = -9
    process.stdout.readline.return_value = ""
    process.stderr.readline.return_value = ""
    return process


@patch("raganything.parser.time.sleep", lambda *_args, **_kwargs: None)
@patch("subprocess.Popen")
def test_mineru_timeout_kills_hung_process(mock_popen):
    process = _hung_process()
    mock_popen.return_value = process
    parser = MineruParser()
    monotonic_calls = {"n": 0}

    def fake_monotonic():
        monotonic_calls["n"] += 1
        # First call is start_time; later checks jump past timeout.
        return 0.0 if monotonic_calls["n"] == 1 else 10.0

    with patch("raganything.parser.time.monotonic", fake_monotonic):
        with pytest.raises(RuntimeError, match="did not finish within 1s") as excinfo:
            parser._run_mineru_command("doc.pdf", "out", timeout=1)

    # TimeoutError is raised after kill, then wrapped by the generic handler.
    assert isinstance(excinfo.value.__cause__, TimeoutError)
    process.kill.assert_called_once()
    process.wait.assert_called()


@patch("subprocess.Popen")
def test_mineru_command_includes_optional_flags(mock_popen):
    mock_popen.return_value = _ready_process()
    parser = MineruParser()

    parser._run_mineru_command(
        "scan.pdf",
        "out",
        method="ocr",
        lang="ch",
        backend="pipeline",
        source="modelscope",
        start_page=0,
        end_page=3,
        formula=False,
        table=False,
        device="cpu",
        vlm_url="http://127.0.0.1:30000",
    )

    cmd = mock_popen.call_args[0][0]
    assert cmd[:6] == ["mineru", "-p", "scan.pdf", "-o", "out", "-m"]
    assert "ocr" in cmd
    assert cmd[cmd.index("-l") + 1] == "ch"
    assert cmd[cmd.index("-b") + 1] == "pipeline"
    assert cmd[cmd.index("--source") + 1] == "modelscope"
    assert cmd[cmd.index("-s") + 1] == "0"
    assert cmd[cmd.index("-e") + 1] == "3"
    assert cmd[cmd.index("-f") + 1] == "false"
    assert cmd[cmd.index("-t") + 1] == "false"
    assert cmd[cmd.index("-d") + 1] == "cpu"
    assert cmd[cmd.index("-u") + 1] == "http://127.0.0.1:30000"


@patch("subprocess.Popen")
def test_mineru_default_formula_and_table_flags_are_omitted(mock_popen):
    mock_popen.return_value = _ready_process()
    parser = MineruParser()

    parser._run_mineru_command("scan.pdf", "out", method="auto")

    cmd = mock_popen.call_args[0][0]
    assert "-f" not in cmd
    assert "-t" not in cmd
    assert "-l" not in cmd
    assert "-b" not in cmd


@patch("subprocess.Popen")
def test_mineru_nonzero_exit_raises_execution_error(mock_popen):
    mock_popen.return_value = _ready_process(return_code=2)
    parser = MineruParser()

    with pytest.raises(MineruExecutionError) as excinfo:
        parser._run_mineru_command("scan.pdf", "out")

    assert excinfo.value.return_code == 2


@patch("subprocess.Popen")
def test_mineru_missing_binary_fails_closed(mock_popen):
    mock_popen.side_effect = FileNotFoundError("mineru")
    parser = MineruParser()

    with pytest.raises(RuntimeError, match="mineru command not found"):
        parser._run_mineru_command("scan.pdf", "out")


def test_parse_pdf_remaps_vlm_and_hybrid_backends_for_output_scan(
    tmp_path, monkeypatch
):
    parser = MineruParser()
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    captured = []

    monkeypatch.setattr(parser, "_run_mineru_command", lambda *a, **k: None)

    def fake_read(output_dir, file_stem, method="auto"):
        captured.append(
            {
                "output_dir": output_dir,
                "file_stem": file_stem,
                "method": method,
            }
        )
        return [{"type": "text", "text": method}], "md"

    monkeypatch.setattr(parser, "_read_output_files", fake_read)

    vlm = parser.parse_pdf(
        pdf, output_dir=str(tmp_path / "out"), method="ocr", backend="vlm-http-client"
    )
    hybrid = parser.parse_pdf(
        pdf, output_dir=str(tmp_path / "out"), method="txt", backend="hybrid-auto"
    )
    pipeline = parser.parse_pdf(
        pdf, output_dir=str(tmp_path / "out"), method="ocr", backend="pipeline"
    )
    missing = parser.parse_pdf(
        pdf, output_dir=str(tmp_path / "out"), method="auto", backend=None
    )

    assert [item["text"] for item in vlm] == ["vlm"]
    assert [item["text"] for item in hybrid] == ["hybrid_auto"]
    assert [item["text"] for item in pipeline] == ["ocr"]
    assert [item["text"] for item in missing] == ["auto"]
    assert [row["method"] for row in captured] == [
        "vlm",
        "hybrid_auto",
        "ocr",
        "auto",
    ]
    assert all(row["file_stem"] == "paper" for row in captured)
