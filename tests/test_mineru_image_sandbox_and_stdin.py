"""MinerU image-path sandbox and subprocess stdin contract.

Table and equation image fields share the same traversal check as img_path.
An empty or null path must stay empty so it is not rewritten to the output
directory. The MinerU child must also be launched with stdin closed so a
prompt on stdin cannot hang a parse that has no timeout.
"""

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import patch


def _load_parser_module():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_sandbox_bb2d", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_parser = _load_parser_module()
MineruParser = _parser.MineruParser


class _Pipe:
    def __init__(self):
        self.closed = False

    def readline(self):
        return ""

    def close(self):
        self.closed = True


class _Process:
    def __init__(self):
        self.stdout = _Pipe()
        self.stderr = _Pipe()

    def poll(self):
        if self.stdout.closed and self.stderr.closed:
            return 0
        return None

    def wait(self):
        return 0


def test_table_and_equation_image_paths_are_sandboxed(tmp_path):
    stem = "manual"
    base = tmp_path / stem / "auto"
    base.mkdir(parents=True)
    safe_table = base / "table.png"
    safe_table.write_bytes(b"png")
    safe_eq = base / "eq.png"
    safe_eq.write_bytes(b"png")
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"secret")

    content = [
        {
            "type": "table",
            "table_img_path": "../secret.png",
            "table_body": "<table></table>",
        },
        {
            "type": "equation",
            "equation_img_path": str(outside),
            "text": "E=mc^2",
        },
        {
            "type": "table",
            "table_img_path": "table.png",
        },
        {
            "type": "equation",
            "equation_img_path": "eq.png",
        },
        {
            "type": "image",
            "img_path": "",
            "table_img_path": None,
        },
    ]
    (base / f"{stem}_content_list.json").write_text(
        json.dumps(content), encoding="utf-8"
    )

    content_list, _md = MineruParser._read_output_files(tmp_path, stem, method="auto")

    assert content_list[0]["table_img_path"] == ""
    assert content_list[0]["table_body"] == "<table></table>"
    assert content_list[1]["equation_img_path"] == ""
    assert content_list[1]["text"] == "E=mc^2"
    assert content_list[2]["table_img_path"] == str(safe_table.resolve())
    assert content_list[3]["equation_img_path"] == str(safe_eq.resolve())
    assert content_list[4]["img_path"] == ""
    assert content_list[4]["table_img_path"] is None


def test_mineru_subprocess_stdin_is_devnull():
    process = _Process()

    with patch.object(_parser.subprocess, "Popen", return_value=process) as popen:
        with patch.object(_parser.time, "sleep", return_value=None):
            MineruParser._run_mineru_command("scan.pdf", "out")

    assert popen.call_count == 1
    kwargs = popen.call_args.kwargs
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.PIPE
    assert popen.call_args.args[0][0] == "mineru"
