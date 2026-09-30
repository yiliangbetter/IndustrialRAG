"""MinerU image paths that are symlinks must not escape the output directory.

A path string with no ``..`` can still point outside the parse output once
the symlink is followed. Harvest keeps an in-tree target and clears a target
that resolves outside, for every image field that shares the sandbox.
"""

import importlib.util
import json
from pathlib import Path


def _load_parser_module():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_symlink_f369", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MineruParser = _load_parser_module().MineruParser


def test_symlink_image_paths_cannot_escape_output_dir(tmp_path):
    stem = "manual"
    base = tmp_path / stem / "auto"
    base.mkdir(parents=True)
    safe = base / "safe.png"
    safe.write_bytes(b"png")
    alias = base / "alias.png"
    alias.symlink_to(safe)
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"secret")
    leak = base / "leak.png"
    leak.symlink_to(outside)

    content = [
        {"type": "image", "img_path": "safe.png"},
        {"type": "image", "img_path": "alias.png"},
        {
            "type": "table",
            "table_img_path": "leak.png",
            "table_body": "<table></table>",
        },
        {"type": "equation", "equation_img_path": "leak.png", "text": "E=mc^2"},
    ]
    (base / f"{stem}_content_list.json").write_text(
        json.dumps(content), encoding="utf-8"
    )

    content_list, _md = MineruParser._read_output_files(tmp_path, stem, method="auto")

    safe_resolved = str(safe.resolve())
    assert content_list[0]["img_path"] == safe_resolved
    assert content_list[1]["img_path"] == safe_resolved
    assert Path(content_list[1]["img_path"]).is_relative_to(base.resolve())
    assert content_list[2]["table_img_path"] == ""
    assert content_list[2]["table_body"] == "<table></table>"
    assert content_list[3]["equation_img_path"] == ""
    assert content_list[3]["text"] == "E=mc^2"
    assert str(outside) not in json.dumps(content_list)
