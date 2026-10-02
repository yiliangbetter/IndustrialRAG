"""MinerU image paths that normalize back inside the output dir must be kept.

The harvest sandbox resolves each path and rejects anything outside the parse
output. A string check for ``..`` or a blanket rejection of absolute paths
would drop figures MinerU already stored inside that directory, so retrieval
and VLM would lose the image.
"""

import json
from pathlib import Path


def _load_mineru_parser():
    """Load MineruParser without importing the heavy raganything package."""
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_intree_paths", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MineruParser


MineruParser = _load_mineru_parser()


def test_resolved_intree_image_paths_are_kept_and_outside_absolute_is_cleared(
    tmp_path,
):
    stem = "manual"
    figure = tmp_path / "images" / "fig.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"png")
    outside = tmp_path.parent / "secret-nameplate.png"
    inside = figure.resolve()

    content_path = tmp_path / f"{stem}_content_list.json"
    content_path.write_text(
        json.dumps(
            [
                {"type": "image", "img_path": "images/../images/fig.png"},
                {"type": "image", "img_path": "./images/fig.png"},
                {"type": "table", "table_img_path": str(inside)},
                {"type": "equation", "equation_img_path": str(outside.resolve())},
            ]
        ),
        encoding="utf-8",
    )

    content_list, _md = MineruParser._read_output_files(tmp_path, stem, method="auto")

    assert content_list[0]["img_path"] == str(inside)
    assert content_list[1]["img_path"] == str(inside)
    assert content_list[2]["table_img_path"] == str(inside)
    assert content_list[3]["equation_img_path"] == ""
