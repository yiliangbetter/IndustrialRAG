"""MinerU JSON harvest must survive bad blocks and an unreadable markdown sidecar.

`_read_output_files` is the only bridge from MinerU's on-disk output into
ingest. A null block or a markdown file that cannot be opened must not drop
the content list, disable image-path sandboxing, or clobber caption aliases.
"""

import importlib.util
import json
from pathlib import Path


def _load_mineru_parser():
    """Load parser.py without importing the raganything package."""
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_harvest", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MineruParser


MineruParser = _load_mineru_parser()


def _write_content_list(path, items):
    path.write_text(json.dumps(items), encoding="utf-8")


def test_non_dict_blocks_do_not_drop_or_unsandbox_siblings(tmp_path):
    stem = "manual"
    image = tmp_path / "fig.png"
    image.write_bytes(b"png")
    _write_content_list(
        tmp_path / f"{stem}_content_list.json",
        [
            None,
            "stray",
            {
                "type": "image",
                "img_path": "../../secret.png",
                "img_caption": ["outside"],
            },
            {
                "type": "image",
                "img_path": "fig.png",
                "image_caption": ["Pump"],
            },
        ],
    )

    content_list, md = MineruParser._read_output_files(tmp_path, stem, method="auto")

    assert md == ""
    assert content_list[0] is None
    assert content_list[1] == "stray"
    assert content_list[2]["img_path"] == ""
    assert content_list[2]["image_caption"] == ["outside"]
    assert content_list[3]["img_path"] == str(image.resolve())
    assert content_list[3]["img_caption"] == ["Pump"]


def test_both_caption_aliases_are_kept_when_present(tmp_path):
    stem = "spec"
    _write_content_list(
        tmp_path / f"{stem}_content_list.json",
        [
            {
                "type": "image",
                "img_path": "",
                "img_caption": ["legacy"],
                "image_caption": ["canonical"],
                "img_footnote": ["old note"],
                "image_footnote": ["new note"],
            }
        ],
    )

    content_list, _md = MineruParser._read_output_files(tmp_path, stem, method="auto")
    block = content_list[0]

    assert block["img_caption"] == ["legacy"]
    assert block["image_caption"] == ["canonical"]
    assert block["img_footnote"] == ["old note"]
    assert block["image_footnote"] == ["new note"]


def test_unreadable_markdown_does_not_drop_content_list(tmp_path):
    stem = "manual"
    _write_content_list(
        tmp_path / f"{stem}_content_list.json",
        [{"type": "text", "text": "keep me"}],
    )
    # A directory named like the markdown sidecar makes open() fail.
    (tmp_path / f"{stem}.md").mkdir()

    content_list, md = MineruParser._read_output_files(tmp_path, stem, method="auto")

    assert content_list == [{"type": "text", "text": "keep me"}]
    assert md == ""
