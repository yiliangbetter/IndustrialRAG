"""MinerU caption aliases must not overwrite a field that is already present.

MinerU 1.x and 2.x use different caption names. The reader copies a value
only when the other name is missing. If both names are present, including an
explicit empty caption, copying one over the other would replace the caption
the newer export already chose.
"""

import importlib.util
import json
from pathlib import Path


def _load_mineru_parser():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_caption_alias_conflict", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MineruParser


MineruParser = _load_mineru_parser()


def test_present_caption_aliases_are_not_overwritten(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    stem = "manual"
    payload = [
        {
            "type": "image",
            "img_path": "images/plate.png",
            "image_caption": ["Nameplate"],
            "img_caption": ["stale v1 caption"],
            "image_footnote": ["See section 2"],
            "img_footnote": ["old note"],
        },
        {
            "type": "image",
            "img_path": "images/blank.png",
            "image_caption": [],
            "img_caption": ["legacy caption"],
            "image_footnote": "",
            "img_footnote": "legacy footnote",
        },
    ]
    (output_dir / f"{stem}_content_list.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    content_list, _md = MineruParser._read_output_files(output_dir, stem, method="auto")

    conflict = content_list[0]
    assert conflict["image_caption"] == ["Nameplate"]
    assert conflict["img_caption"] == ["stale v1 caption"]
    assert conflict["image_footnote"] == ["See section 2"]
    assert conflict["img_footnote"] == ["old note"]

    explicit_empty = content_list[1]
    assert explicit_empty["image_caption"] == []
    assert explicit_empty["img_caption"] == ["legacy caption"]
    assert explicit_empty["image_footnote"] == ""
    assert explicit_empty["img_footnote"] == "legacy footnote"
