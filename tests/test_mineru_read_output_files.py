"""Regression tests for MinerU output harvest and image-path sandboxing.

`_read_output_files` is the only bridge from MinerU's on-disk JSON into
ingest. Wrong subdirectory selection drops the document; missing caption
aliases break citations; unsandboxed `img_path` would read arbitrary files.
"""

import json

from raganything.parser import MineruParser


def _write_content_list(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items), encoding="utf-8")


def test_nested_subdir_scan_finds_json_regardless_of_method(tmp_path):
    stem = "manual"
    nested = tmp_path / stem / "vlm"
    image = nested / "images" / "fig-1.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    _write_content_list(
        nested / f"{stem}_content_list.json",
        [
            {
                "type": "image",
                "img_path": "images/fig-1.png",
                "img_caption": ["Front view"],
            }
        ],
    )
    (nested / f"{stem}.md").write_text("# Manual\n", encoding="utf-8")

    content_list, md = MineruParser._read_output_files(tmp_path, stem, method="auto")

    assert md == "# Manual\n"
    assert content_list[0]["image_caption"] == ["Front view"]
    assert content_list[0]["img_caption"] == ["Front view"]
    assert content_list[0]["img_path"] == str(image.resolve())


def test_v2_image_caption_is_copied_to_legacy_alias(tmp_path):
    stem = "spec"
    json_path = tmp_path / f"{stem}_content_list.json"
    _write_content_list(
        json_path,
        [
            {
                "type": "image",
                "img_path": "",
                "image_caption": ["Nameplate"],
                "image_footnote": ["See §2"],
            }
        ],
    )

    content_list, md = MineruParser._read_output_files(tmp_path, stem, method="auto")

    assert md == ""
    assert content_list[0]["img_caption"] == ["Nameplate"]
    assert content_list[0]["img_footnote"] == ["See §2"]
    assert content_list[0]["image_caption"] == ["Nameplate"]


def test_path_traversal_in_image_fields_is_cleared(tmp_path):
    stem = "doc"
    nested = tmp_path / stem / "pipeline"
    nested.mkdir(parents=True)
    _write_content_list(
        nested / f"{stem}_content_list.json",
        [
            {
                "type": "image",
                "img_path": "../../outside.png",
            },
            {
                "type": "table",
                "table_img_path": "safe.png",
            },
        ],
    )
    (nested / "safe.png").write_bytes(b"x")

    content_list, _ = MineruParser._read_output_files(tmp_path, stem, method="auto")

    assert content_list[0]["img_path"] == ""
    assert content_list[1]["table_img_path"] == str((nested / "safe.png").resolve())


def test_stem_subdir_without_json_does_not_use_flat_sibling(tmp_path):
    """Current contract: an empty stem folder shadows top-level JSON."""
    stem = "report"
    (tmp_path / stem).mkdir()
    _write_content_list(
        tmp_path / f"{stem}_content_list.json",
        [{"type": "text", "text": "should be ignored"}],
    )
    method_json = tmp_path / stem / "ocr" / f"{stem}_content_list.json"
    _write_content_list(method_json, [{"type": "text", "text": "from method dir"}])

    content_list, _ = MineruParser._read_output_files(tmp_path, stem, method="ocr")

    # Scan finds ocr/ because it contains the expected filename.
    assert content_list == [{"type": "text", "text": "from method dir"}]


def test_missing_json_returns_empty_content_list(tmp_path):
    stem = "empty"
    (tmp_path / f"{stem}.md").write_text("notes", encoding="utf-8")

    content_list, md = MineruParser._read_output_files(tmp_path, stem, method="auto")

    assert content_list == []
    assert md == "notes"


def test_corrupt_json_returns_empty_list_without_raising(tmp_path):
    stem = "bad"
    json_path = tmp_path / f"{stem}_content_list.json"
    json_path.write_text("{not-json", encoding="utf-8")

    content_list, md = MineruParser._read_output_files(tmp_path, stem, method="auto")

    assert content_list == []
    assert md == ""


def test_equation_img_path_is_absolutized(tmp_path):
    stem = "eq"
    eq_img = tmp_path / "eq.png"
    eq_img.write_bytes(b"png")
    _write_content_list(
        tmp_path / f"{stem}_content_list.json",
        [{"type": "equation", "equation_img_path": "eq.png", "text": "E=mc^2"}],
    )

    content_list, _ = MineruParser._read_output_files(tmp_path, stem, method="txt")

    assert content_list[0]["equation_img_path"] == str(eq_img.resolve())
    assert content_list[0]["text"] == "E=mc^2"
