"""Regression tests for MinerU `_read_output_files` output normalization.

MinerU layout and field names differ across versions and backends. This helper
is the only place that:
- discovers nested `file_stem/<subdir>/` outputs,
- aliases `img_caption` ↔ `image_caption` (and footnotes),
- rewrites relative image paths to absolute ones,
- clears path-traversal attempts outside the images base directory.

Regressions here drop figures, break multimodal grounding, or reopen local file
read via crafted relative paths in parser JSON.
"""

import json
from pathlib import Path

from raganything.parser import MineruParser


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_read_output_files_scans_nested_subdir_and_fixes_image_paths(tmp_path):
    stem = "manual"
    nested = tmp_path / stem / "vlm"
    images = nested / "images"
    images.mkdir(parents=True)
    image_file = images / "fig1.png"
    image_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    (nested / f"{stem}.md").write_text("# Manual\n", encoding="utf-8")
    _write_json(
        nested / f"{stem}_content_list.json",
        [
            {
                "type": "image",
                "img_path": "images/fig1.png",
                "img_caption": ["Figure one"],
            }
        ],
    )

    content_list, md_content = MineruParser._read_output_files(tmp_path, stem)

    assert md_content == "# Manual\n"
    assert len(content_list) == 1
    item = content_list[0]
    assert Path(item["img_path"]) == image_file.resolve()
    # MinerU 1.x caption field is mirrored to the 2.0 canonical name.
    assert item["img_caption"] == ["Figure one"]
    assert item["image_caption"] == ["Figure one"]


def test_read_output_files_aliases_mineru_v2_caption_fields(tmp_path):
    stem = "report"
    _write_json(
        tmp_path / f"{stem}_content_list.json",
        [
            {
                "type": "image",
                "img_path": "",
                "image_caption": ["v2 caption"],
                "image_footnote": ["v2 note"],
            }
        ],
    )
    (tmp_path / f"{stem}.md").write_text("body", encoding="utf-8")

    content_list, md_content = MineruParser._read_output_files(tmp_path, stem)

    assert md_content == "body"
    item = content_list[0]
    assert item["image_caption"] == ["v2 caption"]
    assert item["img_caption"] == ["v2 caption"]
    assert item["image_footnote"] == ["v2 note"]
    assert item["img_footnote"] == ["v2 note"]


def test_read_output_files_clears_path_traversal_image_paths(tmp_path):
    stem = "doc"
    output_dir = tmp_path / "parser_out"
    outside_dir = tmp_path / "outside"
    output_dir.mkdir()
    outside_dir.mkdir()
    outside = outside_dir / "secret.png"
    outside.write_bytes(b"secret")

    _write_json(
        output_dir / f"{stem}_content_list.json",
        [
            {
                "type": "image",
                "img_path": "../outside/secret.png",
            },
            {
                "type": "table",
                "table_img_path": "../outside/secret.png",
            },
        ],
    )

    content_list, _ = MineruParser._read_output_files(output_dir, stem)

    assert content_list[0]["img_path"] == ""
    assert content_list[1]["table_img_path"] == ""
    assert outside.exists()


def test_read_output_files_falls_back_to_method_subdir_when_scan_misses(tmp_path):
    stem = "paper"
    method_dir = tmp_path / stem / "ocr"
    method_dir.mkdir(parents=True)
    (method_dir / f"{stem}.md").write_text("ocr md", encoding="utf-8")
    _write_json(
        method_dir / f"{stem}_content_list.json",
        [{"type": "text", "text": "from method fallback"}],
    )
    # Nested stem dir exists but has no recognizable content_list JSON yet.
    (tmp_path / stem / "empty_backend").mkdir()

    content_list, md_content = MineruParser._read_output_files(
        tmp_path, stem, method="ocr"
    )

    assert md_content == "ocr md"
    assert content_list == [{"type": "text", "text": "from method fallback"}]
