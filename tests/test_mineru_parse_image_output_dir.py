"""MinerU parse_image output-dir isolation.

PDF unique/omitted dirs are covered by open PRs #151/#165. Image OCR is a
separate entrypoint: scans of the same basename (photo.png from two lines)
must not share a harvest folder, and omitting output_dir must land next to
the image instead of hashing into a plant-wide output tree.
"""

from pathlib import Path

from raganything.parser import MineruParser, Parser


def _stub_mineru(parser, monkeypatch, captured):
    monkeypatch.setattr(parser, "_run_mineru_command", lambda *a, **k: None)

    def fake_read(output_dir, file_stem, method="auto"):
        captured.append(
            {
                "output_dir": Path(output_dir),
                "stem": file_stem,
                "method": method,
            }
        )
        return [{"type": "text", "text": "ocr"}], "md"

    monkeypatch.setattr(parser, "_read_output_files", fake_read)


def test_parse_image_default_output_is_sibling_mineru_output(tmp_path, monkeypatch):
    parser = MineruParser()
    image = tmp_path / "nameplate.png"
    image.write_bytes(b"\x89PNG\r\n")
    captured = []
    _stub_mineru(parser, monkeypatch, captured)

    result = parser.parse_image(image)

    assert result == [{"type": "text", "text": "ocr"}]
    assert captured[0]["output_dir"] == image.parent / "mineru_output"
    assert captured[0]["stem"] == "nameplate"
    assert captured[0]["method"] == "ocr"
    assert captured[0]["output_dir"] != Parser._unique_output_dir(image.parent, image)


def test_parse_image_same_basename_files_get_distinct_output_dirs(
    tmp_path, monkeypatch
):
    parser = MineruParser()
    line_a = tmp_path / "line_a"
    line_b = tmp_path / "line_b"
    line_a.mkdir()
    line_b.mkdir()
    img_a = line_a / "photo.png"
    img_b = line_b / "photo.png"
    img_a.write_bytes(b"\x89PNG\r\na\n")
    img_b.write_bytes(b"\x89PNG\r\nb\n")
    out = tmp_path / "shared_out"
    captured = []
    _stub_mineru(parser, monkeypatch, captured)

    parser.parse_image(img_a, output_dir=str(out))
    parser.parse_image(img_b, output_dir=str(out))

    dirs = [item["output_dir"] for item in captured]
    assert len(dirs) == 2
    assert dirs[0] != dirs[1]
    assert dirs[0].parent == out
    assert dirs[1].parent == out
    assert dirs[0].name.startswith("photo_")
    assert dirs[1].name.startswith("photo_")
    assert dirs[0] == Parser._unique_output_dir(out, img_a)
    assert dirs[1] == Parser._unique_output_dir(out, img_b)
    assert all(item["method"] == "ocr" for item in captured)
