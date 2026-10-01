"""MinerU must normalize scan modes that PNG cannot store as-is.

Print manuals often arrive as grayscale or CMYK TIFF. MinerU only ingests
png/jpeg/jpg, and only RGB and L are written through unchanged. Other modes
have to become RGB PNGs; otherwise OCR raises or receives a file it cannot read.
"""

from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _load_mineru_parser():
    """Load MineruParser without importing the raganything package."""
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_image_modes_d8a7", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MineruParser


MineruParser = _load_mineru_parser()


def _stub_mineru_io(monkeypatch, captured):
    def fake_run(
        cls,
        input_path,
        output_dir,
        method="auto",
        lang=None,
        **kwargs,
    ):
        input_path = Path(input_path)
        captured["input_path"] = input_path
        captured["method"] = method
        captured["suffix"] = input_path.suffix.lower()
        captured["png_signature"] = input_path.read_bytes()[:8]
        with Image.open(input_path) as img:
            captured["mode"] = img.mode

    def fake_read(self, output_dir, name_without_suff, method="auto"):
        captured["read_name"] = name_without_suff
        captured["read_method"] = method
        return ([{"type": "text", "text": "scan text", "page_idx": 0}], "")

    monkeypatch.setattr(MineruParser, "_run_mineru_command", classmethod(fake_run))
    monkeypatch.setattr(MineruParser, "_read_output_files", fake_read)


def test_cmyk_tiff_is_converted_to_rgb_png(monkeypatch, tmp_path):
    captured = {}
    _stub_mineru_io(monkeypatch, captured)
    image = tmp_path / "plate.tiff"
    Image.new("CMYK", (8, 8), (10, 20, 30, 40)).save(image, format="TIFF")

    content = MineruParser().parse_image(image, output_dir=str(tmp_path / "out"))

    assert content[0]["text"] == "scan text"
    assert captured["suffix"] == ".png"
    assert captured["mode"] == "RGB"
    assert captured["png_signature"] == b"\x89PNG\r\n\x1a\n"
    assert captured["method"] == "ocr"
    assert captured["read_name"] == "plate"
    assert captured["read_method"] == "ocr"
    assert not captured["input_path"].exists()
    assert image.exists()


def test_grayscale_tiff_stays_mode_l(monkeypatch, tmp_path):
    captured = {}
    _stub_mineru_io(monkeypatch, captured)
    image = tmp_path / "scan.tif"
    Image.new("L", (8, 8), 40).save(image, format="TIFF")

    MineruParser().parse_image(image, output_dir=str(tmp_path / "out"))

    assert captured["suffix"] == ".png"
    assert captured["mode"] == "L"
    assert captured["read_name"] == "scan"
    assert not captured["input_path"].exists()


def test_bilevel_tiff_is_expanded_to_rgb(monkeypatch, tmp_path):
    captured = {}
    _stub_mineru_io(monkeypatch, captured)
    image = tmp_path / "bitmap.tiff"
    Image.new("1", (8, 8), 0).save(image, format="TIFF")

    MineruParser().parse_image(image, output_dir=str(tmp_path / "out"))

    assert captured["suffix"] == ".png"
    assert captured["mode"] == "RGB"
    assert not captured["input_path"].exists()
