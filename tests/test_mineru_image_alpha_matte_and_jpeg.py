"""MinerU image prep must matte transparency and keep native JPEG bytes.

Scanned diagrams often arrive as transparent TIFF/GIF or as ``.jpeg`` rather
than ``.jpg``. MinerU only ingests png/jpeg/jpg, and only RGB and L are stored
as-is. Transparent pixels have to be composited onto white before the PNG
conversion; pasting the raw RGB of a transparent pixel turns the background
black and OCR reads the wrong marks. A ``.jpeg`` file is already a native
MinerU input, so converting it would change the bytes and the harvest stem.

Distinct from the mode-only RGBA/GIF checks and from uppercase suffix routing,
which stubs ``parse_image`` and never opens the raster.
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
        "_raganything_parser_image_matte_487f", module_path
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
        with Image.open(input_path) as img:
            captured["mode"] = img.mode
            captured["pixels"] = [
                img.getpixel((x, y))
                for y in range(img.size[1])
                for x in range(img.size[0])
            ]

    def fake_read(self, output_dir, name_without_suff, method="auto"):
        captured["read_name"] = name_without_suff
        captured["read_method"] = method
        return ([{"type": "text", "text": "diagram text", "page_idx": 0}], "")

    monkeypatch.setattr(MineruParser, "_run_mineru_command", classmethod(fake_run))
    monkeypatch.setattr(MineruParser, "_read_output_files", fake_read)


def test_rgba_transparency_is_matted_onto_white(monkeypatch, tmp_path):
    captured = {}
    _stub_mineru_io(monkeypatch, captured)
    image = tmp_path / "diagram.tiff"
    canvas = Image.new("RGBA", (2, 2))
    canvas.putpixel((0, 0), (0, 0, 0, 0))
    canvas.putpixel((1, 0), (255, 0, 0, 128))
    canvas.putpixel((0, 1), (0, 0, 255, 255))
    canvas.putpixel((1, 1), (10, 20, 30, 255))
    canvas.save(image, format="TIFF")

    content = MineruParser().parse_image(image, output_dir=str(tmp_path / "out"))

    assert content[0]["text"] == "diagram text"
    assert captured["suffix"] == ".png"
    assert captured["mode"] == "RGB"
    assert captured["pixels"] == [
        (255, 255, 255),
        (255, 127, 127),
        (0, 0, 255),
        (10, 20, 30),
    ]
    assert captured["method"] == "ocr"
    assert captured["read_name"] == "diagram"
    assert captured["read_method"] == "ocr"
    assert not captured["input_path"].exists()
    assert image.exists()


def test_palette_transparency_is_matted_onto_white(monkeypatch, tmp_path):
    captured = {}
    _stub_mineru_io(monkeypatch, captured)
    image = tmp_path / "legend.gif"
    canvas = Image.new("P", (2, 1))
    canvas.putpalette([0, 0, 0, 255, 0, 0] + [0] * (256 * 3 - 6))
    canvas.putpixel((0, 0), 0)
    canvas.putpixel((1, 0), 1)
    canvas.save(image, format="GIF", transparency=0)

    MineruParser().parse_image(image, output_dir=str(tmp_path / "out"))

    assert captured["suffix"] == ".png"
    assert captured["mode"] == "RGB"
    assert captured["pixels"] == [(255, 255, 255), (255, 0, 0)]
    assert captured["read_name"] == "legend"
    assert not captured["input_path"].exists()
    assert image.exists()


def test_jpeg_extension_is_sent_to_mineru_without_conversion(monkeypatch, tmp_path):
    captured = {}
    _stub_mineru_io(monkeypatch, captured)
    image = tmp_path / "photo.jpeg"
    Image.new("RGB", (4, 4), (12, 34, 56)).save(image, format="JPEG")

    MineruParser().parse_image(image, output_dir=str(tmp_path / "out"), lang="ch")

    assert captured["input_path"].resolve() == image.resolve()
    assert captured["suffix"] == ".jpeg"
    assert captured["mode"] == "RGB"
    assert captured["method"] == "ocr"
    assert captured["read_name"] == "photo"
    assert captured["read_method"] == "ocr"
    assert image.exists()
