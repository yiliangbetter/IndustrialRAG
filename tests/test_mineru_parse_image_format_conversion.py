"""MinerU parse_image format gates and Pillow conversion before OCR.

MinerU 2.0 natively accepts only png/jpeg/jpg. Other raster formats must be
converted to PNG or OCR silently drops pages. These tests lock that conversion
and fail-closed behavior without running the mineru binary.
"""

import builtins
from pathlib import Path

import pytest

from raganything.parser import MineruExecutionError, MineruParser

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _stub_mineru_io(monkeypatch, captured):
    """Replace MinerU subprocess + output harvest with in-memory fakes."""

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
        captured["output_dir"] = Path(output_dir)
        captured["method"] = method
        captured["lang"] = lang
        captured["kwargs"] = kwargs
        captured["input_existed"] = input_path.exists()
        captured["input_suffix"] = input_path.suffix.lower()
        if input_path.exists() and input_path.suffix.lower() == ".png":
            with Image.open(input_path) as img:
                captured["converted_mode"] = img.mode

    def fake_read(self, output_dir, name_without_suff, method="auto"):
        captured["read_name"] = name_without_suff
        captured["read_method"] = method
        captured["read_output_dir"] = Path(output_dir)
        return (
            [{"type": "text", "text": "ocr text", "page_idx": 0}],
            "unused.md",
        )

    monkeypatch.setattr(MineruParser, "_run_mineru_command", classmethod(fake_run))
    monkeypatch.setattr(MineruParser, "_read_output_files", fake_read)


def _write_rgb_image(path: Path, fmt: str) -> None:
    Image.new("RGB", (8, 8), (12, 34, 56)).save(path, format=fmt)


class TestMineruParseImageFailClosed:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        parser = MineruParser()
        missing = tmp_path / "gone.png"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            parser.parse_image(missing)

    def test_unsupported_format_raises_value_error(self, tmp_path):
        parser = MineruParser()
        svg = tmp_path / "diagram.svg"
        svg.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>")
        with pytest.raises(ValueError, match="Unsupported image format"):
            parser.parse_image(svg)


class TestMineruParseImageNativeFormats:
    def test_png_is_sent_to_mineru_without_conversion(self, monkeypatch, tmp_path):
        captured = {}
        _stub_mineru_io(monkeypatch, captured)
        image = tmp_path / "native.png"
        _write_rgb_image(image, "PNG")

        parser = MineruParser()
        content = parser.parse_image(image, output_dir=str(tmp_path / "out"), lang="en")

        assert content[0]["text"] == "ocr text"
        assert captured["input_path"].resolve() == image.resolve()
        assert captured["input_existed"] is True
        assert captured["method"] == "ocr"
        assert captured["lang"] == "en"
        assert captured["read_name"] == "native"
        assert captured["read_method"] == "ocr"
        assert image.exists()
        assert captured["output_dir"].parent == tmp_path / "out"
        assert captured["output_dir"].name.startswith("native_")

    def test_jpeg_is_not_converted(self, monkeypatch, tmp_path):
        captured = {}
        _stub_mineru_io(monkeypatch, captured)
        image = tmp_path / "photo.jpg"
        _write_rgb_image(image, "JPEG")

        parser = MineruParser()
        parser.parse_image(image, output_dir=str(tmp_path / "out"))

        assert captured["input_path"].resolve() == image.resolve()
        assert captured["input_suffix"] == ".jpg"
        assert captured["method"] == "ocr"


class TestMineruParseImageConversion:
    @pytest.mark.parametrize(
        ("suffix", "fmt"),
        [
            (".bmp", "BMP"),
            (".tiff", "TIFF"),
            (".tif", "TIFF"),
            (".gif", "GIF"),
            (".webp", "WEBP"),
        ],
    )
    def test_non_native_formats_are_converted_to_png(
        self, monkeypatch, tmp_path, suffix, fmt
    ):
        captured = {}
        _stub_mineru_io(monkeypatch, captured)
        image = tmp_path / f"scan{suffix}"
        _write_rgb_image(image, fmt)

        parser = MineruParser()
        parser.parse_image(image, output_dir=str(tmp_path / "out"))

        assert captured["input_existed"] is True
        assert captured["input_suffix"] == ".png"
        assert captured["input_path"].name.endswith("_converted.png")
        assert captured["method"] == "ocr"
        # Output harvest must keep the original stem, not the converted filename.
        assert captured["read_name"] == "scan"
        # Temporary conversion artifact is cleaned up after OCR.
        assert not captured["input_path"].exists()

    def test_rgba_is_flattened_to_rgb_png(self, monkeypatch, tmp_path):
        captured = {}
        _stub_mineru_io(monkeypatch, captured)
        image = tmp_path / "transparent.webp"
        Image.new("RGBA", (8, 8), (255, 0, 0, 128)).save(image, format="WEBP")

        parser = MineruParser()
        parser.parse_image(image, output_dir=str(tmp_path / "out"))

        assert captured["converted_mode"] in {"RGB", "L"}
        assert "A" not in captured["converted_mode"]

    def test_palette_image_is_converted(self, monkeypatch, tmp_path):
        captured = {}
        _stub_mineru_io(monkeypatch, captured)
        image = tmp_path / "indexed.gif"
        Image.new("P", (8, 8)).save(image, format="GIF")

        parser = MineruParser()
        parser.parse_image(image, output_dir=str(tmp_path / "out"))

        assert captured["input_suffix"] == ".png"
        assert captured["converted_mode"] in {"RGB", "L"}

    def test_missing_pillow_raises_runtime_error(self, monkeypatch, tmp_path):
        image = tmp_path / "scan.bmp"
        _write_rgb_image(image, "BMP")

        real_import = builtins.__import__

        def block_pil(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("No module named 'PIL'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", block_pil)

        parser = MineruParser()
        with pytest.raises(RuntimeError, match="PIL/Pillow is required"):
            parser.parse_image(image)

    def test_conversion_failure_raises_runtime_error(self, monkeypatch, tmp_path):
        image = tmp_path / "broken.bmp"
        image.write_bytes(b"not-an-image")

        parser = MineruParser()
        with pytest.raises(RuntimeError, match="Failed to convert image"):
            parser.parse_image(image)

    def test_mineru_execution_error_propagates(self, monkeypatch, tmp_path):
        image = tmp_path / "native.png"
        _write_rgb_image(image, "PNG")

        def boom(cls, *args, **kwargs):
            raise MineruExecutionError(1, "mineru crashed")

        monkeypatch.setattr(MineruParser, "_run_mineru_command", classmethod(boom))

        parser = MineruParser()
        with pytest.raises(MineruExecutionError, match="mineru crashed"):
            parser.parse_image(image, output_dir=str(tmp_path / "out"))
