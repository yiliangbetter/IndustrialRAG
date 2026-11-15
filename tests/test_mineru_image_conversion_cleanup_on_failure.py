"""Converted MinerU images must not leak temp PNGs when OCR fails.

BMP, TIFF, GIF, and WebP are rewritten to a temporary PNG before MinerU
runs. Native JPEG and PNG files never create that file, and the success-path
cleanup is already covered. This locks the failure path: a MinerU command
error still deletes the temp PNG and its directory, and a cleanup OSError
does not replace the original command failure.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image


def _load_parser_module():
    """Load parser.py without importing the raganything package."""
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    module_name = "_raganything_parser_image_cleanup_1115"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_parser = _load_parser_module()
MineruExecutionError = _parser.MineruExecutionError
MineruParser = _parser.MineruParser


def _write_bmp(path: Path) -> None:
    Image.new("RGB", (2, 2), (9, 8, 7)).save(path, format="BMP")


def _cleanup_converted(path: Path, unlink) -> None:
    if path.exists():
        unlink(path)
    if path.parent.exists():
        try:
            path.parent.rmdir()
        except OSError:
            pass


def test_converted_png_is_removed_when_mineru_fails(monkeypatch, tmp_path):
    image = tmp_path / "scan.bmp"
    _write_bmp(image)
    calls = []

    def boom(input_path, output_dir, method="auto", lang=None, **kwargs):
        calls.append(
            {
                "input_path": Path(input_path),
                "output_dir": Path(output_dir),
                "method": method,
                "lang": lang,
                "kwargs": kwargs,
            }
        )
        raise MineruExecutionError(2, "ocr failed")

    parser = MineruParser()
    monkeypatch.setattr(parser, "_run_mineru_command", boom)

    with pytest.raises(MineruExecutionError, match="ocr failed") as excinfo:
        parser.parse_image(
            image,
            output_dir=str(tmp_path / "out"),
            lang="ch",
            backend="pipeline",
        )

    assert excinfo.value.return_code == 2
    assert excinfo.value.error_msg == "ocr failed"
    assert len(calls) == 1
    converted = calls[0]["input_path"]
    assert converted.name == "scan_converted.png"
    assert converted.resolve() != image.resolve()
    assert calls[0]["method"] == "ocr"
    assert calls[0]["lang"] == "ch"
    assert calls[0]["kwargs"] == {"backend": "pipeline"}
    assert calls[0]["output_dir"].is_dir()
    assert not converted.exists()
    assert not converted.parent.exists()
    assert image.is_file()


def test_cleanup_oserror_does_not_replace_mineru_failure(monkeypatch, tmp_path):
    image = tmp_path / "scan.bmp"
    _write_bmp(image)
    converted_paths = []
    attempted = []
    real_unlink = Path.unlink

    def boom(input_path, output_dir, method="auto", lang=None, **kwargs):
        converted_paths.append(Path(input_path))
        raise MineruExecutionError(3, "ocr aborted")

    def failing_unlink(self, missing_ok=False):
        if self.name.endswith("_converted.png"):
            attempted.append(Path(self))
            raise OSError("temp busy")
        return real_unlink(self, missing_ok=missing_ok)

    parser = MineruParser()
    monkeypatch.setattr(parser, "_run_mineru_command", boom)
    monkeypatch.setattr(Path, "unlink", failing_unlink)

    try:
        with pytest.raises(MineruExecutionError, match="ocr aborted") as excinfo:
            parser.parse_image(image, output_dir=str(tmp_path / "out"))

        assert excinfo.value.return_code == 3
        assert excinfo.value.error_msg == "ocr aborted"
        assert "temp busy" not in str(excinfo.value)
        assert len(converted_paths) == 1
        converted = converted_paths[0]
        assert converted.name == "scan_converted.png"
        assert attempted == [converted]
        assert converted.exists()
        assert converted.parent.is_dir()
        assert image.is_file()
    finally:
        for path in converted_paths:
            _cleanup_converted(path, real_unlink)
