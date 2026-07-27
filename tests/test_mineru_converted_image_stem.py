"""Regression: MinerU parse_image must read output for the converted input stem."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from raganything.parser import MineruParser


def test_parse_image_converted_format_reads_converted_stem(tmp_path, monkeypatch):
    """BMP (and other non-native formats) convert to {stem}_converted.png.

    MinerU writes output under that converted stem. Reading the original stem
    silently returns an empty content list and breaks documented image formats.
    """
    pytest.importorskip("PIL")
    from PIL import Image

    image_path = tmp_path / "scan.bmp"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path, "BMP")
    output_dir = tmp_path / "out"

    def fake_run_mineru(cls, input_path, output_dir, method="auto", lang=None, **kwargs):
        input_path = Path(input_path)
        assert input_path.name == "scan_converted.png"
        stem = input_path.stem
        method_dir = Path(output_dir) / stem / method
        method_dir.mkdir(parents=True, exist_ok=True)
        content = [{"type": "text", "text": "ocr-from-bmp", "page_idx": 0}]
        (method_dir / f"{stem}_content_list.json").write_text(
            json.dumps(content), encoding="utf-8"
        )
        (method_dir / f"{stem}.md").write_text("ocr-from-bmp", encoding="utf-8")

    monkeypatch.setattr(
        MineruParser,
        "_run_mineru_command",
        classmethod(fake_run_mineru),
    )

    result = MineruParser().parse_image(image_path, output_dir=str(output_dir))

    assert len(result) == 1
    assert result[0]["text"] == "ocr-from-bmp"


def test_parse_image_native_png_still_uses_original_stem(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image

    image_path = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(image_path, "PNG")
    output_dir = tmp_path / "out"

    def fake_run_mineru(cls, input_path, output_dir, method="auto", lang=None, **kwargs):
        input_path = Path(input_path)
        assert input_path.name == "photo.png"
        stem = input_path.stem
        method_dir = Path(output_dir) / stem / method
        method_dir.mkdir(parents=True, exist_ok=True)
        content = [{"type": "text", "text": "ocr-from-png", "page_idx": 0}]
        (method_dir / f"{stem}_content_list.json").write_text(
            json.dumps(content), encoding="utf-8"
        )

    monkeypatch.setattr(
        MineruParser,
        "_run_mineru_command",
        classmethod(fake_run_mineru),
    )

    result = MineruParser().parse_image(image_path, output_dir=str(output_dir))
    assert result[0]["text"] == "ocr-from-png"


def test_batch_parser_rejects_empty_content_list():
    from raganything.batch_parser import BatchParser

    batch = BatchParser(
        parser_type="mineru",
        show_progress=False,
        skip_installation_check=True,
    )
    batch.parser = MagicMock()
    batch.parser.parse_document.return_value = []

    success, file_path, error = batch.process_single_file(
        "/tmp/empty.pdf", "/tmp/out", parse_method="auto"
    )

    assert success is False
    assert file_path == "/tmp/empty.pdf"
    assert "No content extracted" in error
