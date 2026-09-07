"""MinerU parse_image must OCR figures and harvest from the ocr/ method dir."""

from pathlib import Path

from raganything.parser import MineruParser


def test_parse_image_uses_ocr_method_for_command_and_harvest(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cls, input_path, output_dir, method="auto", lang=None, **kwargs):
        captured["run_method"] = method
        captured["run_input"] = Path(input_path)
        captured["run_output"] = Path(output_dir)
        captured["run_lang"] = lang

    def fake_read(cls, output_dir, file_stem, method="auto"):
        captured["read_method"] = method
        captured["read_stem"] = file_stem
        captured["read_output"] = Path(output_dir)
        return [{"type": "text", "text": "ocr harvest", "page_idx": 0}], "md"

    monkeypatch.setattr(MineruParser, "_run_mineru_command", classmethod(fake_run))
    monkeypatch.setattr(MineruParser, "_read_output_files", classmethod(fake_read))

    img = tmp_path / "figure.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    out = tmp_path / "parse_out"

    result = MineruParser().parse_image(str(img), output_dir=str(out), lang="en")

    assert captured["run_method"] == "ocr"
    assert captured["read_method"] == "ocr"
    assert captured["run_lang"] == "en"
    assert captured["run_input"] == img
    assert captured["read_stem"] == "figure"
    assert captured["run_output"] == captured["read_output"]
    assert result == [{"type": "text", "text": "ocr harvest", "page_idx": 0}]
