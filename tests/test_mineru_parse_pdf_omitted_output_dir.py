"""parse_pdf without output_dir writes beside the PDF, not a hashed shared dir."""

from pathlib import Path

from raganything.parser import MineruParser


def test_parse_pdf_omitted_output_dir_uses_sibling_mineru_output(tmp_path, monkeypatch):
    captured = {}
    unique_calls = []

    def fake_run(cls, input_path, output_dir, method="auto", lang=None, **kwargs):
        captured["run_output"] = Path(output_dir)
        captured["run_method"] = method
        captured["run_input"] = Path(input_path)

    def fake_read(cls, output_dir, file_stem, method="auto"):
        captured["read_output"] = Path(output_dir)
        captured["read_method"] = method
        captured["read_stem"] = file_stem
        return [{"type": "text", "text": "pdf harvest", "page_idx": 0}], "md"

    def fake_unique(base_dir, file_path):
        unique_calls.append((Path(base_dir), Path(file_path)))
        return Path(base_dir) / "should_not_be_used"

    monkeypatch.setattr(MineruParser, "_run_mineru_command", classmethod(fake_run))
    monkeypatch.setattr(MineruParser, "_read_output_files", classmethod(fake_read))
    monkeypatch.setattr(
        MineruParser, "_unique_output_dir", staticmethod(fake_unique)
    )

    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    result = MineruParser().parse_pdf(str(pdf), method="txt")

    expected = tmp_path / "mineru_output"
    assert unique_calls == []
    assert expected.is_dir()
    assert captured["run_output"] == expected
    assert captured["read_output"] == expected
    assert captured["run_method"] == "txt"
    assert captured["read_method"] == "txt"
    assert captured["read_stem"] == "manual"
    assert captured["run_input"] == pdf
    assert result == [{"type": "text", "text": "pdf harvest", "page_idx": 0}]
