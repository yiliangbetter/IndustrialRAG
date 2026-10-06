"""VLM and hybrid backends remap only the MinerU harvest fallback directory.

MinerU CLI `-m` stays auto/txt/ocr. After the process exits, `vlm-*` and
`hybrid-*` backends select `vlm/` and `hybrid_auto/` when the output scan
misses a content list. Rewriting `-m` itself fails the subprocess; leaving
the harvest method unchanged looks in the wrong folder and returns no blocks.
"""

from pathlib import Path

from raganything.parser import MineruParser


def _capture_parse(parser, monkeypatch):
    captured = {}

    def fake_run(input_path, output_dir, method="auto", lang=None, **kwargs):
        captured["cli_method"] = method
        captured["backend"] = kwargs.get("backend")

    def fake_read(output_dir, file_stem, method="auto"):
        captured["read_method"] = method
        captured["read_stem"] = file_stem
        return [{"type": "text", "text": "harvested"}], ""

    monkeypatch.setattr(parser, "_run_mineru_command", fake_run)
    monkeypatch.setattr(parser, "_read_output_files", fake_read)
    return captured


def _pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    return pdf


def test_vlm_backend_harvest_uses_vlm_dir_without_changing_cli_method(
    tmp_path, monkeypatch
):
    parser = MineruParser()
    captured = _capture_parse(parser, monkeypatch)

    result = parser.parse_pdf(
        _pdf(tmp_path),
        output_dir=str(tmp_path / "out"),
        method="ocr",
        backend="vlm-http-client",
    )

    assert result == [{"type": "text", "text": "harvested"}]
    assert captured["cli_method"] == "ocr"
    assert captured["backend"] == "vlm-http-client"
    assert captured["read_method"] == "vlm"
    assert captured["read_stem"] == "manual"


def test_hybrid_backend_harvest_uses_hybrid_auto_dir(tmp_path, monkeypatch):
    parser = MineruParser()
    captured = _capture_parse(parser, monkeypatch)

    parser.parse_pdf(
        _pdf(tmp_path),
        output_dir=str(tmp_path / "out"),
        method="txt",
        backend="hybrid-engine",
    )

    assert captured["cli_method"] == "txt"
    assert captured["read_method"] == "hybrid_auto"


def test_pipeline_and_unsuffixed_backends_keep_the_parse_method(tmp_path, monkeypatch):
    parser = MineruParser()
    pdf = _pdf(tmp_path)
    out = str(tmp_path / "out")

    pipeline = _capture_parse(parser, monkeypatch)
    parser.parse_pdf(pdf, output_dir=out, method="ocr", backend="pipeline")
    assert pipeline["cli_method"] == "ocr"
    assert pipeline["read_method"] == "ocr"

    plain_vlm = _capture_parse(parser, monkeypatch)
    parser.parse_pdf(pdf, output_dir=out, method="ocr", backend="vlm")
    assert plain_vlm["cli_method"] == "ocr"
    assert plain_vlm["read_method"] == "ocr"

    omitted = _capture_parse(parser, monkeypatch)
    parser.parse_pdf(pdf, output_dir=out, method="auto", backend=None)
    assert omitted["cli_method"] == "auto"
    assert omitted["read_method"] == "auto"
