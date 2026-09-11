"""MinerU parse_pdf must forward CLI kwargs into _run_mineru_command.

Timeout, language, page range, and backend flags are accepted on parse_pdf
and on process_document_complete. Those helpers are tested elsewhere; this
file covers the last hop. Dropping timeout lets stuck model downloads hang
ingest. Rewriting method to "vlm" before spawn would pass an invalid -m
value (MinerU methods are auto/txt/ocr) when backend is vlm-*.
"""

from pathlib import Path

from raganything.parser import MineruParser, Parser


def _stub_parse_pdf(parser, monkeypatch):
    captured = {}

    def fake_run(input_path, output_dir, method="auto", lang=None, **kwargs):
        captured["input_path"] = Path(input_path)
        captured["output_dir"] = Path(output_dir)
        captured["method"] = method
        captured["lang"] = lang
        captured["kwargs"] = kwargs

    def fake_read(output_dir, file_stem, method="auto"):
        captured["read_method"] = method
        captured["read_stem"] = file_stem
        return [{"type": "text", "text": "harvested"}], "md"

    monkeypatch.setattr(parser, "_run_mineru_command", fake_run)
    monkeypatch.setattr(parser, "_read_output_files", fake_read)
    return captured


def test_parse_pdf_forwards_timeout_lang_and_ocr_flags(tmp_path, monkeypatch):
    parser = MineruParser()
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    out = tmp_path / "shared_out"
    captured = _stub_parse_pdf(parser, monkeypatch)

    result = parser.parse_pdf(
        pdf,
        output_dir=str(out),
        method="ocr",
        lang="ch",
        timeout=45,
        formula=False,
        table=False,
        start_page=1,
        end_page=4,
        device="cpu",
        source="local",
        backend="pipeline",
    )

    expected_out = Parser._unique_output_dir(out, pdf)
    assert captured["input_path"] == pdf
    assert captured["output_dir"] == expected_out
    assert captured["method"] == "ocr"
    assert captured["lang"] == "ch"
    assert captured["kwargs"]["timeout"] == 45
    assert captured["kwargs"]["formula"] is False
    assert captured["kwargs"]["table"] is False
    assert captured["kwargs"]["start_page"] == 1
    assert captured["kwargs"]["end_page"] == 4
    assert captured["kwargs"]["device"] == "cpu"
    assert captured["kwargs"]["source"] == "local"
    assert captured["kwargs"]["backend"] == "pipeline"
    assert result == [{"type": "text", "text": "harvested"}]


def test_vlm_backend_keeps_original_cli_method_and_backend(tmp_path, monkeypatch):
    parser = MineruParser()
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    captured = _stub_parse_pdf(parser, monkeypatch)

    parser.parse_pdf(
        pdf,
        output_dir=str(tmp_path / "out"),
        method="ocr",
        backend="vlm-http-client",
        vlm_url="http://127.0.0.1:30000",
        timeout=30,
    )

    # MinerU CLI methods are auto/txt/ocr. VLM is a backend, not -m.
    assert captured["method"] == "ocr"
    assert captured["kwargs"]["backend"] == "vlm-http-client"
    assert captured["kwargs"]["vlm_url"] == "http://127.0.0.1:30000"
    assert captured["kwargs"]["timeout"] == 30


def test_hybrid_backend_keeps_original_cli_method(tmp_path, monkeypatch):
    parser = MineruParser()
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    captured = _stub_parse_pdf(parser, monkeypatch)

    parser.parse_pdf(
        pdf,
        output_dir=str(tmp_path / "out"),
        method="txt",
        backend="hybrid-auto",
    )

    assert captured["method"] == "txt"
    assert captured["kwargs"]["backend"] == "hybrid-auto"
