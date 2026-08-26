"""Tests for DoclingParser.parse_html / parse_office_doc method bodies.

Routing tests mock these methods; this file covers validation, unique output
dirs (#51), kwargs forwarding, and return wiring.
"""

import hashlib
from pathlib import Path

import pytest

from raganything.parser import DoclingParser


def _unique_dir(base: Path, file_path: Path) -> Path:
    resolved = file_path.resolve()
    digest = hashlib.md5(str(resolved).encode()).hexdigest()[:8]
    return Path(base) / f"{resolved.stem}_{digest}"


class TestParseHtml:
    def test_missing_file_raises(self, tmp_path):
        parser = DoclingParser()
        missing = tmp_path / "missing.html"
        with pytest.raises(FileNotFoundError, match="HTML file does not exist"):
            parser.parse_html(missing)

    def test_unsupported_format_raises(self, tmp_path):
        parser = DoclingParser()
        bad = tmp_path / "page.txt"
        bad.write_text("hello", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported HTML format"):
            parser.parse_html(bad)

    def test_xhtml_accepted_uses_unique_output_dir_and_returns_blocks(
        self, monkeypatch, tmp_path
    ):
        parser = DoclingParser()
        html = tmp_path / "page.xhtml"
        html.write_text("<html><body>hi</body></html>", encoding="utf-8")
        out = tmp_path / "shared_out"

        seen = {}
        expected = [{"type": "text", "text": "hi", "page_idx": 0}]

        def fake_run(input_path, output_dir, file_stem, **kwargs):
            seen["input_path"] = Path(input_path)
            seen["output_dir"] = Path(output_dir)
            seen["file_stem"] = file_stem
            seen["kwargs"] = kwargs
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        def fake_read(output_dir, name_without_suff):
            seen["read_args"] = (Path(output_dir), name_without_suff)
            return expected, {"meta": True}

        monkeypatch.setattr(parser, "_run_docling_command", fake_run)
        monkeypatch.setattr(parser, "_read_output_files", fake_read)

        result = parser.parse_html(
            html, output_dir=str(out), image_export_mode="placeholder"
        )

        assert result == expected
        assert seen["input_path"] == html
        assert seen["file_stem"] == "page"
        assert seen["kwargs"] == {"image_export_mode": "placeholder"}
        assert seen["output_dir"] == _unique_dir(out, html)
        assert seen["read_args"] == (seen["output_dir"], "page")

    def test_htm_default_output_dir_without_unique_subdir(self, monkeypatch, tmp_path):
        parser = DoclingParser()
        html = tmp_path / "doc.htm"
        html.write_text("<p>x</p>", encoding="utf-8")
        seen = {}

        def fake_run(input_path, output_dir, file_stem, **kwargs):
            seen["output_dir"] = Path(output_dir)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(parser, "_run_docling_command", fake_run)
        monkeypatch.setattr(
            parser, "_read_output_files", lambda *a, **k: ([{"type": "text"}], None)
        )

        parser.parse_html(html)
        assert seen["output_dir"] == html.parent / "docling_output"


class TestParseOfficeDoc:
    def test_rejects_html_suffix(self, tmp_path):
        parser = DoclingParser()
        html = tmp_path / "page.html"
        html.write_text("<html></html>", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported office format"):
            parser.parse_office_doc(html)

    def test_missing_file_raises(self, tmp_path):
        parser = DoclingParser()
        with pytest.raises(FileNotFoundError, match="Document file does not exist"):
            parser.parse_office_doc(tmp_path / "missing.docx")

    def test_docx_uses_unique_output_dir_and_forwards_kwargs(
        self, monkeypatch, tmp_path
    ):
        parser = DoclingParser()
        docx = tmp_path / "report.docx"
        docx.write_bytes(b"PK\x03\x04")
        out = tmp_path / "out"
        seen = {}
        blocks = [{"type": "text", "text": "office", "page_idx": 0}]

        def fake_run(input_path, output_dir, file_stem, **kwargs):
            seen["input_path"] = Path(input_path)
            seen["output_dir"] = Path(output_dir)
            seen["file_stem"] = file_stem
            seen["kwargs"] = kwargs
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(parser, "_run_docling_command", fake_run)
        monkeypatch.setattr(
            parser, "_read_output_files", lambda *a, **k: (blocks, None)
        )

        assert (
            parser.parse_office_doc(docx, output_dir=str(out), table_mode="accurate")
            == blocks
        )
        assert seen["input_path"] == docx
        assert seen["file_stem"] == "report"
        assert seen["kwargs"] == {"table_mode": "accurate"}
        assert seen["output_dir"] == _unique_dir(out, docx)
