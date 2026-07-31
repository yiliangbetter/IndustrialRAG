"""Regression tests for EnhancedMarkdownConverter validation and routing.

Markdown→PDF is used as an Office/text preprocessing path. Missing-file,
encoding fallback, unknown-method, and backend recommendation bugs are easy
to miss without unit coverage. No open PR claims this module.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from raganything.enhanced_markdown import EnhancedMarkdownConverter, MarkdownConfig


class TestBackendRecommendation:
    def test_recommends_pandoc_when_system_pandoc_available(self):
        converter = EnhancedMarkdownConverter(MarkdownConfig())
        converter.available_backends = {
            "weasyprint": True,
            "pandoc": True,
            "markdown": True,
            "pandoc_system": True,
        }
        assert converter._get_recommended_backend() == "pandoc"

    def test_recommends_weasyprint_when_no_pandoc(self):
        converter = EnhancedMarkdownConverter(MarkdownConfig())
        converter.available_backends = {
            "weasyprint": True,
            "pandoc": False,
            "markdown": True,
            "pandoc_system": False,
        }
        assert converter._get_recommended_backend() == "weasyprint"

    def test_recommends_none_when_no_backends(self):
        converter = EnhancedMarkdownConverter(MarkdownConfig())
        converter.available_backends = {
            "weasyprint": False,
            "pandoc": False,
            "markdown": False,
            "pandoc_system": False,
        }
        assert converter._get_recommended_backend() == "none"

    def test_get_backend_info_shape(self):
        converter = EnhancedMarkdownConverter(MarkdownConfig(page_size="Letter"))
        info = converter.get_backend_info()
        assert "available_backends" in info
        assert "recommended_backend" in info
        assert info["config"]["page_size"] == "Letter"


class TestConvertMarkdownRouting:
    def test_unknown_method_returns_false(self, tmp_path):
        converter = EnhancedMarkdownConverter()
        out = tmp_path / "out.pdf"
        ok = converter.convert_markdown_to_pdf("# hi", str(out), method="not_a_backend")
        assert ok is False
        assert not out.exists()

    def test_auto_uses_recommended_backend(self, tmp_path):
        converter = EnhancedMarkdownConverter()
        converter.available_backends = {
            "weasyprint": False,
            "pandoc": False,
            "markdown": True,
            "pandoc_system": False,
        }
        # recommended is "none" → ValueError path → False
        ok = converter.convert_markdown_to_pdf("# hi", str(tmp_path / "x.pdf"), method="auto")
        assert ok is False

    def test_weasyprint_method_delegates(self, tmp_path):
        converter = EnhancedMarkdownConverter()
        out = str(tmp_path / "w.pdf")
        with patch.object(
            converter, "convert_with_weasyprint", return_value=True
        ) as mocked:
            assert converter.convert_markdown_to_pdf("# t", out, method="weasyprint")
            mocked.assert_called_once_with("# t", out)


class TestConvertFileToPdf:
    def test_missing_input_raises(self, tmp_path):
        converter = EnhancedMarkdownConverter()
        with pytest.raises(FileNotFoundError):
            converter.convert_file_to_pdf(str(tmp_path / "missing.md"))

    def test_utf8_file_default_output_suffix(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\nBody", encoding="utf-8")
        converter = EnhancedMarkdownConverter()
        with patch.object(
            converter, "convert_markdown_to_pdf", return_value=True
        ) as mocked:
            ok = converter.convert_file_to_pdf(str(md))
            assert ok is True
            args, kwargs = mocked.call_args
            assert "# Title" in args[0]
            assert args[1] == str(md.with_suffix(".pdf"))

    def test_encoding_fallback_to_latin1(self, tmp_path):
        md = tmp_path / "latin.md"
        # bytes invalid as utf-8
        md.write_bytes("café".encode("latin-1"))
        converter = EnhancedMarkdownConverter()
        with patch.object(
            converter, "convert_markdown_to_pdf", return_value=True
        ) as mocked:
            assert converter.convert_file_to_pdf(str(md), str(tmp_path / "o.pdf"))
            content = mocked.call_args[0][0]
            assert "caf" in content


class TestProcessMarkdownContent:
    def test_raises_when_markdown_unavailable(self, monkeypatch):
        import raganything.enhanced_markdown as em

        monkeypatch.setattr(em, "MARKDOWN_AVAILABLE", False)
        converter = EnhancedMarkdownConverter()
        with pytest.raises(RuntimeError, match="Markdown library not available"):
            converter._process_markdown_content("# hi")

    def test_html_wraps_custom_css(self, monkeypatch):
        import raganything.enhanced_markdown as em

        if not em.MARKDOWN_AVAILABLE:
            pytest.skip("markdown package not installed")

        converter = EnhancedMarkdownConverter(
            MarkdownConfig(custom_css="body { color: red; }")
        )
        html = converter._process_markdown_content("# Hello")
        assert "<!DOCTYPE html>" in html
        assert "body { color: red; }" in html
        assert "Hello" in html
