"""HTML conversion for EnhancedMarkdownConverter._process_markdown_content.

Backend routing and file-encoding fallbacks are covered elsewhere. This locks
the HTML document contract used before WeasyPrint/Pandoc: tables/fenced code
extensions, custom CSS, charset, and fail-closed when markdown is missing.
"""

import types

import pytest

from raganything.enhanced_markdown import EnhancedMarkdownConverter, MarkdownConfig


def _fake_markdown_module(captured):
    module = types.ModuleType("markdown")

    class FakeMarkdown:
        def __init__(self, extensions, extension_configs):
            captured["extensions"] = list(extensions)
            captured["extension_configs"] = dict(extension_configs)

        def convert(self, content):
            captured["content"] = content
            return "<table><tr><td>A</td></tr></table><pre><code>x=1</code></pre>"

    module.Markdown = FakeMarkdown
    return module


def test_missing_markdown_library_fails_closed(monkeypatch):
    monkeypatch.setattr("raganything.enhanced_markdown.MARKDOWN_AVAILABLE", False)
    converter = EnhancedMarkdownConverter()
    with pytest.raises(RuntimeError, match="Markdown library not available"):
        converter._process_markdown_content("# Title")


def test_wraps_html_document_with_charset_and_body(monkeypatch):
    captured = {}
    monkeypatch.setattr("raganything.enhanced_markdown.MARKDOWN_AVAILABLE", True)
    monkeypatch.setattr(
        "raganything.enhanced_markdown.markdown",
        _fake_markdown_module(captured),
        raising=False,
    )

    converter = EnhancedMarkdownConverter()
    html = converter._process_markdown_content("# Manual\n\n| A |\n| - |\n")

    assert "<!DOCTYPE html>" in html
    assert 'charset="UTF-8"' in html
    assert "<body>" in html
    assert "<table>" in html
    assert "<pre>" in html
    assert captured["content"].startswith("# Manual")


def test_enables_table_and_fenced_code_extensions(monkeypatch):
    captured = {}
    monkeypatch.setattr("raganything.enhanced_markdown.MARKDOWN_AVAILABLE", True)
    monkeypatch.setattr(
        "raganything.enhanced_markdown.markdown",
        _fake_markdown_module(captured),
        raising=False,
    )

    converter = EnhancedMarkdownConverter()
    converter._process_markdown_content("```python\nprint(1)\n```")

    assert "markdown.extensions.tables" in captured["extensions"]
    assert "markdown.extensions.fenced_code" in captured["extensions"]
    assert "markdown.extensions.codehilite" in captured["extensions"]
    assert captured["extension_configs"]["codehilite"]["use_pygments"] is True


def test_custom_css_replaces_default_stylesheet(monkeypatch):
    captured = {}
    monkeypatch.setattr("raganything.enhanced_markdown.MARKDOWN_AVAILABLE", True)
    monkeypatch.setattr(
        "raganything.enhanced_markdown.markdown",
        _fake_markdown_module(captured),
        raising=False,
    )

    converter = EnhancedMarkdownConverter(
        MarkdownConfig(custom_css="body { color: #c00; }")
    )
    html = converter._process_markdown_content("hello")

    assert "body { color: #c00; }" in html
    assert "Segoe UI" not in html
