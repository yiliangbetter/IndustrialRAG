"""Regression tests for EnhancedMarkdownConverter.get_backend_info.

CLI --info and auto routing are claimed elsewhere. This locks the wrapper
contract: recommended_backend follows pandoc_system → weasyprint → none,
and the returned config snapshot matches MarkdownConfig fields operators
rely on when choosing a conversion path.
"""

from raganything.enhanced_markdown import EnhancedMarkdownConverter, MarkdownConfig


def test_get_backend_info_recommends_pandoc_when_system_pandoc_exists(monkeypatch):
    converter = EnhancedMarkdownConverter(
        MarkdownConfig(page_size="Letter", margin="0.5in")
    )
    converter.available_backends = {
        "weasyprint": True,
        "pandoc": True,
        "markdown": True,
        "pandoc_system": True,
    }

    info = converter.get_backend_info()
    assert info["recommended_backend"] == "pandoc"
    assert info["available_backends"]["pandoc_system"] is True
    assert info["config"]["page_size"] == "Letter"
    assert info["config"]["margin"] == "0.5in"
    assert info["config"]["include_toc"] is True
    assert info["config"]["syntax_highlighting"] is True


def test_get_backend_info_falls_back_to_weasyprint_then_none():
    converter = EnhancedMarkdownConverter()
    converter.available_backends = {
        "weasyprint": True,
        "pandoc": False,
        "markdown": True,
        "pandoc_system": False,
    }
    assert converter.get_backend_info()["recommended_backend"] == "weasyprint"

    converter.available_backends["weasyprint"] = False
    assert converter.get_backend_info()["recommended_backend"] == "none"
