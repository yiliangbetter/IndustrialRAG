"""Regression tests for EnhancedMarkdownConverter backend fail-closed paths.

WeasyPrint/Pandoc conversion must raise when backends are missing, return
False (not True) on conversion errors, and always clean up Pandoc temp files.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import raganything.enhanced_markdown as em


@pytest.fixture
def converter(monkeypatch):
    monkeypatch.setattr(em, "WEASYPRINT_AVAILABLE", True)
    monkeypatch.setattr(em, "MARKDOWN_AVAILABLE", True)
    monkeypatch.setattr(em, "PANDOC_AVAILABLE", True)
    conv = em.EnhancedMarkdownConverter()
    conv.available_backends = {
        "weasyprint": True,
        "pandoc": True,
        "pandoc_system": True,
    }
    return conv


def test_convert_with_weasyprint_raises_when_backend_missing(monkeypatch):
    monkeypatch.setattr(em, "WEASYPRINT_AVAILABLE", False)
    conv = em.EnhancedMarkdownConverter()
    with pytest.raises(RuntimeError, match="WeasyPrint not available"):
        conv.convert_with_weasyprint("# hi", "/tmp/out.pdf")


def test_convert_with_weasyprint_returns_false_on_write_failure(
    converter, monkeypatch, tmp_path
):
    html_obj = MagicMock()
    html_obj.write_pdf.side_effect = RuntimeError("render failed")
    # WeasyPrint may be absent in CI; inject the HTML symbol the method uses.
    monkeypatch.setattr(em, "HTML", MagicMock(return_value=html_obj), raising=False)
    monkeypatch.setattr(
        converter,
        "_process_markdown_content",
        MagicMock(return_value="<html>ok</html>"),
    )

    ok = converter.convert_with_weasyprint("# title", str(tmp_path / "out.pdf"))
    assert ok is False
    html_obj.write_pdf.assert_called_once()


def test_convert_with_pandoc_raises_when_backend_missing():
    conv = em.EnhancedMarkdownConverter()
    conv.available_backends = {
        "weasyprint": False,
        "pandoc": False,
        "pandoc_system": False,
    }
    with pytest.raises(RuntimeError, match="Pandoc not available"):
        conv.convert_with_pandoc("# hi", "/tmp/out.pdf")


def test_convert_with_pandoc_returns_false_on_nonzero_exit_and_unlinks_temp(
    converter, monkeypatch, tmp_path
):
    created: list[str] = []

    class FakeTemp:
        def __init__(self, *args, **kwargs):
            self.name = str(tmp_path / "pandoc-input.md")
            created.append(self.name)

        def __enter__(self):
            Path(self.name).write_text("# doc", encoding="utf-8")
            return SimpleNamespace(write=lambda *_a, **_k: None, name=self.name)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(em.tempfile, "NamedTemporaryFile", FakeTemp)

    run = MagicMock(
        return_value=SimpleNamespace(returncode=2, stderr="pandoc boom", stdout="")
    )
    # convert_with_pandoc does a local `import subprocess`.
    monkeypatch.setattr(subprocess, "run", run)

    out = tmp_path / "out.pdf"
    ok = converter.convert_with_pandoc("# title", str(out))
    assert ok is False
    assert created, "temp markdown file was never created"
    assert not Path(created[0]).exists(), "temp markdown file was not cleaned up"
    run.assert_called_once()
    cmd = run.call_args.args[0]
    assert cmd[0] == "pandoc"
    assert "--pdf-engine=wkhtmltopdf" in cmd


def test_convert_with_pandoc_returns_false_on_exception_and_unlinks_temp(
    converter, monkeypatch, tmp_path
):
    created: list[str] = []

    class FakeTemp:
        def __init__(self, *args, **kwargs):
            self.name = str(tmp_path / "pandoc-exc.md")
            created.append(self.name)

        def __enter__(self):
            Path(self.name).write_text("# doc", encoding="utf-8")
            return SimpleNamespace(write=lambda *_a, **_k: None, name=self.name)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(em.tempfile, "NamedTemporaryFile", FakeTemp)
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(side_effect=TimeoutError("pandoc hung")),
    )

    ok = converter.convert_with_pandoc("# title", str(tmp_path / "out.pdf"))
    assert ok is False
    assert created
    assert not Path(created[0]).exists()


def test_convert_markdown_to_pdf_unknown_method_returns_false(converter):
    assert converter.convert_markdown_to_pdf("# x", "/tmp/x.pdf", method="nope") is False
