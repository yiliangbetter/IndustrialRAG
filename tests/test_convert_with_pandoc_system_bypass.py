"""Pandoc system-bypass and markdown-to-PDF routing.

``use_system_pandoc=True`` (method=pandoc_system) must still invoke pandoc when
the availability probe failed. Auto/pandoc routing must not raise out of
``convert_markdown_to_pdf`` when the backend is missing — ingest treats False
as fail-closed. A success path must unlink the temp markdown file.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from raganything.enhanced_markdown import EnhancedMarkdownConverter


def _converter_without_pandoc():
    conv = EnhancedMarkdownConverter()
    conv.available_backends = {
        "weasyprint": False,
        "pandoc": False,
        "markdown": True,
        "pandoc_system": False,
    }
    return conv


def test_use_system_pandoc_bypasses_failed_availability_probe(monkeypatch, tmp_path):
    conv = _converter_without_pandoc()
    run = MagicMock(return_value=SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(subprocess, "run", run)

    out = tmp_path / "out.pdf"
    assert conv.convert_with_pandoc("# title", str(out), use_system_pandoc=True) is True
    run.assert_called_once()
    cmd = run.call_args.args[0]
    assert cmd[0] == "pandoc"
    assert "--pdf-engine=wkhtmltopdf" in cmd
    assert str(out) in cmd


def test_use_system_pandoc_false_still_raises_when_probe_failed():
    conv = _converter_without_pandoc()
    with pytest.raises(RuntimeError, match="Pandoc not available"):
        conv.convert_with_pandoc("# title", "/tmp/out.pdf", use_system_pandoc=False)


def test_pandoc_system_method_forwards_bypass_flag(tmp_path):
    conv = EnhancedMarkdownConverter()
    seen = {}

    def fake(content, output_path, use_system_pandoc=False):
        seen["content"] = content
        seen["output_path"] = output_path
        seen["use_system_pandoc"] = use_system_pandoc
        return True

    conv.convert_with_pandoc = fake
    out = str(tmp_path / "out.pdf")
    assert conv.convert_markdown_to_pdf("# doc", out, method="pandoc_system") is True
    assert seen["content"] == "# doc"
    assert seen["output_path"] == out
    assert seen["use_system_pandoc"] is True


def test_pandoc_method_does_not_set_bypass_flag(tmp_path):
    conv = EnhancedMarkdownConverter()
    seen = {}

    def fake(content, output_path, use_system_pandoc=False):
        seen["use_system_pandoc"] = use_system_pandoc
        return True

    conv.convert_with_pandoc = fake
    assert (
        conv.convert_markdown_to_pdf("# doc", str(tmp_path / "out.pdf"), method="pandoc")
        is True
    )
    assert seen["use_system_pandoc"] is False


def test_convert_markdown_to_pdf_pandoc_returns_false_when_backend_raises(
    tmp_path,
):
    conv = _converter_without_pandoc()

    def boom(*_args, **_kwargs):
        raise RuntimeError("Pandoc not available")

    conv.convert_with_pandoc = boom
    assert (
        conv.convert_markdown_to_pdf(
            "# doc", str(tmp_path / "out.pdf"), method="pandoc"
        )
        is False
    )


def test_convert_with_pandoc_success_unlinks_temp_markdown(monkeypatch, tmp_path):
    conv = EnhancedMarkdownConverter()
    conv.available_backends["pandoc_system"] = True
    created = []
    real_named = __import__("tempfile").NamedTemporaryFile

    def tracking_named(*args, **kwargs):
        handle = real_named(*args, **kwargs)
        created.append(handle.name)
        return handle

    monkeypatch.setattr(
        "tempfile.NamedTemporaryFile",
        tracking_named,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(return_value=SimpleNamespace(returncode=0, stdout="", stderr="")),
    )

    assert conv.convert_with_pandoc("# ok", str(tmp_path / "out.pdf")) is True
    assert created
    from pathlib import Path

    assert not Path(created[0]).exists()


def test_check_backends_records_system_pandoc_probe(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(return_value=SimpleNamespace(returncode=0, stdout="pandoc 3", stderr="")),
    )
    conv = EnhancedMarkdownConverter()
    assert conv.available_backends.get("pandoc_system") is True

    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(side_effect=FileNotFoundError("pandoc")),
    )
    conv = EnhancedMarkdownConverter()
    assert conv.available_backends.get("pandoc_system") is False
