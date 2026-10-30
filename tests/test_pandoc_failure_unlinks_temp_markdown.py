"""Failed Pandoc conversions must not leave the source markdown on disk."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

MODULE_NAME = "_raganything_enhanced_markdown_pandoc_fail"


def _load_enhanced_markdown():
    path = Path(__file__).resolve().parents[1] / "raganything" / "enhanced_markdown.py"
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _converter_ready_for_pandoc():
    module = _load_enhanced_markdown()
    converter = module.EnhancedMarkdownConverter()
    # The availability probe is covered elsewhere. These tests start after
    # Pandoc has already been accepted so a failed run still creates a temp file.
    converter.available_backends["pandoc_system"] = True
    return converter


def _capture_run(monkeypatch, side_effect):
    seen = {}

    def fake_run(cmd, **kwargs):
        markdown_path = Path(cmd[1])
        seen["cmd"] = list(cmd)
        seen["timeout"] = kwargs.get("timeout")
        seen["text"] = kwargs.get("text")
        seen["capture_output"] = kwargs.get("capture_output")
        seen["content"] = markdown_path.read_text(encoding="utf-8")
        seen["path"] = markdown_path
        if isinstance(side_effect, BaseException):
            raise side_effect
        return side_effect

    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


def _assert_pandoc_invocation(seen, output_path: Path, markdown: str):
    cmd = seen["cmd"]
    assert cmd[0] == "pandoc"
    assert cmd[1].endswith(".md")
    assert cmd[2:] == [
        "-o",
        str(output_path),
        "--pdf-engine=wkhtmltopdf",
        "--standalone",
        "--toc",
        "--number-sections",
    ]
    assert seen["timeout"] == 60
    assert seen["text"] is True
    assert seen["capture_output"] is True
    assert seen["content"] == markdown
    assert seen["path"].name.endswith(".md")


def test_nonzero_pandoc_exit_returns_false_and_unlinks_temp(monkeypatch, tmp_path):
    converter = _converter_ready_for_pandoc()
    markdown = "# Torque\n\nTighten the coupling to 42 Nm.\n"
    output = tmp_path / "manual.pdf"
    seen = _capture_run(
        monkeypatch,
        SimpleNamespace(returncode=2, stdout="", stderr="wkhtmltopdf missing"),
    )

    ok = converter.convert_with_pandoc(markdown, str(output))

    assert ok is False
    _assert_pandoc_invocation(seen, output, markdown)
    assert not seen["path"].exists()


def test_pandoc_timeout_returns_false_and_unlinks_temp(monkeypatch, tmp_path):
    converter = _converter_ready_for_pandoc()
    markdown = "Seal replacement procedure.\n"
    output = tmp_path / "seal.pdf"
    seen = _capture_run(
        monkeypatch,
        subprocess.TimeoutExpired(cmd=["pandoc"], timeout=60),
    )

    ok = converter.convert_with_pandoc(markdown, str(output))

    assert ok is False
    _assert_pandoc_invocation(seen, output, markdown)
    assert not seen["path"].exists()


def test_temp_cleanup_oserror_does_not_mask_failed_conversion(monkeypatch, tmp_path):
    converter = _converter_ready_for_pandoc()
    markdown = "Bearing clearance table.\n"
    output = tmp_path / "bearing.pdf"
    seen = _capture_run(
        monkeypatch,
        SimpleNamespace(returncode=1, stdout="", stderr="pdf engine failed"),
    )
    real_unlink = os.unlink

    def unlink_busy(path):
        seen["unlink_path"] = path
        raise OSError("temp file busy")

    monkeypatch.setattr(os, "unlink", unlink_busy)
    try:
        ok = converter.convert_with_pandoc(markdown, str(output))
    finally:
        leftover = seen.get("path")
        if leftover is not None and leftover.exists():
            real_unlink(leftover)

    assert ok is False
    assert seen["unlink_path"] == str(seen["path"])
