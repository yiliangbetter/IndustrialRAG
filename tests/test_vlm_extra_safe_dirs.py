"""Regression tests for VLM image-path extra_safe_dirs allowlisting.

#128 covers cwd / working_dir / parser_output_dir sandboxing. These tests lock
the remaining allowlist: extra_safe_dirs must admit images that sit outside
those defaults, and a bad extra-dir entry must not block a later valid one.
aquery_vlm_enhanced must forward extra_safe_dirs into the path processor.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from raganything.query import QueryMixin


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg % args if args else msg)

    def error(self, msg, *args, **kwargs):
        self.errors.append(msg % args if args else msg)


class DummyQuery(QueryMixin):
    def __init__(self, config=None):
        self.config = config
        self.logger = FakeLogger()


def _png_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"image-bytes"


def _config(working_dir, output_dir):
    return type(
        "Config",
        (),
        {
            "working_dir": str(working_dir),
            "parser_output_dir": str(output_dir),
        },
    )()


@pytest.mark.asyncio
async def test_extra_safe_dirs_allows_image_outside_default_sandbox(
    tmp_path, monkeypatch
):
    working_dir = tmp_path / "workspace"
    output_dir = tmp_path / "parser_output"
    cwd = tmp_path / "cwd"
    extra_dir = tmp_path / "operator-export"
    working_dir.mkdir()
    output_dir.mkdir()
    cwd.mkdir()
    extra_dir.mkdir()
    monkeypatch.chdir(cwd)

    image = extra_dir / "figure.png"
    image.write_bytes(_png_bytes())
    prompt = f"Image Path: {image}"

    query = DummyQuery(config=_config(working_dir, output_dir))
    blocked, blocked_count = await query._process_image_paths_for_vlm(prompt)
    assert blocked_count == 0
    assert "[VLM_IMAGE_" not in blocked
    assert str(image) in blocked

    allowed, allowed_count = await query._process_image_paths_for_vlm(
        prompt, extra_safe_dirs=[str(extra_dir)]
    )
    assert allowed_count == 1
    assert "[VLM_IMAGE_1]" in allowed
    assert query._current_images_base64 == [
        base64.b64encode(_png_bytes()).decode("utf-8")
    ]


@pytest.mark.asyncio
async def test_bad_extra_safe_dir_does_not_block_later_valid_dir(
    tmp_path, monkeypatch
):
    working_dir = tmp_path / "workspace"
    output_dir = tmp_path / "parser_output"
    cwd = tmp_path / "cwd"
    extra_dir = tmp_path / "allowed-export"
    working_dir.mkdir()
    output_dir.mkdir()
    cwd.mkdir()
    extra_dir.mkdir()
    monkeypatch.chdir(cwd)

    image = extra_dir / "figure.png"
    image.write_bytes(_png_bytes())
    prompt = f"Image Path: {image}"

    query = DummyQuery(config=_config(working_dir, output_dir))
    enhanced, count = await query._process_image_paths_for_vlm(
        prompt,
        extra_safe_dirs=[None, str(extra_dir)],
    )
    assert count == 1
    assert "[VLM_IMAGE_1]" in enhanced


@pytest.mark.asyncio
async def test_aquery_vlm_enhanced_forwards_extra_safe_dirs():
    captured = {}

    class DummyVlmQuery(QueryMixin):
        def __init__(self):
            self.vision_model_func = object()
            self.logger = FakeLogger()
            self.lightrag = SimpleNamespace(aquery=self._aquery)

        async def _ensure_lightrag_initialized(self):
            return {"success": True}

        async def _aquery(self, query, param, system_prompt=None):
            return "raw retrieval prompt"

        async def _process_image_paths_for_vlm(self, prompt, extra_safe_dirs=None):
            captured["prompt"] = prompt
            captured["extra_safe_dirs"] = extra_safe_dirs
            return prompt, 0

    rag = DummyVlmQuery()
    extra = ["/mnt/manuals/figures"]
    result = await rag.aquery_vlm_enhanced(
        "what is this figure?",
        extra_safe_dirs=extra,
    )
    assert result == "raw retrieval prompt"
    assert captured["prompt"] == "raw retrieval prompt"
    assert captured["extra_safe_dirs"] == extra
