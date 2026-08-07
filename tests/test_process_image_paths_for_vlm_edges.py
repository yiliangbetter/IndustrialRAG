"""Regression tests for QueryMixin._process_image_paths_for_vlm edge paths.

Open PRs cover sandbox deny/extra-allow (#64) and aquery_vlm_enhanced glue (#95).
This file covers remaining high-risk edges: allow working_dir / parser_output_dir,
encode soft-failures that must not invent markers, and invalid short path formats.
"""

from __future__ import annotations

import base64

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
async def test_process_image_paths_allows_working_dir_images(tmp_path, monkeypatch):
    working_dir = tmp_path / "workspace"
    output_dir = tmp_path / "parser_output"
    cwd = tmp_path / "cwd"
    working_dir.mkdir()
    output_dir.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    image = working_dir / "figure.png"
    image_bytes = _png_bytes()
    image.write_bytes(image_bytes)

    query = DummyQuery(config=_config(working_dir, output_dir))
    prompt = f"Evidence\nImage Path: {image}\nEnd"

    processed_prompt, image_count = await query._process_image_paths_for_vlm(prompt)

    assert image_count == 1
    assert f"Image Path: {image}\n[VLM_IMAGE_1]" in processed_prompt
    assert query._current_images_base64 == [
        base64.b64encode(image_bytes).decode("utf-8")
    ]


@pytest.mark.asyncio
async def test_process_image_paths_allows_parser_output_dir_images(tmp_path, monkeypatch):
    working_dir = tmp_path / "workspace"
    output_dir = tmp_path / "parser_output"
    cwd = tmp_path / "cwd"
    working_dir.mkdir()
    output_dir.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    image = output_dir / "page-1.png"
    image_bytes = _png_bytes()
    image.write_bytes(image_bytes)

    query = DummyQuery(config=_config(working_dir, output_dir))
    prompt = f"Evidence\nImage Path: {image}\nEnd"

    processed_prompt, image_count = await query._process_image_paths_for_vlm(prompt)

    assert image_count == 1
    assert f"Image Path: {image}\n[VLM_IMAGE_1]" in processed_prompt
    assert query._current_images_base64 == [
        base64.b64encode(image_bytes).decode("utf-8")
    ]


@pytest.mark.asyncio
async def test_process_image_paths_keeps_original_when_encode_returns_empty(
    tmp_path, monkeypatch
):
    working_dir = tmp_path / "workspace"
    output_dir = tmp_path / "parser_output"
    working_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.chdir(working_dir)

    image = working_dir / "figure.png"
    image.write_bytes(_png_bytes())

    query = DummyQuery(config=_config(working_dir, output_dir))
    prompt = f"Evidence\nImage Path: {image}\nEnd"

    monkeypatch.setattr(
        "raganything.query.encode_image_to_base64", lambda _path: ""
    )

    processed_prompt, image_count = await query._process_image_paths_for_vlm(prompt)

    assert processed_prompt == prompt
    assert image_count == 0
    assert query._current_images_base64 == []
    assert any("Failed to encode image" in msg for msg in query.logger.errors)


@pytest.mark.asyncio
async def test_process_image_paths_keeps_original_when_encode_raises(
    tmp_path, monkeypatch
):
    working_dir = tmp_path / "workspace"
    output_dir = tmp_path / "parser_output"
    working_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.chdir(working_dir)

    image = working_dir / "figure.png"
    image.write_bytes(_png_bytes())

    query = DummyQuery(config=_config(working_dir, output_dir))
    prompt = f"Evidence\nImage Path: {image}\nEnd"

    def boom(_path):
        raise OSError("disk read failed")

    monkeypatch.setattr("raganything.query.encode_image_to_base64", boom)

    processed_prompt, image_count = await query._process_image_paths_for_vlm(prompt)

    assert processed_prompt == prompt
    assert image_count == 0
    assert query._current_images_base64 == []
    assert any("Failed to process image" in msg for msg in query.logger.errors)


@pytest.mark.asyncio
async def test_process_image_paths_skips_paths_that_fail_validation(
    tmp_path, monkeypatch
):
    working_dir = tmp_path / "workspace"
    output_dir = tmp_path / "parser_output"
    working_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.chdir(working_dir)

    # Extension matches the regex, but the file is missing → validate_image_file=False.
    missing = working_dir / "missing-figure.png"
    query = DummyQuery(config=_config(working_dir, output_dir))
    prompt = f"Evidence\nImage Path: {missing}\nEnd"

    processed_prompt, image_count = await query._process_image_paths_for_vlm(prompt)

    assert processed_prompt == prompt
    assert image_count == 0
    assert query._current_images_base64 == []
    assert any(
        "Image validation failed or path unsafe" in msg for msg in query.logger.warnings
    )


@pytest.mark.asyncio
async def test_process_image_paths_partial_success_keeps_unsafe_path_literal(
    tmp_path, monkeypatch
):
    """One safe image should still be encoded when a sibling path is blocked."""
    working_dir = tmp_path / "workspace"
    output_dir = tmp_path / "parser_output"
    outside_dir = tmp_path / "outside"
    cwd = tmp_path / "cwd"
    working_dir.mkdir()
    output_dir.mkdir()
    outside_dir.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    safe_image = working_dir / "safe.png"
    safe_bytes = _png_bytes()
    safe_image.write_bytes(safe_bytes)
    blocked_image = outside_dir / "blocked.png"
    blocked_image.write_bytes(_png_bytes())

    query = DummyQuery(config=_config(working_dir, output_dir))
    prompt = (
        f"First\nImage Path: {blocked_image}\n"
        f"Second\nImage Path: {safe_image}\nEnd"
    )

    processed_prompt, image_count = await query._process_image_paths_for_vlm(prompt)

    assert image_count == 1
    assert processed_prompt.count("[VLM_IMAGE_") == 1
    assert f"Image Path: {blocked_image}\nSecond" in processed_prompt
    assert f"Image Path: {safe_image}\n[VLM_IMAGE_1]" in processed_prompt
    assert query._current_images_base64 == [
        base64.b64encode(safe_bytes).decode("utf-8")
    ]
    assert any(
        "Blocking image path outside safe directories" in msg
        for msg in query.logger.warnings
    )
