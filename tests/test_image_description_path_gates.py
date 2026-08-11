"""Image generate_description_only path validation soft-fallback.

Open PR #109 covers vision prompt wiring and encode-failure fallback.
These tests lock the earlier gates: missing img_path and missing image file
must soft-fallback on main (2-tuple) rather than raise into individual/batch
ingest orchestration.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from raganything.modalprocessors import ImageModalProcessor


def _make_image(caption_func=None):
    proc = ImageModalProcessor.__new__(ImageModalProcessor)
    proc.modal_caption_func = caption_func or AsyncMock()
    return proc


@pytest.mark.asyncio
async def test_image_generate_description_only_soft_fallback_when_path_missing():
    caption_func = AsyncMock()
    proc = _make_image(caption_func)

    modal_content = {"image_caption": ["no path provided"]}
    caption, entity = await proc.generate_description_only(
        modal_content,
        "image",
        entity_name="Missing Path Figure",
    )

    caption_func.assert_not_called()
    assert caption == str(modal_content)
    assert entity["entity_name"] == "Missing Path Figure"
    assert entity["entity_type"] == "image"
    assert entity["summary"].startswith("Image content:")


@pytest.mark.asyncio
async def test_image_generate_description_only_soft_fallback_when_file_missing(
    tmp_path,
):
    caption_func = AsyncMock()
    proc = _make_image(caption_func)
    missing = tmp_path / "does-not-exist.png"
    assert not missing.exists()

    modal_content = {"img_path": str(missing)}
    caption, entity = await proc.generate_description_only(
        modal_content,
        "image",
        entity_name="Gone Figure",
    )

    caption_func.assert_not_called()
    assert caption == str(modal_content)
    assert entity["entity_name"] == "Gone Figure"
    assert entity["entity_type"] == "image"
    assert entity["summary"].startswith("Image content:")


@pytest.mark.asyncio
async def test_image_generate_description_only_fallback_hashes_default_name():
    proc = _make_image(AsyncMock())
    modal_content = {"img_path": str(Path("/nonexistent/figure.png"))}

    caption, entity = await proc.generate_description_only(modal_content, "image")

    assert caption == str(modal_content)
    assert entity["entity_name"].startswith("image_")
    assert entity["entity_type"] == "image"
