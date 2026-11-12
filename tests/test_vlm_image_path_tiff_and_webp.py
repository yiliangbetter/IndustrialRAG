"""VLM image-path suffixes must keep the longer TIFF extension.

``_process_image_paths_for_vlm`` captures ``tiff`` before ``tif``. If the
shorter alternative is tried first, ``plate.tiff`` is read as ``plate.tif``
and a sibling file is encoded instead. ``.webp`` is in the same suffix list
and must still be encoded after a TIFF path.
"""

import base64

import pytest

from raganything.query import QueryMixin


def _query(working_dir):
    query = QueryMixin()
    query.config = type(
        "Config",
        (),
        {
            "working_dir": str(working_dir),
            "parser_output_dir": str(working_dir),
        },
    )()
    query.logger = type(
        "Logger",
        (),
        {
            "info": lambda *args, **kwargs: None,
            "warning": lambda *args, **kwargs: None,
            "error": lambda *args, **kwargs: None,
            "debug": lambda *args, **kwargs: None,
        },
    )()
    return query


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode()


@pytest.mark.asyncio
async def test_tiff_path_encodes_tiff_bytes_not_sibling_tif(tmp_path):
    tif_path = tmp_path / "plate.tif"
    tiff_path = tmp_path / "plate.tiff"
    webp_path = tmp_path / "diagram.webp"
    tif_path.write_bytes(b"SHORT-TIF")
    tiff_path.write_bytes(b"LONG-TIFF")
    webp_path.write_bytes(b"WEBP-BYTES")

    query = _query(tmp_path)
    prompt = (
        f"Context before\nImage Path: {tiff_path}\n"
        f"Image Path: {webp_path}\nContext after\n"
    )

    enhanced, count = await query._process_image_paths_for_vlm(prompt)

    assert count == 2
    assert query._current_images_base64 == [_b64(b"LONG-TIFF"), _b64(b"WEBP-BYTES")]
    assert f"Image Path: {tiff_path}\n[VLM_IMAGE_1]" in enhanced
    assert f"Image Path: {webp_path}\n[VLM_IMAGE_2]" in enhanced
    assert "Context before" in enhanced
    assert "Context after" in enhanced
    assert _b64(b"SHORT-TIF") not in query._current_images_base64


@pytest.mark.asyncio
async def test_short_tif_suffix_is_still_encoded(tmp_path):
    tif_path = tmp_path / "scan.tif"
    tif_path.write_bytes(b"SHORT-TIF")

    query = _query(tmp_path)
    prompt = f"Image Path: {tif_path}\n"

    enhanced, count = await query._process_image_paths_for_vlm(prompt)

    assert count == 1
    assert query._current_images_base64 == [_b64(b"SHORT-TIF")]
    assert f"Image Path: {tif_path}\n[VLM_IMAGE_1]" in enhanced
