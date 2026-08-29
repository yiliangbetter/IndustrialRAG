"""ImageModalProcessor base64 encoding must fail closed for missing files.

Query image description calls processor._encode_image_to_base64, not the
utils helper. An empty string skips the vision caption and falls back to
path/caption text; a raised exception would abort the whole multimodal query.
"""

import base64

from raganything.modalprocessors import ImageModalProcessor


def _processor():
    proc = ImageModalProcessor.__new__(ImageModalProcessor)
    return proc


def test_missing_file_returns_empty_string():
    assert _processor()._encode_image_to_base64("/nonexistent/figure.png") == ""


def test_existing_file_returns_base64(tmp_path):
    image = tmp_path / "figure.png"
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    image.write_bytes(payload)

    encoded = _processor()._encode_image_to_base64(str(image))

    assert encoded
    assert base64.b64decode(encoded) == payload
