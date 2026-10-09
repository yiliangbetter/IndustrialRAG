"""Image validation size boundary and case-insensitive extensions.

VLM query encoding calls ``validate_image_file`` before reading a file into
the prompt. An off-by-one size check rejects a manual scan that is exactly
at the limit, or accepts a payload one byte over it. Suffix checks that are
case-sensitive drop Windows exports such as ``figure.JPG``.
"""

from raganything.utils import validate_image_file


def test_file_at_size_limit_is_accepted_and_one_byte_over_is_rejected(tmp_path):
    limit = 1 * 1024 * 1024
    at_limit = tmp_path / "scan.png"
    at_limit.write_bytes(b"x" * limit)
    over_limit = tmp_path / "scan-over.png"
    over_limit.write_bytes(b"x" * (limit + 1))

    assert validate_image_file(str(at_limit), max_size_mb=1) is True
    assert validate_image_file(str(over_limit), max_size_mb=1) is False


def test_uppercase_and_mixed_case_image_extensions_are_accepted(tmp_path):
    names = ["figure.JPG", "photo.JPEG", "plate.Png", "drawing.TIFF", "icon.WebP"]
    for name in names:
        image = tmp_path / name
        image.write_bytes(b"\x00" * 32)
        assert validate_image_file(str(image)) is True, name
