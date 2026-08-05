"""Regression: refuse to base64-encode non-image local files for VLM calls.

Without validation, multimodal ingest and aquery_with_multimodal could read
arbitrary paths (e.g. /etc/passwd) and forward bytes to an external vision model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from raganything.modalprocessors import ImageModalProcessor
from raganything.query import QueryMixin
from raganything.utils import encode_image_to_base64


def test_encode_image_to_base64_rejects_non_image_secret(tmp_path):
    secret = tmp_path / "credentials.env"
    secret.write_text("OPENAI_API_KEY=sk-secret-value\n", encoding="utf-8")

    assert encode_image_to_base64(str(secret)) == ""


def test_encode_image_to_base64_still_accepts_image(tmp_path):
    img = tmp_path / "figure.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

    encoded = encode_image_to_base64(str(img))
    assert encoded
    assert "iVBOR" in encoded or len(encoded) > 0


def test_image_modal_processor_encode_rejects_passwd_like_file(tmp_path):
    secret = tmp_path / "passwd"
    secret.write_text("root:x:0:0:root:/root:/bin/bash\n", encoding="utf-8")

    processor = ImageModalProcessor.__new__(ImageModalProcessor)
    assert processor._encode_image_to_base64(str(secret)) == ""


@pytest.mark.asyncio
async def test_describe_image_for_query_does_not_read_non_image(tmp_path):
    secret = tmp_path / "secrets.txt"
    payload = "super-secret-token-do-not-leak"
    secret.write_text(payload, encoding="utf-8")

    calls = []

    class FakeProcessor:
        def _encode_image_to_base64(self, image_path: str) -> str:
            calls.append(("encode", image_path))
            # Would leak if called; return marker to make failures obvious
            with open(image_path, "rb") as f:
                return f.read().decode("utf-8", errors="replace")

        async def modal_caption_func(self, *args, **kwargs):
            calls.append(("vlm", kwargs.get("image_data", "")))
            return "should-not-be-called"

    mixin = QueryMixin.__new__(QueryMixin)
    mixin.logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )

    description = await mixin._describe_image_for_query(
        FakeProcessor(),
        {"img_path": str(secret), "image_caption": ["cap"]},
    )

    assert calls == []
    assert payload not in description
    assert "Image path:" in description


@pytest.mark.asyncio
async def test_describe_image_for_query_encodes_valid_image(tmp_path):
    img = tmp_path / "ok.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 32)

    calls = []

    class FakeProcessor:
        def _encode_image_to_base64(self, image_path: str) -> str:
            calls.append(image_path)
            return "YmFzZTY0"

        async def modal_caption_func(self, *args, **kwargs):
            return "a diagram of a pump"

    mixin = QueryMixin.__new__(QueryMixin)
    mixin.logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )

    description = await mixin._describe_image_for_query(
        FakeProcessor(),
        {"img_path": str(img)},
    )

    assert calls == [str(img)]
    assert description == "a diagram of a pump"
