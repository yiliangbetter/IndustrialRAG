from __future__ import annotations

import pytest

from raganything.utils import insert_text_content_with_multimodal_content


class FailingLightRAG:
    async def ainsert(self, **kwargs):
        raise RuntimeError("storage unavailable")


@pytest.mark.asyncio
async def test_insert_text_content_with_multimodal_content_propagates_insert_errors():
    with pytest.raises(RuntimeError, match="storage unavailable"):
        await insert_text_content_with_multimodal_content(
            FailingLightRAG(),
            input="important document text",
            multimodal_content=[{"type": "image", "img_path": "/tmp/a.png"}],
            ids="doc-1",
            file_paths="doc.pdf",
        )
