"""insert_text_content_with_multimodal_content must not raise on ainsert failure.

LightRAG's public ainsert historically lacked multimodal_content. The helper
logs and continues so API ingest still marks a document complete. A future
change that starts raising would turn a version mismatch into a hard ingest
failure for every LightRAG-API document.
"""

import pytest

from raganything.utils import insert_text_content_with_multimodal_content


class BoomLightRAG:
    def __init__(self):
        self.calls = 0

    async def ainsert(self, **kwargs):
        self.calls += 1
        raise RuntimeError("storage down")


@pytest.mark.asyncio
async def test_multimodal_insert_swallows_ainsert_failure():
    lightrag = BoomLightRAG()

    await insert_text_content_with_multimodal_content(
        lightrag,
        input="caption",
        multimodal_content=[{"type": "table", "table_body": "a|b"}],
        ids="doc-mm",
        file_paths="slide.pptx",
        scheme_name="minerU",
    )

    assert lightrag.calls == 1


@pytest.mark.asyncio
async def test_plain_insert_still_raises_on_ainsert_failure():
    from raganything.utils import insert_text_content

    lightrag = BoomLightRAG()

    with pytest.raises(RuntimeError, match="storage down"):
        await insert_text_content(
            lightrag,
            input="hello",
            ids="doc-1",
            file_paths="paper.pdf",
        )
