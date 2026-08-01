"""Regression tests for insert_text_content parameter forwarding.

Pure-text ingestion is shared by document processors and batch scripts.
Dropped kwargs (ids, file_paths, split options) cause wrong citations, ID
collisions, or uncontrolled chunking. Covers the pure-text helper only —
multimodal ainsert exception handling is owned by open critical-bug PRs.
"""

import pytest

pytest.importorskip("lightrag")

from raganything.utils import insert_text_content


class FakeLightRAG:
    def __init__(self):
        self.calls = []

    async def ainsert(self, **kwargs):
        self.calls.append(kwargs)


@pytest.mark.asyncio
async def test_insert_text_content_forwards_all_kwargs():
    lightrag = FakeLightRAG()

    await insert_text_content(
        lightrag,
        input=["doc a", "doc b"],
        split_by_character="\n",
        split_by_character_only=True,
        ids=["id-a", "id-b"],
        file_paths=["/a.pdf", "/b.pdf"],
    )

    assert lightrag.calls == [
        {
            "input": ["doc a", "doc b"],
            "file_paths": ["/a.pdf", "/b.pdf"],
            "split_by_character": "\n",
            "split_by_character_only": True,
            "ids": ["id-a", "id-b"],
        }
    ]


@pytest.mark.asyncio
async def test_insert_text_content_defaults_optional_kwargs():
    lightrag = FakeLightRAG()

    await insert_text_content(lightrag, input="single document")

    assert lightrag.calls == [
        {
            "input": "single document",
            "file_paths": None,
            "split_by_character": None,
            "split_by_character_only": False,
            "ids": None,
        }
    ]


@pytest.mark.asyncio
async def test_insert_text_content_propagates_ainsert_errors():
    class FailingLightRAG:
        async def ainsert(self, **kwargs):
            raise RuntimeError("storage down")

    with pytest.raises(RuntimeError, match="storage down"):
        await insert_text_content(FailingLightRAG(), input="x")
