"""LightRAG insert helpers must forward ids, paths, and split options.

ingest of parsed text is a shared core flow. Dropping file_paths or ids here
breaks citations and document status; dropping split flags changes chunking.
"""

import pytest

from raganything.utils import (
    insert_text_content,
    insert_text_content_with_multimodal_content,
)


class FakeLightRAG:
    def __init__(self):
        self.calls = []

    async def ainsert(self, **kwargs):
        self.calls.append(kwargs)


@pytest.mark.asyncio
async def test_insert_text_content_forwards_all_arguments():
    lightrag = FakeLightRAG()

    await insert_text_content(
        lightrag,
        input="hello world",
        split_by_character="\n",
        split_by_character_only=True,
        ids="doc-1",
        file_paths="paper.pdf",
    )

    assert lightrag.calls == [
        {
            "input": "hello world",
            "file_paths": "paper.pdf",
            "split_by_character": "\n",
            "split_by_character_only": True,
            "ids": "doc-1",
        }
    ]


@pytest.mark.asyncio
async def test_insert_text_content_accepts_list_payloads():
    lightrag = FakeLightRAG()

    await insert_text_content(
        lightrag,
        input=["a", "b"],
        ids=["doc-a", "doc-b"],
        file_paths=["a.pdf", "b.pdf"],
    )

    assert lightrag.calls[0]["input"] == ["a", "b"]
    assert lightrag.calls[0]["ids"] == ["doc-a", "doc-b"]
    assert lightrag.calls[0]["file_paths"] == ["a.pdf", "b.pdf"]
    assert lightrag.calls[0]["split_by_character"] is None
    assert lightrag.calls[0]["split_by_character_only"] is False


@pytest.mark.asyncio
async def test_insert_text_content_with_multimodal_forwards_scheme_and_content():
    lightrag = FakeLightRAG()
    multimodal = [{"type": "table", "table_body": "a|b"}]

    await insert_text_content_with_multimodal_content(
        lightrag,
        input="caption",
        multimodal_content=multimodal,
        split_by_character=None,
        split_by_character_only=False,
        ids="doc-mm",
        file_paths="slide.pptx",
        scheme_name="minerU",
    )

    assert lightrag.calls == [
        {
            "input": "caption",
            "multimodal_content": multimodal,
            "file_paths": "slide.pptx",
            "split_by_character": None,
            "split_by_character_only": False,
            "ids": "doc-mm",
            "scheme_name": "minerU",
        }
    ]
