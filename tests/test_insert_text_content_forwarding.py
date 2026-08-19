"""insert_text_content must forward LightRAG kwargs and not swallow errors.

Silent success after ainsert failure would mark documents processed with
empty graph/vector state. The plain-text helper is the production path for
LLM ingestion; multimodal insert is only asserted on the success path.
"""

import pytest

from raganything.utils import (
    insert_text_content,
    insert_text_content_with_multimodal_content,
)


class RecordingRAG:
    def __init__(self):
        self.calls = []

    async def ainsert(self, **kwargs):
        self.calls.append(kwargs)


class FailingRAG:
    async def ainsert(self, **kwargs):
        raise RuntimeError("ainsert exploded")


@pytest.mark.asyncio
async def test_insert_text_content_forwards_all_kwargs():
    rag = RecordingRAG()
    await insert_text_content(
        rag,
        input="body",
        split_by_character="\n",
        split_by_character_only=True,
        ids="doc-1",
        file_paths="manual.pdf",
    )
    assert rag.calls == [
        {
            "input": "body",
            "file_paths": "manual.pdf",
            "split_by_character": "\n",
            "split_by_character_only": True,
            "ids": "doc-1",
        }
    ]


@pytest.mark.asyncio
async def test_insert_text_content_propagates_ainsert_errors():
    with pytest.raises(RuntimeError, match="ainsert exploded"):
        await insert_text_content(FailingRAG(), input="body")


@pytest.mark.asyncio
async def test_multimodal_insert_forwards_scheme_and_content_on_success():
    rag = RecordingRAG()
    multimodal = [{"type": "image", "img_path": "/abs/fig.png"}]
    await insert_text_content_with_multimodal_content(
        rag,
        input="caption",
        multimodal_content=multimodal,
        ids=["doc-2"],
        file_paths=["fig.pdf"],
        scheme_name="mineru",
        split_by_character=None,
        split_by_character_only=False,
    )
    assert rag.calls == [
        {
            "input": "caption",
            "multimodal_content": multimodal,
            "file_paths": ["fig.pdf"],
            "split_by_character": None,
            "split_by_character_only": False,
            "ids": ["doc-2"],
            "scheme_name": "mineru",
        }
    ]
