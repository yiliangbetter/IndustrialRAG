"""Regression tests for LightRAG text-insert forwarding helpers."""

import pytest

from raganything import utils as utils_module


class FakeLightRAG:
    def __init__(self, *, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    async def ainsert(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on is not None:
            raise self.fail_on


@pytest.mark.asyncio
async def test_insert_text_content_forwards_all_args():
    rag = FakeLightRAG()

    await utils_module.insert_text_content(
        rag,
        input=["hello", "world"],
        split_by_character="\n",
        split_by_character_only=True,
        ids=["doc-1", "doc-2"],
        file_paths=["a.pdf", "b.pdf"],
    )

    assert rag.calls == [
        {
            "input": ["hello", "world"],
            "file_paths": ["a.pdf", "b.pdf"],
            "split_by_character": "\n",
            "split_by_character_only": True,
            "ids": ["doc-1", "doc-2"],
        }
    ]


@pytest.mark.asyncio
async def test_insert_text_content_with_multimodal_forwards_extra_args():
    rag = FakeLightRAG()
    multimodal = [{"type": "table", "table_body": "| a |"}]

    await utils_module.insert_text_content_with_multimodal_content(
        rag,
        input="body",
        multimodal_content=multimodal,
        split_by_character=None,
        split_by_character_only=False,
        ids="doc-1",
        file_paths="manual.pdf",
        scheme_name="mineru",
    )

    assert rag.calls == [
        {
            "input": "body",
            "multimodal_content": multimodal,
            "file_paths": "manual.pdf",
            "split_by_character": None,
            "split_by_character_only": False,
            "ids": "doc-1",
            "scheme_name": "mineru",
        }
    ]


@pytest.mark.asyncio
async def test_insert_text_content_with_multimodal_swallows_ainsert_errors(caplog):
    rag = FakeLightRAG(
        fail_on=TypeError("unexpected keyword argument multimodal_content")
    )

    with caplog.at_level("INFO"):
        await utils_module.insert_text_content_with_multimodal_content(
            rag,
            input="body",
            multimodal_content=[{"type": "image"}],
            ids="doc-1",
            file_paths="manual.pdf",
        )

    assert len(rag.calls) == 1
    assert any("Error:" in record.message for record in caplog.records)
    assert any(
        "update the raganything branch of lightrag" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_insert_text_content_propagates_ainsert_errors():
    rag = FakeLightRAG(fail_on=RuntimeError("insert failed"))

    with pytest.raises(RuntimeError, match="insert failed"):
        await utils_module.insert_text_content(
            rag,
            input="body",
            ids="doc-1",
            file_paths="manual.pdf",
        )
