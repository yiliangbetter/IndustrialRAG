"""Table generate_description_only soft-fallback contracts.

Complements #106 (table success-path prompt wiring) and #109 (equation/image
description soft-fallback). Asserts current main-branch behavior: description
errors return a 2-tuple fallback rather than aborting batch stage-1 ingest.
#75 changes soft-fallback to fail-closed and is intentionally not required.
"""

from unittest.mock import AsyncMock

import pytest
from lightrag.utils import compute_mdhash_id

from raganything.modalprocessors import TableModalProcessor


def _make_table(caption_func):
    proc = TableModalProcessor.__new__(TableModalProcessor)
    proc.modal_caption_func = caption_func
    return proc


@pytest.mark.asyncio
async def test_table_generate_description_only_soft_fallback_on_llm_error():
    """LLM failures must not abort stage-1; preserve caller entity_name."""
    proc = _make_table(AsyncMock(side_effect=RuntimeError("llm down")))
    modal_content = {
        "img_path": "/t.png",
        "table_caption": ["Specs"],
        "table_body": "| A | B |\n| 1 | 2 |",
        "table_footnote": [],
    }

    caption, entity = await proc.generate_description_only(
        modal_content=modal_content,
        content_type="table",
        entity_name="Pump Specs",
    )

    assert caption == str(modal_content)
    assert entity["entity_name"] == "Pump Specs"
    assert entity["entity_type"] == "table"
    assert entity["summary"].startswith("Table content:")
    assert str(modal_content)[:100] in entity["summary"]


@pytest.mark.asyncio
async def test_table_generate_description_only_soft_fallback_hashes_entity_name():
    """Without entity_name, fallback must use a deterministic table_* hash id."""
    proc = _make_table(AsyncMock(side_effect=RuntimeError("llm down")))
    modal_content = {"table_body": "| rpm | 1500 |"}

    caption, entity = await proc.generate_description_only(
        modal_content=modal_content,
        content_type="table",
    )

    expected_name = f"table_{compute_mdhash_id(str(modal_content))}"
    assert caption == str(modal_content)
    assert entity["entity_name"] == expected_name
    assert entity["entity_type"] == "table"
    assert entity["summary"].startswith("Table content:")


@pytest.mark.asyncio
async def test_table_generate_description_only_soft_fallback_on_format_error():
    """Brace-rich table_body that breaks .format() must soft-fallback, not raise."""
    proc = _make_table(AsyncMock())
    # Unescaped braces in table_body cause str.format KeyError/ValueError.
    modal_content = {
        "table_body": "| template | {unclosed | value |",
        "table_caption": ["Broken"],
    }

    caption, entity = await proc.generate_description_only(
        modal_content=modal_content,
        content_type="table",
        entity_name="Broken Table",
    )

    assert caption == str(modal_content)
    assert entity["entity_name"] == "Broken Table"
    assert entity["entity_type"] == "table"
    assert entity["summary"].startswith("Table content:")
    proc.modal_caption_func.assert_not_awaited()
