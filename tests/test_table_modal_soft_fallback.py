"""Tests for TableModalProcessor outer soft-fallback paths.

LLM/caption failures must not abort table ingest: the processor returns a
fallback entity so the rest of the document can still be indexed.
"""

from __future__ import annotations

import json

import pytest

from raganything.modalprocessors import TableModalProcessor


class BareTableProcessor(TableModalProcessor):
    """Skip LightRAG-backed BaseModalProcessor.__init__."""

    def __init__(self, caption_func=None):
        self.modal_caption_func = caption_func
        self.context_extractor = None


def _valid_table_response(entity_name="spec_table"):
    return json.dumps(
        {
            "detailed_description": "Voltage ratings for the drive.",
            "entity_info": {
                "entity_name": entity_name,
                "entity_type": "table",
                "summary": "Drive voltage table",
            },
        }
    )


@pytest.mark.asyncio
async def test_generate_description_only_falls_back_when_caption_func_raises():
    async def boom(*args, **kwargs):
        raise RuntimeError("llm unavailable")

    proc = BareTableProcessor(caption_func=boom)
    content = {"table_body": "| V | A |\n| 220 | 5 |", "table_caption": ["Ratings"]}

    caption, entity = await proc.generate_description_only(
        content, "table", entity_name="ratings_table"
    )

    assert caption == str(content)
    assert entity["entity_name"] == "ratings_table"
    assert entity["entity_type"] == "table"
    assert "Table content:" in entity["summary"]


@pytest.mark.asyncio
async def test_generate_description_only_parses_json_string_content():
    captured = {}

    async def fake_caption(prompt, system_prompt=None):
        captured["prompt"] = prompt
        return _valid_table_response()

    proc = BareTableProcessor(caption_func=fake_caption)
    raw = json.dumps(
        {
            "table_body": "col1|col2",
            "table_caption": ["caption"],
            "img_path": "/tmp/t.png",
            "table_footnote": ["note"],
        }
    )

    caption, entity = await proc.generate_description_only(raw, "table")

    assert caption == "Voltage ratings for the drive."
    assert entity["entity_name"] == "spec_table (table)"
    assert "col1|col2" in captured["prompt"]
    assert "/tmp/t.png" in captured["prompt"]


@pytest.mark.asyncio
async def test_generate_description_only_non_json_string_uses_table_body():
    captured = {}

    async def fake_caption(prompt, system_prompt=None):
        captured["prompt"] = prompt
        return _valid_table_response("plain")

    proc = BareTableProcessor(caption_func=fake_caption)
    caption, entity = await proc.generate_description_only(
        "not-json-table-body", "table"
    )

    assert "not-json-table-body" in captured["prompt"]
    assert caption == "Voltage ratings for the drive."
    assert entity["entity_type"] == "table"


@pytest.mark.asyncio
async def test_process_multimodal_content_falls_back_when_chunk_write_fails():
    async def ok_caption(prompt, system_prompt=None):
        return _valid_table_response()

    proc = BareTableProcessor(caption_func=ok_caption)

    async def boom_chunk(*args, **kwargs):
        raise RuntimeError("storage down")

    proc._create_entity_and_chunk = boom_chunk

    content = {"table_body": "a|b", "table_caption": ["c"]}
    caption, entity = await proc.process_multimodal_content(
        content, "table", file_path="manual.pdf", entity_name="kept_name"
    )

    assert caption == str(content)
    assert entity["entity_name"] == "kept_name"
    assert entity["entity_type"] == "table"
