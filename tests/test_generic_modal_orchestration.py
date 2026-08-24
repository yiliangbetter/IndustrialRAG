"""Generic modal description + entity/chunk write path, including soft fallback."""

import json

import pytest

from raganything.modalprocessors import GenericModalProcessor


def _make_processor(**attrs) -> GenericModalProcessor:
    proc = GenericModalProcessor.__new__(GenericModalProcessor)
    proc.content_source = None
    proc.content_format = "auto"
    for name, value in attrs.items():
        setattr(proc, name, value)
    return proc


JSON_RESPONSE = json.dumps(
    {
        "detailed_description": "A temperature trend chart",
        "entity_info": {
            "entity_name": "temp-chart",
            "entity_type": "chart",
            "summary": "Temperature over time",
        },
    }
)


@pytest.mark.asyncio
async def test_generate_description_only_parses_json_and_suffixes_type():
    async def caption(prompt, system_prompt=None):
        assert "raw-chart-bytes" in prompt
        return JSON_RESPONSE

    proc = _make_processor(modal_caption_func=caption)
    description, entity = await proc.generate_description_only(
        "raw-chart-bytes", "chart"
    )

    assert description == "A temperature trend chart"
    assert entity["entity_name"] == "temp-chart (chart)"
    assert entity["entity_type"] == "chart"
    assert entity["summary"] == "Temperature over time"


@pytest.mark.asyncio
async def test_generate_description_only_uses_context_prompt():
    captured = {}

    async def caption(prompt, system_prompt=None):
        captured["prompt"] = prompt
        return JSON_RESPONSE

    proc = _make_processor(modal_caption_func=caption)
    proc._get_context_for_item = lambda item_info: "surrounding paragraph"

    await proc.generate_description_only(
        "payload", "chart", item_info={"page_idx": 2}, entity_name="forced-name"
    )

    assert "surrounding paragraph" in captured["prompt"]
    assert "payload" in captured["prompt"]


@pytest.mark.asyncio
async def test_generate_description_only_falls_back_when_caption_raises():
    async def caption(*args, **kwargs):
        raise RuntimeError("llm unavailable")

    proc = _make_processor(modal_caption_func=caption)
    description, entity = await proc.generate_description_only(
        "raw-chart", "chart", entity_name="preset"
    )

    assert description == "raw-chart"
    assert entity["entity_name"] == "preset"
    assert entity["entity_type"] == "chart"
    assert "raw-chart" in entity["summary"]


@pytest.mark.asyncio
async def test_process_multimodal_content_forwards_chunk_to_storage():
    proc = _make_processor()
    captured = {}

    async def fake_description(*args, **kwargs):
        return "desc", {
            "entity_name": "temp-chart (chart)",
            "entity_type": "chart",
            "summary": "Temperature over time",
        }

    async def fake_create(
        modal_chunk,
        entity_info,
        file_path,
        batch_mode=False,
        doc_id=None,
        chunk_order_index=0,
    ):
        captured["chunk"] = modal_chunk
        captured["entity"] = entity_info
        captured["file_path"] = file_path
        captured["doc_id"] = doc_id
        captured["chunk_order_index"] = chunk_order_index
        return (
            "Temperature over time",
            {"entity_name": entity_info["entity_name"], "chunk_id": "chunk-1"},
            None,
        )

    proc.generate_description_only = fake_description
    proc._create_entity_and_chunk = fake_create

    summary, meta, extra = await proc.process_multimodal_content(
        "payload",
        "chart",
        file_path="manual.pdf",
        doc_id="doc-1",
        chunk_order_index=4,
    )

    assert "Chart Content Analysis" in captured["chunk"]
    assert "payload" in captured["chunk"]
    assert "desc" in captured["chunk"]
    assert captured["file_path"] == "manual.pdf"
    assert captured["doc_id"] == "doc-1"
    assert captured["chunk_order_index"] == 4
    assert summary == "Temperature over time"
    assert meta["chunk_id"] == "chunk-1"
    assert extra is None


@pytest.mark.asyncio
async def test_process_multimodal_content_swallows_storage_failure():
    proc = _make_processor()

    async def fake_description(*args, **kwargs):
        return "desc", {
            "entity_name": "n",
            "entity_type": "chart",
            "summary": "s",
        }

    async def fake_create(*args, **kwargs):
        raise RuntimeError("chunk storage down")

    proc.generate_description_only = fake_description
    proc._create_entity_and_chunk = fake_create

    description, entity = await proc.process_multimodal_content(
        "payload", "chart", entity_name="keep-me"
    )

    assert description == "payload"
    assert entity["entity_name"] == "keep-me"
    assert entity["entity_type"] == "chart"
