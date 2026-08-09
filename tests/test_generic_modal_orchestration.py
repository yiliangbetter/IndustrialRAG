"""GenericModalProcessor orchestration and description prompt wiring.

Covers the untyped/generic modal path left uncovered by typed image/table/
equation process_multimodal_content tests (#103) and fail-closed description
coverage (#75). Asserts current main-branch contracts (including soft-fallback
on description errors).
"""

import json
from unittest.mock import AsyncMock

import pytest

from raganything.modalprocessors import GenericModalProcessor
from raganything.prompt import PROMPTS


class _BareModal:
    """Skip LightRAG-backed BaseModalProcessor.__init__."""

    def __init__(self):
        pass


def _as_generic():
    proc = _BareModal()
    proc.__class__ = GenericModalProcessor
    return proc


def _make_generic_with_caption(caption_func):
    proc = GenericModalProcessor.__new__(GenericModalProcessor)
    proc.modal_caption_func = caption_func
    return proc


@pytest.mark.asyncio
async def test_generic_process_multimodal_content_builds_chunk_and_forwards_ids():
    proc = _as_generic()
    entity_info = {
        "entity_name": "Audio Clip (audio)",
        "entity_type": "audio",
        "summary": "A short waveform note",
    }
    captured = {}

    async def fake_generate_description_only(
        modal_content, content_type, item_info=None, entity_name=None
    ):
        assert content_type == "audio"
        assert entity_name == "Audio Clip"
        assert modal_content == {"text": "beep at 440Hz"}
        return "enhanced audio caption", entity_info

    async def fake_create_entity_and_chunk(
        modal_chunk,
        entity_info_arg,
        file_path,
        batch_mode=False,
        doc_id=None,
        chunk_order_index=0,
    ):
        captured["modal_chunk"] = modal_chunk
        captured["entity_info"] = entity_info_arg
        captured["file_path"] = file_path
        captured["batch_mode"] = batch_mode
        captured["doc_id"] = doc_id
        captured["chunk_order_index"] = chunk_order_index
        return "enhanced audio caption", {"chunk_id": "chunk-g"}, []

    proc.generate_description_only = fake_generate_description_only
    proc._create_entity_and_chunk = fake_create_entity_and_chunk

    result = await proc.process_multimodal_content(
        {"text": "beep at 440Hz"},
        "audio",
        file_path="manual.pdf",
        entity_name="Audio Clip",
        batch_mode=True,
        doc_id="doc-g",
        chunk_order_index=4,
    )

    expected_chunk = PROMPTS["generic_chunk"].format(
        content_type="Audio",
        content=str({"text": "beep at 440Hz"}),
        enhanced_caption="enhanced audio caption",
    )
    assert captured["modal_chunk"] == expected_chunk
    assert captured["entity_info"] == entity_info
    assert captured["file_path"] == "manual.pdf"
    assert captured["batch_mode"] is True
    assert captured["doc_id"] == "doc-g"
    assert captured["chunk_order_index"] == 4
    assert result[0] == "enhanced audio caption"


@pytest.mark.asyncio
async def test_generic_generate_description_only_wires_prompt_without_context():
    calls = {}

    async def caption_func(prompt, system_prompt=None):
        calls["prompt"] = prompt
        calls["system_prompt"] = system_prompt
        return json.dumps(
            {
                "detailed_description": "Chart showing torque curves",
                "entity_info": {
                    "entity_name": "Torque Chart",
                    "entity_type": "chart",
                    "summary": "Torque vs RPM",
                },
            }
        )

    proc = _make_generic_with_caption(caption_func)
    caption, entity = await proc.generate_description_only(
        modal_content={"values": [1, 2, 3]},
        content_type="chart",
        entity_name="Torque Chart",
    )

    expected_prompt = PROMPTS["generic_prompt"].format(
        content_type="chart",
        entity_name="Torque Chart",
        content=str({"values": [1, 2, 3]}),
    )
    assert calls["prompt"] == expected_prompt
    assert calls["system_prompt"] == PROMPTS["GENERIC_ANALYSIS_SYSTEM"].format(
        content_type="chart"
    )
    assert caption == "Chart showing torque curves"
    assert entity["entity_name"] == "Torque Chart"
    assert entity["entity_type"] == "chart"
    assert entity["summary"] == "Torque vs RPM"


@pytest.mark.asyncio
async def test_generic_generate_description_only_uses_context_prompt():
    calls = {}

    async def caption_func(prompt, system_prompt=None):
        calls["prompt"] = prompt
        return json.dumps(
            {
                "detailed_description": "Annotated schematic detail",
                "entity_info": {
                    "entity_name": "Schematic Detail",
                    "entity_type": "diagram",
                    "summary": "Detail callout",
                },
            }
        )

    proc = _make_generic_with_caption(caption_func)
    proc._get_context_for_item = lambda item_info: "nearby paragraph about pumps"

    caption, entity = await proc.generate_description_only(
        modal_content="detail-A",
        content_type="diagram",
        item_info={"page_idx": 2, "index": 5},
        entity_name="Schematic Detail",
    )

    expected_prompt = PROMPTS["generic_prompt_with_context"].format(
        context="nearby paragraph about pumps",
        content_type="diagram",
        entity_name="Schematic Detail",
        content="detail-A",
    )
    assert calls["prompt"] == expected_prompt
    assert caption == "Annotated schematic detail"
    assert entity["entity_name"] == "Schematic Detail"


@pytest.mark.asyncio
async def test_generic_generate_description_only_defaults_entity_name_in_prompt():
    calls = {}

    async def caption_func(prompt, system_prompt=None):
        calls["prompt"] = prompt
        return json.dumps(
            {
                "detailed_description": "Generic blob",
                "entity_info": {
                    "entity_name": "Blob",
                    "entity_type": "custom",
                    "summary": "blob summary",
                },
            }
        )

    proc = _make_generic_with_caption(caption_func)
    caption, entity = await proc.generate_description_only(
        modal_content="x",
        content_type="custom",
    )

    assert "descriptive name for this custom" in calls["prompt"]
    assert caption == "Generic blob"
    # Without caller entity_name, parser appends "(entity_type)"
    assert entity["entity_name"] == "Blob (custom)"


@pytest.mark.asyncio
async def test_generic_generate_description_only_soft_fallback_on_llm_error():
    """Main-branch contract: description errors return a 2-tuple fallback."""
    proc = _make_generic_with_caption(AsyncMock(side_effect=RuntimeError("llm down")))

    caption, entity = await proc.generate_description_only(
        modal_content={"k": "v"},
        content_type="widget",
        entity_name="Widget-1",
    )

    assert caption == str({"k": "v"})
    assert entity["entity_name"] == "Widget-1"
    assert entity["entity_type"] == "widget"
    assert entity["summary"].startswith("widget content:")


@pytest.mark.asyncio
async def test_table_generate_description_only_wires_prompt_and_system():
    """Table description prompt wiring; distinct from chunk-template tests (#103)."""
    from raganything.modalprocessors import TableModalProcessor

    calls = {}

    async def caption_func(prompt, system_prompt=None):
        calls["prompt"] = prompt
        calls["system_prompt"] = system_prompt
        return json.dumps(
            {
                "detailed_description": "Spec table for pump model X",
                "entity_info": {
                    "entity_name": "Pump Specs",
                    "entity_type": "table",
                    "summary": "Pump ratings",
                },
            }
        )

    proc = TableModalProcessor.__new__(TableModalProcessor)
    proc.modal_caption_func = caption_func
    proc._get_context_for_item = lambda item_info: "section: ratings"

    caption, entity = await proc.generate_description_only(
        modal_content={
            "img_path": "/t.png",
            "table_caption": ["Specs"],
            "table_body": "| A | B |\n| 1 | 2 |",
            "table_footnote": ["psi"],
        },
        content_type="table",
        item_info={"page_idx": 1},
        entity_name="Pump Specs",
    )

    expected_prompt = PROMPTS["table_prompt_with_context"].format(
        context="section: ratings",
        entity_name="Pump Specs",
        table_img_path="/t.png",
        table_caption=["Specs"],
        table_body="| A | B |\n| 1 | 2 |",
        table_footnote=["psi"],
    )
    assert calls["prompt"] == expected_prompt
    assert calls["system_prompt"] == PROMPTS["TABLE_ANALYSIS_SYSTEM"]
    assert caption == "Spec table for pump model X"
    assert entity["entity_name"] == "Pump Specs"
