"""Outer soft-fallback for typed/generic process_multimodal_content.

Happy-path chunk forwarding is covered by open PRs (#103/#106). These tests
lock the main-branch contract when description succeeds but chunk/entity
persistence fails: callers still get a 2-tuple fallback instead of a raise.
Distinct from generate_description_only soft-fallback (#106/#109) and fail-
closed production changes (#75).
"""

from unittest.mock import AsyncMock

import pytest

from raganything.modalprocessors import (
    EquationModalProcessor,
    GenericModalProcessor,
    ImageModalProcessor,
    TableModalProcessor,
)


class _BareModal:
    """Skip LightRAG-backed BaseModalProcessor.__init__."""

    def __init__(self):
        pass


def _as_processor(cls):
    proc = _BareModal()
    proc.__class__ = cls
    return proc


async def _failing_create_entity_and_chunk(*args, **kwargs):
    raise RuntimeError("storage upsert failed")


@pytest.mark.asyncio
async def test_image_process_multimodal_content_soft_fallback_on_chunk_failure():
    proc = _as_processor(ImageModalProcessor)

    async def fake_generate_description_only(*args, **kwargs):
        return "enhanced caption", {
            "entity_name": "Pump (image)",
            "entity_type": "image",
            "summary": "pump",
        }

    proc.generate_description_only = fake_generate_description_only
    proc._create_entity_and_chunk = _failing_create_entity_and_chunk

    modal_content = {
        "img_path": "/abs/pump.png",
        "image_caption": ["Pump"],
        "image_footnote": [],
    }
    caption, entity = await proc.process_multimodal_content(
        modal_content,
        "image",
        entity_name="Pump Diagram",
    )

    assert caption == str(modal_content)
    assert entity["entity_name"] == "Pump Diagram"
    assert entity["entity_type"] == "image"
    assert entity["summary"].startswith("Image content:")


@pytest.mark.asyncio
async def test_image_process_multimodal_content_fallback_hashes_default_name():
    proc = _as_processor(ImageModalProcessor)
    proc.generate_description_only = AsyncMock(
        return_value=(
            "caption",
            {"entity_name": "X (image)", "entity_type": "image", "summary": "x"},
        )
    )
    proc._create_entity_and_chunk = _failing_create_entity_and_chunk

    modal_content = {"img_path": "/a.png"}
    caption, entity = await proc.process_multimodal_content(modal_content, "image")

    assert caption == str(modal_content)
    assert entity["entity_name"].startswith("image_")
    assert entity["entity_type"] == "image"


@pytest.mark.asyncio
async def test_table_process_multimodal_content_soft_fallback_on_chunk_failure():
    proc = _as_processor(TableModalProcessor)
    proc.generate_description_only = AsyncMock(
        return_value=(
            "table summary",
            {
                "entity_name": "Specs (table)",
                "entity_type": "table",
                "summary": "ratings",
            },
        )
    )
    proc._create_entity_and_chunk = _failing_create_entity_and_chunk

    modal_content = {
        "table_caption": ["Specs"],
        "table_body": "| A | B |",
        "table_footnote": [],
    }
    caption, entity = await proc.process_multimodal_content(
        modal_content,
        "table",
        entity_name="Spec Table",
    )

    assert caption == str(modal_content)
    assert entity["entity_name"] == "Spec Table"
    assert entity["entity_type"] == "table"
    assert entity["summary"].startswith("Table content:")


@pytest.mark.asyncio
async def test_equation_process_multimodal_content_soft_fallback_on_chunk_failure():
    proc = _as_processor(EquationModalProcessor)
    proc.generate_description_only = AsyncMock(
        return_value=(
            "eq summary",
            {
                "entity_name": "Ohm (equation)",
                "entity_type": "equation",
                "summary": "V=IR",
            },
        )
    )
    proc._create_entity_and_chunk = _failing_create_entity_and_chunk

    modal_content = {"text": "V = I R", "text_format": "latex"}
    caption, entity = await proc.process_multimodal_content(
        modal_content,
        "equation",
        entity_name="Ohms Law",
    )

    assert caption == str(modal_content)
    assert entity["entity_name"] == "Ohms Law"
    assert entity["entity_type"] == "equation"
    assert entity["summary"].startswith("Equation content:")


@pytest.mark.asyncio
async def test_generic_process_multimodal_content_soft_fallback_on_chunk_failure():
    proc = _as_processor(GenericModalProcessor)
    proc.generate_description_only = AsyncMock(
        return_value=(
            "audio caption",
            {
                "entity_name": "Clip (audio)",
                "entity_type": "audio",
                "summary": "beep",
            },
        )
    )
    proc._create_entity_and_chunk = _failing_create_entity_and_chunk

    modal_content = {"text": "beep at 440Hz"}
    caption, entity = await proc.process_multimodal_content(
        modal_content,
        "audio",
        entity_name="Audio Clip",
    )

    assert caption == str(modal_content)
    assert entity["entity_name"] == "Audio Clip"
    assert entity["entity_type"] == "audio"
    assert entity["summary"].startswith("audio content:")
