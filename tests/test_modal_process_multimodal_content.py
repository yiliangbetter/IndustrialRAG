"""Happy-path orchestration for typed modal process_multimodal_content.

Verifies image/table/equation processors build chunk templates from description
output and forward doc_id / chunk_order_index into _create_entity_and_chunk.
Distinct from fail-closed soft-fallback coverage (#75) and JSON parse (#79).
"""

import pytest

from raganything.modalprocessors import (
    EquationModalProcessor,
    ImageModalProcessor,
    TableModalProcessor,
)
from raganything.prompt import PROMPTS


class _BareModal:
    """Skip LightRAG-backed BaseModalProcessor.__init__."""

    def __init__(self):
        pass


def _as_processor(cls):
    proc = _BareModal()
    proc.__class__ = cls
    return proc


@pytest.mark.asyncio
async def test_image_process_multimodal_content_builds_chunk_and_forwards_ids():
    proc = _as_processor(ImageModalProcessor)
    entity_info = {
        "entity_name": "Pump Diagram (image)",
        "entity_type": "image",
        "summary": "A pump schematic",
    }
    captured = {}

    async def fake_generate_description_only(
        modal_content, content_type, item_info=None, entity_name=None
    ):
        assert content_type == "image"
        assert entity_name == "Pump Diagram"
        return "enhanced image caption", entity_info

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
        return "enhanced image caption", {"chunk_id": "chunk-1"}, []

    proc.generate_description_only = fake_generate_description_only
    proc._create_entity_and_chunk = fake_create_entity_and_chunk

    modal_content = {
        "img_path": "/abs/pump.png",
        "image_caption": ["Pump"],
        "image_footnote": ["see notes"],
    }
    result = await proc.process_multimodal_content(
        modal_content,
        "image",
        file_path="manual.pdf",
        entity_name="Pump Diagram",
        batch_mode=True,
        doc_id="doc-1",
        chunk_order_index=3,
    )

    expected_chunk = PROMPTS["image_chunk"].format(
        image_path="/abs/pump.png",
        captions="Pump",
        footnotes="see notes",
        enhanced_caption="enhanced image caption",
    )
    assert captured["modal_chunk"] == expected_chunk
    assert captured["entity_info"] == entity_info
    assert captured["file_path"] == "manual.pdf"
    assert captured["batch_mode"] is True
    assert captured["doc_id"] == "doc-1"
    assert captured["chunk_order_index"] == 3
    assert result[0] == "enhanced image caption"


@pytest.mark.asyncio
async def test_image_process_multimodal_content_accepts_json_string_and_aliases():
    proc = _as_processor(ImageModalProcessor)
    captured = {}

    async def fake_generate_description_only(*args, **kwargs):
        return "caption", {
            "entity_name": "X (image)",
            "entity_type": "image",
            "summary": "X",
        }

    async def fake_create_entity_and_chunk(modal_chunk, *args, **kwargs):
        captured["modal_chunk"] = modal_chunk
        return "caption", {}, []

    proc.generate_description_only = fake_generate_description_only
    proc._create_entity_and_chunk = fake_create_entity_and_chunk

    await proc.process_multimodal_content(
        '{"img_path":"/a.png","img_caption":["Alt"],"img_footnote":[]}',
        "image",
    )

    assert "/a.png" in captured["modal_chunk"]
    assert "Alt" in captured["modal_chunk"]
    assert "None" in captured["modal_chunk"]  # empty footnotes default


@pytest.mark.asyncio
async def test_table_process_multimodal_content_builds_chunk():
    proc = _as_processor(TableModalProcessor)
    captured = {}

    async def fake_generate_description_only(*args, **kwargs):
        return "table summary", {
            "entity_name": "Spec Table (table)",
            "entity_type": "table",
            "summary": "table summary",
        }

    async def fake_create_entity_and_chunk(
        modal_chunk,
        entity_info_arg,
        file_path,
        batch_mode=False,
        doc_id=None,
        chunk_order_index=0,
    ):
        captured["modal_chunk"] = modal_chunk
        captured["file_path"] = file_path
        captured["doc_id"] = doc_id
        captured["chunk_order_index"] = chunk_order_index
        return "table summary", {}, []

    proc.generate_description_only = fake_generate_description_only
    proc._create_entity_and_chunk = fake_create_entity_and_chunk

    await proc.process_multimodal_content(
        {
            "img_path": "/t.png",
            "table_caption": ["Specs"],
            "table_body": "| A | B |\n|---|---|\n| 1 | 2 |",
            "table_footnote": ["units"],
        },
        "table",
        file_path="doc.pdf",
        doc_id="doc-t",
        chunk_order_index=1,
    )

    expected = PROMPTS["table_chunk"].format(
        table_img_path="/t.png",
        table_caption="Specs",
        table_body="| A | B |\n|---|---|\n| 1 | 2 |",
        table_footnote="units",
        enhanced_caption="table summary",
    )
    assert captured["modal_chunk"] == expected
    assert captured["file_path"] == "doc.pdf"
    assert captured["doc_id"] == "doc-t"
    assert captured["chunk_order_index"] == 1


@pytest.mark.asyncio
async def test_equation_process_multimodal_content_builds_chunk():
    proc = _as_processor(EquationModalProcessor)
    captured = {}

    async def fake_generate_description_only(*args, **kwargs):
        return "eq summary", {
            "entity_name": "Ohm (equation)",
            "entity_type": "equation",
            "summary": "eq summary",
        }

    async def fake_create_entity_and_chunk(modal_chunk, *args, **kwargs):
        captured["modal_chunk"] = modal_chunk
        return "eq summary", {}, []

    proc.generate_description_only = fake_generate_description_only
    proc._create_entity_and_chunk = fake_create_entity_and_chunk

    await proc.process_multimodal_content(
        {"text": "V = IR", "text_format": "latex"},
        "equation",
        file_path="eq.pdf",
    )

    expected = PROMPTS["equation_chunk"].format(
        equation_text="V = IR",
        equation_format="latex",
        enhanced_caption="eq summary",
    )
    assert captured["modal_chunk"] == expected
