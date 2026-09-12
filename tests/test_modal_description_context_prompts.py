"""Table/equation/generic captions must include nearby page context.

#174 locks ImageModalProcessor vision_prompt_with_context. These processors
use the same contract: generate_description_only switches to the
``*_prompt_with_context`` template when ``_get_context_for_item`` returns
text, and stays on the plain template when item_info is omitted or context
is empty.

A regression here stores torque tables, formulas, or chart blocks without
the adjacent procedure text that makes them searchable.
"""

from __future__ import annotations

import pytest

from raganything.modalprocessors import (
    EquationModalProcessor,
    GenericModalProcessor,
    TableModalProcessor,
)


class _RecordingCaption:
    def __init__(self):
        self.calls = []

    async def __call__(self, prompt, system_prompt=None, **kwargs):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return '{"detailed_description": "ok", "entity_info": {}}'


def _processor(cls, caption, parse_attr):
    proc = cls.__new__(cls)
    proc.modal_caption_func = caption
    setattr(
        proc,
        parse_attr,
        lambda response, entity_name=None, content_type=None: (
            "caption",
            {
                "entity_name": entity_name or "entity",
                "entity_type": "modal",
                "summary": "caption",
            },
        ),
    )
    return proc


def _should_not_extract(item_info):
    raise AssertionError("must not extract context without item_info")


@pytest.mark.asyncio
async def test_table_prompt_includes_surrounding_context():
    caption = _RecordingCaption()
    proc = _processor(TableModalProcessor, caption, "_parse_table_response")
    nearby = "Torque the caliper bolts to 28 N·m after seating the pads."
    proc._get_context_for_item = lambda item_info: nearby

    await proc.generate_description_only(
        {
            "table_body": "| Bolt | N·m |\n| --- | --- |\n| M8 | 28 |",
            "table_caption": ["Caliper torque"],
            "table_footnote": ["Dry threads"],
            "img_path": "/tmp/torque.png",
        },
        "table",
        item_info={"page_idx": 7, "index": 3},
        entity_name="caliper-torque",
    )

    assert len(caption.calls) == 1
    prompt = caption.calls[0]["prompt"]
    assert "Context from surrounding content:" in prompt
    assert nearby in prompt
    assert "Caliper torque" in prompt
    assert "28" in prompt
    assert "/tmp/torque.png" in prompt
    assert "Dry threads" in prompt


@pytest.mark.asyncio
async def test_table_missing_item_info_uses_plain_prompt():
    caption = _RecordingCaption()
    proc = _processor(TableModalProcessor, caption, "_parse_table_response")
    proc._get_context_for_item = _should_not_extract

    await proc.generate_description_only(
        {"table_body": "| A | B |\n| 1 | 2 |", "table_caption": ["Specs"]},
        "table",
        item_info=None,
    )

    prompt = caption.calls[0]["prompt"]
    assert "Context from surrounding content:" not in prompt
    assert "Specs" in prompt
    assert "| A | B |" in prompt


@pytest.mark.asyncio
async def test_table_empty_context_uses_plain_prompt():
    caption = _RecordingCaption()
    proc = _processor(TableModalProcessor, caption, "_parse_table_response")
    proc._get_context_for_item = lambda item_info: ""

    await proc.generate_description_only(
        {"table_body": "a,b\n1,2"},
        "table",
        item_info={"page_idx": 0},
    )

    prompt = caption.calls[0]["prompt"]
    assert "Context from surrounding content:" not in prompt
    assert "a,b" in prompt


@pytest.mark.asyncio
async def test_equation_prompt_includes_surrounding_context():
    caption = _RecordingCaption()
    proc = _processor(EquationModalProcessor, caption, "_parse_equation_response")
    nearby = "Q is hydraulic flow in L/min; ΔP is the measured pressure drop."
    proc._get_context_for_item = lambda item_info: nearby

    await proc.generate_description_only(
        {"text": "Q = C_d A \\sqrt{2\\Delta P / \\rho}", "text_format": "latex"},
        "equation",
        item_info={"page_idx": 12, "index": 1},
        entity_name="orifice-flow",
    )

    prompt = caption.calls[0]["prompt"]
    assert "Context from surrounding content:" in prompt
    assert nearby in prompt
    assert "Q = C_d A" in prompt
    assert "latex" in prompt


@pytest.mark.asyncio
async def test_equation_missing_item_info_uses_plain_prompt():
    caption = _RecordingCaption()
    proc = _processor(EquationModalProcessor, caption, "_parse_equation_response")
    proc._get_context_for_item = _should_not_extract

    await proc.generate_description_only(
        {"text": "F = ma", "text_format": "latex"},
        "equation",
        item_info=None,
    )

    prompt = caption.calls[0]["prompt"]
    assert "Context from surrounding content:" not in prompt
    assert "F = ma" in prompt


@pytest.mark.asyncio
async def test_generic_prompt_includes_surrounding_context():
    caption = _RecordingCaption()
    proc = _processor(GenericModalProcessor, caption, "_parse_generic_response")
    nearby = "The bleed sequence starts at the right-rear caliper."
    proc._get_context_for_item = lambda item_info: nearby

    await proc.generate_description_only(
        {"type": "chart", "content": "Bleed order: RR → LR → RF → LF"},
        "chart",
        item_info={"page_idx": 3, "index": 0},
        entity_name="bleed-order",
    )

    prompt = caption.calls[0]["prompt"]
    assert "Context from surrounding content:" in prompt
    assert nearby in prompt
    assert "Bleed order: RR → LR → RF → LF" in prompt
    assert "chart" in prompt


@pytest.mark.asyncio
async def test_generic_missing_item_info_uses_plain_prompt():
    caption = _RecordingCaption()
    proc = _processor(GenericModalProcessor, caption, "_parse_generic_response")
    proc._get_context_for_item = _should_not_extract

    await proc.generate_description_only(
        {"type": "code", "content": "SET TORQUE 28"},
        "code",
        item_info=None,
    )

    prompt = caption.calls[0]["prompt"]
    assert "Context from surrounding content:" not in prompt
    assert "SET TORQUE 28" in prompt
    assert "code" in prompt
