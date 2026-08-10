"""Equation/Image generate_description_only prompt and vision wiring.

Complements #106 (generic/table description prompts) and #103 (typed
process_multimodal_content chunk templates). Asserts current main-branch
contracts, including soft-fallback on description errors (#75 changes that
to fail-closed and is intentionally not required here).
"""

import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from raganything.modalprocessors import EquationModalProcessor, ImageModalProcessor
from raganything.prompt import PROMPTS


def _make_equation(caption_func):
    proc = EquationModalProcessor.__new__(EquationModalProcessor)
    proc.modal_caption_func = caption_func
    return proc


def _make_image(caption_func):
    proc = ImageModalProcessor.__new__(ImageModalProcessor)
    proc.modal_caption_func = caption_func
    return proc


def _write_tiny_png(path: Path) -> None:
    # Minimal valid 1x1 PNG
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    path.write_bytes(png_bytes)


@pytest.mark.asyncio
async def test_equation_generate_description_only_wires_prompt_without_context():
    calls = {}

    async def caption_func(prompt, system_prompt=None):
        calls["prompt"] = prompt
        calls["system_prompt"] = system_prompt
        return json.dumps(
            {
                "detailed_description": "Ohm's law relating voltage and current",
                "entity_info": {
                    "entity_name": "Ohms Law",
                    "entity_type": "equation",
                    "summary": "V = IR",
                },
            }
        )

    proc = _make_equation(caption_func)
    caption, entity = await proc.generate_description_only(
        modal_content={"text": "V = I R", "text_format": "latex"},
        content_type="equation",
        entity_name="Ohms Law",
    )

    expected_prompt = PROMPTS["equation_prompt"].format(
        equation_text="V = I R",
        equation_format="latex",
        entity_name="Ohms Law",
    )
    assert calls["prompt"] == expected_prompt
    assert calls["system_prompt"] == PROMPTS["EQUATION_ANALYSIS_SYSTEM"]
    assert caption == "Ohm's law relating voltage and current"
    assert entity["entity_name"] == "Ohms Law"
    assert entity["entity_type"] == "equation"
    assert entity["summary"] == "V = IR"


@pytest.mark.asyncio
async def test_equation_generate_description_only_uses_context_prompt():
    calls = {}

    async def caption_func(prompt, system_prompt=None):
        calls["prompt"] = prompt
        calls["system_prompt"] = system_prompt
        return json.dumps(
            {
                "detailed_description": "Torque balance for shaft design",
                "entity_info": {
                    "entity_name": "Torque Balance",
                    "entity_type": "equation",
                    "summary": "Sum of torques",
                },
            }
        )

    proc = _make_equation(caption_func)
    proc._get_context_for_item = lambda item_info: "section: shaft loading"

    caption, entity = await proc.generate_description_only(
        modal_content={"text": "T = F * r", "text_format": "plain"},
        content_type="equation",
        item_info={"page_idx": 3, "index": 1},
        entity_name="Torque Balance",
    )

    expected_prompt = PROMPTS["equation_prompt_with_context"].format(
        context="section: shaft loading",
        equation_text="T = F * r",
        equation_format="plain",
        entity_name="Torque Balance",
    )
    assert calls["prompt"] == expected_prompt
    assert calls["system_prompt"] == PROMPTS["EQUATION_ANALYSIS_SYSTEM"]
    assert caption == "Torque balance for shaft design"
    assert entity["entity_name"] == "Torque Balance"


@pytest.mark.asyncio
async def test_equation_generate_description_only_defaults_entity_name_and_json_string():
    calls = {}

    async def caption_func(prompt, system_prompt=None):
        calls["prompt"] = prompt
        return json.dumps(
            {
                "detailed_description": "Energy conservation identity",
                "entity_info": {
                    "entity_name": "Energy Id",
                    "entity_type": "equation",
                    "summary": "E conserved",
                },
            }
        )

    proc = _make_equation(caption_func)
    caption, entity = await proc.generate_description_only(
        modal_content=json.dumps({"text": "E1 = E2", "text_format": "latex"}),
        content_type="equation",
    )

    assert "descriptive name for this equation" in calls["prompt"]
    assert "E1 = E2" in calls["prompt"]
    assert caption == "Energy conservation identity"
    # Without caller entity_name, parser appends "(entity_type)"
    assert entity["entity_name"] == "Energy Id (equation)"


@pytest.mark.asyncio
async def test_equation_generate_description_only_soft_fallback_on_llm_error():
    """Main-branch contract: description errors return a 2-tuple fallback."""
    proc = _make_equation(AsyncMock(side_effect=RuntimeError("llm down")))

    caption, entity = await proc.generate_description_only(
        modal_content={"text": "a^2 + b^2 = c^2", "text_format": "latex"},
        content_type="equation",
        entity_name="Pythagoras",
    )

    assert caption == str({"text": "a^2 + b^2 = c^2", "text_format": "latex"})
    assert entity["entity_name"] == "Pythagoras"
    assert entity["entity_type"] == "equation"
    assert entity["summary"].startswith("Equation content:")


@pytest.mark.asyncio
async def test_image_generate_description_only_wires_vision_prompt_and_base64(
    tmp_path,
):
    image_path = tmp_path / "figure.png"
    _write_tiny_png(image_path)
    expected_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    calls = {}

    async def caption_func(prompt, image_data=None, system_prompt=None):
        calls["prompt"] = prompt
        calls["image_data"] = image_data
        calls["system_prompt"] = system_prompt
        return json.dumps(
            {
                "detailed_description": "Pump cutaway diagram",
                "entity_info": {
                    "entity_name": "Pump Cutaway",
                    "entity_type": "image",
                    "summary": "Internal pump view",
                },
            }
        )

    proc = _make_image(caption_func)
    caption, entity = await proc.generate_description_only(
        modal_content={
            "img_path": str(image_path),
            "image_caption": ["Pump internals"],
            "image_footnote": ["see §3"],
        },
        content_type="image",
        entity_name="Pump Cutaway",
    )

    expected_prompt = PROMPTS["vision_prompt"].format(
        entity_name="Pump Cutaway",
        image_path=str(image_path),
        captions=["Pump internals"],
        footnotes=["see §3"],
    )
    assert calls["prompt"] == expected_prompt
    assert calls["image_data"] == expected_b64
    assert calls["system_prompt"] == PROMPTS["IMAGE_ANALYSIS_SYSTEM"]
    assert caption == "Pump cutaway diagram"
    assert entity["entity_name"] == "Pump Cutaway"
    assert entity["entity_type"] == "image"


@pytest.mark.asyncio
async def test_image_generate_description_only_uses_context_prompt_and_aliases(
    tmp_path,
):
    image_path = tmp_path / "alias.png"
    _write_tiny_png(image_path)
    calls = {}

    async def caption_func(prompt, image_data=None, system_prompt=None):
        calls["prompt"] = prompt
        calls["image_data"] = image_data
        return json.dumps(
            {
                "detailed_description": "Wiring schematic detail",
                "entity_info": {
                    "entity_name": "Wiring Detail",
                    "entity_type": "image",
                    "summary": "Connector pinout",
                },
            }
        )

    proc = _make_image(caption_func)
    proc._get_context_for_item = lambda item_info: "nearby text about connectors"

    caption, entity = await proc.generate_description_only(
        modal_content={
            "img_path": str(image_path),
            "img_caption": ["Connector"],
            "img_footnote": ["pin 1"],
        },
        content_type="image",
        item_info={"page_idx": 1, "index": 2},
        entity_name="Wiring Detail",
    )

    expected_prompt = PROMPTS["vision_prompt_with_context"].format(
        context="nearby text about connectors",
        entity_name="Wiring Detail",
        image_path=str(image_path),
        captions=["Connector"],
        footnotes=["pin 1"],
    )
    assert calls["prompt"] == expected_prompt
    assert calls["image_data"]
    assert caption == "Wiring schematic detail"
    assert entity["entity_name"] == "Wiring Detail"


@pytest.mark.asyncio
async def test_image_generate_description_only_soft_fallback_when_encode_fails(
    tmp_path,
):
    """Main-branch contract: encode failure is soft-fallback, not raised."""
    image_path = tmp_path / "unreadable.png"
    _write_tiny_png(image_path)

    proc = _make_image(AsyncMock())
    proc._encode_image_to_base64 = lambda path: ""

    caption, entity = await proc.generate_description_only(
        modal_content={"img_path": str(image_path)},
        content_type="image",
        entity_name="Broken Encode",
    )

    assert caption == str({"img_path": str(image_path)})
    assert entity["entity_name"] == "Broken Encode"
    assert entity["entity_type"] == "image"
    assert entity["summary"].startswith("Image content:")
    proc.modal_caption_func.assert_not_awaited()
