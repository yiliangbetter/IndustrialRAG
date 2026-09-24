import base64
import json
from unittest.mock import AsyncMock

import pytest
from lightrag.utils import compute_mdhash_id

from raganything.modalprocessors import (
    EquationModalProcessor,
    GenericModalProcessor,
    ImageModalProcessor,
    TableModalProcessor,
)
from raganything.prompt import PROMPTS

VALID_RESPONSE = json.dumps(
    {
        "detailed_description": "enhanced",
        "entity_info": {
            "entity_name": "model name",
            "entity_type": "asset",
            "summary": "short summary",
        },
    }
)


def _processor(processor_type, caption_func=None):
    processor = object.__new__(processor_type)
    processor.modal_caption_func = caption_func
    return processor


def _content_for(processor_type, image_path):
    if processor_type is ImageModalProcessor:
        return {
            "img_path": str(image_path),
            "image_caption": ["caption"],
            "image_footnote": ["footnote"],
        }
    if processor_type is TableModalProcessor:
        return {
            "img_path": "table.png",
            "table_caption": ["caption"],
            "table_body": "A | B",
            "table_footnote": ["footnote"],
        }
    if processor_type is EquationModalProcessor:
        return {"text": "x = 1", "text_format": "latex"}
    return {"kind": "audio", "value": 7}


def _expected_prompt(processor_type, content, context):
    shared = {"context": context} if context else {}
    if processor_type is ImageModalProcessor:
        values = {
            "entity_name": "chosen",
            "image_path": content["img_path"],
            "captions": content["image_caption"],
            "footnotes": content["image_footnote"],
        }
        keys = "vision_prompt", "vision_prompt_with_context"
    elif processor_type is TableModalProcessor:
        values = {
            "entity_name": "chosen",
            "table_img_path": content["img_path"],
            "table_caption": content["table_caption"],
            "table_body": content["table_body"],
            "table_footnote": content["table_footnote"],
        }
        keys = "table_prompt", "table_prompt_with_context"
    elif processor_type is EquationModalProcessor:
        values = {
            "entity_name": "chosen",
            "equation_text": content["text"],
            "equation_format": content["text_format"],
        }
        keys = "equation_prompt", "equation_prompt_with_context"
    else:
        values = {
            "entity_name": "chosen",
            "content_type": "audio",
            "content": str(content),
        }
        keys = "generic_prompt", "generic_prompt_with_context"
    return PROMPTS[keys[bool(context)]].format(**shared, **values)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("processor_type", "content_type", "system_key"),
    [
        (ImageModalProcessor, "image", "IMAGE_ANALYSIS_SYSTEM"),
        (TableModalProcessor, "table", "TABLE_ANALYSIS_SYSTEM"),
        (EquationModalProcessor, "equation", "EQUATION_ANALYSIS_SYSTEM"),
        (GenericModalProcessor, "audio", "GENERIC_ANALYSIS_SYSTEM"),
    ],
)
@pytest.mark.parametrize("context", ["", "nearby text"])
async def test_generation_preserves_prompt_selection_and_model_kwargs(
    tmp_path, processor_type, content_type, system_key, context
):
    image_path = tmp_path / "image.bin"
    image_path.write_bytes(b"image bytes")
    content = _content_for(processor_type, image_path)
    calls = []

    async def caption_func(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return VALID_RESPONSE

    processor = _processor(processor_type, caption_func)
    processor._get_context_for_item = lambda _item: context

    result = await processor.generate_description_only(
        content, content_type, {"index": 2}, "chosen"
    )

    assert result == (
        "enhanced",
        {"entity_name": "chosen", "entity_type": "asset", "summary": "short summary"},
    )
    assert calls[0][0] == _expected_prompt(processor_type, content, context)
    expected_system = PROMPTS[system_key]
    if processor_type is GenericModalProcessor:
        expected_system = expected_system.format(content_type=content_type)
    assert calls[0][1]["system_prompt"] == expected_system
    if processor_type is ImageModalProcessor:
        assert calls[0][1]["image_data"] == base64.b64encode(b"image bytes").decode()
    else:
        assert set(calls[0][1]) == {"system_prompt"}


@pytest.mark.parametrize(
    ("processor_type", "method_name", "args", "content_type"),
    [
        (ImageModalProcessor, "_parse_response", (), "image"),
        (TableModalProcessor, "_parse_table_response", (), "table"),
        (EquationModalProcessor, "_parse_equation_response", (), "equation"),
        (GenericModalProcessor, "_parse_generic_response", ("audio",), "audio"),
    ],
)
def test_modality_parser_names_preserve_success_and_fallback(
    processor_type, method_name, args, content_type
):
    processor = _processor(processor_type)
    parse = getattr(processor, method_name)

    assert parse(VALID_RESPONSE, None, *args) == (
        "enhanced",
        {
            "entity_name": "model name (asset)",
            "entity_type": "asset",
            "summary": "short summary",
        },
    )

    raw = "<think>private reasoning</think>plain fallback"
    description, entity = parse(raw, "forced", *args)
    assert description == "plain fallback"
    assert entity == {
        "entity_name": "forced",
        "entity_type": content_type,
        "summary": "plain fallback",
    }


def _expected_chunk(processor_type, content):
    if processor_type is ImageModalProcessor:
        return PROMPTS["image_chunk"].format(
            image_path=content["img_path"],
            captions="caption",
            footnotes="footnote",
            enhanced_caption="enhanced",
        )
    if processor_type is TableModalProcessor:
        return PROMPTS["table_chunk"].format(
            table_img_path=content["img_path"],
            table_caption="caption",
            table_body=content["table_body"],
            table_footnote="footnote",
            enhanced_caption="enhanced",
        )
    if processor_type is EquationModalProcessor:
        return PROMPTS["equation_chunk"].format(
            equation_text=content["text"],
            equation_format=content["text_format"],
            enhanced_caption="enhanced",
        )
    return PROMPTS["generic_chunk"].format(
        content_type="Audio",
        content=str(content),
        enhanced_caption="enhanced",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("processor_type", "content_type"),
    [
        (ImageModalProcessor, "image"),
        (TableModalProcessor, "table"),
        (EquationModalProcessor, "equation"),
        (GenericModalProcessor, "audio"),
    ],
)
async def test_process_formats_chunk_and_delegates_or_falls_back(
    tmp_path, processor_type, content_type
):
    image_path = tmp_path / "image.bin"
    image_path.write_bytes(b"image bytes")
    content = _content_for(processor_type, image_path)
    entity = {
        "entity_name": "chosen",
        "entity_type": content_type,
        "summary": "short summary",
    }
    processor = _processor(processor_type)
    processor.generate_description_only = AsyncMock(return_value=("enhanced", entity))
    processor._create_entity_and_chunk = AsyncMock(return_value=("stored", entity, []))

    result = await processor.process_multimodal_content(
        content,
        content_type,
        "manual.pdf",
        "chosen",
        {"index": 4},
        True,
        "doc-1",
        9,
    )

    assert result == ("stored", entity, [])
    processor.generate_description_only.assert_awaited_once_with(
        content, content_type, {"index": 4}, "chosen"
    )
    processor._create_entity_and_chunk.assert_awaited_once_with(
        _expected_chunk(processor_type, content),
        entity,
        "manual.pdf",
        True,
        "doc-1",
        9,
    )

    processor._create_entity_and_chunk.side_effect = RuntimeError("storage failed")
    fallback = await processor.process_multimodal_content(
        content, content_type, entity_name=None
    )
    text = str(content)
    label = (
        content_type
        if processor_type is GenericModalProcessor
        else content_type.title()
    )
    assert fallback == (
        text,
        {
            "entity_name": f"{content_type}_{compute_mdhash_id(text)}",
            "entity_type": content_type,
            "summary": f"{label} content: {text[:100]}",
        },
    )
