"""Image captions must include nearby page context in the vision prompt.

``ImageModalProcessor.generate_description_only`` switches to
``vision_prompt_with_context`` when ``_get_context_for_item`` returns text.
Dropping that branch (or formatting without ``{context}``) makes figure
descriptions ignore adjacent procedure steps — the usual industrial-manual
failure mode where a diagram is stored without the torque/bleed instructions
printed next to it.
"""

from __future__ import annotations

import pytest

from raganything.modalprocessors import ImageModalProcessor


class _RecordingCaption:
    def __init__(self):
        self.calls = []

    async def __call__(self, prompt, image_data=None, system_prompt=None, **kwargs):
        self.calls.append(
            {
                "prompt": prompt,
                "image_data": image_data,
                "system_prompt": system_prompt,
            }
        )
        return '{"detailed_description": "ok", "entity_info": {}}'


def _processor(caption):
    proc = ImageModalProcessor.__new__(ImageModalProcessor)
    proc.modal_caption_func = caption
    proc._encode_image_to_base64 = lambda path: "ZmFrZS1iYXNlNjQ="
    proc._parse_response = lambda response, entity_name=None: (
        "caption",
        {
            "entity_name": entity_name or "figure",
            "entity_type": "image",
            "summary": "caption",
        },
    )
    return proc


@pytest.mark.asyncio
async def test_vision_prompt_includes_surrounding_context(tmp_path):
    image = tmp_path / "caliper.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    caption = _RecordingCaption()
    proc = _processor(caption)
    nearby = "Bleed the brake line before replacing the caliper."
    proc._get_context_for_item = lambda item_info: nearby

    await proc.generate_description_only(
        {
            "img_path": str(image),
            "image_caption": ["Front caliper"],
            "image_footnote": ["See §4.2"],
        },
        "image",
        item_info={"page_idx": 4, "index": 2},
        entity_name="front-caliper",
    )

    assert len(caption.calls) == 1
    prompt = caption.calls[0]["prompt"]
    assert "Context from surrounding content:" in prompt
    assert nearby in prompt
    assert "Front caliper" in prompt
    assert str(image) in prompt
    assert caption.calls[0]["image_data"] == "ZmFrZS1iYXNlNjQ="


@pytest.mark.asyncio
async def test_missing_item_info_uses_plain_vision_prompt(tmp_path):
    image = tmp_path / "pump.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    caption = _RecordingCaption()
    proc = _processor(caption)

    def _should_not_extract(item_info):
        raise AssertionError("must not extract context without item_info")

    proc._get_context_for_item = _should_not_extract

    await proc.generate_description_only(
        {"img_path": str(image), "image_caption": ["Pump"]},
        "image",
        item_info=None,
    )

    prompt = caption.calls[0]["prompt"]
    assert "Additional context:" in prompt
    assert "Context from surrounding content:" not in prompt
    assert "Pump" in prompt
