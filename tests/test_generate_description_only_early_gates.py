"""Early fail/fallback gates for modal generate_description_only.

Batch multimodal ingest (stage 1) only stores what these methods return.
A missing image path, encode failure, or caption-LLM exception must fall
back to a typed entity instead of raising and aborting the whole document.
"""

import json

import pytest
from lightrag.utils import compute_mdhash_id

from raganything.modalprocessors import (
    EquationModalProcessor,
    GenericModalProcessor,
    ImageModalProcessor,
    TableModalProcessor,
)


def _bare(cls):
    """Construct a processor without LightRAG storage wiring."""
    proc = cls.__new__(cls)
    proc.content_source = None
    return proc


def _ok_caption(prompt, **kwargs):
    return json.dumps(
        {
            "detailed_description": "ok-description",
            "entity_info": {
                "entity_name": "NamedEntity",
                "entity_type": "equation",
                "summary": "ok-summary",
            },
        }
    )


class TestImageDescriptionEarlyGates:
    @pytest.mark.asyncio
    async def test_missing_img_path_falls_back_without_calling_vision(self):
        proc = _bare(ImageModalProcessor)
        calls = []

        async def caption(*args, **kwargs):
            calls.append(1)
            return "unused"

        proc.modal_caption_func = caption

        caption_text, entity = await proc.generate_description_only(
            {"type": "image", "image_caption": ["cap"]},
            "image",
        )

        assert calls == []
        assert caption_text == str({"type": "image", "image_caption": ["cap"]})
        assert entity["entity_type"] == "image"
        assert entity["entity_name"].startswith("image_")
        assert "Image content:" in entity["summary"]

    @pytest.mark.asyncio
    async def test_raw_string_content_has_no_path_and_falls_back(self):
        proc = _bare(ImageModalProcessor)
        proc.modal_caption_func = _ok_caption

        caption_text, entity = await proc.generate_description_only(
            "not-json-and-not-a-path",
            "image",
            entity_name="ForcedName",
        )

        assert caption_text == "not-json-and-not-a-path"
        assert entity["entity_name"] == "ForcedName"
        assert entity["entity_type"] == "image"

    @pytest.mark.asyncio
    async def test_missing_file_falls_back(self, tmp_path):
        proc = _bare(ImageModalProcessor)
        proc.modal_caption_func = _ok_caption
        missing = str(tmp_path / "gone.png")

        caption_text, entity = await proc.generate_description_only(
            {"img_path": missing},
            "image",
        )

        assert missing in caption_text or caption_text == str({"img_path": missing})
        assert entity["entity_type"] == "image"
        assert (
            entity["entity_name"]
            == f"image_{compute_mdhash_id(str({'img_path': missing}))}"
        )

    @pytest.mark.asyncio
    async def test_encode_failure_falls_back_after_file_exists(self, tmp_path):
        image = tmp_path / "fig.png"
        image.write_bytes(b"\x89PNG\r\n")
        proc = _bare(ImageModalProcessor)
        calls = []

        async def caption(*args, **kwargs):
            calls.append(1)
            return "unused"

        proc.modal_caption_func = caption
        proc._encode_image_to_base64 = lambda path: ""

        caption_text, entity = await proc.generate_description_only(
            {"img_path": str(image)},
            "image",
        )

        assert calls == []
        assert entity["entity_type"] == "image"
        assert "Image content:" in entity["summary"]
        assert caption_text == str({"img_path": str(image)})


class TestEquationDescriptionEarlyGates:
    @pytest.mark.asyncio
    async def test_json_string_uses_text_and_format_fields(self):
        proc = _bare(EquationModalProcessor)
        seen = {}

        async def caption(prompt, **kwargs):
            seen["prompt"] = prompt
            seen["system"] = kwargs.get("system_prompt")
            return json.dumps(
                {
                    "detailed_description": "mass-energy",
                    "entity_info": {
                        "entity_name": "MassEnergy",
                        "entity_type": "equation",
                        "summary": "E equals mc squared",
                    },
                }
            )

        proc.modal_caption_func = caption

        caption_text, entity = await proc.generate_description_only(
            json.dumps({"text": "E=mc^2", "text_format": "latex"}),
            "equation",
        )

        assert "E=mc^2" in seen["prompt"]
        assert "latex" in seen["prompt"]
        assert caption_text == "mass-energy"
        assert entity["entity_name"] == "MassEnergy (equation)"
        assert entity["entity_type"] == "equation"

    @pytest.mark.asyncio
    async def test_raw_non_json_string_does_not_use_body_as_equation_text(self):
        """Current contract: JSONDecodeError wraps the string as {'equation': ...},
        so equation_text is None rather than the raw LaTeX."""
        proc = _bare(EquationModalProcessor)
        seen = {}

        async def caption(prompt, **kwargs):
            seen["prompt"] = prompt
            return json.dumps(
                {
                    "detailed_description": "fallback-desc",
                    "entity_info": {
                        "entity_name": "Eq",
                        "entity_type": "equation",
                        "summary": "sum",
                    },
                }
            )

        proc.modal_caption_func = caption

        await proc.generate_description_only("E=mc^2", "equation")

        assert "Equation: None" in seen["prompt"]
        assert "E=mc^2" not in seen["prompt"]

    @pytest.mark.asyncio
    async def test_caption_exception_falls_back_with_hash_name(self):
        proc = _bare(EquationModalProcessor)

        async def boom(*args, **kwargs):
            raise RuntimeError("llm down")

        proc.modal_caption_func = boom
        content = {"text": "a^2+b^2=c^2", "text_format": "latex"}

        caption_text, entity = await proc.generate_description_only(content, "equation")

        assert caption_text == str(content)
        assert entity["entity_type"] == "equation"
        assert entity["entity_name"] == f"equation_{compute_mdhash_id(str(content))}"
        assert "Equation content:" in entity["summary"]


class TestTableDescriptionEarlyGates:
    @pytest.mark.asyncio
    async def test_raw_string_is_treated_as_table_body(self):
        proc = _bare(TableModalProcessor)
        seen = {}

        async def caption(prompt, **kwargs):
            seen["prompt"] = prompt
            return json.dumps(
                {
                    "detailed_description": "two columns",
                    "entity_info": {
                        "entity_name": "SpecTable",
                        "entity_type": "table",
                        "summary": "specs",
                    },
                }
            )

        proc.modal_caption_func = caption

        caption_text, entity = await proc.generate_description_only(
            "col1|col2\na|b",
            "table",
        )

        assert "col1|col2" in seen["prompt"]
        assert caption_text == "two columns"
        assert entity["entity_name"] == "SpecTable (table)"

    @pytest.mark.asyncio
    async def test_caption_exception_preserves_entity_name(self):
        proc = _bare(TableModalProcessor)

        async def boom(*args, **kwargs):
            raise RuntimeError("llm down")

        proc.modal_caption_func = boom

        caption_text, entity = await proc.generate_description_only(
            {"table_body": "a|b"},
            "table",
            entity_name="KeepMe",
        )

        assert caption_text == str({"table_body": "a|b"})
        assert entity["entity_name"] == "KeepMe"
        assert entity["entity_type"] == "table"


class TestGenericDescriptionEarlyGates:
    @pytest.mark.asyncio
    async def test_caption_exception_falls_back_with_content_type(self):
        proc = _bare(GenericModalProcessor)

        async def boom(*args, **kwargs):
            raise RuntimeError("llm down")

        proc.modal_caption_func = boom
        content = {"type": "chart", "title": "throughput"}

        caption_text, entity = await proc.generate_description_only(content, "chart")

        assert caption_text == str(content)
        assert entity["entity_type"] == "chart"
        assert entity["entity_name"] == f"chart_{compute_mdhash_id(str(content))}"
        assert "chart content:" in entity["summary"]
