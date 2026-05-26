import base64

import pytest

from raganything.query import QueryMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class DummyQuery(QueryMixin):
    def __init__(self, config=None):
        self.config = config
        self.logger = FakeLogger()


def test_multimodal_cache_key_includes_system_prompt_and_normalizes_paths(tmp_path):
    query = DummyQuery()
    long_table = "col\n" + ("value," * 60)

    base_content = [
        {
            "type": "image",
            "img_path": str(tmp_path / "first" / "diagram.png"),
            "table_data": long_table,
        }
    ]
    same_basename_content = [
        {
            "type": "image",
            "img_path": str(tmp_path / "second" / "diagram.png"),
            "table_data": long_table,
        }
    ]

    safety_key = query._generate_multimodal_cache_key(
        "  What changed?  ",
        base_content,
        "mix",
        system_prompt="Answer as a safety reviewer",
    )
    finance_key = query._generate_multimodal_cache_key(
        "What changed?",
        base_content,
        "mix",
        system_prompt="Answer as a finance reviewer",
    )
    same_prompt_portable_path_key = query._generate_multimodal_cache_key(
        "What changed?",
        same_basename_content,
        "mix",
        system_prompt="Answer as a safety reviewer",
    )

    assert safety_key != finance_key
    assert safety_key == same_prompt_portable_path_key


@pytest.mark.asyncio
async def test_process_image_paths_blocks_valid_images_outside_safe_dirs(tmp_path):
    working_dir = tmp_path / "workspace"
    output_dir = tmp_path / "parser_output"
    outside_dir = tmp_path / "outside"
    working_dir.mkdir()
    output_dir.mkdir()
    outside_dir.mkdir()

    outside_image = outside_dir / "blocked.png"
    outside_image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"image bytes")

    config = type(
        "Config",
        (),
        {
            "working_dir": str(working_dir),
            "parser_output_dir": str(output_dir),
        },
    )()
    query = DummyQuery(config=config)
    prompt = f"Evidence\nImage Path: {outside_image}\nEnd"

    processed_prompt, image_count = await query._process_image_paths_for_vlm(prompt)

    assert processed_prompt == prompt
    assert image_count == 0
    assert query._current_images_base64 == []


@pytest.mark.asyncio
async def test_process_image_paths_allows_extra_safe_dirs(tmp_path):
    working_dir = tmp_path / "workspace"
    output_dir = tmp_path / "parser_output"
    allowed_dir = tmp_path / "allowed"
    working_dir.mkdir()
    output_dir.mkdir()
    allowed_dir.mkdir()

    image = allowed_dir / "allowed.png"
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"image bytes"
    image.write_bytes(image_bytes)

    config = type(
        "Config",
        (),
        {
            "working_dir": str(working_dir),
            "parser_output_dir": str(output_dir),
        },
    )()
    query = DummyQuery(config=config)
    prompt = f"Evidence\nImage Path: {image}\nEnd"

    processed_prompt, image_count = await query._process_image_paths_for_vlm(
        prompt, extra_safe_dirs=[str(allowed_dir)]
    )

    assert image_count == 1
    assert f"Image Path: {image}\n[VLM_IMAGE_1]" in processed_prompt
    assert query._current_images_base64 == [
        base64.b64encode(image_bytes).decode("utf-8")
    ]
