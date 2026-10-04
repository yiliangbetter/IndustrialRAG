"""VLM prompt loading must not follow a symlink that stays inside a safe directory.

``_process_image_paths_for_vlm`` allowlists the resolved path. A symlink whose
target is also inside the working directory would pass that check and its
bytes would be base64-encoded into the vision prompt. ``validate_image_file``
rejects every symlink first; this test locks that the caller still does.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from raganything.query import QueryMixin


class _Logger:
    def __init__(self):
        self.warnings = []

    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg % args if args else msg)

    def error(self, msg, *args, **kwargs):
        pass


class _Query(QueryMixin):
    def __init__(self, working_dir: Path, output_dir: Path):
        self.config = type(
            "Config",
            (),
            {
                "working_dir": str(working_dir),
                "parser_output_dir": str(output_dir),
            },
        )()
        self.logger = _Logger()


@pytest.mark.asyncio
async def test_in_tree_symlink_is_not_encoded_while_real_image_is(
    tmp_path, monkeypatch
):
    working_dir = tmp_path / "workspace"
    output_dir = tmp_path / "parser_output"
    cwd = tmp_path / "cwd"
    working_dir.mkdir()
    output_dir.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    real = working_dir / "nameplate.png"
    real_bytes = b"\x89PNG\r\n\x1a\nreal-nameplate"
    real.write_bytes(real_bytes)
    link = working_dir / "alias.png"
    link.symlink_to(real)

    query = _Query(working_dir, output_dir)
    prompt = f"Manual\nImage Path: {link}\n" f"Next\nImage Path: {real}\nEnd"

    processed, image_count = await query._process_image_paths_for_vlm(prompt)

    assert image_count == 1
    assert f"Image Path: {link}\nNext" in processed
    assert "[VLM_IMAGE_" not in processed.split("Next", 1)[0]
    assert f"Image Path: {real}\n[VLM_IMAGE_1]" in processed
    assert query._current_images_base64 == [base64.b64encode(real_bytes).decode()]
    assert any(
        "Image validation failed or path unsafe" in msg for msg in query.logger.warnings
    )
