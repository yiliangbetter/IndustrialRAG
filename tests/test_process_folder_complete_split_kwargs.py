"""process_folder_complete must forward split options to each file ingest.

Folder ingest is how plant manuals are loaded. If split_by_character stays on
the folder wrapper, nested PDFs chunk differently from single-file ingest and
section citations break only in batch runs.
"""

from pathlib import Path

import pytest

from raganything.batch import BatchMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _make_batch(tmp_path):
    class DummyBatch(BatchMixin):
        pass

    dummy = DummyBatch()
    dummy.logger = FakeLogger()
    dummy.config = type(
        "Config",
        (),
        {
            "parser_output_dir": str(tmp_path / "parsed"),
            "parse_method": "ocr",
            "supported_file_extensions": [".pdf"],
            "recursive_folder_processing": True,
            "max_concurrent_files": 2,
            "parser": "mineru",
        },
    )()
    dummy.process_calls = []

    async def _ensure():
        return {"success": True}

    async def _process(file_path, **kwargs):
        dummy.process_calls.append({"file_path": file_path, **kwargs})

    dummy._ensure_lightrag_initialized = _ensure
    dummy.process_document_complete = _process
    return dummy


@pytest.mark.asyncio
async def test_process_folder_complete_forwards_split_and_parse_method(tmp_path):
    dummy = _make_batch(tmp_path)
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(b"%PDF")
    (folder / "b.pdf").write_bytes(b"%PDF")

    await dummy.process_folder_complete(
        str(folder),
        split_by_character="\n",
        split_by_character_only=True,
    )

    assert len(dummy.process_calls) == 2
    names = {Path(call["file_path"]).name for call in dummy.process_calls}
    assert names == {"a.pdf", "b.pdf"}
    for call in dummy.process_calls:
        assert call["parse_method"] == "ocr"
        assert call["split_by_character"] == "\n"
        assert call["split_by_character_only"] is True
        assert call["output_dir"] == str(tmp_path / "parsed")
        assert call["file_name"] is None
