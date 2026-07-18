"""Regression tests for ``scripts/rag_pipeline_parse_graph_chat.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_parse_graph_chat_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pipeline():
    return _load_script_module()


@pytest.mark.asyncio
async def test_ingest_folder_preserves_nested_parser_output_path(
    pipeline, tmp_path
):
    input_folder = tmp_path / "docs"
    source_file = input_folder / "manuals" / "chapter-1" / "guide.pdf"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"%PDF")
    parser_output_dir = tmp_path / "parser-output"

    content_list = [{"type": "text", "text": "Installation steps"}]
    rag = MagicMock()
    rag.parse_document = AsyncMock(return_value=(content_list, "doc-123"))
    rag.insert_content_list = AsyncMock()
    config = SimpleNamespace(
        supported_file_extensions=[".pdf"],
        display_content_stats=False,
    )
    logger = MagicMock()

    result = await pipeline._ingest_folder(
        rag,
        config,
        logger,
        input_folder=input_folder,
        parser_output_dir=parser_output_dir,
        parse_method="ocr",
        parse_extra={"lang": "en"},
        recursive=True,
        limit=0,
        skip_multimodal=True,
    )

    nested_output_dir = parser_output_dir / "manuals" / "chapter-1"
    assert result == (1, 0)
    assert nested_output_dir.is_dir()
    rag.parse_document.assert_awaited_once_with(
        str(source_file),
        output_dir=str(nested_output_dir),
        parse_method="ocr",
        display_stats=False,
        lang="en",
    )
    rag.insert_content_list.assert_awaited_once_with(
        content_list,
        file_path="manuals/chapter-1/guide.pdf",
        doc_id="doc-123",
        skip_multimodal_processing=True,
    )
