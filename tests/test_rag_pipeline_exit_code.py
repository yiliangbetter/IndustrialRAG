"""rag_pipeline_parse_graph_chat must fail closed on ingest failures."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"


def _load_script_module():
    name = "rag_pipeline_parse_graph_chat_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.asyncio
async def test_async_main_exits_nonzero_when_ingest_has_failures(tmp_path: Path):
    mod = _load_script_module()
    input_folder = tmp_path / "in"
    input_folder.mkdir()
    (input_folder / "a.pdf").write_bytes(b"%PDF")

    rag = MagicMock()
    rag.finalize_storages = AsyncMock()
    config = MagicMock()
    config.parser = "mineru"
    logger = MagicMock()

    argv = [
        "rag_pipeline_parse_graph_chat.py",
        "--input-folder",
        str(input_folder),
        "--working-dir",
        str(tmp_path / "wd"),
        "--parser-output-dir",
        str(tmp_path / "parse"),
        "--ingest-only",
        "--no-recursive",
    ]

    with (
        patch.object(mod.sys, "argv", argv),
        patch.object(
            mod, "_build_rag", AsyncMock(return_value=(rag, config, logger))
        ),
        patch.object(mod, "_mineru_parse_kwargs", return_value={}),
        patch.object(mod, "_ingest_folder", AsyncMock(return_value=(1, 2))),
    ):
        with pytest.raises(SystemExit) as exc:
            await mod.async_main()
        assert exc.value.code == 1


@pytest.mark.asyncio
async def test_async_main_returns_when_ingest_all_ok(tmp_path: Path):
    mod = _load_script_module()
    input_folder = tmp_path / "in"
    input_folder.mkdir()
    (input_folder / "a.pdf").write_bytes(b"%PDF")

    rag = MagicMock()
    rag.finalize_storages = AsyncMock()
    config = MagicMock()
    config.parser = "mineru"
    logger = MagicMock()

    argv = [
        "rag_pipeline_parse_graph_chat.py",
        "--input-folder",
        str(input_folder),
        "--working-dir",
        str(tmp_path / "wd"),
        "--parser-output-dir",
        str(tmp_path / "parse"),
        "--ingest-only",
        "--no-recursive",
    ]

    with (
        patch.object(mod.sys, "argv", argv),
        patch.object(
            mod, "_build_rag", AsyncMock(return_value=(rag, config, logger))
        ),
        patch.object(mod, "_mineru_parse_kwargs", return_value={}),
        patch.object(mod, "_ingest_folder", AsyncMock(return_value=(2, 0))),
    ):
        await mod.async_main()

    rag.finalize_storages.assert_awaited()
