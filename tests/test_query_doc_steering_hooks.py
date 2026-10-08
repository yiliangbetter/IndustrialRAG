from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import query_doc_steering


@pytest.mark.asyncio
async def test_catalog_rerank_wrapper_forwards_new_keyword_arguments(monkeypatch):
    from lightrag import operate, utils

    received = {}

    async def original(*args, **kwargs):
        received.update(kwargs)
        return ["ok"]

    monkeypatch.setattr(utils, "process_chunks_unified", original)
    monkeypatch.setattr(operate, "process_chunks_unified", original)

    query_doc_steering.install_catalog_rerank_threshold()
    callback = object()
    result = await utils.process_chunks_unified(
        "普通维护问题",
        [],
        object(),
        {},
        progress_callback=callback,
    )

    assert result == ["ok"]
    assert received["progress_callback"] is callback
