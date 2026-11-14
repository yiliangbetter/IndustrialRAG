"""Startup log for a newly built LightRAG must not print credentials.

``get_config_info`` redaction is a different copy of this filter. This test
covers the log line inside ``RAGAnything._ensure_lightrag_initialized``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from raganything.raganything import RAGAnything


class RecordingLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg % args if args else str(msg))

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg % args if args else str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(msg % args if args else str(msg))

    def debug(self, msg, *args, **kwargs):
        return None


class FakeParseCache:
    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.initialized = False
        FakeParseCache.created.append(self)

    async def initialize(self):
        self.initialized = True


class FakeLightRAG:
    constructed = []

    def __init__(self, **kwargs):
        FakeLightRAG.constructed.append(kwargs)
        self.workspace = "plant-workspace"
        self.key_string_value_json_storage_cls = FakeParseCache

    async def initialize_storages(self):
        self.storages_initialized = True


async def plant_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    return "ok"


async def plant_embed(texts):
    return []


def SECRET_RERANK_FUNC(query, **kwargs):
    return []


def _reject_parser_probe():
    raise AssertionError("parser probe should not run")


def _rag(tmp_path):
    rag = object.__new__(RAGAnything)
    rag.logger = RecordingLogger()
    rag.config = SimpleNamespace(
        allow_embedding_only_ingestion=False,
        parser="mineru",
    )
    rag.doc_parser = SimpleNamespace(check_installation=_reject_parser_probe)
    rag._parser_installation_checked = True
    rag.lightrag = None
    rag.llm_model_func = plant_llm
    rag.embedding_func = plant_embed
    rag.working_dir = str(tmp_path / "rag-storage")
    rag.parse_cache = None
    rag.modal_processors = {}
    rag._initialize_processors = lambda: setattr(rag, "processors_ready", True)
    rag.lightrag_kwargs = {
        "top_k": 7,
        "addon_params": {"language": "Chinese"},
        "llm_model_kwargs": {"api_key": "sk-live-do-not-log"},
        "vector_db_storage_cls_kwargs": {"password": "db-password-do-not-log"},
        "rerank_model_func": SECRET_RERANK_FUNC,
    }
    return rag


@pytest.mark.asyncio
async def test_new_lightrag_init_log_omits_secrets_and_callables(tmp_path):
    FakeLightRAG.constructed.clear()
    FakeParseCache.created.clear()
    rag = _rag(tmp_path)
    pipeline_init = AsyncMock()

    with (
        patch("raganything.raganything.LightRAG", FakeLightRAG),
        patch(
            "lightrag.kg.shared_storage.initialize_pipeline_status",
            pipeline_init,
        ),
    ):
        result = await rag._ensure_lightrag_initialized()

    assert result == {"success": True}
    assert pipeline_init.await_count == 1
    assert rag.processors_ready is True
    assert FakeParseCache.created[0].initialized is True
    assert FakeParseCache.created[0].kwargs["embedding_func"] is plant_embed
    assert FakeParseCache.created[0].kwargs["namespace"] == "parse_cache"

    built = FakeLightRAG.constructed[0]
    assert built["working_dir"] == str(tmp_path / "rag-storage")
    assert built["top_k"] == 7
    assert built["addon_params"] == {"language": "Chinese"}
    assert built["llm_model_func"] is plant_llm
    assert built["embedding_func"] is plant_embed
    assert built["llm_model_kwargs"] == {"api_key": "sk-live-do-not-log"}
    assert built["vector_db_storage_cls_kwargs"] == {
        "password": "db-password-do-not-log"
    }
    assert built["rerank_model_func"] is SECRET_RERANK_FUNC

    init_logs = [
        line
        for line in rag.logger.infos
        if line.startswith("Initializing LightRAG with parameters:")
    ]
    assert len(init_logs) == 1
    logged = init_logs[0]
    assert "top_k" in logged
    assert "Chinese" in logged
    assert str(tmp_path / "rag-storage") in logged
    for secret in (
        "sk-live-do-not-log",
        "db-password-do-not-log",
        "llm_model_kwargs",
        "vector_db_storage_cls_kwargs",
        "SECRET_RERANK_FUNC",
        "plant_llm",
        "plant_embed",
    ):
        assert secret not in logged
