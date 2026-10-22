"""Embedding-only LightRAG startup must not depend on a parser or an LLM.

`allow_embedding_only_ingestion` skips the parser install probe and installs an
empty LLM placeholder when no model function was given. An embedder is still
required, and a normal (non-embedding-only) failed probe stays retryable.
"""

from types import SimpleNamespace

import pytest


class ProbeParser:
    def __init__(self, installed: bool):
        self.installed = installed
        self.checks = 0

    def check_installation(self):
        self.checks += 1
        return self.installed


def _make_rag(monkeypatch, tmp_path, *, parser, allow_embedding_only, llm, embedding):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module
    from raganything.config import RAGAnythingConfig

    monkeypatch.setattr(rag_module, "get_parser", lambda _name: parser)
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **kwargs: None)

    config = RAGAnythingConfig(
        working_dir=str(tmp_path / "rag_workdir"),
        parser="mineru",
        allow_embedding_only_ingestion=allow_embedding_only,
    )
    rag = rag_module.RAGAnything(
        config=config,
        llm_model_func=llm,
        embedding_func=embedding,
    )
    rag.doc_parser = parser
    rag.lightrag = None
    return rag, rag_module


def _reject_lightrag_construction(monkeypatch, rag_module):
    def boom(*args, **kwargs):
        raise AssertionError("LightRAG should not be constructed")

    monkeypatch.setattr(rag_module, "LightRAG", boom)


class _RecordingLightRAG:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.workspace = "ws"
        self.key_string_value_json_storage_cls = _RecordingCache

    async def initialize_storages(self):
        self.initialized = True


class _RecordingCache:
    created = None

    def __init__(self, **kwargs):
        _RecordingCache.created = kwargs

    async def initialize(self):
        self.ready = True


def _patch_new_lightrag(monkeypatch, rag, rag_module):
    _RecordingCache.created = None
    calls = {"pipeline": 0, "processors": 0}

    async def pipeline_status():
        calls["pipeline"] += 1

    monkeypatch.setattr(rag_module, "LightRAG", _RecordingLightRAG)
    import lightrag.kg.shared_storage as shared_storage

    monkeypatch.setattr(shared_storage, "initialize_pipeline_status", pipeline_status)

    def init_processors():
        calls["processors"] += 1
        rag.modal_processors = {"generic": object()}

    rag._initialize_processors = init_processors
    return calls


@pytest.mark.asyncio
async def test_failed_parser_probe_is_retried_without_building_lightrag(
    monkeypatch, tmp_path
):
    parser = ProbeParser(installed=False)
    rag, rag_module = _make_rag(
        monkeypatch,
        tmp_path,
        parser=parser,
        allow_embedding_only=False,
        llm=object(),
        embedding=object(),
    )
    _reject_lightrag_construction(monkeypatch, rag_module)

    first = await rag._ensure_lightrag_initialized()
    second = await rag._ensure_lightrag_initialized()

    assert first == {
        "success": False,
        "error": (
            "Parser 'mineru' is not properly installed. "
            "Please install it using 'pip install' or 'uv pip install'."
        ),
    }
    assert second == first
    assert parser.checks == 2
    assert rag._parser_installation_checked is False
    assert rag.lightrag is None


@pytest.mark.asyncio
async def test_embedding_only_skips_failing_parser_probe(monkeypatch, tmp_path):
    parser = ProbeParser(installed=False)
    rag, rag_module = _make_rag(
        monkeypatch,
        tmp_path,
        parser=parser,
        allow_embedding_only=True,
        llm=object(),
        embedding=object(),
    )
    _reject_lightrag_construction(monkeypatch, rag_module)
    rag.lightrag = SimpleNamespace(_storages_status=SimpleNamespace(name="INITIALIZED"))
    rag.parse_cache = object()
    rag.modal_processors = {"generic": object()}

    result = await rag._ensure_lightrag_initialized()

    assert result == {"success": True}
    assert parser.checks == 0
    assert rag._parser_installation_checked is True


@pytest.mark.asyncio
async def test_embedding_only_installs_placeholder_and_keeps_explicit_llm(
    monkeypatch, tmp_path
):
    embed = object()

    async def explicit_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        return "real-answer"

    placeholder_rag, rag_module = _make_rag(
        monkeypatch,
        tmp_path,
        parser=ProbeParser(installed=False),
        allow_embedding_only=True,
        llm=None,
        embedding=embed,
    )
    placeholder_calls = _patch_new_lightrag(monkeypatch, placeholder_rag, rag_module)

    placeholder_result = await placeholder_rag._ensure_lightrag_initialized()

    assert placeholder_result == {"success": True}
    assert placeholder_rag.doc_parser.checks == 0
    built = placeholder_rag.lightrag
    assert isinstance(built, _RecordingLightRAG)
    assert built.kwargs["working_dir"] == placeholder_rag.working_dir
    assert built.kwargs["embedding_func"] is embed
    placeholder = built.kwargs["llm_model_func"]
    assert placeholder is placeholder_rag.llm_model_func
    assert (
        await placeholder("prompt", system_prompt="sys", history_messages=["prior"])
        == ""
    )
    assert placeholder_calls == {"pipeline": 1, "processors": 1}
    cache_kwargs = _RecordingCache.created
    assert cache_kwargs["namespace"] == "parse_cache"
    assert cache_kwargs["workspace"] == "ws"
    assert cache_kwargs["embedding_func"] is embed

    explicit_rag, explicit_module = _make_rag(
        monkeypatch,
        tmp_path,
        parser=ProbeParser(installed=False),
        allow_embedding_only=True,
        llm=explicit_llm,
        embedding=embed,
    )
    _patch_new_lightrag(monkeypatch, explicit_rag, explicit_module)

    explicit_result = await explicit_rag._ensure_lightrag_initialized()

    assert explicit_result == {"success": True}
    assert explicit_rag.doc_parser.checks == 0
    assert explicit_rag.lightrag.kwargs["llm_model_func"] is explicit_llm
    assert await explicit_llm("q") == "real-answer"


@pytest.mark.asyncio
async def test_embedding_only_still_requires_embedding_func(monkeypatch, tmp_path):
    parser = ProbeParser(installed=False)
    rag, rag_module = _make_rag(
        monkeypatch,
        tmp_path,
        parser=parser,
        allow_embedding_only=True,
        llm=None,
        embedding=None,
    )
    _reject_lightrag_construction(monkeypatch, rag_module)

    result = await rag._ensure_lightrag_initialized()

    assert result == {
        "success": False,
        "error": "embedding_func must be provided when LightRAG is not pre-initialized",
    }
    assert parser.checks == 0
    assert rag.lightrag is None


@pytest.mark.asyncio
async def test_missing_llm_does_not_install_placeholder_or_reprobe(
    monkeypatch, tmp_path
):
    parser = ProbeParser(installed=True)
    rag, rag_module = _make_rag(
        monkeypatch,
        tmp_path,
        parser=parser,
        allow_embedding_only=False,
        llm=None,
        embedding=object(),
    )
    _reject_lightrag_construction(monkeypatch, rag_module)

    first = await rag._ensure_lightrag_initialized()
    second = await rag._ensure_lightrag_initialized()

    assert first == {
        "success": False,
        "error": "llm_model_func must be provided when LightRAG is not pre-initialized",
    }
    assert second == first
    assert rag.llm_model_func is None
    assert parser.checks == 1
    assert rag._parser_installation_checked is True
    assert rag.lightrag is None
