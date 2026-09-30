from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from raganything import runtime_factory as factory
from raganything.config import RAGAnythingConfig

_PROVIDER_ENV = (
    "OPENAI_API_KEY",
    "LLM_BINDING_API_KEY",
    "EMBEDDING_API_KEY",
    "LLM_BINDING_HOST",
    "OPENAI_BASE_URL",
    "EMBEDDING_BINDING_HOST",
    "LLM_MODEL",
    "VISION_MODEL",
    "EMBEDDING_BACKEND",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "KEYWORD_LLM_ENABLE_THINKING",
)


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)


def test_provider_settings_preserve_key_and_base_url_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "binding-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-compatible.example/v1")

    default = factory.ProviderSettings.from_env(
        factory.RuntimeOptions(project_root=tmp_path)
    )
    binding_first = factory.ProviderSettings.from_env(
        factory.RuntimeOptions(project_root=tmp_path, prefer_binding_api_key=True)
    )
    binding_only_url = factory.ProviderSettings.from_env(
        factory.RuntimeOptions(project_root=tmp_path, allow_openai_base_url=False)
    )

    assert default.llm_key == "openai-key"
    assert binding_first.llm_key == "binding-key"
    assert default.llm_base_url == "https://openai-compatible.example/v1"
    assert binding_only_url.llm_base_url is None
    assert default.embedding_dim == 1536
    assert default.embedding_model == "text-embedding-3-small"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, {}),
        ({"stream": False}, {"stream": False}),
        ({"stream": True}, {"stream": True, "enable_cot": True}),
        (
            {"stream": True, "enable_cot": False},
            {"stream": True, "enable_cot": False},
        ),
    ],
)
def test_streaming_cot_default_preserves_caller_override(kwargs, expected):
    assert factory._with_streaming_cot(kwargs) == expected


def test_keyword_thinking_hint_preserves_caller_override():
    assert factory._with_keyword_thinking({}, None) == {}
    assert factory._with_keyword_thinking({}, False) == {
        "extra_body": {"enable_thinking": False}
    }
    assert factory._with_keyword_thinking(
        {"extra_body": {"enable_thinking": True, "other": "value"}}, False
    ) == {"extra_body": {"enable_thinking": True, "other": "value"}}


def test_keyword_thinking_setting_rejects_invalid_bool(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("KEYWORD_LLM_ENABLE_THINKING", "sometimes")

    with pytest.raises(SystemExit, match="KEYWORD_LLM_ENABLE_THINKING"):
        factory.ProviderSettings.from_env(factory.RuntimeOptions(project_root=tmp_path))


def test_provider_settings_require_host_specific_embedding_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://embed.example/v1")

    with pytest.raises(SystemExit, match="EMBEDDING_API_KEY"):
        factory.ProviderSettings.from_env(factory.RuntimeOptions(project_root=tmp_path))

    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    settings = factory.ProviderSettings.from_env(
        factory.RuntimeOptions(project_root=tmp_path)
    )
    assert settings.embedding_dim == 1024
    assert settings.embedding_model == "BAAI/bge-m3"


@pytest.mark.asyncio
async def test_create_runtime_wires_openai_models_and_light_rag(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embed-key")
    monkeypatch.setenv("LLM_BINDING_HOST", "https://llm.example/v1")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://embed.example/v1")
    monkeypatch.setenv("LLM_MODEL", "text-model")
    monkeypatch.setenv("VISION_MODEL", "vision-model")
    monkeypatch.setenv("KEYWORD_LLM_ENABLE_THINKING", "false")

    completions = []
    hf_home_calls = []

    async def complete(*args, **kwargs):
        completions.append((args, kwargs))
        return "response"

    class FakeEmbedding:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLightRAG:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.initialized = False

        async def initialize_storages(self):
            self.initialized = True

    class FakeRAGAnything:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    reranker = object()
    deps = factory._Dependencies(
        light_rag=FakeLightRAG,
        rag_anything=FakeRAGAnything,
        embedding_func=FakeEmbedding,
        openai_complete=complete,
        openai_embed=lambda *_args, **_kwargs: None,
        local_embedding=lambda *_args, **_kwargs: None,
        ensure_hf_home=hf_home_calls.append,
        build_reranker=lambda: reranker,
        logger=SimpleNamespace(),
    )
    monkeypatch.setattr(factory, "_load_dependencies", lambda: deps)

    config = RAGAnythingConfig(working_dir=str(tmp_path / "rag"))
    runtime = await factory.create_rag_runtime(
        config,
        factory.RuntimeOptions(
            project_root=tmp_path,
            embedding_func_max_async=3,
            embedding_batch_num=7,
            enable_llm_cache=False,
            enable_rerank=True,
        ),
    )

    light_rag = runtime.rag.kwargs["lightrag"]
    assert light_rag.initialized is True
    assert light_rag.kwargs["working_dir"] == str(tmp_path / "rag")
    assert light_rag.kwargs["rerank_model_func"] is reranker
    assert light_rag.kwargs["enable_llm_cache"] is False
    assert light_rag.kwargs["embedding_func_max_async"] == 3
    assert light_rag.kwargs["embedding_batch_num"] == 7
    assert hf_home_calls == [tmp_path]

    embedding = light_rag.kwargs["embedding_func"]
    assert embedding.kwargs["embedding_dim"] == 1536
    assert embedding.kwargs["func"].keywords == {
        "model": "text-embedding-3-small",
        "api_key": "embed-key",
        "base_url": "https://embed.example/v1",
    }

    llm = runtime.rag.kwargs["llm_model_func"]
    vision = runtime.rag.kwargs["vision_model_func"]
    assert inspect.iscoroutinefunction(llm)
    assert await llm("question", system_prompt="system") == "response"
    assert completions[-1] == (
        ("text-model", "question"),
        {
            "system_prompt": "system",
            "history_messages": [],
            "api_key": "llm-key",
            "base_url": "https://llm.example/v1",
        },
    )
    assert await llm("stream question", stream=True) == "response"
    assert completions[-1][1]["stream"] is True
    assert completions[-1][1]["enable_cot"] is True
    assert await llm("private stream", stream=True, enable_cot=False) == "response"
    assert completions[-1][1]["enable_cot"] is False

    keyword_llm = light_rag.kwargs["role_llm_configs"]["keyword"]["func"]
    assert (
        await keyword_llm(
            "keywords",
            response_format={"type": "json_object"},
            extra_body={"provider_option": "kept"},
        )
        == "response"
    )
    assert completions[-1][1]["extra_body"] == {
        "provider_option": "kept",
        "enable_thinking": False,
    }
    assert (
        await keyword_llm(
            "keywords",
            extra_body={"enable_thinking": True},
        )
        == "response"
    )
    assert completions[-1][1]["extra_body"] == {"enable_thinking": True}

    assert await llm("final answer", stream=True) == "response"
    assert "extra_body" not in completions[-1][1]
    assert completions[-1][1]["enable_cot"] is True

    assert await vision("inspect", image_data="abc") == "response"
    args, kwargs = completions[-1]
    assert args == ("vision-model", "")
    assert kwargs["messages"][1]["content"][1]["image_url"]["url"] == (
        "data:image/jpeg;base64,abc"
    )


@pytest.mark.asyncio
async def test_create_runtime_preserves_sync_wrapper_and_hf_backend(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    local_embedding = object()
    local_calls = []

    async def complete(*_args, **_kwargs):
        return "response"

    class FakeLightRAG:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def initialize_storages(self):
            pass

    deps = factory._Dependencies(
        light_rag=FakeLightRAG,
        rag_anything=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
        embedding_func=lambda **_kwargs: pytest.fail("OpenAI embedding used"),
        openai_complete=complete,
        openai_embed=lambda *_args, **_kwargs: None,
        local_embedding=lambda *args, **kwargs: (
            local_calls.append((args, kwargs)) or local_embedding
        ),
        ensure_hf_home=lambda _root: None,
        build_reranker=lambda: pytest.fail("reranker unexpectedly enabled"),
        logger=SimpleNamespace(),
    )
    monkeypatch.setattr(factory, "_load_dependencies", lambda: deps)

    runtime = await factory.create_rag_runtime(
        RAGAnythingConfig(working_dir=str(tmp_path)),
        factory.RuntimeOptions(project_root=tmp_path, await_model_calls=False),
    )

    llm = runtime.rag.kwargs["llm_model_func"]
    assert not inspect.iscoroutinefunction(llm)
    assert await llm("question") == "response"
    assert local_calls == [((1024,), {"embedding_model": "BAAI/bge-m3"})]
    assert runtime.rag.kwargs["embedding_func"] is local_embedding
    assert "role_llm_configs" not in runtime.rag.kwargs["lightrag"].kwargs


def test_cli_scripts_use_the_shared_runtime_factory():
    root = Path(__file__).resolve().parent.parent
    scripts = (
        "rag_pipeline_parse_graph_chat.py",
        "reingest_uploaded_documents_ocr.py",
        "run_demo_question_bank.py",
        "batch_ingest_content_lists_with_graph.py",
    )
    for name in scripts:
        source = (root / "scripts" / name).read_text(encoding="utf-8")
        assert "create_rag_runtime(" in source
        assert "LightRAG(" not in source
        assert "openai_complete_if_cache" not in source
