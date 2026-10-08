"""Shared construction of the OpenAI/HF-backed RAG runtime used by CLI scripts."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from .config import RAGAnythingConfig


def _optional_env_bool(name: str) -> bool | None:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{name} must be true/false when set.")


def _with_keyword_thinking(
    kwargs: dict[str, Any], enabled: bool | None
) -> dict[str, Any]:
    """Apply a provider thinking hint without overriding an explicit caller value."""
    if enabled is None:
        return kwargs
    extra_body = dict(kwargs.get("extra_body") or {})
    extra_body.setdefault("enable_thinking", enabled)
    return {**kwargs, "extra_body": extra_body}


@dataclass(frozen=True)
class RuntimeOptions:
    """Caller-specific LightRAG options that should not be hidden in the factory."""

    project_root: Path
    embedding_func_max_async: int = 1
    embedding_batch_num: int = 1
    enable_llm_cache: bool = True
    enable_rerank: bool = False
    prefer_binding_api_key: bool = False
    allow_openai_base_url: bool = True
    await_model_calls: bool = True


@dataclass(frozen=True)
class ProviderSettings:
    """Resolved model/provider settings, kept separate for validation and testing."""

    llm_key: str
    embedding_key: str
    llm_base_url: str | None
    embedding_base_url: str | None
    llm_model: str
    vision_model: str
    embedding_backend: str
    embedding_dim: int
    embedding_model: str
    keyword_llm_enable_thinking: bool | None

    @classmethod
    def from_env(cls, options: RuntimeOptions) -> ProviderSettings:
        key_names = (
            ("LLM_BINDING_API_KEY", "OPENAI_API_KEY")
            if options.prefer_binding_api_key
            else ("OPENAI_API_KEY", "LLM_BINDING_API_KEY")
        )
        llm_key = next(
            (value for name in key_names if (value := os.getenv(name, "").strip())),
            "",
        )
        if not llm_key:
            raise SystemExit("Set OPENAI_API_KEY or LLM_BINDING_API_KEY.")

        llm_base_url = os.getenv("LLM_BINDING_HOST", "").strip()
        if not llm_base_url and options.allow_openai_base_url:
            llm_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        embedding_host = os.getenv("EMBEDDING_BINDING_HOST", "").strip()
        backend = os.getenv("EMBEDDING_BACKEND", "openai").strip().lower()
        embedding_key = os.getenv("EMBEDDING_API_KEY", "").strip() or llm_key
        if (
            backend != "hf"
            and embedding_host
            and not os.getenv("EMBEDDING_API_KEY", "").strip()
        ):
            raise SystemExit(
                "EMBEDDING_BINDING_HOST is set; set EMBEDDING_API_KEY for that host."
            )

        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        defaults = (
            (1024, "BAAI/bge-m3")
            if backend == "hf"
            else (1536, "text-embedding-3-small")
        )
        return cls(
            llm_key=llm_key,
            embedding_key=embedding_key,
            llm_base_url=llm_base_url or None,
            embedding_base_url=embedding_host or llm_base_url or None,
            llm_model=llm_model,
            vision_model=os.getenv("VISION_MODEL", llm_model),
            embedding_backend=backend,
            embedding_dim=int(os.getenv("EMBEDDING_DIM", str(defaults[0]))),
            embedding_model=os.getenv("EMBEDDING_MODEL", defaults[1]).strip(),
            keyword_llm_enable_thinking=_optional_env_bool(
                "KEYWORD_LLM_ENABLE_THINKING"
            ),
        )


@dataclass(frozen=True)
class RAGRuntime:
    """Objects created together for ingestion/query scripts."""

    rag: Any
    config: RAGAnythingConfig
    logger: Any


@dataclass(frozen=True)
class _Dependencies:
    light_rag: Callable[..., Any]
    rag_anything: Callable[..., Any]
    embedding_func: Callable[..., Any]
    openai_complete: Callable[..., Any]
    openai_embed: Callable[..., Any]
    local_embedding: Callable[..., Any]
    ensure_hf_home: Callable[..., Any]
    build_reranker: Callable[..., Any]
    logger: Any


def _with_streaming_cot(kwargs: dict[str, Any]) -> dict[str, Any]:
    if not kwargs.get("stream") or "enable_cot" in kwargs:
        return kwargs
    return {**kwargs, "enable_cot": True}


def _load_dependencies() -> _Dependencies:
    """Load heavyweight ML dependencies only after CLI environment setup."""
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc, logger

    from .local_hf_embedding import (
        ensure_hf_home_from_repo_fallback,
        make_local_hf_embedding_func,
    )
    from .pipeline_rerank import build_rerank_model_func_from_env
    from .raganything import RAGAnything

    return _Dependencies(
        light_rag=LightRAG,
        rag_anything=RAGAnything,
        embedding_func=EmbeddingFunc,
        openai_complete=openai_complete_if_cache,
        openai_embed=openai_embed.func,
        local_embedding=make_local_hf_embedding_func,
        ensure_hf_home=ensure_hf_home_from_repo_fallback,
        build_reranker=build_rerank_model_func_from_env,
        logger=logger,
    )


async def create_rag_runtime(
    config: RAGAnythingConfig,
    options: RuntimeOptions,
) -> RAGRuntime:
    """Create and initialize RAGAnything plus its LightRAG/model dependencies."""
    deps = _load_dependencies()
    deps.ensure_hf_home(options.project_root)
    settings = ProviderSettings.from_env(options)

    def call_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        history = [] if history_messages is None else history_messages
        kwargs = _with_streaming_cot(kwargs)
        return deps.openai_complete(
            settings.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history,
            api_key=settings.llm_key,
            base_url=settings.llm_base_url,
            **kwargs,
        )

    def call_vision(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if messages:
            return deps.openai_complete(
                settings.vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=settings.llm_key,
                base_url=settings.llm_base_url,
                **kwargs,
            )
        if image_data:
            content = [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                },
            ]
            return deps.openai_complete(
                settings.vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=content,
                api_key=settings.llm_key,
                base_url=settings.llm_base_url,
                **kwargs,
            )
        return call_llm(prompt, system_prompt, history_messages, **kwargs)

    if options.await_model_calls:

        async def llm_model_func(*args, **kwargs):
            return await call_llm(*args, **kwargs)

        async def vision_model_func(*args, **kwargs):
            return await call_vision(*args, **kwargs)

    else:
        llm_model_func = call_llm
        vision_model_func = call_vision

    keyword_llm_model_func = None
    if settings.keyword_llm_enable_thinking is not None:
        if options.await_model_calls:

            async def keyword_llm_model_func(*args, **kwargs):
                return await llm_model_func(
                    *args,
                    **_with_keyword_thinking(
                        kwargs, settings.keyword_llm_enable_thinking
                    ),
                )

        else:

            def keyword_llm_model_func(*args, **kwargs):
                return llm_model_func(
                    *args,
                    **_with_keyword_thinking(
                        kwargs, settings.keyword_llm_enable_thinking
                    ),
                )

    if settings.embedding_backend == "hf":
        embedding = deps.local_embedding(
            settings.embedding_dim,
            embedding_model=settings.embedding_model,
        )
    else:
        embedding = deps.embedding_func(
            embedding_dim=settings.embedding_dim,
            max_token_size=8192,
            func=partial(
                deps.openai_embed,
                model=settings.embedding_model,
                api_key=settings.embedding_key,
                base_url=settings.embedding_base_url,
            ),
        )

    light_rag_kwargs = {
        "working_dir": config.working_dir,
        "llm_model_func": llm_model_func,
        "embedding_func": embedding,
        "enable_llm_cache": options.enable_llm_cache,
        "embedding_func_max_async": options.embedding_func_max_async,
        "embedding_batch_num": options.embedding_batch_num,
    }
    if options.enable_rerank:
        light_rag_kwargs["rerank_model_func"] = deps.build_reranker()
    if keyword_llm_model_func is not None:
        light_rag_kwargs["role_llm_configs"] = {
            "keyword": {"func": keyword_llm_model_func}
        }
    light_rag = deps.light_rag(**light_rag_kwargs)
    await light_rag.initialize_storages()

    rag = deps.rag_anything(
        config=config,
        lightrag=light_rag,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding,
    )
    return RAGRuntime(rag=rag, config=config, logger=deps.logger)
