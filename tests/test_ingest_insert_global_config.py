from types import SimpleNamespace

import numpy as np
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

from raganything.ingest_insert import build_lightrag_global_config


def test_global_config_uses_lightrag_builder_for_runtime_roles():
    role_funcs = {"extract": object(), "keyword": object(), "query": object()}
    rag = SimpleNamespace(
        _build_global_config=lambda: {"role_llm_funcs": role_funcs},
        stale="not-used",
    )

    config = build_lightrag_global_config(rag)

    assert config["role_llm_funcs"] is role_funcs
    assert "stale" not in config


def test_global_config_falls_back_for_older_lightrag():
    rag = SimpleNamespace(llm_model_func="legacy", chunk_token_size=1200)

    config = build_lightrag_global_config(rag)

    assert config == {"llm_model_func": "legacy", "chunk_token_size": 1200}
    assert config is not rag.__dict__


def test_global_config_matches_installed_lightrag_contract(tmp_path):
    async def llm_model_func(_prompt, **_kwargs):
        return ""

    async def embedding_func(texts):
        return np.zeros((len(texts), 3), dtype=np.float32)

    rag = LightRAG(
        working_dir=str(tmp_path),
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(3, embedding_func),
    )

    role_funcs = build_lightrag_global_config(rag)["role_llm_funcs"]

    assert set(role_funcs) == {"extract", "keyword", "query", "vlm"}
    assert all(callable(func) for func in role_funcs.values())
