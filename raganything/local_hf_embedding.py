"""
Local Hugging Face embeddings (e.g. BAAI/bge-m3) via sentence-transformers.

Set EMBEDDING_BACKEND=hf and EMBEDDING_DIM / EMBEDDING_MODEL / HF_HOME as needed.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import numpy as np


def ensure_hf_home_from_repo_fallback(repo_root: str | os.PathLike[str] | None) -> None:
    """If HF_HOME is unset, use <repo_root>/.hf_cache when that directory exists."""
    if os.getenv("HF_HOME"):
        return
    base = repo_root
    if base is None:
        return
    candidate = os.path.join(str(base), ".hf_cache")
    if os.path.isdir(candidate):
        os.environ["HF_HOME"] = candidate


def make_local_hf_embedding_func(embedding_dim: int, embedding_model: str | None = None):
    """Build a LightRAG-compatible EmbeddingFunc for a local HF sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
        from lightrag.utils import EmbeddingFunc
    except ImportError as e:
        raise ImportError(
            "sentence-transformers (and numpy) are required for EMBEDDING_BACKEND=hf. "
            'Install with: pip install "raganything[local-embed]" '
            'or pip install "sentence-transformers>=3.0.0".'
        ) from e

    model_id = embedding_model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    hf_home = os.getenv("HF_HOME")
    holder: dict[str, Any] = {"model": None}

    def _get_model():
        if holder["model"] is None:
            kwargs: dict[str, Any] = {}
            if hf_home:
                kwargs["cache_folder"] = hf_home
            holder["model"] = SentenceTransformer(model_id, **kwargs)
        return holder["model"]

    max_token_size = int(os.getenv("HF_EMBED_MAX_TOKEN", "8192"))

    def _encode_sync(text_list: list[str]) -> np.ndarray:
        model = _get_model()
        vecs = model.encode(
            text_list,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        out = np.asarray(vecs, dtype=np.float32)
        if out.ndim == 2 and out.shape[1] != embedding_dim:
            raise ValueError(
                f"Embedding shape {out.shape[1]} != EMBEDDING_DIM={embedding_dim}. "
                "Set EMBEDDING_DIM to match the model (1024 for BAAI/bge-m3)."
            )
        return out

    async def hf_embed(texts):
        return await asyncio.to_thread(_encode_sync, list(texts))

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        func=hf_embed,
    )
