"""
Local Hugging Face embeddings (e.g. BAAI/bge-m3) via sentence-transformers.

Set EMBEDDING_BACKEND=hf and EMBEDDING_DIM / EMBEDDING_MODEL / HF_HOME as needed.

Device: unset `HF_EMBED_DEVICE` / `EMBEDDING_DEVICE` to auto-pick CUDA (NVIDIA)
or MPS (Apple Silicon) when available; otherwise CPU. Override with e.g.
`HF_EMBED_DEVICE=cuda` or `cpu`.

Offline: set `HF_EMBED_OFFLINE=1` (or global `HF_HUB_OFFLINE=1`) after the model
is fully cached so loads skip network checks to huggingface.co.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _hub_snapshot_dir(repo_id: str, hf_home: str) -> Path | None:
    """Resolve HF hub cache snapshot path for ``org/name`` (no network)."""
    if "/" not in repo_id.strip():
        return None
    org, name = repo_id.strip().split("/", 1)
    if not org or not name:
        return None
    repo_dir = Path(hf_home) / "hub" / f"models--{org}--{name}"
    ref = repo_dir / "refs" / "main"
    if not ref.is_file():
        return None
    rev = ref.read_text(encoding="utf-8").strip()
    if not rev:
        return None
    snap = repo_dir / "snapshots" / rev
    if snap.is_dir() and (snap / "modules.json").is_file():
        return snap
    return None


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


def _resolve_embedding_device() -> str:
    explicit = (
        os.getenv("HF_EMBED_DEVICE") or os.getenv("EMBEDDING_DEVICE") or ""
    ).strip()
    if explicit:
        return explicit.lower()
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def make_local_hf_embedding_func(
    embedding_dim: int, embedding_model: str | None = None
):
    """Build a LightRAG-compatible EmbeddingFunc for a local HF sentence-transformers model."""
    offline = (os.getenv("HF_EMBED_OFFLINE") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if offline:
        # Set before importing sentence_transformers / transformers (reduces hub probes).
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
    device = _resolve_embedding_device()
    holder: dict[str, Any] = {"model": None}

    def _get_model():
        if holder["model"] is None:
            kwargs: dict[str, Any] = {"device": device}
            load_id = model_id
            if offline and hf_home:
                snap = _hub_snapshot_dir(model_id, hf_home)
                if snap is not None:
                    load_id = str(snap)
                    logger.info(
                        "HF embed offline: loading from local snapshot %s", load_id
                    )
                else:
                    kwargs["local_files_only"] = True
                    if hf_home:
                        kwargs["cache_folder"] = hf_home
                    logger.warning(
                        "HF_EMBED_OFFLINE=1 but no hub snapshot under HF_HOME=%s for %s; "
                        "using repo id + local_files_only (may still probe hub).",
                        hf_home,
                        model_id,
                    )
            elif offline:
                kwargs["local_files_only"] = True
            elif hf_home:
                kwargs["cache_folder"] = hf_home
                if offline:
                    kwargs["local_files_only"] = True

            holder["model"] = SentenceTransformer(load_id, **kwargs)
            logger.info(
                "Local HF embedding model loaded: %s on device=%s",
                load_id,
                getattr(holder["model"], "device", device),
            )
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
