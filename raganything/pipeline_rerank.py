"""Build LightRAG ``rerank_model_func`` from environment (used by pipeline scripts).

LightRAG only reranks when ``QueryParam.enable_rerank`` is true (see ``RERANK_BY_DEFAULT``)
and ``rerank_model_func`` is set on ``LightRAG``. Without the latter, retrieval logs a
warning and skips reranking.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "build_rerank_model_func_from_env",
    "hf_cross_encoder_rerank",
    "release_cross_encoder",
    "release_rerank_after_query_if_enabled",
    "rerank_release_after_gate",
    "rerank_release_after_predict",
    "rerank_release_after_query",
]

logger = logging.getLogger(__name__)

_cross_encoder_id: str | None = None
_cross_encoder_device: str | None = None
_cross_encoder_dtype_key: str | None = None
_cross_encoder: Any = None


def _resolve_rerank_device() -> str:
    explicit = (
        os.getenv("RERANK_HF_DEVICE") or os.getenv("RERANK_DEVICE") or ""
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


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes")


def rerank_release_after_predict() -> bool:
    """When true, drop CrossEncoder singleton after each ``hf_cross_encoder_rerank`` predict."""
    return _env_flag("RERANK_RELEASE_AFTER_PREDICT")


def rerank_release_after_gate() -> bool:
    """When true, drop CrossEncoder singleton when ``evaluate_clarify_gate`` finishes."""
    return _env_flag("RERANK_RELEASE_AFTER_GATE")


def rerank_release_after_query() -> bool:
    """When true, Web server releases CrossEncoder once per HTTP query (see ``rag_web_server``)."""
    return _env_flag("RERANK_RELEASE_AFTER_QUERY")


def release_rerank_after_query_if_enabled() -> None:
    """Release HF CrossEncoder after a full Web query when ``RERANK_RELEASE_AFTER_QUERY=1``."""
    if not rerank_release_after_query():
        return
    binding = (os.getenv("RERANK_BINDING") or "").strip().lower()
    if binding not in ("hf", "local", "cross_encoder", "sentence_transformers"):
        return
    release_cross_encoder()


def _rerank_torch_dtype() -> Any | None:
    """CrossEncoder ``model_kwargs['torch_dtype']``; ``None`` keeps library default (FP32)."""
    raw = (
        (os.getenv("RERANK_TORCH_DTYPE") or os.getenv("RERANK_DTYPE") or "")
        .strip()
        .lower()
    )
    if not raw:
        return None
    if raw in ("auto", "config"):
        return "auto"
    try:
        import torch
    except ImportError:
        logger.warning("RERANK_TORCH_DTYPE=%r ignored (torch not installed)", raw)
        return None
    if raw in ("bf16", "bfloat16"):
        return torch.bfloat16
    if raw in ("fp16", "float16", "half"):
        return torch.float16
    if raw in ("fp32", "float32", "full"):
        return torch.float32
    logger.warning("Unknown RERANK_TORCH_DTYPE=%r, ignoring", raw)
    return None


def _rerank_torch_dtype_key() -> str:
    dtype = _rerank_torch_dtype()
    return "fp32-default" if dtype is None else str(dtype)


def _rerank_batch_size() -> int:
    raw = (os.getenv("RERANK_BATCH_SIZE") or "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return 8


def _cross_encoder_predict(ce: Any, pairs: list[tuple[str, str]]) -> Any:
    """Run CrossEncoder.predict with P1 tuning (batch, no progress bar, inference_mode)."""
    predict_kw: dict[str, Any] = {
        "batch_size": _rerank_batch_size(),
        "show_progress_bar": False,
    }
    try:
        import torch

        with torch.inference_mode():
            return ce.predict(pairs, **predict_kw)
    except ImportError:
        return ce.predict(pairs, **predict_kw)


def release_cross_encoder() -> None:
    """Drop the global CrossEncoder singleton and free GPU memory if applicable."""
    global \
        _cross_encoder_id, \
        _cross_encoder_device, \
        _cross_encoder_dtype_key, \
        _cross_encoder
    if _cross_encoder is None:
        return
    device = (_cross_encoder_device or _resolve_rerank_device() or "").lower()
    try:
        from raganything.query_timing_trace import trace_event

        trace_event("rerank_release", device=device)
    except ImportError:
        pass
    ce = _cross_encoder
    _cross_encoder = None
    _cross_encoder_id = None
    _cross_encoder_device = None
    _cross_encoder_dtype_key = None
    del ce
    gc.collect()
    if device.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    logger.debug(
        "RERANK hf: released CrossEncoder (device was %s)", device or "unknown"
    )


def _hf_hub_offline_requested() -> bool:
    return (
        (os.getenv("HF_EMBED_OFFLINE") or "").strip().lower() in ("1", "true", "yes")
        or (os.getenv("HF_HUB_OFFLINE") or "").strip().lower() in ("1", "true", "yes")
        or (os.getenv("TRANSFORMERS_OFFLINE") or "").strip().lower()
        in ("1", "true", "yes")
        or (os.getenv("RERANK_HF_OFFLINE") or "").strip().lower()
        in ("1", "true", "yes")
    )


def _hub_snapshot_dir_cross_encoder(repo_id: str, hf_home: str) -> Path | None:
    """Resolve HF hub cache snapshot for a CrossEncoder repo (has ``config.json``, not ``modules.json``)."""
    rid = repo_id.strip()
    if "/" not in rid:
        return None
    org, name = rid.split("/", 1)
    if not org or not name:
        return None
    repo_dir = Path(hf_home) / "hub" / f"models--{org}--{name}"
    if not repo_dir.is_dir():
        return None
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        rev = ref.read_text(encoding="utf-8").strip()
        if rev:
            snap = repo_dir / "snapshots" / rev
            if snap.is_dir() and (snap / "config.json").is_file():
                return snap
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        return None
    candidates = [
        p for p in snaps.iterdir() if p.is_dir() and (p / "config.json").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _get_cross_encoder(model_id: str) -> Any:
    global \
        _cross_encoder_id, \
        _cross_encoder_device, \
        _cross_encoder_dtype_key, \
        _cross_encoder
    device = _resolve_rerank_device()
    dtype_key = _rerank_torch_dtype_key()
    if (
        _cross_encoder is not None
        and _cross_encoder_id == model_id
        and _cross_encoder_device == device
        and _cross_encoder_dtype_key == dtype_key
    ):
        try:
            from raganything.query_timing_trace import trace_event

            trace_event("rerank_reuse", model=model_id, device=device)
        except ImportError:
            pass
        return _cross_encoder
    try:
        from raganything.query_timing_trace import trace_event

        trace_event("rerank_load_start", model=model_id, device=device)
    except ImportError:
        pass
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise SystemExit(
            "RERANK_BINDING=hf requires sentence-transformers. "
            "Install with: uv sync --extra local-embed"
        ) from e

    hf_home = (os.getenv("HF_HOME") or "").strip()
    snap = _hub_snapshot_dir_cross_encoder(model_id, hf_home) if hf_home else None
    offline = _hf_hub_offline_requested()

    load_id = model_id
    kwargs: dict[str, Any] = {}
    if snap is not None:
        load_id = str(snap.resolve())
        kwargs["local_files_only"] = True
        logger.info("RERANK hf: loading CrossEncoder from local snapshot %s", load_id)
    else:
        if hf_home:
            kwargs["cache_folder"] = hf_home
        if offline:
            kwargs["local_files_only"] = True
            logger.info(
                "RERANK hf: offline mode, no snapshot under HF_HOME for %s — repo id + local_files_only",
                model_id,
            )
        else:
            logger.info(
                "RERANK hf: loading CrossEncoder from hub/cache for %s", model_id
            )
    kwargs["device"] = device
    torch_dtype = _rerank_torch_dtype()
    if torch_dtype is not None:
        kwargs["model_kwargs"] = {"torch_dtype": torch_dtype}
    try:
        _cross_encoder = CrossEncoder(load_id, **kwargs)
    except Exception as e:
        if offline or kwargs.get("local_files_only"):
            raise SystemExit(
                f"RERANK_BINDING=hf failed to load {model_id!r} offline. "
                f"Pre-download with: huggingface-cli download {model_id} "
                f"(HF_HOME={hf_home or 'unset'} must contain the hub cache). "
                f"Original error: {e}"
            ) from e
        raise

    _cross_encoder_id = model_id
    _cross_encoder_device = device
    _cross_encoder_dtype_key = dtype_key
    logger.info(
        "RERANK hf: CrossEncoder loaded on device=%s dtype=%s batch_size=%s",
        device,
        torch_dtype or "fp32-default",
        _rerank_batch_size(),
    )
    try:
        from raganything.query_timing_trace import trace_event

        trace_event("rerank_load_done", model=model_id, device=device)
    except ImportError:
        pass
    return _cross_encoder


async def hf_cross_encoder_rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    model: str = "BAAI/bge-reranker-base",
    **_kwargs: Any,
) -> list[dict[str, Any]]:
    """Rerank with a local ``sentence_transformers.CrossEncoder`` (no HTTP API)."""
    if not documents:
        return []
    try:
        ce = _get_cross_encoder(model)
        pairs = [(query, d) for d in documents]
        try:
            from raganything.query_timing_trace import trace_event

            trace_event("rerank_predict_start", model=model, pairs=len(pairs))
        except ImportError:
            pass
        t_predict = time.perf_counter()
        scores = await asyncio.to_thread(_cross_encoder_predict, ce, pairs)
        predict_s = round(time.perf_counter() - t_predict, 2)
        try:
            from raganything.query_timing_trace import trace_event

            trace_event(
                "rerank_predict_done",
                model=model,
                pairs=len(pairs),
                predict_s=predict_s,
            )
        except ImportError:
            pass
        try:
            scores_list = scores.tolist()  # type: ignore[union-attr]
        except Exception:
            scores_list = [float(s) for s in scores]
        order = sorted(
            range(len(scores_list)),
            key=lambda i: float(scores_list[i]),
            reverse=True,
        )
        if top_n is not None and top_n > 0:
            order = order[:top_n]
        return [{"index": i, "relevance_score": float(scores_list[i])} for i in order]
    finally:
        if rerank_release_after_predict():
            release_cross_encoder()


def build_rerank_model_func_from_env() -> Callable[..., Any] | None:
    """Return a partial rerank coroutine, or ``None`` if reranking is disabled."""
    binding = (os.getenv("RERANK_BINDING") or "").strip().lower()
    if binding in ("", "none", "off", "false", "0", "disabled"):
        return None

    model = (os.getenv("RERANK_MODEL") or "").strip()
    api_key = (os.getenv("RERANK_BINDING_API_KEY") or "").strip() or None
    base_url = (os.getenv("RERANK_BINDING_HOST") or "").strip() or None

    if binding in ("hf", "local", "cross_encoder", "sentence_transformers"):
        mid = model or "BAAI/bge-reranker-base"
        return partial(hf_cross_encoder_rerank, model=mid)

    if binding == "jina":
        from lightrag.rerank import jina_rerank

        if not (api_key or os.getenv("JINA_API_KEY", "").strip()):
            raise SystemExit(
                "RERANK_BINDING=jina requires RERANK_BINDING_API_KEY or JINA_API_KEY."
            )
        kw: dict[str, Any] = {}
        if model:
            kw["model"] = model
        if api_key is not None:
            kw["api_key"] = api_key
        if base_url:
            kw["base_url"] = base_url
        return partial(jina_rerank, **kw)

    if binding in ("cohere", "cohere_rerank"):
        from lightrag.rerank import cohere_rerank

        if not (api_key or os.getenv("COHERE_API_KEY", "").strip()):
            raise SystemExit(
                "RERANK_BINDING=cohere requires RERANK_BINDING_API_KEY or COHERE_API_KEY."
            )
        kw = {
            "model": model or "rerank-v3.5",
            "api_key": api_key,
        }
        if base_url:
            kw["base_url"] = base_url
        chunking = (os.getenv("RERANK_ENABLE_CHUNKING") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if chunking:
            kw["enable_chunking"] = True
            mtpd = (os.getenv("RERANK_MAX_TOKENS_PER_DOC") or "").strip()
            if mtpd.isdigit():
                kw["max_tokens_per_doc"] = int(mtpd)
        return partial(cohere_rerank, **kw)

    if binding in ("aliyun", "dashscope", "ali"):
        from lightrag.rerank import ali_rerank

        if not (api_key or os.getenv("DASHSCOPE_API_KEY", "").strip()):
            raise SystemExit(
                "RERANK_BINDING=aliyun requires RERANK_BINDING_API_KEY or DASHSCOPE_API_KEY."
            )
        kw = {
            "model": model or "gte-rerank-v2",
            "api_key": api_key,
        }
        if base_url:
            kw["base_url"] = base_url
        return partial(ali_rerank, **kw)

    raise SystemExit(
        f"Unknown RERANK_BINDING={binding!r}. "
        "Use: hf | jina | cohere | aliyun | none"
    )
