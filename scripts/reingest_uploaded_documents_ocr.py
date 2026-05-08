#!/usr/bin/env python
"""Re-ingest documents from uploaded_documents with OCR (parse_method=ocr).

Pre-downloads MinerU pipeline weights when possible (mineru-models-download).
Uses the same LLM / embedding setup as scripts/run_demo_question_bank.py.

Env: MINERU_MODEL_SOURCE=huggingface|modelscope (default huggingface),
     MINERU_LANG / OCR_LANG, MINERU_BACKEND, MINERU_DEVICE, WORKING_DIR, OUTPUT_DIR.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(dotenv_path=_ROOT / ".env", override=False)


def _ensure_venv_path() -> None:
    venv_bin = _ROOT / ".venv" / "bin"
    if venv_bin.is_dir():
        os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")


def _ensure_hf_hub_timeouts() -> None:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")


def _prefer_hf_hub_offline_for_embeddings() -> None:
    """Use local HF cache only when embedding runs from HF_HOME (avoids blocked huggingface.co)."""
    if os.getenv("EMBEDDING_BACKEND", "openai").strip().lower() != "hf":
        return
    if not (os.getenv("HF_HOME") or "").strip():
        return
    if os.getenv("HF_HUB_OFFLINE"):
        return
    if os.getenv("HF_FORCE_ONLINE", "").lower() in ("1", "true", "yes"):
        return
    os.environ["HF_HUB_OFFLINE"] = "1"


def _ensure_loopback_no_proxy() -> None:
    parts: list[str] = []
    for key in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(key, "")
        if raw:
            parts.extend(x.strip() for x in raw.split(",") if x.strip())
    for host in ("127.0.0.1", "localhost", "::1"):
        if host not in parts:
            parts.append(host)
    merged = ",".join(dict.fromkeys(parts))
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


def _download_mineru_pipeline_models() -> None:
    src = os.getenv("MINERU_MODEL_SOURCE", "huggingface").strip().lower()
    if src not in ("huggingface", "modelscope"):
        src = "huggingface"
    cmd = ["mineru-models-download", "-s", src, "-m", "pipeline"]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=False, cwd=str(_ROOT))


def _resolve_keys() -> tuple[str, str]:
    llm_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("LLM_BINDING_API_KEY", "").strip()
    )
    emb_key = os.getenv("EMBEDDING_API_KEY", "").strip() or llm_key
    return llm_key, emb_key


def _mineru_parse_kwargs(parser_name: str) -> dict:
    out: dict = {}
    lang = os.getenv("MINERU_LANG", "").strip() or os.getenv("OCR_LANG", "").strip()
    if lang:
        out["lang"] = lang
    for key, env in (
        ("backend", "MINERU_BACKEND"),
        ("source", "MINERU_SOURCE"),
        ("device", "MINERU_DEVICE"),
    ):
        v = os.getenv(env, "").strip()
        if v:
            out[key] = v
    if parser_name.lower() == "mineru" and "backend" not in out:
        out["backend"] = "pipeline"
    if parser_name.lower() == "mineru" and "device" not in out and sys.platform == "darwin":
        out["device"] = "cpu"
    return out


async def main() -> None:
    _ensure_venv_path()
    _prefer_hf_hub_offline_for_embeddings()
    _ensure_hf_hub_timeouts()
    _ensure_loopback_no_proxy()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder",
        type=Path,
        default=_ROOT / "uploaded_documents",
        help="Directory containing originals (default: uploaded_documents)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Parser output dir (default: OUTPUT_DIR env or ./output)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many files (sorted paths), for testing",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Do not run mineru-models-download first",
    )
    args = parser.parse_args()
    folder: Path = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    if not args.skip_model_download:
        _download_mineru_pipeline_models()

    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc, logger
    from raganything import RAGAnything, RAGAnythingConfig
    from raganything.local_hf_embedding import (
        ensure_hf_home_from_repo_fallback,
        make_local_hf_embedding_func,
    )

    ensure_hf_home_from_repo_fallback(_ROOT)

    llm_key, emb_key = _resolve_keys()
    if not llm_key:
        raise SystemExit(
            "Set OPENAI_API_KEY or LLM_BINDING_API_KEY in the environment or .env"
        )

    working_dir = Path(os.getenv("WORKING_DIR", "./rag_storage")).resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else Path(os.getenv("OUTPUT_DIR", "./output")).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_backend = os.getenv("EMBEDDING_BACKEND", "openai").strip().lower()

    base_url = os.getenv("LLM_BINDING_HOST", "").strip() or None
    emb_host = os.getenv("EMBEDDING_BINDING_HOST", "").strip()
    embedding_base_url = emb_host if emb_host else base_url
    if embedding_backend != "hf":
        if emb_host and not os.getenv("EMBEDDING_API_KEY", "").strip():
            raise SystemExit(
                "EMBEDDING_BINDING_HOST is set; set EMBEDDING_API_KEY to the key for that host."
            )

    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    vision_model = os.getenv("VISION_MODEL", llm_model)
    if embedding_backend == "hf":
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    else:
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
        embedding_model = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        ).strip()

    config = RAGAnythingConfig(
        working_dir=str(working_dir),
        parser=os.getenv("PARSER", "mineru"),
        parse_method="ocr",
        parser_output_dir=str(output_dir),
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        max_concurrent_files=int(os.getenv("MAX_CONCURRENT_FILES", "1")),
    )
    parse_extra = _mineru_parse_kwargs(config.parser)

    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=llm_key,
            base_url=base_url,
            **kwargs,
        )

    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if history_messages is None:
            history_messages = []
        if messages:
            return openai_complete_if_cache(
                vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=llm_key,
                base_url=base_url,
                **kwargs,
            )
        if image_data:
            return openai_complete_if_cache(
                vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}",
                                },
                            },
                        ],
                    },
                ],
                api_key=llm_key,
                base_url=base_url,
                **kwargs,
            )
        return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    if embedding_backend == "hf":
        embedding_func = make_local_hf_embedding_func(
            embedding_dim,
            embedding_model=embedding_model,
        )
    else:
        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=partial(
                openai_embed.func,
                model=embedding_model,
                api_key=emb_key,
                base_url=embedding_base_url,
            ),
        )

    embedding_func_max_async = int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "8"))
    embedding_batch_num = int(os.getenv("EMBEDDING_BATCH_NUM", "10"))

    lightrag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        enable_llm_cache=True,
        embedding_func_max_async=embedding_func_max_async,
        embedding_batch_num=embedding_batch_num,
    )
    await lightrag.initialize_storages()

    rag = RAGAnything(
        config=config,
        lightrag=lightrag,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    logger.info(
        "Starting OCR re-ingest: folder=%s working_dir=%s parser=%s",
        folder,
        working_dir,
        config.parser,
    )

    exts = {
        e.lower() if e.startswith(".") else f".{e.lower()}"
        for e in config.supported_file_extensions
    }
    files: list[Path] = []
    for ext in exts:
        files.extend(folder.glob(f"*{ext}"))
    files = sorted({p.resolve() for p in files if p.is_file()})
    if not files:
        raise SystemExit(f"No supported files in {folder} (extensions: {exts})")
    if args.limit is not None:
        files = files[: max(0, args.limit)]

    for fp in files:
        logger.info("Processing (OCR): %s", fp.name)
        await rag.process_document_complete(
            str(fp),
            output_dir=str(output_dir),
            parse_method="ocr",
            **parse_extra,
        )

    logger.info("OCR re-ingest finished (%d files) under %s", len(files), folder)


if __name__ == "__main__":
    asyncio.run(main())
