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
    if (
        parser_name.lower() == "mineru"
        and "device" not in out
        and sys.platform == "darwin"
    ):
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

    from raganything import RAGAnythingConfig
    from raganything.runtime_factory import RuntimeOptions, create_rag_runtime

    working_dir = Path(os.getenv("WORKING_DIR", "./rag_storage")).resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else Path(os.getenv("OUTPUT_DIR", "./output")).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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
    runtime = await create_rag_runtime(
        config,
        RuntimeOptions(
            project_root=_ROOT,
            embedding_func_max_async=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "8")),
            embedding_batch_num=int(os.getenv("EMBEDDING_BATCH_NUM", "10")),
            allow_openai_base_url=False,
            await_model_calls=False,
        ),
    )
    rag, logger = runtime.rag, runtime.logger

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
