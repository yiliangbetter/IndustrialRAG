#!/usr/bin/env python
"""Export MinerU parse artifacts (*_content_list.json, md, images) with no LightRAG / no LLM.

Uses the same layout as batch_parser: ``--output/<stem>/`` plus MinerU subfolders (e.g. ``office/``, ``auto/``).

Defaults aim to reduce garbage text vs forced OCR:
  - ``--method auto`` lets MinerU pick txt vs ocr per page (better for digital PDFs / Word→PDF).
  - ``--lang ch`` for Chinese manuals.

Set ``MINERU_MODEL_SOURCE=modelscope`` when Hugging Face is unreachable (matches ~/.cache/modelscope weights).

Example::

    export MINERU_MODEL_SOURCE=modelscope
    python scripts/export_parse_json_no_llm.py \\
        --input uploaded_documents --output output/data_upload_test_v4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _venv_path() -> None:
    candidates = [_ROOT / ".venv" / "bin", _ROOT.parent.parent / ".venv" / "bin"]
    for bin_dir in candidates:
        if bin_dir.is_dir():
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            return


def _loopback_no_proxy() -> None:
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


def main() -> int:
    _venv_path()
    _loopback_no_proxy()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "uploaded_documents",
        help="File or directory of documents",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "output" / "data_upload_test_v4",
        help="Base output directory (mirrors data_upload_test_v3 layout)",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["auto", "txt", "ocr"],
        default="auto",
        help="MinerU parse method (auto recommended for mixed digital/scanned)",
    )
    parser.add_argument("--lang", "-l", default="ch", help="MinerU language tag")
    parser.add_argument(
        "--backend",
        "-b",
        default="pipeline",
        help="MinerU backend (default pipeline)",
    )
    parser.add_argument(
        "--source",
        "-s",
        choices=["huggingface", "modelscope", "local"],
        default=os.getenv("MINERU_MODEL_SOURCE", "modelscope"),
        help="MinerU --source (default: env MINERU_MODEL_SOURCE or modelscope)",
    )
    parser.add_argument(
        "--device", "-d", default=None, help="MinerU device e.g. cpu, mps"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel files (default 1 for heavy pipeline)",
    )
    parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=28800,
        help="ThreadPool wait between completions (seconds); raise for huge manuals",
    )
    parser.add_argument(
        "--no-formula",
        action="store_true",
        help="MinerU -f false",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="MinerU -t false",
    )
    args = parser.parse_args()

    os.environ.setdefault("MINERU_MODEL_SOURCE", args.source)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from raganything.batch_parser import BatchParser

    inp = args.input.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    paths = [str(inp)] if inp.is_file() else [str(inp)]

    bp = BatchParser(
        parser_type="mineru",
        max_workers=max(1, args.max_workers),
        show_progress=True,
        timeout_per_file=max(300, args.timeout_per_file),
    )

    extra = dict(
        lang=args.lang or None,
        backend=args.backend,
        source=args.source,
        formula=not args.no_formula,
        table=not args.no_table,
    )
    if args.device:
        extra["device"] = args.device

    result = bp.process_batch(
        file_paths=paths,
        output_dir=str(out),
        parse_method=args.method,
        recursive=True,
        **extra,
    )
    print(result.summary())
    if result.failed_files:
        for fp in result.failed_files:
            print("FAILED:", fp, result.errors.get(fp, ""))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
