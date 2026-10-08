#!/usr/bin/env python3
"""Time evaluate_clarify_gate (question -> recommendations UI)."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
load_dotenv(_ROOT / ".env", override=False)

spec = importlib.util.spec_from_file_location(
    "rag_pipeline_parse_graph_chat", _ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
)
rpc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rpc)

CASES = [
    ("1", "高速智能封边机维护保养手册适用于哪些产品型号？"),
    ("17", "所有机型中，各自哪些部件需要清除残胶"),
]


async def main() -> None:
    from raganything.clarify_gate import ClarifyRequired, evaluate_clarify_gate

    wd = Path(os.getenv("RAG_WEB_WORKING_DIR") or (_ROOT / "data" / "rag_storage")).resolve()
    pod = Path(os.getenv("RAG_WEB_PARSER_OUTPUT_DIR") or (_ROOT / "data" / "pipeline_parse")).resolve()
    rag, _, _ = await rpc._build_rag(wd, pod)
    try:
        for cid, q in CASES:
            t0 = time.perf_counter()
            gate = await evaluate_clarify_gate(rag.lightrag, q, mode="mix")
            ms = int((time.perf_counter() - t0) * 1000)
            if isinstance(gate, ClarifyRequired):
                gen = gate.data.get("generation") or {}
                cands = gate.data.get("candidates") or []
                print(
                    f"#{cid} gate_ms={ms} outcome={gate.gate_outcome} "
                    f"reason={gate.data.get('gate_reason')} cands={len(cands)} "
                    f"llm_validations={gen.get('llm_validations_used')} "
                    f"probes={gen.get('probes_used')} rounds={gen.get('rounds_used')}"
                )
                for c in cands:
                    text = (c.get("text") or "")[:70]
                    print(f"  {c.get('id')}: {text}")
            else:
                print(f"#{cid} gate_ms={ms} bypass={gate.reason}")
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
