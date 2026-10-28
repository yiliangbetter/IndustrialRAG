# IndustrialRAG evaluation system

This repository uses a layered evaluation system. There is deliberately no
single “RAG score”: retrieval, grounding, answer behavior, images, latency,
and product interaction outcomes are recorded and judged separately.

## What is implemented

The versioned contract lives in `raganything/evaluation_schema.py` and covers:

- cases, expected action, claims, evidence anchors, and expected images;
- page, text-span, bounding-box, modality, document hash, and chunk locators;
- reproducible run manifests (dataset/corpus/code/model/config identity);
- per-case traces, retrieved evidence, images, timings, tokens, metrics, and
  errors;
- canonical JSON and JSONL loading with strict validation.

The tracked canonical datasets are:

| Dataset | Cases | Purpose |
|---|---:|---|
| `tests/fixtures/evaluation/shili17.json` | 17 | Web-path answer/image regressions |
| `tests/fixtures/evaluation/nanxing_latency.json` | 4 | Grounded answer and latency checks |
| `tests/fixtures/evaluation/voice29.json` | 29 | Technical voice-script regressions |

All 50 cases have namespaced IDs. `raganything/evaluation_legacy.py` recreates
the previous runner inputs exactly, so migration does not silently change the
existing acceptance rules.

`rag_web_server.execute_query_with_trace()` executes the same clarification,
retrieval, reranking, streamed generation, and image-finalization path as the
browser SSE endpoint. It records answer, thinking, ordered timed events,
clarification/rejection/error outcomes, images, placements, the hook snapshot,
and the ordered chunks retained by the answer-model pipeline. A trace labels
those chunks with `context_source=llm_chunks_for_images` and
`contexts_complete=true`. Older debug snapshots and retrieved documents remain
explicitly incomplete fallbacks; they are never presented as exact model input.

## Running it

Install the query and optional RAGAS dependencies:

```bash
uv sync --extra eval
```

Validate a dataset without loading models:

```bash
uv run python scripts/run_rag_evaluation.py --validate-only
```

Run a diagnostic production-path evaluation of the migrated fixture:

```bash
uv run python scripts/run_rag_evaluation.py \
  --dataset tests/fixtures/evaluation/shili17.json \
  --working-dir data/kb_new/rag_storage \
  --parser-output-dir data/kb_new/pipeline_parse
```

After adding expert-reviewed typed evidence anchors, enable the release gate:

```bash
uv run python scripts/run_rag_evaluation.py \
  --dataset path/to/release-gold.json \
  --working-dir data/kb_new/rag_storage \
  --parser-output-dir data/kb_new/pipeline_parse \
  --corpus-id shili-kb-2026-10 \
  --corpus-hash sha256:<frozen-corpus-digest> \
  --fail-on-hard-gate
```

The runner processes cases sequentially and writes `run.json`, `run.jsonl`, and
`run.md` under `logs/evaluation_runs/<timestamp>/`. JSON/JSONL are validated by
the same schema used to load them. It atomically checkpoints those artifacts after every
case, converts unexpected per-case exceptions into error traces, and records a
storage-finalization failure without discarding completed paid work.
`--fail-on-hard-gate` requires a real `sha256:` digest with 64 hexadecimal
characters and a non-empty RAG storage directory; it exits with status 2 if any
case fails.

Run optional RAGAS diagnostics over the same production path:

```bash
uv run python scripts/eval_ragas_webpath.py --collect-only
uv run python scripts/eval_ragas_webpath.py
```

The first command records model-bound contexts without judge calls. The second
adds faithfulness, answer relevance, answer correctness, context recall, and
context precision. Rows that cannot be scored remain in the run and are listed
with an explicit reason; they are never silently discarded. Coverage is
reported separately for eligibility, fully evaluated cases, and every metric,
so a judge returning `NaN` cannot masquerade as 100% coverage. Collection is
checkpointed after each case before judge calls. Metrics are not averaged into
a dimensionally meaningless composite.

The migrated compatibility suites remain available:

```bash
uv run python scripts/benchmark_nanxing_rag.py
uv run python scripts/run_voice_script_tests.py --group 技术
uv run python scripts/run_web_path_q1_17.py
```

## Scoring and release gates

`raganything/evaluation_metrics.py` provides deterministic, dependency-free
diagnostics:

- expected-vs-observed action;
- normalized lexical vital/optional/forbidden claim checks;
- gold-document recall;
- required-image recall;
- a safety-oriented hard gate.

The generic hard gate requires a completed trace, the correct action, every
vital claim, no forbidden claim, complete model-bound answer evidence, at
least one typed gold evidence anchor, at least one explicit correctness
criterion, every exact gold evidence anchor, and every required image with
typed supporting evidence. Anchor matching honors every
specified locator, including document hash, page, chunk, span, bounding box,
and modality; a matching filename alone is insufficient when finer labels are
present. Lexical normalization preserves meaningful punctuation in values and
identifiers such as `0.6`, `5-10`, and `AB-12`. Lexical claim matching is still
only lexical, not an entailment judge.

`raganything/evaluation_legacy_metrics.py` executes the original suite-specific
checks retained in `metadata.legacy`. For migrated cases, the legacy criteria
supply the explicit correctness check, while the augmented generic gate still
requires status, action, complete model context, and typed gold evidence. The
final gate is their conjunction; a non-empty but wrong answer cannot pass
merely because older cases lack typed claim annotations. The migrated fixtures
currently remain diagnostic/compatibility datasets, not release authorities,
until domain experts add typed evidence anchors.

Recommended release policy:

1. Require zero execution errors and 100% trace coverage.
2. Require every critical case to pass its hard gate.
3. Set separate thresholds for retrieval/evidence, grounding, correctness,
   abstention, images, and latency by slice.
4. Compare changes with paired case-level deltas and 95% bootstrap confidence
   intervals; do not accept a change from a macro average alone.
5. Require human adjudication for judge disagreements and all safety-critical
   failures.

Useful slices include direct lookup, exact identifiers, numeric/table
reasoning, multi-document synthesis, visual evidence, conflicting versions,
ambiguous follow-ups, and insufficient-evidence/refusal cases.

The canonical runner currently evaluates one product turn and deliberately
does not auto-accept a clarification offer. A case that reaches clarification
is recorded as that observed action. The Nanxing compatibility benchmark still
replays its historical `keep_original` follow-up for latency parity. Non-empty
`conversation` histories are rejected at preflight rather than silently
dropped; add an explicit product-session replay runner before using multi-turn
cases as a release gate.

## Adding a gold case

Prefer stable document identity and precise evidence. A minimal answer case is:

```json
{
  "schema_version": "rag-evaluation-case/v1",
  "case_id": "maintenance:lubrication-001",
  "query": "主轴多久润滑一次？",
  "expected_action": "answer",
  "reference_answer": "主轴每周润滑一次。",
  "vital_claims": ["主轴每周润滑一次"],
  "forbidden_claims": ["主轴每天润滑一次"],
  "evidence": [
    {
      "doc_id": "maintenance-manual-v3",
      "hash": "sha256:<document-digest>",
      "page": 12,
      "span": {"quote": "主轴每周润滑一次"},
      "modality": "table"
    }
  ],
  "severity": "critical",
  "tags": ["maintenance", "numeric", "table"],
  "split": "holdout"
}
```

Use a hidden, expert-reviewed set as the release authority. Synthetic cases are
useful for coverage and perturbation testing, but should stay in a separate dev
or stress split and must not be the only release gate.

## Current limitations and next steps

- Most migrated cases lack reviewed page/region evidence. They preserve legacy
  checks, but their canonical hard gate intentionally fails until they are
  upgraded; the compatibility runners remain available for historical parity.
- A corpus hash is not inferred from a large index. Release mode requires the
  digest, but the operator must still derive it from a frozen ingestion
  manifest that covers source files, parser/OCR outputs, and index settings.
- Query cases run sequentially. The model stream currently outlives the narrow
  internal lock, so parallel in-process evaluation could mix global cleanup.
- CI validates schemas, migrations, scorers, trace assembly, and adapters but
  does not call paid/live models.
- LLM judges are not a release gate yet. Before enabling them, calibrate each
  rubric against stratified human labels, pin model/prompt versions, blind
  system identity, randomize pair order, and monitor agreement and bias.
- Add gold-context, no-retrieval, lexical, dense, hybrid/reranked, and (when
  applicable) graph/agent baselines under matched token and latency budgets.

## Research basis

The design follows the diagnostic direction established by
[RAGAS](https://aclanthology.org/2024.eacl-demo.16/), claim-level decomposition
in [RAGChecker](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html),
citation correctness/completeness in
[ALCE](https://aclanthology.org/2023.emnlp-main.398/), multi-turn failure slices
in [mtRAG](https://aclanthology.org/2025.tacl-1.36/), and evaluator calibration
concerns demonstrated by
[GroUSE](https://aclanthology.org/2025.coling-main.304/) and
[JudgeBench](https://openreview.net/forum?id=G0dksFayVq). For uncertainty,
prediction-powered inference and paired case-level resampling are preferable to
unqualified point estimates.
