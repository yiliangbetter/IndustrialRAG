# Qwen rerank 耗时优化 — 实现说明

> **版本**：P0 + P1（tag `Qwen_rerank_optimized_P0+P1`）
> **方案与验收数据**：[`澄清门控Qwen_rerank优化方案.md`](./澄清门控Qwen_rerank优化方案.md)
> **诊断报告**：[`澄清门控gate耗时诊断报告_20260629.md`](./澄清门控gate耗时诊断报告_20260629.md)
> **澄清门控**：[`澄清门控实现说明_v4.md`](./澄清门控实现说明_v4.md)

本文档说明 **已落地的代码改动**：改了什么、在哪改、调用链如何、**不改变**哪些语义（chunk 集合、`rerank_score`、门控 `final_score`）。

---

## 1. 背景与根因

| 现象 | 说明 |
|------|------|
| 长进程内 Qwen `CrossEncoder.predict` 越跑越慢 | 同问 repeat 可从 ~30s 劣化到 180s+ |
| 孤立脚本稳定 | `standalone_rerank_stress.py` ~7.5s/轮 |
| VRAM ~14–16GB 常驻 | 非泄漏，而是 **单例常驻 GPU** |

**根因**：`pipeline_rerank.py` 用模块级全局变量缓存 `CrossEncoder`；优化前每次 `predict` 后不释放，CUDA 侧状态在连续 query 后恶化。

**不改**：继续用 `Qwen/Qwen3-Reranker-0.6B`，不换成 BGE（分数尺度与门控阈值不一致）。

---

## 2. 改动文件一览

| 文件 | 改动 |
|------|------|
| `raganything/pipeline_rerank.py` | 设备选择、release、P1 predict、rerank 主路径 |
| `raganything/clarify_gate.py` | gate 结束可选 release（P0 B3） |
| `config/env.example` | 新增 env 注释项 |

**未改**：`clarify_gate` 的 `final_score` 计算、`MIN_RERANK_SCORE`、merge pool 大小、LightRAG 检索逻辑。

---

## 3. 调用链（rerank 何时执行）

```text
LightRAG.aquery / aquery_data
  → lightrag.utils.apply_rerank_if_enabled
  → rerank_model_func（build_rerank_model_func_from_env 注入）
  → hf_cross_encoder_rerank
       → _get_cross_encoder（可能加载模型）
       → _cross_encoder_predict（P1）
       → [P0] release_cross_encoder（若 RERANK_RELEASE_AFTER_PREDICT=1）

澄清门控 evaluate_clarify_gate
  → probe_llm_retrieval_full → aquery_data（同上，触发 rerank）
  → [P0] finally: release_cross_encoder（若 RERANK_RELEASE_AFTER_GATE=1）
```

`rerank_model_func` 挂载点（Web / 批测共用）：

```397:410:scripts/rag_pipeline_parse_graph_chat.py
    from raganything.pipeline_rerank import build_rerank_model_func_from_env

    rerank_model_func = build_rerank_model_func_from_env()
    ...
        rerank_model_func=rerank_model_func,
```

---

## 4. 全局单例与设备（优化前已有 + `RERANK_HF_DEVICE`）

模块级三变量缓存 CrossEncoder；**model_id 或 device 变化** 会触发重新加载。

```28:30:raganything/pipeline_rerank.py
_cross_encoder_id: str | None = None
_cross_encoder_device: str | None = None
_cross_encoder: Any = None
```

### 4.1 `_resolve_rerank_device`

与 embedding 的 `HF_EMBED_DEVICE` **独立**；未设置时自动选 CUDA → MPS → CPU。

```33:48:raganything/pipeline_rerank.py
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
```

**注意**：须写 `cuda`，不是 `gpu`。

### 4.2 `_get_cross_encoder` 传入 `device=`

加载时写入 `kwargs["device"]`；缓存命中条件包含 device：

```149:157:raganything/pipeline_rerank.py
def _get_cross_encoder(model_id: str) -> Any:
    global _cross_encoder_id, _cross_encoder_device, _cross_encoder
    device = _resolve_rerank_device()
    if (
        _cross_encoder is not None
        and _cross_encoder_id == model_id
        and _cross_encoder_device == device
    ):
        return _cross_encoder
```

```187:203:raganything/pipeline_rerank.py
    kwargs["device"] = device
    try:
        _cross_encoder = CrossEncoder(load_id, **kwargs)
    ...
    _cross_encoder_id = model_id
    _cross_encoder_device = device
    logger.info("RERANK hf: CrossEncoder loaded on device=%s", device)
    return _cross_encoder
```

---

## 5. P0 — 题间释放 CrossEncoder

### 5.1 环境开关

```51:62:raganything/pipeline_rerank.py
def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes")


def rerank_release_after_predict() -> bool:
    """When true, drop CrossEncoder singleton after each ``hf_cross_encoder_rerank`` predict."""
    return _env_flag("RERANK_RELEASE_AFTER_PREDICT")


def rerank_release_after_gate() -> bool:
    """When true, drop CrossEncoder singleton when ``evaluate_clarify_gate`` finishes."""
    return _env_flag("RERANK_RELEASE_AFTER_GATE")
```

| 变量 | 触发时机 |
|------|----------|
| `RERANK_RELEASE_AFTER_PREDICT=1` | 每次 `hf_cross_encoder_rerank` 的 `predict` 结束后（B1） |
| `RERANK_RELEASE_AFTER_GATE=1` | `evaluate_clarify_gate` 主流程 `finally`（B3，兜底） |

两者可同时开启；gate 路径上可能 **predict 释放一次 + gate 结束再释放一次**（第二次多为 no-op）。

### 5.2 `release_cross_encoder`

```88:108:raganything/pipeline_rerank.py
def release_cross_encoder() -> None:
    """Drop the global CrossEncoder singleton and free GPU memory if applicable."""
    global _cross_encoder_id, _cross_encoder_device, _cross_encoder
    if _cross_encoder is None:
        return
    device = (_cross_encoder_device or _resolve_rerank_device() or "").lower()
    ce = _cross_encoder
    _cross_encoder = None
    _cross_encoder_id = None
    _cross_encoder_device = None
    del ce
    gc.collect()
    if device.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    logger.debug("RERANK hf: released CrossEncoder (device was %s)", device or "unknown")
```

要点：

- 清空三个全局变量，避免下次 `_get_cross_encoder` 误用旧实例
- CUDA 上 `empty_cache()`；CPU 跳过
- **代价**：下次 rerank 需重新 `CrossEncoder(...)`，约 **~2s**（本地 snapshot）

### 5.3 B1：每次 rerank 后 release

`hf_cross_encoder_rerank` 用 `try/finally` 保证 predict 异常时也会尝试释放：

```206:236:raganything/pipeline_rerank.py
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
        scores = await asyncio.to_thread(_cross_encoder_predict, ce, pairs)
        ...
        return [
            {"index": i, "relevance_score": float(scores_list[i])} for i in order
        ]
    finally:
        if rerank_release_after_predict():
            release_cross_encoder()
```

**语义不变**：

- `documents` 列表仍 **全部** 打分（除非 LightRAG 上层已截断 pool）
- 返回的 `relevance_score` 与优化前同一模型、同一对 `(query, doc)` 应一致
- 仅 **生命周期** 变化：不常驻 GPU

### 5.4 B3：gate 结束 release

`clarify_gate.py` 在 probe 路径外包 `try/finally`（`use_candidate` 校验短路 **不** 经过此处）：

```29:32:raganything/clarify_gate.py
from raganything.pipeline_rerank import (
    release_cross_encoder,
    rerank_release_after_gate,
)
```

```717:725:raganything/clarify_gate.py
    try:
        return await _evaluate_clarify_gate_probed(
            lightrag,
            q,
            mode=mode,
        )
    finally:
        if rerank_release_after_gate():
            release_cross_encoder()
```

`_evaluate_clarify_gate_probed` 内至少一次 `probe_llm_retrieval_full`（原问 mix+rerank）；路线 1 还有推荐 probe。B3 保证 gate 整段结束后 GPU 上无残留 CrossEncoder。

---

## 6. P1 — predict 调优（不改变 chunk / 分数）

### 6.1 `_rerank_batch_size`

```65:70:raganything/pipeline_rerank.py
def _rerank_batch_size() -> int:
    raw = (os.getenv("RERANK_BATCH_SIZE") or "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    device = _resolve_rerank_device()
    return 8 if device == "cpu" else 32
```

`batch_size` 只影响 **算分批处理粒度**，不改变参与 rerank 的文档条数。

### 6.2 `_cross_encoder_predict`

```73:85:raganything/pipeline_rerank.py
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
```

| 参数 | 作用 |
|------|------|
| `batch_size` | GPU 默认 32，CPU 默认 8 |
| `show_progress_bar=False` | 避免 tqdm 在批测日志里刷屏 |
| `torch.inference_mode()` | 推理模式，略减开销 |

**优化前**（P1 之前）为：

```python
scores = await asyncio.to_thread(ce.predict, pairs)
```

P1 后改为：

```python
scores = await asyncio.to_thread(_cross_encoder_predict, ce, pairs)
```

验收结论：P0+P1 相对仅 P0，端到端 probe **无稳定显著缩短**；P1 价值在于路径规范、可配 batch。

---

## 7. 推荐生产 `.env`

```env
RERANK_BINDING=hf
RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B
RERANK_HF_DEVICE=cuda

# P0 — 消除长进程劣化（建议都开）
RERANK_RELEASE_AFTER_PREDICT=1
RERANK_RELEASE_AFTER_GATE=1

# P1 — 可选；不设则用 cuda→32 / cpu→8
# RERANK_BATCH_SIZE=32
```

与 embedding 并列示例（仅 rerank 上 GPU、embed 可仍 CPU）：

```env
HF_EMBED_DEVICE=cpu
RERANK_HF_DEVICE=cuda
```

改 `.env` 后须 **重启** Web 或重跑批测进程（单例在进程内初始化）。

`config/env.example` 中对应注释：

```15:24:config/env.example
RERANK_BY_DEFAULT=true
RERANK_BINDING=hf
RERANK_MODEL=BAAI/bge-reranker-base
# Rerank 设备：cpu | cuda | cuda:0（不设则自动选 CUDA）。与 HF_EMBED_DEVICE 独立。
# RERANK_HF_DEVICE=cpu
# Qwen CUDA 长进程 rerank 劣化缓解（见 docs/澄清门控Qwen_rerank优化方案.md）：
# RERANK_RELEASE_AFTER_PREDICT=1
# RERANK_RELEASE_AFTER_GATE=1
# P1 predict 调优（不改变打分 chunk 集合与分数）：
# RERANK_BATCH_SIZE=32
```

---

## 8. 公开 API（可供测试或其它模块调用）

`pipeline_rerank.py` 的 `__all__`：

```18:24:raganything/pipeline_rerank.py
__all__ = [
    "build_rerank_model_func_from_env",
    "hf_cross_encoder_rerank",
    "release_cross_encoder",
    "rerank_release_after_gate",
    "rerank_release_after_predict",
]
```

---

## 9. 行为对比表

| 维度 | 优化前 | P0 + P1 后 |
|------|--------|------------|
| CrossEncoder 生命周期 | 进程内常驻至退出 | 每次 predict 后可释放 |
| VRAM（rerank） | ~14–16 GB 常驻 | ~2–3 GB（不持有模型时） |
| 单次 predict 后 | 可能热机劣化 | 稳定；下次加载 ~2s |
| merge pool / chunk 数 | 由 LightRAG 决定 | **相同** |
| `rerank_score` / `final_score` | CrossEncoder 输出 | **相同**（同模型同输入） |
| 门控 direct/offer/reject | 由阈值决定 | **相同**（批测 outcomes 一致） |

---

## 10. 验收命令与参考 log

```powershell
cd D:\dev\RAGdemo\RAG-Anything

# P0：同问 repeat，比值应 < ~1.5，无分钟级暴增
uv run python scripts/bench_probe_timing_green8.py --repeat-each 2

# P0+P1 对照（可选）
uv run python scripts/bench_probe_timing_green8.py --repeat-each 2 `
  --out-report logs/bench_probe_timing_green8_p1_20260630.txt

# 孤立 rerank 基线（~7.5s/轮，79 chunk）
uv run python scripts/standalone_rerank_stress.py --model Qwen/Qwen3-Reranker-0.6B --rounds 8 --pool-size 79

# 端到端 gate
uv run python scripts/bench_clarify_green8.py --source green8 --gate-only
```

| log | 说明 |
|-----|------|
| `logs/bench_probe_timing_green8_repeat2.txt` | 优化前（劣化明显） |
| `logs/bench_probe_timing_green8_20260630_133338.txt` | P0 验收 |
| `logs/bench_probe_timing_green8_p1_20260630.txt` | P0+P1 |
| `logs/bench_clarify_green8_20260630_133531.txt` | P0 后 green8 全程 |

启动日志确认：

```text
INFO:raganything.pipeline_rerank:RERANK hf: CrossEncoder loaded on device=cuda
```

每次 rerank 后若开启 P0，同进程内下一次检索会再次出现 `CrossEncoder loaded`（reload）。

---

## 11. 未实现项（方案中的 P2）

**C1 子进程 rerank**：每题独立进程跑 CrossEncoder，进程退出自然释放 CUDA。当前 P0 已满足生产稳定性需求，P2 仅作备选。可参考 `scripts/standalone_rerank_stress.py` 拆 worker。

---

## 12. Git 标签

| Tag | 内容 |
|-----|------|
| `Qwen_rerank_optimized` | P0（release + `RERANK_HF_DEVICE`） |
| `Qwen_rerank_optimized_P0+P1` | P0 + P1（batch / inference_mode） |

---

## 13. 与澄清门控的关系

- 门控 **只读** rerank 结果算 `final_score`，不调用 `release_*` 自身逻辑（除 B3 的 `finally`）。
- 路线 2（`CLARIFY_CANDIDATE_SKIP_PROBE=1`）下 gate 仅 **原问 1 次** rerank；answer 阶段另一次 rerank，各自触发 B1 release。
- 配图、推荐 LLM **不经过** `pipeline_rerank` 的改动路径。

详见 [`澄清门控实现说明_v4.md`](./澄清门控实现说明_v4.md) §5、§11。

---

*文档版本：impl-qwen-rerank-20260630 · commit/tag `Qwen_rerank_optimized_P0+P1`*
