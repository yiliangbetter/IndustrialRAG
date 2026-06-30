# 澄清门控 · Qwen rerank 劣化优化方案

> 日期：2026-06-30（P0 已落地并验证）  
> 前提：**继续使用 Qwen rerank**（`Qwen/Qwen3-Reranker-0.6B`），不切换到 BGE（BGE 虽快但分数尺度与 Qwen 不一致，易混淆阈值）。  
> 关联诊断：[`docs/澄清门控gate耗时诊断报告_20260629.md`](./澄清门控gate耗时诊断报告_20260629.md)  
> 路线 2（skip probe）：[`docs/澄清门控优化路线.md`](./澄清门控优化路线.md)

---

## 实施进度

| 项 | 状态 | 说明 |
|----|------|------|
| **P0 B1** `release_cross_encoder` + `RERANK_RELEASE_AFTER_PREDICT` | ✅ 已上线 | `raganything/pipeline_rerank.py` |
| **P0 B3** `RERANK_RELEASE_AFTER_GATE` | ✅ 已上线 | `evaluate_clarify_gate` 的 `finally` |
| **`RERANK_HF_DEVICE`** | ✅ 已上线 | rerank 与 embed 设备独立配置 |
| **CPU rerank 对照试验** | ✅ 已做 | `logs/bench_clarify_green8_20260630_100814.txt` |
| **P0 CUDA 验收** | ✅ 通过 | `logs/bench_probe_timing_green8_20260630_133338.txt` |
| **green8 端到端 gate** | ✅ 通过 | `logs/bench_clarify_green8_20260630_133531.txt` |
| **P1 B2** batch / inference_mode | ⏳ 未做 | |
| **P2 C1** 子进程 rerank | ⏳ 未做 | |

### P0 验收摘要（CUDA + release，`133338` repeat×2）

| 指标 | 优化前 `repeat2` | P0 后 `133338` |
|------|------------------|----------------|
| first-pass probe med | 35.2s | **27.5s** |
| first-pass probe max | **226.8s** (#14) | **63.8s** (#4) |
| #14 repeat | 227s → 309s | 33s → 40s |
| #7 repeat | 32s → 122s (3.85×) | 25s → 44s (1.81×) |
| VRAM | ~14–16 GB | **~2.6–3.1 GB** |

**结论**：长进程 rerank 热机劣化已消除；剩余 #4/#7 波动来自 mix 检索路径，非 rerank 单例问题。

### 推荐生产配置

```env
RERANK_HF_DEVICE=cuda
RERANK_RELEASE_AFTER_PREDICT=1
RERANK_RELEASE_AFTER_GATE=1
```

---
## 1. 要解决的问题

| 现象 | 数据 |
|------|------|
| 同一 Python 进程内，Qwen CrossEncoder rerank **越跑越慢** | batch 从 ~3s 劣化到 90–180s |
| 孤立脚本 rerank **稳定** | `standalone_rerank_stress.py` ~7.5s/轮（79 chunk） |
| gate 耗时波动大 | probe（`original_probe_s`）占 gate 绝大部分；#2 ~36s vs #4 ~71–184s |
| **不是**显存泄漏 | VRAM ~14–16GB 全程稳定 |

根因：**LightRAG 长进程内 CrossEncoder 单例常驻 CUDA**，连续 query 后 GPU 侧状态/缓存导致 predict 变慢；与 merge pool 大小、offer/reject 路径关系不大。

---

## 2. 优化优先级（仅 Qwen）

### P0 — 题间释放 CrossEncoder ✅ 已实施

**B1：`release_cross_encoder()` + 环境开关** — **已完成**

- `raganything/pipeline_rerank.py`：
  - `release_cross_encoder()`：`del` 全局单例 + `gc.collect()` + CUDA 时 `empty_cache()`
  - `RERANK_RELEASE_AFTER_PREDICT=1`：每次 `hf_cross_encoder_rerank` 的 `predict` 结束后 release
- **实测**（`133338`）：repeat 不再分钟级；VRAM ~3GB

**B3：gate 结束释放** — **已完成**

- `evaluate_clarify_gate` 在 probe 路径 `finally` 中按 `RERANK_RELEASE_AFTER_GATE=1` 再 release 一次

### P1 — predict 本身提速（不改变单例策略时仍有帮助）

**B2：predict 参数**

- `RERANK_BATCH_SIZE`（默认 32 或按 GPU/CPU 调）：传给 `CrossEncoder.predict(..., batch_size=...)`
- `torch.inference_mode()` 包裹 predict（若 ST 版本未内置）
- `show_progress_bar=False`（避免 tqdm 开销）

### P2 — 仍不够时

**C1：子进程 rerank**

- 每题或每 N 次 predict 在 **独立子进程** 跑 rerank，进程结束自然释放 CUDA
- 仍用 Qwen；代价是每题 ~2s 加载 + IPC 传 chunk 文本
- 可参考 `scripts/standalone_rerank_stress.py` 拆成 worker

### 中期（与 rerank 劣化正交）

| 措施 | 说明 |
|------|------|
| 澄清专用轻量 probe | 门控只需 final 三档；缩小 TOP_K / entity chunk |
| 短句 keyword 稳定 | ≤N 字口语题用原问或缓存 LLM 抽词（缓解 #7 类波动） |
| gate / answer 分离批测 | Web 已分离；`bench_clarify_green8` 串行 gate+answer 会放大「热机」效应 |

### 明确不做

- **不把 gate rerank 换成 BGE**（用户决策：分数尺度不一致）
- 单纯降 `MIN_RERANK_SCORE` / 加大 `CHUNK_TOP_K`：不解决耗时劣化

---

## 3. CPU rerank 试验（对比 CUDA 劣化 vs 稳定慢）

### 3.1 配置

在 `.env` 中增加（与 `HF_EMBED_DEVICE=cpu` 并列，**只影响 rerank**）：

```env
RERANK_HF_DEVICE=cpu
# 可选：CPU 上适当减小 batch，避免内存尖峰
# RERANK_BATCH_SIZE=8
```

说明：

- `RERANK_HF_DEVICE`：显式指定 `cpu` / `cuda` / `cuda:0` 等；**不设则自动选 CUDA**（与改前行为一致）
- 别名：`RERANK_DEVICE` 与 `RERANK_HF_DEVICE` 等价
- 改 `.env` 后需 **重启** Web 服务或重新跑 bench 脚本（单例在进程启动后加载）

实现位置：`raganything/pipeline_rerank.py` → `_get_cross_encoder()` 将 `device=` 传给 `CrossEncoder(...)`。

### 3.2 不建议的粗暴方式

```powershell
# 清空 CUDA_VISIBLE_DEVICES 会让整个进程看不到 GPU（embedding 若也在 GPU 会一起掉）
# 仅当「整机只想 CPU」时临时用；日常请用 RERANK_HF_DEVICE=cpu
$env:CUDA_VISIBLE_DEVICES=""
```

你当前 embedding 已是 CPU（`HF_EMBED_DEVICE=cpu`），用 `RERANK_HF_DEVICE=cpu` 即可，无需动 `CUDA_VISIBLE_DEVICES`。

### 3.3 测速步骤

**Step 1 — 孤立 rerank 基线（可选，最快）**

当前 `standalone_rerank_stress.py` 默认 auto CUDA；CPU 试验可直接改脚本一行 `CrossEncoder(..., device="cpu")`，或依赖 Step 2 的进程内路径。

**Step 2 — 进程内 probe（与 gate 同路径，推荐）**

```powershell
cd D:\dev\RAGdemo\RAG-Anything

# 1) 确认 .env：RERANK_HF_DEVICE=cpu，RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B
# 2) 同问连跑，看是否还有「第二轮暴增」
uv run python scripts/bench_probe_timing_green8.py --repeat-each 2

# 3) 与 CUDA 对比：改回 RERANK_HF_DEVICE=cuda 或删掉该行，再跑一遍，保存两份 log
```

**Step 3 — 完整 gate（可选）**

```powershell
uv run python scripts/bench_clarify_green8.py --source green8 --gate-only
```

### 3.4 如何解读结果

| 对比项 | CUDA（当前） | CPU（试验） |
|--------|--------------|-------------|
| 单次 predict | 首题 ~7s，热机后 30–180s | 通常 **更慢但稳定**（无 GPU 劣化） |
| repeat #4 | 32s → 26s 或 32s → 71s+ | 两轮应接近，波动小 |
| VRAM | ~14–16GB | rerank 不占 GPU；总 VRAM 下降 |
| 是否值得上生产 | 需 B1 修复劣化 | 若 CPU 稳定 < CUDA 热机时间，可 **embed+rerank 全 CPU** 换可预测性 |

记录 log 到 `logs/bench_probe_timing_green8_cpu_*.txt`，与 `logs/bench_probe_timing_green8_repeat2.txt` 对照。

---

## 4. 实施 B1 后的验证清单

```powershell
# A. B1 生效后：repeat 不应再出现 10× 劣化
uv run python scripts/bench_probe_timing_green8.py --repeat-each 2

# B. 题间 GC 是否额外有帮助（B1 已 release 时可作对照）
uv run python scripts/bench_probe_timing_green8.py --gc-between --repeat-each 2

# C. gate-only 端到端
uv run python scripts/bench_clarify_green8.py --source green8 --gate-only

# D. 孤立 rerank 基线（任意时刻复现 ~7.5s）
uv run python scripts/standalone_rerank_stress.py --model Qwen/Qwen3-Reranker-0.6B --rounds 8 --pool-size 79
```

**通过标准（建议）**：

- 同问 repeat 两次，`probe_s` 比值 < 1.5（例如 40s / 45s）
- #4 reject 类题不再出现 180s 级 `original_probe_s`（见 query dump）

---

## 5. 相关代码与日志

| 路径 | 说明 |
|------|------|
| `raganything/pipeline_rerank.py` | `RERANK_HF_DEVICE`、`release_cross_encoder`、B1/B3 开关 |
| `raganything/clarify_gate.py` | gate v4、B3 gate 结束 release |
| `scripts/bench_probe_timing_green8.py` | probe 耗时 + VRAM |
| `scripts/bench_clarify_green8.py` | gate/answer 分阶段批测 |
| `scripts/standalone_rerank_stress.py` | 孤立 rerank |
| `logs/bench_probe_timing_green8_repeat2.txt` | P0 前 CUDA repeat 对照 |
| `logs/bench_probe_timing_green8_20260630_133338.txt` | **P0 后验收** |
| `logs/bench_clarify_green8_20260630_133531.txt` | P0 后 green8 端到端 |
| `logs/query_dumps/20260630_094523_仿形效果不好_*.json` | P0 前 #4 reject 183s 样例 |

---

## 6. 一句话备忘

**P0 已落地：CUDA 上开 `RERANK_RELEASE_AFTER_PREDICT=1` + `RERANK_RELEASE_AFTER_GATE=1` 可消除 Qwen rerank 热机劣化；坚持 Qwen 不换 BGE；后续若仍慢看 mix 检索（#4/#7），P1 batch 或中期 keyword 稳定。**
