# 配图 Placement 语义合并方案

> **目的**：将 inline 配图挂位从「多分支 + Markdown 排版依赖」收敛为「答案语义行 + 文本偏移」单路径，避免 Q15/Q16/Q17 因 `-` / `<li>` 等格式差异表现不一致。  
> **关联**：[`配图query锚定实施方案.md`](配图query锚定实施方案.md) · `scripts/image_query_refs.py` · `web/static/app.js`

---

## Checkpoint（阶段 A 前基线）

| 项 | 内容 |
|----|------|
| **Git commit** | `345f5c1` — `fix(images): Q15 KV pool, placement fixes, and inline insert by line` |
| **分支** | `lhq-rag-dev` |
| **已知问题** | `build_inline_placements` 有 5 条并行分支；无 `-` bullet 时 Q16 曾退化到 `cross_manual`；Web 依赖 `li`/`p`/`br` DOM 猜插入点 |
| **阶段 A 后 commit** | （实施后填写） |

回退示例：

```bash
git checkout 345f5c1
# 或
git revert <phase-A-commit>
```

---

## 设计原则

1. **挂位跟答案写什么**，不跟 **怎么排**（有无 `-`、`###`、单 `<p>`）。
2. **选图**（`images_for_api` / KV 补池 / pair 补图）与 **挂位**（placement）分离；本轮只动挂位。
3. 禁止领域词表；用语义行结构（`**机型**：…`、`### 机型` 段内 `- **部件**`）与既有 `source_key` / caption 对齐函数。
4. 每条 placement 必须带 **`match_start` / `match_end`**（`_answer_text_for_placement` 内偏移）及整行 `anchor_text`。

---

## 现状（阶段 A 前）

`build_inline_placements()` 优先级：

| 顺序 | 函数 | 典型题型 | 格式假设 |
|------|------|----------|----------|
| 1 | `_build_machine_component_inline_placements` | Q17 | pair + component listing query |
| 2 | `_build_machine_bullet_manual_placements` | Q15 | 必须有 `- **机型**：` |
| 3 | `_build_cross_manual_inline_placements` | 退化 | `**加粗**` 片段 ≈ 机型名 |
| 4 | `_build_listing_inline_placements` | 列表 | listing span + caption |
| 5 | 通用 caption 锚点 + `_fallback_end_placements` | 单图 | 文末兜底 |

前端：`findBlockForAnchor` → `findInsertAfterForPlacement`（`345f5c1` 已加按 `<br>` 分行）。

---

## 阶段 A：后端合并（本实施）

### 新增

| 符号 | 说明 |
|------|------|
| `_LogicLine` | `start, end, text, machine, subject` |
| `_line_spans_in_body(body)` | 纯文本行 → `(start, end, line_text)` |
| `_subjects_from_machine_field_value(value, query)` | 从 `用于保养**部件**` / 保养周期行抽 subject；过滤 `_is_maintenance_cycle_value` |
| `_answer_logic_lines(answer, query)` | **唯一抽取器**：机型字段行（有无 `-` 同规）、`### 机型` 段内部件 bullet |
| `_image_score_for_logic_line(line, img)` | 统一打分：`manual_hint` + `_pair_component_ref_align` / `_label_matches_listing_target` |
| `_build_semantic_inline_placements(...)` | 遍历 logic lines，贪心配未占用图 |
| `_build_caption_fallback_placements(...)` | 无 logic line 时，沿用原 caption/listing span 单图逻辑 |
| `scripts/replay_placement_dumps.py` | 离线回放 `logs/query_dumps/*.json`，不跑 LLM |

### 收敛

| 函数 | 阶段 A 后 |
|------|-----------|
| `build_inline_placements` | 只调 `_build_semantic_inline_placements` → 失败则 `_fallback_end_placements` |
| `_build_machine_component_inline_placements` 等 | **保留代码**，顶层不再调用（便于 diff / 回退） |

### 不动

- `images_for_api` / `filter_docs_cited_by_answer` / KV 补池
- `query_progress_hooks.finalize_inline_images` 接口
- 前端（阶段 B 再做 offset 主通道）

---

## 阶段 B（已完成）：前端偏移优先

| 动作 | 文件 | 说明 |
|------|------|------|
| **新增** | `normalizePlacementLine` / `placementLinesMatch` | 比对时去掉 `**`、`-`、`1.` 等排版差异 |
| **新增** | `lineAtPlacementOffset` | 以 `match_start` 从 `answerRaw` 取逻辑行 |
| **新增** | `findInsertPointByOffset` | 在 `li` / `p>br` 中找对应行，在其后插入 |
| **收敛** | `findInsertPointForPlacement` | offset 优先；失败再 `findBlockForAnchor` 兜底 |
| **收敛** | `applyInlineImages` | 只调 `findInsertPointForPlacement` |

**不改**（阶段 B 初版）：`image_query_refs.py` 及 placement 产出逻辑。

**后续（tag `OK0614_no_clarify`）**：后端 placement 与前端贴图均有增量修复，见下方 Checkpoint C。

---

## Checkpoint C（Q3/Q4/Q11 回归 · 2026-06-14）

| 项 | 内容 |
|----|------|
| **Git tag** | `OK0614_no_clarify`（commit `fe9bf94`） |
| **后端** | 周期/步骤 query 门控；`_cycle_caption_block_placement`；Q11 单手册列举 |
| **前端** | `insertFigureAfterAnchor` 段落级插入；三层兜底 |
| **验收** | Q3/Q4/Q11 dump replay + Web 手测 Q3 |

---

## Checkpoint（阶段 B 完成）

| 项 | 内容 |
|----|------|
| **Git commit** | `625af82` — `feat(web): phase B offset-primary inline image insertion` |
| **前置** | `b0590cc` 阶段 A |
| **改动文件** | `web/static/app.js` |

回退阶段 B：`git revert <phase-B-commit>`

---

## 回归清单

### 1. 离线 replay（改完第一步）

```bash
$env:PYTHONIOENCODING="utf-8"
python scripts/replay_placement_dumps.py
python scripts/replay_placement_dumps.py --glob "*透平*"
python scripts/replay_placement_dumps.py --glob "*电控板*"
python scripts/replay_placement_dumps.py --glob "*残胶*"
```

| 用例 | 验收 |
|------|------|
| Q15 | ≥4 placement；含高速智能；`anchor_text` 为整行 |
| Q16 | 3 placement；无 bullet dump 与有 bullet dump 均整行锚点 |
| Q11 | 3 placement；压带轮/仿形靠板/涂胶轴各 1；涂胶电机可选 |
| Q3 | 周期短答：整段或 bullet 锚点；非括号内子串 |
| Q4 | 步骤题：首条编号步骤或 query 主题配对；非段首导语 |
| Q17 | 每 (机型, 部件) 1 placement；高速智能 3 部件不抢图 |

### 2. Web 批测

```bash
$env:PYTHONIOENCODING="utf-8"
python scripts/run_web_path_q1_17.py --ids 15,16,17
python scripts/run_web_path_q1_17.py
```

### 3. 手测

`RAG_QUERY_DEBUG_DUMP=1`，Q15/Q16 各 2 次，查 `inline_placements`。

---

## 风险

| 风险 | 控制 |
|------|------|
| Q1–Q14 单图退化 | `_build_caption_fallback_placements` 保留原 caption 锚点逻辑 |
| Q13 多图 | `used_indices` + `used_ranges` 与现实现一致 |
| 回退 | 基于 `345f5c1`；旧 `_build_*` 函数仍留文件中 |

---

## Checkpoint（阶段 A 完成）

| 项 | 内容 |
|----|------|
| **Git commit** | `b0590cc` — `refactor(images): phase A semantic inline placement merge` |
| **前置基线** | `345f5c1` — placement 修补 + Web 按行插入 |
| **主要改动** | `build_inline_placements` 仅调 `_build_semantic_inline_placements`；新增 `_answer_logic_lines` |
| **回放脚本** | `scripts/replay_placement_dumps.py` |

回退阶段 A：`git revert b0590cc` 或 `git checkout 345f5c1`

---

## 实施记录

| 日期 | 内容 | commit / tag |
|------|------|----------------|
| 2026-06-14 | 文档创建 | — |
| 2026-06-14 | 阶段 B：Web `match_start` 偏移优先插入 | `625af82` |
| 2026-06-14 | Q3/Q4 周期·步骤 placement + Web 块级插入兜底 | `OK0614_no_clarify` |
