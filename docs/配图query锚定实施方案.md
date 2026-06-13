# 配图 Query 锚定实施方案（阶段 1 已完成）

> **目的**：记录 2026-06-12 讨论的配图改进方向与分步计划，便于追溯与回退。  
> **关联**：[`查询配图设计方案.md`](查询配图设计方案.md) · [`测试例参考答案.md`](测试例参考答案.md)  
> **批测脚本**：`scripts/run_web_path_q1_17.py`

---

## Checkpoint（实施前基线）

| 项 | 内容 |
|----|------|
| **Git commit** | `1d3b5f4` — `feat(query,images): Plan B citation images and Q1-17 batch baseline` |
| **分支** | `lhq-rag-dev`（提交时未 push） |
| **批测结果** | 全量 Q1–Q17：**14/17**（文字 17/17）；报告 `logs/web_path_q1_17/20260612_122334_all.md` |
| **配图失败** | Q5、Q9、Q17 |
| **本 commit 已含** | Plan B（LLM chunk 不收窄给配图）、citation 池扩至 rerank、span 策略（Q15 四图 / Q17 部件列举）、Q5 concise prompt、参考答案更新、批测脚本 |

回退示例：

```bash
git checkout 1d3b5f4
# 或
git revert <later-commits>
```

---

## Checkpoint（阶段 1）

| 项 | 内容 |
|----|------|
| **Git commit** | `4837143` — `feat(images): phase 1 query-section anchor for terse-answer figures` |
| **批测结果** | 全量 Q1–Q17：**17/17**；报告 `logs/web_path_q1_17/20260612_161319_all.md` |
| **相对基线** | 修 Q5/Q9/Q17 配图；Q1–Q16 无回归 |
| **主要改动** | `_anchor_chunks_by_query_section`、`cite_pool`/`anchor_pool` 分离、`figure_pool`、`_figure_context_from_answer_docs` query 回退、多机型手册名 span |
| **开关** | `RAG_IMAGE_QUERY_SECTION_ANCHOR=1`（`config/env.example`） |

回退阶段 1：`git revert <phase-1-commit>` 或 `git checkout 1d3b5f4` 后 cherry-pick 所需提交。

---

## 问题诊断（2026-06-12）

### 现象

- **Q5**：答案精简为「刮刀」「切勿划伤压带轮表面涂层」，文字正确但 **0 图**（`no_answer_cited_chunks`）。
- **Q9**：答案仅「美孚长效液压油」，**0 图**（同上）。
- **Q17**：答案跨机型列举部件正确，但图仅 2 张，未达 ≥3（缺涂胶轴、激光出光口等）。

### 根因（共识）

1. **配图过度依赖答案字面匹配**：citation 要求答案与 chunk 有较长子串重合，或列举题要求答案有加粗「部件名 / 图注同款」；答案 paraphrase 后（如只写刮刀、涂层）无法锚定到手册 **「2.1.5 清理压带轮残胶」** 整段及内联图。
2. **单题 early return**：`filter_docs_cited_by_answer` 在 `scored` 为空且非 listing 时直接 `return []`，后续的 query 补图逻辑 **未执行**（Q5/Q9 全量失败原因之一）。
3. **列举题 span 与 pool 覆盖**：Q17 用部件 span 从 rerank 池补 chunk，但池外手册段落（如高速自动机激光出光口）仍可能缺失。
4. **灌库无问题**：库内 chunk 已含 `[图片]` 与正文关联；问题在 **查询侧 citation / 锚定策略**，非灌库缺失。

### 用户期望（产品语义）

从 **query**（如「清理压带轮残胶用什么工具」）应能定位到手册 **节号/段落**（如 `2.1.5`），再取该段内联图；答案即使不写「压带轮残胶清理」字样，只要内容同属该段（刮刀、勿划伤涂层），配图仍应正确。

---

## 设计原则（已确认，实施须遵守）

### 1. 答案 citation = 主（划定范围）

- 配图 **必须** 来自「与本次回答相关」的 chunk 内的 `[图片]`。
- **答案** 用于：流式结束后 **收窄**「写答案时见过的片段」中 **与回答语义一致** 的子集（防答案讲 A、从 B 段抠图）。
- **不把答案当作扩大搜图范围的辅助信号**（避免 Q11「此外…刮刀片」、Q5 展开邻节等 **误导配图**）。

### 2. Query = 辅（节级 / 段落锚定）

- 当答案 **过短** 或 **未复述图注/节标题** 时，用 **query** 在 **已收窄的候选集内** 对齐到 **节号 + 保养内容行 + 保养步骤** 所在段落。
- Query **不得** 独立开启「全 rerank 池 / 全库」第二条搜图线，避免问法歧义把图配到 **与答案无关** 的 chunk。

### 3. 候选集边界（硬约束）

Query 辅助锚定 **仅允许** 在以下集合内选 chunk（优先级从高到低）：

1. 答案 citation 已保留的 chunk（`kept`）；
2. 与 `kept` **同一来源手册、同一节号**（如均为 `2.1.5`）的邻接 chunk（灌库 coalesce 已保证同节多段可合并）；
3. **不** 因 query 单独从 rerank 全池引入 **citation 未触及且与答案无弱一致** 的新手册段。

> 弱一致：例如 Q5 答案含「压带轮」「刮刀」，query 含「压带轮残胶」→ 允许在同手册 `2.1.5` 段内用 query 对齐；**不允许** query 含「残胶」就跳到 `2.1.4 仿形靠模`  unless 答案或 citation 已涉及该节。

### 4. 与现有 Plan A / Plan B 的关系

- **Plan B**（`1d3b5f4`）：送给 LLM 的 chunk 快照不人为删节，配图与写答案 **同源** — **保留**。
- **Route A**：只解析 cited chunk 内 inline `[图片]` — **保留**；不恢复 LEGACY `content_list` 全库扫图作为主路径。
- **待收敛**：`1d3b5f4` 中为修 Q15/Q17 引入的 **answer span_keep / answer_topic supplement** 需按上文原则 **收缩或改写**（列举题 Q11/Q17 可保留 **query+listing 结构**，但 span 来源应优先 query 与 citation，而非答案加粗扩池）。

---

## 分步实施方案

### 阶段 0：文档与基线（已完成）

- [x] 提交 `1d3b5f4` 作为回退点
- [x] 全量批测并记录 `20260612_122334_all.md`
- [x] 本文档

### 阶段 1：修单题「0 图」断链（低风险，优先）✅ 已完成

**目标**：Q5、Q9 类「短答案 + 单保养条目」恢复配图，不扩大误配面。

| 步骤 | 改动要点 | 验收 |
|------|----------|------|
| 1.1 | 去掉 / 改写 `filter_docs_cited_by_answer` 中 `not scored and not span_keep` 的 **提前 return**；在空 `kept` 时进入 query 辅锚分支 | Q5、Q9 不再 `no_answer_cited_chunks` |
| 1.2 | 新增 `_anchor_chunks_by_query_section(query, pool, *, answer_gate)`：在 pool 内用 query subject needles + 节标题（`2.1.5 …`）+ `保养内容：` 对齐，**且** chunk 与 answer 有最低 term/步骤重合（gate，非扩池） | Q5 → 图注「压带轮残胶清理」；Q9 → 注油泵相关图 |
| 1.3 | Query 辅锚 **仅写回 `kept`**，不修改送给答案模型的 chunk；配图仍走 `resolve_query_images` | Web 路径与批测一致 |
| 1.4 | `_figure_context_from_answer_docs`：短答案单题在 topic 未命中时 **query 对齐回退**（不作用于 listing / 多机型对比）；多机型 **手册名匹配** 补 span | Q4 不误配；Q15 四图稳定 |
| 1.5 | `finalize_inline_images`：`cite_pool` / `anchor_pool` 分离；`figure_pool` + 多机型 rerank 含图 chunk 扩 anchor；多机型跳过 `query_aligned` 删 chunk | Q2 无图；Q15 不回归 |

**风险控**：answer_gate 用已有 `discriminative_terms` / 步骤句重合，**不用**答案加粗 span 列表。

**批测**：`logs/web_path_q1_17/20260612_161319_all.md` — **17/17**（基线 `122334` 为 14/17）。

**轻量去硬编码（阶段 2 前）**：见变更记录；子集 `20260613_004822` **5/5**，全量 `20260613_010606` **15/17**（Q13/Q17 为 LLM 文字方差）。

### 阶段 2：收敛 answer-span 扩池逻辑（中风险）

**目标**：保留 Q15 四图、Q11 三图能力，去掉「答案另起一段」导致的误配图。

| 步骤 | 改动要点 | 验收 |
|------|----------|------|
| 2.1 | 保留 `_answer_primary_listing_body`（截断「此外…」）— 已用于减误配 | Q11 不出现「刮刀片检查」 |
| 2.2 | 列举题 span_keep：span 优先来自 **query 保养条目**（检索正文 `保养内容：`）与 **citation kept** 的交集；答案加粗仅作 **次序提示** | Q11 稳定 3 张残胶图 |
| 2.3 | 多机型周期题（Q15）：维持 **机型 span + query 词「电控」** 过滤，不改为纯 query 漫游 | Q15 稳定 4 张电控板图 |

### 阶段 3：Q17 跨手册列举补全（较高复杂度）

**目标**：答案列举的各部件，在对应手册段落有图则应配上；仍不脱离 citation/query 边界。

| 步骤 | 改动要点 | 验收 |
|------|----------|------|
| 3.1 | 列举题：每个 **query 对齐的保养条目**（非答案加粗）在 **citation pool ∪ 同手册同节邻段** 内取 1 张图 | 智能机：压带轮 / 仿形 / 涂胶轴 / 电机（若有） |
| 3.2 | 跨手册：citation 已引用手册来源（References PDF）限制补图 **source_hints** | 高速自动机：激光出光口图仅来自该手册 |
| 3.3 | （可选）span 在 pool 内未命中时，**受限** content_list bbox 补图：仅 `source_hints` + label 与 **query 条目** 对齐，且答案正文含对应机型/部件 **弱一致** | Q17 ≥3 张且不乱配 |

### 阶段 4：批测、文档与开关

| 步骤 | 改动要点 | 验收 | 状态 |
|------|----------|------|------|
| 4.1 | 全量 `run_web_path_q1_17.py` ≥ **17/17** 或明确记录例外 | 新报告入 `logs/web_path_q1_17/` | ✅ 阶段 1 已达成（`161319`） |
| 4.2 | 更新 `查询配图设计方案.md` §0 三层收窄表述，与本文 **query 辅锚** 一致 | 设计 / 实现文档同步 | 待做 |
| 4.3 | `config/env.example` 增加 query 节锚相关开关（如 `RAG_IMAGE_QUERY_SECTION_ANCHOR=true`） | 可回退单步行为 | ✅ 阶段 1 |

---

## 实施顺序建议

```
阶段 1（Q5/Q9 断链） → 阶段 2（收敛 span） → 阶段 3（Q17） → 阶段 4（批测+文档）
```

每阶段结束：**单独 commit + 跑 `--ids` 回归 + 记录报告路径**，避免与 `1d3b5f4` 混杂难以 bisect。

---

## 明确不做（本方案）

- **不用答案加粗 / span 从 rerank 全池扩图** 作为主路径（仅列举题在严格边界下保留受限 span）。
- **不让 query 单独主导全库配图**（无 citation / 无 answer_gate 的 chunk 不入选）。
- **不恢复** LEGACY `supplement_refs_from_content_lists` 无边界全目录扫图。
- **不改灌库** coalesce / 文内 `[图片]` 格式（除非发现库内缺图，当前判断为无此问题）。

---

## 变更记录

| 日期 | Commit | 说明 |
|------|--------|------|
| 2026-06-12 | `1d3b5f4` | Plan B + citation 扩池 + span 策略基线；全量 14/17；**本方案起点** |
| 2026-06-12 | `4837143` | 阶段 1：query 节锚、figure 扫描 query 回退、多机型 anchor_pool / 跳过 query_aligned 过滤 |
| 2026-06-12 | — | 阶段 1 批测：`20260612_161319_all.md` **17/17**（基线 14/17；修 Q5/Q9/Q17 配图，无 Q1–Q16 回归） |
| 2026-06-13 | `ab4ed96` | 轻量去硬编码：删保养 query 词表→`_pool_has_query_aligned_figure_chunks`；`封边机 in topic`→`topic in machine_names`；多候选才启用 subject_needles |
| 2026-06-13 | — | 去硬编码批测：子集 `20260613_004822` **5/5**；全量 `20260613_010606` **15/17** |
| （待填） | — | 阶段 2：span 收敛 |
| （待填） | — | 阶段 3：Q17 跨手册 |
| （待填） | — | 阶段 4：17/17 批测与文档 |
