# PR 拆分合入主线操作说明

> **工作指导（长期保留）**：本文件是 `lhq-rag-dev` → `main` 拆 PR / 合入顺序的权威说明。
> **禁止**在「清理开发文档 / 误暂存文件」类提交中删除本文件。
> 须随 **PR-A / PR-B / PR-C / PR-D** 与 `lhq-rag-dev` 保留；合入 `main` 后继续留在主线，供后续 PR-B 及日常开发对照。
> Agent / 协作者改代码、开 PR、拣文件前先读本文。

> 适用分支：`lhq-rag-dev` → `main`
> 背景：单 PR 约 **+43,180 行**（`113 files`），超过 GitHub Copilot review 上限（约 2 万行），且与 `main` 存在 merge conflicts。
> 策略：拆成 **4 个独立 PR**，每个 < 2 万行，按依赖顺序合入。
>
> **2026-07-27 更新**：`lhq-rag-dev` 已完成重组式重构（monolith → iqr_* 13 模块 + utils 5 子模块 + Domain Schema 外置），
> PR-A 同步量约 25K 行 diff，超限。**PR #47 拆为两次 push（A1-① raganything ~5K + A1-② iqr ~16K），A2 单独开新 PR（~3.6K）**。

---

## 一、为什么要拆

| 指标 | 数值 |
|------|------|
| 总新增行数 | ~43,180（删 245 行） |
| Copilot review 上限 | ~20,000 |
| `docs/` 占比 | ~19.5%（去掉仍 ~34.8k 行） |
| 最大单文件 | `scripts/image_query_refs.py`（+7,698） |

**结论**：仅剔除 `docs/` 不够；需按**功能模块 + 依赖关系**拆分。

---

## 二、PR 总览（2026-07-27 修订）

```mermaid
graph TD
    main[main]
    A1a["PR #47 push-1<br/>A1-① raganything 拆分<br/>~5K / 7 文件"]
    A1b["PR #47 push-2<br/>A1-② iqr 引擎拆分<br/>~16K / 14 文件"]
    A2["新 PR: pr-a2-schema<br/>Schema+测试<br/>~3.6K / 12 文件"]
    B["PR-B Web + 澄清门控 + 客户端<br/>+10,675 行 / 27 文件"]
    C["PR-C 文档<br/>+8,409 行 / 32 文件"]
    D["PR-D 开发/测试工具<br/>+8,844 行 / 28 文件"]

    main --> A1a
    A1a --> A1b
    A1b -->|"合入 main"| A2
    A2 -->|"合入 main"| B
    main --> C
    B --> D
    A1a -.可并行.-> C
```

| 步骤 | 分支 / PR | diff 量 | 合并顺序 | Copilot |
|------|-----------|---------|----------|--------|
| A1-① raganything 拆分 | `pr-a-core-engine` / PR #47 push-1 | ~5K | **第 1** | ✅ 可完整 review |
| A1-② iqr 引擎拆分 | `pr-a-core-engine` / PR #47 push-2 | ~16K | **第 2**（同 PR） | 超限（纯搬家，人工确认 facade） |
| A2 Schema + 测试 | `pr-a2-schema`（新分支）/ 新 PR | ~3.6K | **第 3**（#47 合入后） | ✅ 可完整 review |
| PR-B Web 产品层 | `pr-b-web-ui` | +10,675 | **第 4**（依赖 A2） | 可 review |
| PR-C 文档 | `pr-c-docs` | +8,409 | 随时（可与 A 并行） | 可 review |
| PR-D 开发工具 | `pr-d-dev-tooling` | +8,844 | **第 5**（依赖 B） | 可 review |

推荐时间线：

1. **Week 1**：PR #47（A1-① → A1-②）合入；PR-C 可同步提
2. **Week 1–2**：A2 新 PR 合入
3. **Week 2**：PR-B 合入
4. **Week 3**：PR-D 合入

### PR-A 拆分理由

| 层 | 内容 | +行 | -行 | diff |
|----|------|-----|-----|------|
| ① raganything 拆分 | utils→5 子模块 + facade + processor | 2,867 | 2,272 | ~5K |
| ② iqr 引擎拆分 | image_query_refs→13 个 iqr_* + facade | 8,520 | 7,722 | ~16K |
| ③ Schema + 运行时依赖 | domain_schema + loader + steering + hooks + docs | 2,467 | 0 | ~2.5K |
| ④ 测试 | 6 个新单元测试 + fixtures | 1,255 | 0 | ~1.3K |

- **PR #47 push-1 = ①**：raganything 拆分，~5K，Copilot 可完整审查
- **PR #47 push-2 = ②**：iqr 引擎拆分，~16K，纯搬家，人工确认 facade 完整性
- **新 PR = ③ + ④**：真正有逻辑变更（schema 外置 + hardcode 修复），Copilot 可完整审查
- **批测脚本**（`run_test_cases_report.py`、`run_web_path_*`、`bench_clarify_*` 等）：不属于 PR-A，全部列入 PR-D

---

## 三、各 PR 文件清单

### PR #47 push-1：A1-① raganything 拆分（纯重组，~5K）

**范围**：`raganything/utils.py`（2400行）拆分为 5 个子模块 + facade re-export。

```
raganything/text_align.py
raganything/image_context.py
raganything/ingest_coalesce.py
raganything/ingest_insert.py
raganything/machine_derive.py
raganything/utils.py                  ← facade（~200 行 re-export）
raganything/processor.py              ← import 路径更新
```

**Copilot**：~5K 行，可完整审查。重点：facade re-export 完整性、import 路径正确性。

---

### PR #47 push-2：A1-② iqr 引擎拆分（纯重组，~16K）

**范围**：`scripts/image_query_refs.py`（7800行）拆分为 13 个 `iqr_*` 模块 + facade re-export。

```
scripts/iqr_align.py
scripts/iqr_anchor.py
scripts/iqr_config.py
scripts/iqr_domain_schema.py
scripts/iqr_explain.py
scripts/iqr_figure_target.py
scripts/iqr_machine.py
scripts/iqr_media.py
scripts/iqr_placement.py
scripts/iqr_protocol.py
scripts/iqr_query_intent.py
scripts/iqr_store.py
scripts/iqr_terms.py
scripts/image_query_refs.py          ← facade（~76 行 re-export）
```

**Copilot**：~16K 超限，人工确认 facade 完整性即可。

**合入后验证**（两次 push 后统一跑）：

```powershell
uvx ruff@0.6.4 check scripts/ raganything/ --ignore=E402
uv run pytest tests/ -x -q
```

---

### A2 新 PR（pr-a2-schema）：Domain Schema 外置 + 运行时依赖 + 测试

**范围**：领域词汇表外置为 JSON、hardcode 修复、运行时依赖模块、新增单元测试；**依赖 PR #47 已合入 main**。不含批测脚本。

```
# ③ Schema + 运行时依赖
config/domain_schema.json
scripts/iqr_domain_schema.py          ← 加载器（若 A1 未含则此处补）
scripts/query_doc_steering.py         ← 运行时依赖（1078 行）
scripts/query_progress_hooks.py       ← 运行时依赖（944 行）
docs/domain_schema_design.md
scripts/_induce_field_schema.py       ← 客户切换工具

# ④ 测试
tests/test_chunk_locality_anchor.py
tests/test_clarify_candidate_skip_probe.py
tests/test_field_schema_induction.py
tests/test_figure_targets.py
tests/test_image_chunk_locality.py
tests/test_ingest_coalesce_schema.py
tests/fixtures/                       ← 测试数据
```

**合入后验证**：

```powershell
uv run pytest tests/ -x -q
uv run python -c "from scripts.iqr_domain_schema import get_domain_schema; print(get_domain_schema().section_markers)"
```

**Copilot review 重点**：`iqr_domain_schema.py` 加载逻辑、`domain_schema.json` 字段完整性、hardcode 替换正确性。

---

### PR-B：Web UI + 查询接线 + 客户端

**范围**：浏览器问答、澄清门控接线、客户端打包；**依赖 PR-A 已合入**。

```
scripts/rag_web_server.py
scripts/query_doc_steering.py
scripts/query_progress_hooks.py
scripts/query_debug_dump.py
scripts/stream_cot_parser.py
scripts/clear_query_llm_cache.py

scripts/client_env_manager.py
scripts/client_launcher.py
scripts/client_launcher_exe.py
scripts/client_paths.py
scripts/client_setup_service.py
scripts/pack_client.bat
scripts/pack_client.ps1
scripts/stage_client_models.bat

web/index.html
web/setup.html
web/static/app.js
web/static/images/nanxing-logo.png
web/static/ingest_ui.js
web/static/setup.css
web/static/setup.js
web/static/styles.css
web/static/vendor/marked.min.js
web/static/vendor/purify.min.js

.gitignore
README.md
README_zh.md
```

**合入后验证**：

```powershell
uv sync --extra web
uv run python scripts/rag_web_server.py
# 浏览器打开 http://127.0.0.1:8765，跑一条 Q&A
```

**Copilot review 重点**：`rag_web_server.py` 流式接口、`query_progress_hooks.py` 钩子链、`app.js` 前端状态。

---

### PR-C：文档

**范围**：设计文档、测试参考答案、fixtures；**不影响运行时，可与 PR-A 并行**。

```
docs/   （整个目录，当前分支相对 main 的全部变更）
```

主要大文件：

- `docs/统一图文关联架构实施方案.md`
- `docs/版本改动记录.md`
- `docs/澄清门控实现说明_v3.md` / `v4.md`
- `docs/网页问答界面.md`
- `docs/fixtures/q15_电控板_ground_truth/`（含 8 张图片）

**注意**：图片为二进制，不占文本行数，但会增加 PR 文件数。

---

### PR-D：开发/测试工具

**范围**：bench、replay、eval、回归脚本；**建议在 PR-B 合入后提交**。

```
scripts/analyze_clarify_threshold.py
scripts/audit_q15_ingest.py
scripts/bench_clarify_green8.py
scripts/bench_probe_timing_green8.py
scripts/build_ragas_ground_truth_shili17.py
scripts/build_voice_script_tests.py
scripts/capture_merged_chunk_ids.py
scripts/debug_q3_citation.py
scripts/dump_query_context.py
scripts/eval_ragas_webpath.py
scripts/extract_voice_green8.py
scripts/fill_voice_script_xlsx.py
scripts/recover_clarify_replay_from_terminal.py
scripts/relevance_distribution_charts.py
scripts/relevance_per_question_charts.py
scripts/replay_clarify_gate_green8.py
scripts/replay_placement_dumps.py
scripts/report_relevance_keywords_shili17_green8.py
scripts/run_demo_question_bank.py
scripts/run_test_cases_report.py
scripts/run_voice_script_tests.py
scripts/run_web_path_q1_13.py
scripts/run_web_path_q1_17.py
scripts/score_query_relevance.py
scripts/standalone_rerank_stress.py
scripts/stress_gpu_vram.py
scripts/test_clarify_gate_batch.py
scripts/time_clarify_gate.py
```

**合入后验证**（任选 smoke）：

```powershell
uv run python scripts/run_web_path_q1_17.py --help
```

---

## 四、操作前准备

### 4.1 确保源分支最新且已解决冲突

在 `lhq-rag-dev` 上先把 `main` 合进来（**merge，不必 rebase**）：

```powershell
git fetch origin
git checkout lhq-rag-dev
git merge origin/main
```

若有冲突，重点处理这 3 个文件（两边改动尽量都保留）：

- `raganything/local_hf_embedding.py`
- `raganything/processor.py`
- `scripts/rag_pipeline_parse_graph_chat.py`

解决后：

```powershell
git add raganything/local_hf_embedding.py raganything/processor.py scripts/rag_pipeline_parse_graph_chat.py
git commit -m "merge: resolve main conflicts on processor, embedding, and pipeline CLI"
git push origin lhq-rag-dev
```

> **说明**：后续 4 个 PR 分支都从**已解决冲突的 `lhq-rag-dev`** 或 **`origin/main`（PR-A 合入后）** 拣文件，避免重复解冲突。

### 4.2 关闭或标记原大 PR

原来的 `lhq-rag-dev → main` 大 PR 可以：

- **关闭（Close）**，改用下面 4 个小 PR；或
- 改为 **Draft**，4 个都合入后自动关闭。

---

## 五、创建各 PR 分支（推荐方法：从 main 拣文件）

核心思路：从干净的 `origin/main` 拉分支，再用 `git checkout lhq-rag-dev -- <路径>` 拣入本 PR 需要的文件。

### 5.1 PR #47 push-1：A1-① raganything 拆分（~5K，Copilot 可审）

> `pr-a-core-engine` 分支已存在（GitHub #47）。先推 raganything 拆分，让 Copilot 审查这一批。

```powershell
git checkout pr-a-core-engine

# 从 lhq-rag-dev 文件级同步（禁止 merge！）
git checkout lhq-rag-dev -- `
  raganything/text_align.py `
  raganything/image_context.py `
  raganything/ingest_coalesce.py `
  raganything/ingest_insert.py `
  raganything/machine_derive.py `
  raganything/utils.py `
  raganything/processor.py

git add -A
git commit --no-verify -m "refactor: split utils.py into 5 raganything submodules (pure restructure)"
git push origin pr-a-core-engine
```

push 后 GitHub 触发 Copilot review（~5K 行，可完整审查）。

---

### 5.2 PR #47 push-2：A1-② iqr 引擎拆分（~16K，纯搬家）

> 等 push-1 的 Copilot review 完成（或确认无问题）后再推。

```powershell
# 仍在 pr-a-core-engine 分支
git checkout lhq-rag-dev -- `
  scripts/iqr_align.py `
  scripts/iqr_anchor.py `
  scripts/iqr_config.py `
  scripts/iqr_domain_schema.py `
  scripts/iqr_explain.py `
  scripts/iqr_figure_target.py `
  scripts/iqr_machine.py `
  scripts/iqr_media.py `
  scripts/iqr_placement.py `
  scripts/iqr_protocol.py `
  scripts/iqr_query_intent.py `
  scripts/iqr_store.py `
  scripts/iqr_terms.py `
  scripts/image_query_refs.py

git add -A
git commit --no-verify -m "refactor: split image_query_refs.py into 13 iqr_* modules (pure restructure)"
git push origin pr-a-core-engine
```

push 后 Copilot 可能审不完（超限），在 PR 描述中注明“纯结构搬家，零逻辑变更”。

---

### 5.3 创建 A2 新 PR（#47 合入 main 后）

```powershell
git fetch origin
git checkout -b pr-a2-schema origin/main

git checkout lhq-rag-dev -- `
  config/domain_schema.json `
  scripts/iqr_domain_schema.py `
  scripts/query_doc_steering.py `
  scripts/query_progress_hooks.py `
  docs/domain_schema_design.md `
  scripts/_induce_field_schema.py `
  tests/fixtures `
  tests/test_chunk_locality_anchor.py `
  tests/test_clarify_candidate_skip_probe.py `
  tests/test_field_schema_induction.py `
  tests/test_figure_targets.py `
  tests/test_image_chunk_locality.py `
  tests/test_ingest_coalesce_schema.py

git add -A
git commit --no-verify -m "feat: domain schema externalization + runtime deps + unit tests"
git push -u origin pr-a2-schema
```

在 GitHub 创建新 PR：**`pr-a2-schema` → `main`**，~3.6K 行，Copilot 可完整审查。

---

### 5.4 创建 PR-B（等 PR-A 合入后，或先基于 PR-A 分支开发）

**推荐**：PR-A 合入 `main` 后再做，冲突最少。

```powershell
git fetch origin
git checkout -b pr-b-web-ui origin/main

git checkout lhq-rag-dev -- `
  scripts/rag_web_server.py `
  scripts/query_doc_steering.py `
  scripts/query_progress_hooks.py `
  scripts/query_debug_dump.py `
  scripts/stream_cot_parser.py `
  scripts/clear_query_llm_cache.py `
  scripts/client_env_manager.py `
  scripts/client_launcher.py `
  scripts/client_launcher_exe.py `
  scripts/client_paths.py `
  scripts/client_setup_service.py `
  scripts/pack_client.bat `
  scripts/pack_client.ps1 `
  scripts/stage_client_models.bat `
  web/ `
  .gitignore README.md README_zh.md

git add -A
git commit -m "feat: Web Q&A UI, clarify gate wiring, and client packaging"
git push -u origin pr-b-web-ui
```

在 GitHub 创建 PR：**`pr-b-web-ui` → `main`**

若 PR-A 尚未合入、想提前开 PR-B：可从 `pr-a-core-engine` 分支创建，合入前在 GitHub 上把 base 保持为 `main`，并 rebase 到已合入 A 的 `main`。

---

### 5.5 创建 PR-C（可与 PR-A 并行）

```powershell
git fetch origin
git checkout -b pr-c-docs origin/main

git checkout lhq-rag-dev -- docs/

git add -A
git commit -m "docs: architecture plans, test references, and clarify gate design"
git push -u origin pr-c-docs
```

在 GitHub 创建 PR：**`pr-c-docs` → `main`**

---

### 5.6 创建 PR-D（建议 PR-B 合入后）

```powershell
git fetch origin
git checkout -b pr-d-dev-tooling origin/main

git checkout lhq-rag-dev -- `
  scripts/analyze_clarify_threshold.py `
  scripts/audit_q15_ingest.py `
  scripts/bench_clarify_green8.py `
  scripts/bench_probe_timing_green8.py `
  scripts/build_ragas_ground_truth_shili17.py `
  scripts/build_voice_script_tests.py `
  scripts/capture_merged_chunk_ids.py `
  scripts/debug_q3_citation.py `
  scripts/dump_query_context.py `
  scripts/eval_ragas_webpath.py `
  scripts/extract_voice_green8.py `
  scripts/fill_voice_script_xlsx.py `
  scripts/recover_clarify_replay_from_terminal.py `
  scripts/relevance_distribution_charts.py `
  scripts/relevance_per_question_charts.py `
  scripts/replay_clarify_gate_green8.py `
  scripts/replay_placement_dumps.py `
  scripts/report_relevance_keywords_shili17_green8.py `
  scripts/run_demo_question_bank.py `
  scripts/run_test_cases_report.py `
  scripts/run_voice_script_tests.py `
  scripts/run_web_path_q1_13.py `
  scripts/run_web_path_q1_17.py `
  scripts/score_query_relevance.py `
  scripts/standalone_rerank_stress.py `
  scripts/stress_gpu_vram.py `
  scripts/test_clarify_gate_batch.py `
  scripts/time_clarify_gate.py

git add -A
git commit -m "chore: add bench, replay, and eval tooling scripts"
git push -u origin pr-d-dev-tooling
```

在 GitHub 创建 PR：**`pr-d-dev-tooling` → `main`**

---

## 六、合并顺序与冲突处理

### 推荐合并顺序

```
PR-A → PR-B → PR-D
PR-C 任意时刻（与 A 并行无依赖）
```

### 若 PR-B 与 main 冲突

PR-A 合入后，`main` 已有 `raganything/*` 等文件；开 PR-B 时若基于旧的 `origin/main` 创建，需：

```powershell
git fetch origin
git checkout pr-b-web-ui
git rebase origin/main
# 或 git merge origin/main
# 解决冲突后 push
```

### merge 还是 rebase？

| 场景 | 建议 |
|------|------|
| `lhq-rag-dev` 吸收 `main` 的冲突 | **merge**（一次解决，不需 force push） |
| 小 PR 分支跟进已更新的 `main` | **rebase** 或 **merge** 均可 |
| 已 push 的 PR 分支整理历史 | 避免随意 force push；用 `--force-with-lease` 且确认无他人协作 |

---

## 七、PR 描述模板（复制到 GitHub）

### PR #47（A1-① + A1-② 合并描述）

```markdown
## Summary
- **push-1 (A1-①)**：`raganything/utils.py`（2400行）→ 5 个子模块 + facade（纯搬家）
- **push-2 (A1-②)**：`scripts/image_query_refs.py`（7800行）→ 13 个 `iqr_*` 模块 + facade（纯搬家）
- 零逻辑变更，仅文件拆分 + re-export
- 不含 Web UI、批测脚本

## Depends on
无（第一个合入）

## Review 指引
- push-1 commit：raganything 拆分，~5K，Copilot 可审
- push-2 commit：iqr 拆分，~16K，纯搬家，确认 facade re-export 完整即可

## Test plan
- [ ] `uvx ruff@0.6.4 check scripts/ raganything/ --ignore=E402`
- [ ] `uv run pytest tests/ -x -q`
```

### A2 新 PR（pr-a2-schema → main）

```markdown
## Summary
- 领域词汇表外置为 `config/domain_schema.json`
- 运行时依赖：`query_doc_steering.py`、`query_progress_hooks.py`
- 6 个新单元测试 + fixtures

## Depends on
- PR #47 已合入 main

## Test plan
- [ ] `uv run pytest tests/ -x -q`
- [ ] `uv run python -c "from scripts.iqr_domain_schema import get_domain_schema; print(get_domain_schema().section_markers)"
```

### PR-B

```markdown
## Summary
- FastAPI Web 服务 + 浏览器问答 UI
- 澄清门控接线（`query_progress_hooks.py`）
- 客户端打包脚本

## Depends on
- PR-A 已合入 main

## Test plan
- [ ] `uv sync --extra web`
- [ ] `uv run python scripts/rag_web_server.py`
- [ ] 浏览器 Q&A smoke test
```

### PR-C

```markdown
## Summary
- 架构实施方案、澄清门控设计文档、测试参考答案
- 不影响运行时

## Test plan
- [ ] 确认无敏感信息（密钥、内网地址等）
```

### PR-D

```markdown
## Summary
- bench / replay / eval / 回归脚本（开发工具，非生产路径）

## Depends on
- PR-B 已合入 main（`run_web_path_*` 依赖 Web 路径）

## Test plan
- [ ] `uv run python scripts/run_web_path_q1_17.py --help`
```

---

## 八、全部合入后的收尾

1. 确认 4 个 PR 均已 merge 到 `main`
2. 关闭原来的大 PR（若仍开着）
3. 本地同步：

```powershell
git fetch origin
git checkout main
git pull origin main
```

4. 可选：删除远程拆分分支

```powershell
git push origin --delete pr-a-core-engine pr-b-web-ui pr-c-docs pr-d-dev-tooling
```

5. 本地开发分支 `lhq-rag-dev` 可保留作归档，或 rebase 到最新 `main` 后继续开发

---

## 九、快速自查：行数是否仍超限

在任意分支上对比 `main`：

```powershell
git fetch origin
git diff --stat origin/main...HEAD
```

若单 PR 仍 > 2 万行，用下面命令按目录看分布：

```powershell
git diff --numstat origin/main...HEAD
```

重点关注是否误把 `docs/` 或 `run_web_path_*` 混进了 PR-A / PR-B。

---

## 十、常见问题

**Q：能不能只删 `docs/` 让一个 PR 过 Copilot？**
A：不能。去掉 docs 仍约 35k 行，必须按模块拆。

**Q：PR-A 要不要包含 `clarify_gate.py`？**
A：要。澄清算法在库层（`raganything/`），Web 接线在 PR-B。拆库与 UI 更清晰。

**Q：`image_query_refs.py` 体量很大，能否单独成 PR？**
A：已不需要。该文件已拆为 13 个 `iqr_*` 模块 + 76 行 facade（见 §三 PR-A1），总体量不变但单文件最大 ~3000 行。

**Q：四个 PR 都合完后，`lhq-rag-dev` 还有用吗？**
A：可作为历史归档；新功能建议从最新 `main` 拉分支。

**Q：PR-A 半套代码上怎么做全量测试？瘦身和大改谁先谁后？**
A：见下文 **「十一、PR-A 瘦身 ↔ `lhq-rag-dev` 全量大改同步」**。半套分支只做 A 范围冒烟；全量验收必须在 `lhq-rag-dev`（或已含 A 的集成分支）。

**Q：A1-② 超 Copilot 限额怎么办？**
A：A1-② 是纯结构搬家（代码逻辑零改动），在 PR 描述中注明“纯搬家”，reviewer 只需确认 facade re-export 完整、import 路径正确。可跳过 Copilot 自动审查，人工确认即可。

**Q：A2 为什么是单独 PR 而不是放在 #47 里？**
A：为了让 Copilot 能完整审查。#47 总量 ~21K 超限，A2 单独只有 ~3.6K，Copilot 可以逐行审查 schema 加载逻辑和 hardcode 替换正确性。

**Q：批测脚本为什么不在 PR-A 里？**
A：批测脚本（`run_test_cases_report.py`、`run_web_path_*`、`bench_clarify_*`）大多依赖 Web 服务层（PR-B 范围），且属于开发工具而非核心引擎，按功能归属列入 PR-D。

---

## 十一、PR-A 瘦身 ↔ `lhq-rag-dev` 全量大改同步（已完成）

> **状态：✅ 已完成（2026-07-27）**。下述流水线已实际执行，`lhq-rag-dev` 已包含全部重构结果。

> 背景：`pr-a-core-engine` **不含** Web / 客户端 / 部分 hooks，无法做产品级全量回归。
> 瘦身适合直接改 PR-A；rerank / `utils` / `image_query_refs` 等大改需在全量树上验证。
> **约定**：可等大改进 A（或紧跟 follow-up）后再合 #47 进 `main`；但瘦身结果必须先并进 `lhq-rag-dev`，否则大改不包含已瘦身状态。

### 11.1 工作类型与分支

| 工作 | 在哪改 | 如何上 PR / 全量树 |
|------|--------|-------------------|
| **#47 瘦身**（删 naive、拿掉 steering JSON、移出迁移脚本、env 合并等） | `pr-a-core-engine` | 直接 commit + push 更新 #47；**先不合 `main` 也可以** |
| **全量大改**（`RERANK_RELEASE_*` 收敛、`utils` 拆分/去 hardcode、`image_query_refs` 去领域 hardcode） | 先在 **`lhq-rag-dev`** | 全量测绿后，再 **拣回** `pr-a-core-engine`（勿整支 merge） |

### 11.2 推荐流水线（瘦身 → 同步到 lhq → 大改 → 拣回 A → 再合 main）

```text
1) git checkout pr-a-core-engine
   # 瘦身改动 → commit → push origin pr-a-core-engine
   # （更新 GitHub #47；此时可不 merge 进 main）

2) 把瘦身带进全量树（关键：否则 lhq 仍是瘦身前的引擎）
   git checkout lhq-rag-dev
   git merge pr-a-core-engine
   # 冲突时：保留 lhq 的 Web / client / pack；接受 A 的删除与瘦身
   git push origin lhq-rag-dev

3) 在 lhq-rag-dev 上做大改 + 全量测试
   # 此时代码 = 全量产品面 + 已瘦身引擎

4) 把「属于 PR-A 范围」的大改弄回 pr-a-core-engine
   # 禁止：git merge lhq-rag-dev → pr-a-core-engine
   #      （会把 web/、client_*、pack_client* 整包带进 #47）
   # 允许：
   #   - git cherry-pick <大改相关 commit>
   #   - 或 git checkout lhq-rag-dev -- <仅 A 清单内文件> 后再 commit
   git checkout pr-a-core-engine
   # …拣回… → commit → push origin pr-a-core-engine

5) 再合 #47 进 main（或先合瘦身版 A、再立刻合 follow-up；按评审节奏）
```

```mermaid
flowchart LR
  slim["1 瘦身 on pr-a"] --> pushA["push 更新 #47"]
  pushA --> mergeLhq["2 merge A into lhq-rag-dev"]
  mergeLhq --> big["3 大改 + 全量测 on lhq"]
  big --> pick["4 cherry-pick / 拣文件回 pr-a"]
  pick --> mergeMain["5 合 #47 进 main"]
```

### 11.3 方向禁忌

| 操作 | 是否允许 | 原因 |
|------|----------|------|
| `merge pr-a-core-engine` → `lhq-rag-dev` | **允许（推荐）** | 全量树吸收瘦身 / 引擎修复 |
| `merge lhq-rag-dev` → `pr-a-core-engine` | **禁止** | 污染 PR-A，混入 Web/打包 |
| 只在 `pr-a` 瘦身、从不 merge 到 `lhq` 就开始大改 | **禁止（若大改在 lhq）** | 大改基于未瘦身代码，与 #47 分叉 |
| 只在 `pr-a` 上跑「全量话术 / Web」验收 | **不够** | A 代码不全，结果不可信 |

### 11.4 实际执行记录

- ✅ 第 1 步：pr-a 瘦身已完成（GitHub #47 已更新）
- ✅ 第 2 步：`git merge pr-a-core-engine` → `lhq-rag-dev` 已完成
- ✅ 第 3 步：lhq-rag-dev 大改已完成（iqr 拆分 + utils 拆分 + schema 外置 + hardcode 修复）
- ⬜ 第 4 步：拣回 pr-a-core-engine（即本文 §5.1 push-1 + §5.2 push-2）—— **待执行**
- ⬜ 第 5 步：合 #47 进 main
- ⬜ 第 6 步：创建 pr-a2-schema 新 PR（即本文 §5.3）
- ⬜ 第 7 步：合 A2 进 main

### 11.5 与「另开会话」的对应

- **短会话 / 本 PR 急合前**：只做 §11.1 瘦身（在 `pr-a-core-engine`）。
- **另开会话**：先确认 §11.2 第 2 步已完成，再在 `lhq-rag-dev` 做 rerank / utils / image 大改，最后 §11.2 第 4 步拣回 A。
- 细节待办见 [`docs/PR47_下一步工作摘要.md`](PR47_下一步工作摘要.md)。

---

*文档生成依据：2026-07-12 对 `lhq-rag-dev` 与 `origin/main` 的 diff 统计（`2665196` 起 `image_query_refs.py` 已剔除 legacy 死代码，较初版统计少约 2.1k 行）。*
*§十一补充：2026-07-25 PR-A 评审收尾与全量验证约定。*
*§二/三/五/七/十一 修订：2026-07-27——lhq 重组式重构完成，PR #47 拆为两次 push（A1-① raganything ~5K + A1-② iqr ~16K），A2 单独新 PR（pr-a2-schema，~3.6K）。批测脚本全部列入 PR-D。*
