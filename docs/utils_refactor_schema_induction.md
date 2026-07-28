# utils.py 拆分与去 hardcode 重构方案（schema 归纳版）

> 分支：lhq-rag-dev ｜ 状态：待实施 ｜ 关联：PR-A(#47) 评审意见"utils.py 过大、领域词硬编码"
>
> 本方案取代早期"字段名 env 可配"的思路：把汉字字面量搬进环境变量并没有解开耦合，
> 只是换了存放位置。正解是让算法从文档自身结构归纳出字段角色，代码中不保留任何业务汉字常量。

## 1. 背景与问题

`raganything/utils.py` 现状 2367 行，混装六类职责，且灌库段落聚合（coalesce）逻辑
硬编码了维保手册的字段名：

```python
_MAINTENANCE_FIELD_PREFIXES = ("保养周期：", "保养内容：", "保养步骤：")   # L408
...
if stripped.startswith("保养") and "：" in stripped:                      # L295
```

两个问题：

1. **体量与职责**：单文件承载通用工具、词项对齐、图文配对、段落聚合、表格切分、
   灌库入口六类职责，评审与维护困难。
2. **领域硬编码**：coalesce 的记录边界/闭合判定绑定了"保养×3"三个具体字段名。
   换一个领域的手册（哪怕字段结构完全相同）逻辑就失效。

### 1.1 数据证据：硬编码现在就在漏判

对 `data/pipeline_parse` 7 份手册 content_list 与 `data/rag_storage/kv_store_text_chunks.json`
全部 1234 个 chunk 的扫描结果（分析脚本：`scripts/_induce_field_schema.py`、
`scripts/_induce_run_signature.py`，输出：`logs/_tmp_schema_induce.txt`、`logs/_tmp_run_sig.txt`）：

| 手册 | 字段行形态 | 现 hardcode 覆盖情况 |
|---|---|---|
| 高速智能封边机 | 保养周期/保养内容/**保养步骤** ×61 组 | ✅ 完整覆盖 |
| 加工中心 | 保养周期/保养内容/**操作步骤** ×39 组 | ❌ `操作步骤：` 39 处不被识别 |
| 数控六面钻 | 保养周期/保养内容/**操作步骤** ×16 组 | ❌ `操作步骤：` 16 处不被识别 |
| 双端/自动/高速自动封边机、PC报警 | 无记录式字段行 | —（不受影响） |

后果：加工中心/数控六面钻的 `操作步骤：` 行不参与 step-closed 判定与过合并切分，
这两份手册的小节聚合质量低于高速智能封边机——**去 hardcode 同时是修 bug**。

## 2. 重构总体思路：两阶段两 commit

| 阶段 | 内容 | 行为变化 | 验收 |
|---|---|---|---|
| 阶段一 | 纯搬家：utils.py 拆 4 个模块 + 门面 re-export | **零** | 7 份手册 coalesce 输出逐字节 diff 为空 + 209 项跟踪测试全绿 |
| 阶段二 | schema 归纳重写，删除全部领域汉字常量 | 仅限白名单差异 | 差异白名单核对 + 新增归纳测试 + q1-17 回归 |

两阶段分别独立 commit，出问题可单独回滚阶段二而不影响拆分。

## 3. 阶段一：模块拆分（纯搬家）

### 3.1 目标结构

```
raganything/
├── utils.py            # A 类通用工具 + 完整 re-export 门面（对外 API 不变）
├── text_align.py       # B 类：词项对齐原语
├── image_context.py    # C 类：图片-文本配对（MinerU bbox 布局）
├── ingest_coalesce.py  # D+E 类：灌库段落聚合 + 表格感知切分
└── ingest_insert.py    # F 类：doc-scoped 灌库入口
```

依赖方向单向无环：`text_align ← image_context ← ingest_coalesce ← ingest_insert`，
`utils.py` 仅保留 A 类实现并 re-export 其余四模块全部公有名与被外部引用的私有名。

### 3.2 职责归属（按现 utils.py 区块）

| 新模块 | 迁入内容（代表符号） |
|---|---|
| `text_align.py` | `discriminative_terms`、`substantive_bigrams`、`text_term_alignment_symmetric`、`short_label_bag_aligns`、`image_label_for_item`、`image_label_text` 及其私有辅助 |
| `image_context.py` | `separate_content`、`anchor_context_for_image`、`plan_text_image_assignments`、`best_image_for_text_item`、`context_text_for_image`、`resolve_image_caption`、`resolve_image_footnote`、`_looks_like_section_heading`、`_text_image_layout_distance`（含 500/520 页宽常量，阶段二处理） |
| `ingest_coalesce.py` | `coalesce_text_image_segments` 十步管线全部、`coalesce_parts_for_embedding_ingest`、`compute_table_aware_ingest_segments`、`_TABLE_INGEST_MARKER`、`_INGEST_SEGMENT_DELIMITER`、全部 `_maintenance_*` / `_split_*` / `_collect_*` 辅助、`build_image_ref_block`、`flatten_image_refs_for_skip_multimodal` |
| `ingest_insert.py` | `insert_text_content`、`insert_text_content_with_multimodal_content`、`compute_ingest_chunk_id`、`get_processor_for_type` |
| `utils.py`（留守） | 编码/哈希/路径等 A 类通用函数 + `from .xxx import *` 门面 |

### 3.3 兼容性约束

- 6 个外部调用方（processor.py、query.py、table_matrix.py、modalprocessors.py、
  raganything.py、scripts/image_query_refs.py、tests/test_core_modules.py）**零改动**；
  全部经 `raganything.utils` 门面继续导入。
- 被外部引用的私有名必须保留 re-export：`table_matrix.py` 引用 `_TABLE_INGEST_MARKER`。
- 搬家过程只允许：移动代码、增加 import/`__all__`、模块 docstring。
  不允许改函数体、改名、改默认值、改常量。

### 3.4 阶段一验收（金标准）

1. 拆分前用 `scripts/_sim_coalesce_full.py` 对 7 份手册 content_list 跑
   coalesce 全管线，dump 输出为金标准文件。
2. 拆分后重跑，**逐字节 diff 必须为空**。
3. `git ls-files tests` 跟踪测试 209 项全绿。
4. `uv tool run ruff@0.6.4 check .` 通过。
5. import smoke：`python -c "import raganything.utils"` 及 6 个调用方模块。

## 4. 阶段二：schema 归纳重写（去 hardcode 核心）

### 4.1 关键洞察：字面量是结构角色的代理

三个汉字前缀在代码中真正承担的是**记录结构中的位置角色**：

| hardcode | 结构角色 | 判定用途 |
|---|---|---|
| `保养周期：` | **initiator**（记录起始键） | 再次出现 ⇒ 上一条记录结束，切开过合并段 |
| `保养内容：` | **metadata**（记录中间键） | 与 initiator 一起构成"元数据行"，向后归并 |
| `保养步骤：` | **closer**（记录收尾过程键） | 出现 ⇒ 本小节 step-closed，后续字段行开新节 |

角色由**键在记录序列中的位置**定义，与文字内容无关。因此可以从文档自身归纳。

### 4.2 归纳算法（run-signature induction）

在 coalesce 入口对输入 segments 做一次预扫描：

```
1. 字段行识别：首行匹配通用正则 ^([^\s：:，。、,.]{2,8})：  （任何"短键：值"行）
2. run 切分：连续字段行构成一个 run（图片引用段不打断，普通文本段打断），
   得到键序列签名，如 (保养周期, 保养内容, 操作步骤)
3. schema 判定：出现次数 >= 3 且含 >= 2 个不同键的签名为候选；
   取支配签名家族（与最高频签名同 initiator 的候选集合）
4. 角色分配：家族签名首键 = initiator，最高频签名末键 = closer，
   其余键 = metadata
5. 归纳失败（无支配家族）⇒ schema = None，字段合并逻辑整体短路为 no-op
```

产物为不可变数据类：

```python
@dataclass(frozen=True)
class DocFieldSchema:
    initiator: str          # 记录起始键，如 "保养周期"
    closer: str             # 记录收尾键，如 "保养步骤" / "操作步骤"
    metadata_keys: frozenset[str]   # 其余记录键

    def field_key(self, line: str) -> str | None: ...   # 行首键 ∈ schema 才算字段行
    def is_metadata(self, key: str) -> bool: ...
```

schema 在 `coalesce_text_image_segments` / `coalesce_parts_for_embedding_ingest`
入口内部归纳并向下传参，**公共 API 签名不变**。

### 4.3 数据验证结果（7 份手册全部正确）

`scripts/_induce_run_signature.py` 实测：

| 手册 | 支配签名 | 归纳结果 |
|---|---|---|
| 高速智能封边机 | `保养周期>保养内容>保养步骤` ×61 | initiator=保养周期, closer=保养步骤（与现 hardcode 等价） |
| 加工中心 | `保养周期>保养内容>操作步骤` ×39 | closer=**操作步骤**（自动修复漏判） |
| 数控六面钻 | 同上 ×15 | closer=**操作步骤**（自动修复漏判） |
| 其余 4 份 | 无重复签名 | schema=None，行为与现状一致 |

两个鲁棒性关键点（真实数据中的坑，算法已覆盖）：

1. **数控六面钻的 `操作步骤：` 值为整段散文**（无编号子步骤）。若按"后随编号行"
   等内容特征判定过程角色会漏掉它；按记录内位置判定则天然正确。
2. **加工中心存在干扰签名 `第一步>第二步` ×3**。支配家族按 initiator 过滤后
   自动排除，不污染 schema。

### 4.4 函数级重写映射

| 现函数（utils.py 行号） | 重写后（ingest_coalesce.py） |
|---|---|
| `_maintenance_field_prefix` (L447) | `schema.field_key(line)`；schema=None ⇒ None |
| `_is_orphan_maintenance_field_segment` (L455) | 段内全部行均为 schema 字段行 |
| `_is_orphan_maintenance_metadata_only_segment` (L468) | 全部行键 ∈ metadata_keys ∪ {initiator} |
| `_maintenance_section_step_closed` (L488) | closer 键行出现在段内 |
| `_section_heading_needs_field_merge` (L494) | closer 未出现，且 initiator+metadata 未齐 |
| `_segment_has_maintenance_cycle` (L508) | initiator 键行出现在段内 |
| `_split_overmerged_maintenance_segment` (L527) 两条切分规则 | ① 再见 initiator 且当前节已含 initiator ⇒ 切；② 见 metadata 键且当前节已 closed ⇒ 切 |
| `_follows_maintenance_subsection` (L602) | 下一段首行键 ∈ metadata ∪ {initiator} |
| `_has_backward_procedure_for_heading` (L613) | 回看窗口内存在 closer 段 |
| `_collect_maintenance_section_parts` (L633) 两条终止规则 | 同 split 的两条规则参数化 |
| `_looks_like_section_heading` 的 `startswith("保养")` (L295，image_context.py) | 通用"`键：`行不作图注"判断（无需 schema，任何 kv 行都不是图注） |

命名同步去领域化：`_maintenance_*` → `_record_field_*` / `_field_schema_*`。
删除 `_MAINTENANCE_FIELD_PREFIXES` 常量。重写完成后，
`raganything/` 生产代码中不再存在任何业务领域汉字常量
（`[图片]`/`图片路径：`/`[Table]` 等自产自销协议标记除外，见 §4.6）。

### 4.5 阶段二验收：可预言的差异白名单

对 7 份手册重跑 coalesce，逐份对比金标准（实测结果，commit 2 已确认）：

| 手册 | 实测差异 | 处置 |
|---|---|---|
| 高速智能封边机 | schema 归纳部分**逐字节不变**；image_context 页宽改动引起 **3 处同节兄弟图片段换位**（3.5.1/3.9.1/3.13.3，段数 128=128 不变、正文无增删，仅图片绑定的`关联正文`与兄弟段先后顺序变化，见 §4.6） | 审阅后接受 |
| 双端/自动/高速自动封边机 | **逐字节不变**（schema=None 路径） | — |
| PC报警 | 仅 1 行：封面图`脚注：南兴装备股票代码：002757`被 kv-line 泛化判为结构行、不再作脚注（本就不是图注） | 接受 |
| 加工中心、数控六面钻 | **仅**"操作步骤参与 step-closed/切分"引起的段落重组（126→117、76→72） | 逐处人工审阅 |

其余验收：

1. 新增 `tests/test_field_schema_induction.py`：
   - 7 份手册真实签名的归纳用例（3 份出 schema、4 份拒绝）
   - 干扰签名过滤（第一步/第二步）
   - 散文型 closer（数控六面钻形态）
   - 退化输入（空、全图片、单键重复）
2. 209 项跟踪测试全绿（涉及 hardcode 行为的既有用例按白名单更新断言）。
3. `scripts/run_web_path_q1_17.py` 端到端回归，q7/q15 等聚合敏感用例人工核对。
4. ruff@0.6.4 通过。

### 4.6 明确不改的部分（边界）

- **解析层**：MinerU content_list 产物与解析流程不动。
- **序列化协议**：`[图片]`、`图片路径：`、`页码：`、`图注：`、`脚注：`、`关联正文：`、
  `[Table]`、`<<<RAG_SEG_BOUNDARY>>>` 是系统自产自销的格式约定（KB 已有 442 处依赖），
  不属于领域 hardcode，保持字面值并集中于 ingest_coalesce.py 常量区。
- **落库接口**：`insert_text_content*`、chunk id 计算方式、向量库/图库写入不动。
- **查询链路**：query.py / image_query_refs.py / rerank 不经过 coalesce，不受影响。
- **阈值魔数不 env 化**：`_COALESCE_*` 窗口/长度阈值保持代码常量（避免制造无人调优的开关）。
- 例外：`_text_image_layout_distance` 的 500/520 页宽魔数改为按当页 bbox 推算
  （`max(x1)` 归一化比例），属阶段二内的小项，同受差异白名单约束。
  **实测结果（已审阅接受）**：7 份中仅高速智能封边机受影响——3 个"一节多图"
  小节（3.5.1 p17 / 3.9.1 p23 / 3.13.3 p31）的兄弟图片段发生相对换位，且每张图
  绑定的`关联正文`随之改变（改后首图多绑到`保养周期`首字段）。段数与正文均不变，
  兄弟段换位对检索中性（同标题同正文，查询命中该节会同时召回各图）。
  **接受判据：只要图片的`关联正文`锚点合理即可**；逐段对照见
  `logs/_tmp_reorder_review.txt`（dump 脚本 `scripts/_tmp_reorder_review.py`）。
  **后续测试若发现高速智能封边机上述 3 节图文召回异常，优先怀疑此处。**

### 4.7 环境变量方案作废声明

早期讨论的 `RAG_INGEST_FIELD_CYCLE/SUMMARY/PROCEDURE` 三个环境变量**不实施**：
把字段名从代码常量搬到 env 常量没有消除耦合，运维仍需为每个新文档域改配置。
schema 归纳方案下无需任何字段名配置。

## 5. 灌库影响与重灌要求

- 本次改动全部位于**灌库时**的段落聚合环节；已灌 KB（1234 chunk、向量、图谱）不受
  代码合入影响。
- 阶段二合入后，**加工中心、数控六面钻两份手册需重灌**方能吃到修复
  （聚合结果变 ⇒ chunk 内容变 ⇒ chunk id 变）。其余 5 份输出不变，无需重灌。
- 重灌操作沿用 `scripts/batch_ingest_content_lists_local_hf.py` 现有流程，
  重灌前先删除两份手册的旧 doc（`scripts/list_ingested_docs.py` 查 doc id）。

## 6. 提交与拣回

1. commit 1（阶段一）：`refactor: split utils.py into text_align/image_context/ingest_coalesce/ingest_insert with re-export facade`
2. commit 2（阶段二）：`refactor: replace maintenance field hardcode with per-doc field schema induction`
3. 两 commit 均在 lhq-rag-dev 完成并测绿后，按既有流程以文件拣回方式同步到
   pr-a-core-engine（`git checkout lhq-rag-dev -- <文件>`），**不 merge 分支**。
4. push 时机等用户指令。

## 7. 风险与回滚

| 风险 | 缓解 |
|---|---|
| 归纳阈值（≥3 次、≥2 键）对未来小文档过严/过松 | 阈值集中为 `_SCHEMA_MIN_REPEAT`/`_SCHEMA_MIN_KEYS` 模块常量，测试覆盖边界值 |
| 未来文档出现多 schema 混排（一份文档两种记录型） | 当前取支配家族，其余键降级为普通文本；测试固化该行为，出现真实需求再扩展 |
| 拆分引入隐性循环 import | 依赖方向单向（§3.1），阶段一 import smoke 验证 |
| 阶段二差异超出白名单 | 金标准 diff 强制逐处审阅；仅回滚 commit 2 即可恢复 |
| 两份手册重灌遗漏导致新旧 chunk 并存 | §5 重灌步骤写入本方案，执行时按 doc id 先删后灌并复核 chunk 计数 |
| 高速智能封边机 3 节兄弟图片段换位/`关联正文`改绑（§4.6）后续出现图文召回异常 | 已记录为已知接受差异；若 q1-17 回归或线上发现 3.5.1/3.9.1/3.13.3 召回异常，可单独回退 image_context 页宽改动（保留 kv-line 泛化）即恢复字节一致 |
