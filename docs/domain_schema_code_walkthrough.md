# 领域 Schema 代码级详解：每个词是如何在代码中生效的

> 配套文档：[`domain_schema_design.md`](./domain_schema_design.md)（设计/方案层）。
> 本文是**代码级**详解：逐字段给出真实代码块、行号、数据流与具体示例。

---

## 0. 总览：三层结构

```
┌─────────────────────────────────────────────┐
│ ① 配置层  config/domain_schema.json          │  唯一权威词表（11 个字段）
└──────────────────┬──────────────────────────┘
                   │ json.load + lru_cache + env 覆盖 + fallback
                   ▼
┌─────────────────────────────────────────────┐
│ ② 加载层  scripts/iqr_domain_schema.py       │  DomainSchema 单例 `schema`
└──────────────────┬──────────────────────────┘
                   │ from iqr_domain_schema import schema as _domain_schema
                   ▼
┌─────────────────────────────────────────────┐
│ ③ 消费层  iqr_terms / iqr_align / iqr_anchor │  查询解析、图匹配、锚定
│           iqr_figure_target / iqr_query_intent / iqr_protocol
└─────────────────────────────────────────────┘
```

代码**只引用** `schema.xxx`，不写死任何领域词。换领域 = 换 JSON，代码零改动。

---

## 1. 加载层：JSON 如何变成 `schema` 对象

文件：[`scripts/iqr_domain_schema.py`](../scripts/iqr_domain_schema.py)

### 1.1 加载入口（优先级：env > 默认路径 > 内置 fallback）

```python
# iqr_domain_schema.py:141-155
@lru_cache(maxsize=1)                      # 进程级单例，只读一次
def _load_schema() -> DomainSchema:
    path_str = os.environ.get("RAG_DOMAIN_SCHEMA", "")     # ① 环境变量覆盖
    path = Path(path_str) if path_str else _DEFAULT_PATH   # ② config/domain_schema.json
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return DomainSchema(data)
        except (json.JSONDecodeError, OSError):
            pass                                            # 解析失败 → fallback
    return DomainSchema(_FALLBACK)                          # ③ 内置默认（= 旧 hardcode）

schema: DomainSchema = _load_schema()      # 模块级单例，供全项目 import
```

### 1.2 构造：预建 set/tuple 以 O(1) 查询

```python
# iqr_domain_schema.py:54-64
def __init__(self, data: dict[str, Any]) -> None:
    self._data = data
    self._section_markers_set   = set(data.get("section_markers") or [])
    self._field_keys_set        = set(data.get("structural_field_keys") or [])
    self._action_prefixes_tuple = tuple(data.get("action_prefixes") or [])
    self._footnote_labels_set   = set(data.get("footnote_labels") or [])
    self._connectors_set        = set(data.get("paragraph_connectors") or [])
    self._truncate_markers      = list(data.get("filename_truncate_markers") or [])
    self._special_patterns      = list(data.get("special_query_patterns") or [])
    self._machine_class_suffixes= list(data.get("machine_class_suffixes") or [])
```

### 1.3 统一 import 方式（6 个消费模块一致）

```python
from iqr_domain_schema import schema as _domain_schema
# 见 iqr_terms.py:19 / iqr_align.py:29 / iqr_anchor.py:32
#    iqr_figure_target.py:98 / iqr_query_intent.py:13 / iqr_protocol.py:13
```

---

## 2. 逐字段详解（11 个字段）

下面对每个字段给出：JSON 值 → accessor → 消费代码（带行号）→ 具体示例。

---

### 2.1 `section_markers`（手册章节模板词）

```json
"section_markers": ["保养步骤", "保养内容", "保养周期"]
```

**用途**：识别手册中「保养步骤：/保养内容：/保养周期：」这类固定章节行，用于章节切分、topic 截断、引用对齐。这是**被消费最多**的字段。

**消费点 A — 编译成正则（查询意图）** [`iqr_query_intent.py:21-41`](../scripts/iqr_query_intent.py#L21-L41)

```python
def _build_section_marker_re() -> re.Pattern[str]:
    alt = "|".join(re.escape(m) for m in _domain_schema.section_markers)
    return re.compile(rf"(?:{alt})[：:]\s*([^\n]{{2,48}})", re.IGNORECASE)

_SECTION_ALT = "|".join(re.escape(m) for m in _domain_schema.section_markers)
_SECTION_MARKER_SPLIT_RE = re.compile(_SECTION_ALT)          # 用于 split
_SECTION_MARKER_CONTENT_RE = re.compile(rf"(?:{_SECTION_ALT})[：:]\s*([^\n]+)")
```

> 示例：对文本 `"保养周期：每班一次\n保养部位：压带轮"`，`_SECTION_MARKER_CONTENT_RE`
> 会匹配出 `保养周期→每班一次`、`保养部位…`（若"保养部位"在表内）。

**消费点 B — 协议层 split** [`iqr_protocol.py:16-18`](../scripts/iqr_protocol.py#L16-L18)

```python
_SECTION_MARKER_SPLIT_RE = re.compile(
    "|".join(re.escape(m) for m in _domain_schema.section_markers)
)
```

**消费点 C — topic 截断** [`iqr_terms.py:169-173`](../scripts/iqr_terms.py#L169-L173)

```python
tail = re.split(
    "|".join(re.escape(marker) for marker in _domain_schema.section_markers),
    m.group(1), maxsplit=1,
)[0].strip()
```

> 示例：`"清理压带轮保养步骤：…"` → 在 `"保养步骤"` 处截断 → 得 `"清理压带轮"`。

**消费点 D — 引用对齐打分** [`iqr_align.py:498-505`](../scripts/iqr_align.py#L498-L505)

```python
for field in _domain_schema.section_markers:
    for match in re.finditer(rf"{field}[：:]([^\n]+)", content):
        val = _normalize_citation_blob(match.group(1).strip())
        ...
```

**消费点 E — 图目标筛选** [`iqr_figure_target.py:2623 / 2908 / 3214`](../scripts/iqr_figure_target.py#L2623)

```python
if any(m in text for m in _domain_schema.section_markers):   # 2623 章节行不算短引用
    return False
...
re.split(r"|".join(re.escape(m) for m in _domain_schema.section_markers), value, ...)  # 2908
```

---

### 2.2 `structural_field_keys`（结构标签后缀）

```json
"structural_field_keys": ["周期","步骤","内容","方式","部位","部件","工具","标准","依据","要求","方法","说明"]
```

**用途**：判断一个 bullet/标题是否是「字段标签」（如"保养周期""使用工具"），而非真正的图主体（如"压带轮"）。标签不应被当作配图目标。

**加载**：在 [`iqr_query_intent.py:53`](../scripts/iqr_query_intent.py#L53) 转成 tuple 供 `endswith` 用：

```python
_STRUCTURAL_LABEL_SUFFIXES = tuple(_domain_schema.structural_field_keys)
```

**消费** — [`iqr_figure_target.py:474-477`](../scripts/iqr_figure_target.py#L474-L477)（该 tuple 在 :113 被 import）：

```python
def _is_answer_structural_label(span: str) -> bool:
    """Answer field label (ends in a structural suffix), not a figure subject."""
    key = _normalize_label_key(span)
    return bool(key) and key.endswith(_STRUCTURAL_LABEL_SUFFIXES)
```

> 示例：`span="保养周期"` → `endswith(("周期",…))` 为真 → 判定为标签 → **不**配图。
> `span="压带轮"` → 不以任何后缀结尾 → 是主体 → 可配图。

该判定在图目标提取中被反复用于过滤（:301、:407、:507、:528、:571、:670、:721）。

---

### 2.3 `action_prefixes`（维护动作动词）★ 上一轮已详述

```json
"action_prefixes": ["清理", "检查", "更换", "调整", "清洁"]
```

**用途**：从「检查压带轮残胶」中剥掉开头动词，得到纯对象「压带轮残胶」，用于图匹配。

**消费** — [`iqr_terms.py:338-342`](../scripts/iqr_terms.py#L338-L342)：

```python
for prefix in _domain_schema.action_prefixes:
    if cjk.startswith(prefix) and len(cjk) > len(prefix) + 2:
        cjk = cjk[len(prefix):]      # "检查压带轮残胶" → "压带轮残胶"
        break
```

下游：`_query_primary_object_term`(:368) / `_query_object_terms`(:345) →
`_figure_matches_query_object`(:379) 判断图注是否含对象词 → iqr_align(:412/:456)、
iqr_figure_target(:3277) 选图。

---

### 2.4 `footnote_labels`（脚注标签）

```json
"footnote_labels": ["注", "备注"]
```

**用途**：解析答案字段时跳过脚注行（"注：…"/"备注：…"），不把脚注当部件/机型。

**消费** — [`iqr_figure_target.py:697-700`](../scripts/iqr_figure_target.py#L697-L700)：

```python
if field_name in _domain_schema.footnote_labels or re.fullmatch(r"注\d*", field_name):
    continue    # 跳过脚注字段
```

---

### 2.5 `paragraph_connectors`（段落连接词）

```json
"paragraph_connectors": ["此外", "另外", "同时", "除此之外"]
```

**用途**：答案正文中遇到「此外…」等补充段落时截断，只保留主列举部分。

**消费** — [`iqr_figure_target.py:393-396`](../scripts/iqr_figure_target.py#L393-L396)：

```python
for marker in _domain_schema.paragraph_connectors:
    match = re.search(rf"(?:^|\n)\s*{marker}", text)
    if match:
        text = text[: match.start()]   # 截掉补充段落
```

---

### 2.6 `filename_truncate_markers`（PDF 文件名截断标记）

```json
"filename_truncate_markers": ["维护保养"]
```

**用途**：从 PDF 标题取机型名：`"NB680维护保养.pdf"` → 截掉「维护保养」→ `"NB680"`。

**消费** — loader 内置方法 [`iqr_domain_schema.py:133-138`](../scripts/iqr_domain_schema.py#L133-L138)：

```python
def truncate_filename(self, title: str) -> str:
    result = title
    for marker in self._truncate_markers:
        result = result.split(marker)[0]
    return result.split(".pdf")[0].split(".PDF")[0].strip()
```

被 [`iqr_figure_target.py:614`](../scripts/iqr_figure_target.py#L614) 调用（见 2.7）。

---

### 2.7 `machine_class_suffixes`（设备类别后缀）

```json
"machine_class_suffixes": ["封边机", "钻", "中心"]
```

**用途**：判断一段文本是否提及某类设备（即使该机型尚未灌库），用于机型识别兜底。

**消费** — [`iqr_figure_target.py:599-614`](../scripts/iqr_figure_target.py#L599-L614)：

```python
def _is_machine_class_span(text: str) -> bool:
    blob = (text or "").strip()
    return any(s in blob for s in _domain_schema.machine_class_suffixes)   # :600

def _machine_from_section_title(title: str) -> str:
    ...
    if _is_machine_class_span(title):
        return _domain_schema.truncate_filename(title)   # :614 结合 2.6 取机型名
    return ""
```

> 示例：标题 `"NB680封边机"` → 含「封边机」→ 是设备类 → `truncate_filename` 得 `"NB680"`。

---

### 2.8 `special_query_patterns`（特殊查询模式）

```json
"special_query_patterns": ["开机前"]
```

**用途**：命中「开机前」这类查询时走特殊路径（preflight 检查类），影响锚定与图保留策略。

**loader 谓词** [`iqr_domain_schema.py:126-131`](../scripts/iqr_domain_schema.py#L126-L131)：

```python
def matches_special_pattern(self, query: str) -> str | None:
    for pat in self._special_patterns:
        if pat in query:
            return pat
    return None
```

**消费点 A — 锚定** [`iqr_anchor.py:1128-1130`](../scripts/iqr_anchor.py#L1128-L1130)：

```python
if _domain_schema.matches_special_pattern(q) and not _is_listing_scope_query(q):
    meta["reason"] = "preflight_query"
    return out, meta        # 开机前类查询走 preflight 分支
```

**消费点 B — 图保留过滤** [`iqr_figure_target.py:2859-2864`](../scripts/iqr_figure_target.py#L2859-L2864)：

```python
if _domain_schema.matches_special_pattern(query or "") and not span_keep:
    kept = [doc for doc in kept
            if not extract_image_refs_from_context(_doc_content(doc).strip())]
```

---

### 2.9 `catalog_page_marker`（目录/适用型号页标记）

```json
"catalog_page_marker": "本手册适用产品型号"
```

**用途**：识别手册第一页「适用型号」目录页，作为 catalog 查询的标记。

**消费** — [`iqr_query_intent.py:56-61`](../scripts/iqr_query_intent.py#L56-L61)（作为 import 失败的兜底）：

```python
try:
    from query_doc_steering import _CATALOG_MODEL_MARKER, detect_table_filter_signal
except ImportError:
    _CATALOG_MODEL_MARKER = _domain_schema.catalog_page_marker
```

---

### 2.10 `image_block_fields`（图片块元数据字段名）

```json
"image_block_fields": ["图片路径", "页码", "关联正文", "图注", "脚注"]
```

**用途**：灌库模板里图片块的字段名。对齐打分时，这些元数据行不算正文引用。

**消费** — [`iqr_align.py:463-464`](../scripts/iqr_align.py#L463-L464)：

```python
if line.startswith(tuple(["[图片]", *_domain_schema.image_block_fields])):
    return 0.0    # 图片元数据行不参与引用对齐打分
```

---

### 2.11 `domain` / `version`（元数据）

```json
"domain": "industrial_equipment_maintenance",
"version": "1.0",
```

仅用于日志/调试标识当前领域（`schema.domain`），不参与匹配逻辑。

---

## 3. 端到端示例：一次查询的完整数据流

查询：`"开机前需要检查压带轮残胶吗？"`

```
config/domain_schema.json
  ├─ special_query_patterns["开机前"] ──► iqr_anchor:1128  判定 preflight 查询
  ├─ action_prefixes["检查"]        ──► iqr_terms:338     剥离动词 → "压带轮残胶"
  └─ section_markers["保养…"]       ──► iqr_query_intent  章节切分/意图分类

iqr_terms._action_object_cjk      "检查压带轮残胶" → "压带轮残胶"
        ▼
iqr_terms._query_primary_object_term → "压带轮残胶"
        ▼
iqr_terms._figure_matches_query_object  图注含"压带轮残胶"? → True/False
        ▼
iqr_align / iqr_figure_target     候选图块筛选
        ▼
iqr_placement.resolve_query_images 返回最终配图 + debug
```

同一查询里 **3 个不同字段** 各自在不同环节起作用，互不耦合——这正是外置的价值。

---

## 4. 兼容与扩展

- **fallback**：JSON 缺失/损坏 → `_FALLBACK`（= 旧 hardcode），零配置可用。
- **env 覆盖**：`RAG_DOMAIN_SCHEMA=/path/x.json`，优先级最高。
- **加新字段**：① JSON 加键；② `DomainSchema.__init__` 预建 + `@property`；③ 消费模块引用。
- **换领域**：只改 JSON（参见 design.md §3.5 CNC 示例），代码零改动。
- **多租户**：当前 per-process 单例；如需并存，把 `_load_schema()` 改为按 path 缓存多实例。

---

## 5. 字段 × 消费模块 速查表

| 字段 | accessor | 主要消费模块:行 |
|------|----------|----------------|
| section_markers | `.section_markers` | query_intent:21-41, protocol:16, terms:169, align:498, figure_target:2623/2908/3214 |
| structural_field_keys | `.structural_field_keys` | query_intent:53 → figure_target:477 |
| action_prefixes | `.action_prefixes` | terms:338 |
| footnote_labels | `.footnote_labels` | figure_target:697 |
| paragraph_connectors | `.paragraph_connectors` | figure_target:393 |
| filename_truncate_markers | `.truncate_filename()` | figure_target:614 |
| special_query_patterns | `.matches_special_pattern()` | anchor:1128, figure_target:2859 |
| machine_class_suffixes | `.machine_class_suffixes` | figure_target:600 |
| catalog_page_marker | `.catalog_page_marker` | query_intent:61 |
| image_block_fields | `.image_block_fields` | align:463 |
| domain/version | `.domain` | 日志/调试 |

---

## 附录 A：真实运行示例（实测输出）

以下不是推演，而是**实际执行** `python _tmp_schema_demo.py`（`scripts/` 加入 `sys.path` 后直接 import 各模块）得到的真实输出，验证上述每个字段确实在代码中生效。

### A.1 加载层：字段确实从 JSON 读出

```
== 1. schema fields loaded from config/domain_schema.json ==
domain            = industrial_equipment_maintenance
section_markers   = ['保养步骤', '保养内容', '保养周期']
action_prefixes   = ('清理', '检查', '更换', '调整', '清洁')
machine_suffixes  = ['封边机', '钻', '中心']
```

### A.2 `filename_truncate_markers` + `machine_class_suffixes`

```
== 2. truncate_filename ==
  truncate_filename('NB680维护保养.pdf')      = 'NB680'
  truncate_filename('M750D封边机维护保养.PDF') = 'M750D封边机'
```

> 「维护保养」被截掉；「封边机」不在截断表内故保留——两个字段各司其职。

### A.3 `special_query_patterns`

```
== 3. matches_special_pattern ==
  matches_special_pattern('开机前需要检查压带轮残胶吗？') = '开机前'
  matches_special_pattern('压带轮残胶怎么清理？')         = None
```

> 含「开机前」→ 命中，走 preflight 分支（iqr_anchor:1128）；否则返回 None。

### A.4 `action_prefixes`：动词剥离

```
== 4. iqr_terms._action_object_cjk ==
  _action_object_cjk('检查压带轮残胶') = '压带轮残胶'   # 剥掉「检查」
  _action_object_cjk('清理压带轮残胶') = '压带轮残胶'   # 剥掉「清理」
  _action_object_cjk('压带轮残胶')     = '压带轮残胶'   # 无前缀，原样
  _query_primary_object_term('检查压带轮残胶') = '压带轮残胶'
```

### A.5 `structural_field_keys`：标签 vs 主体

```
== 5. iqr_figure_target._is_answer_structural_label ==
  _is_answer_structural_label('保养周期') = True    # 以「周期」结尾 → 标签，不配图
  _is_answer_structural_label('使用工具') = True    # 以「工具」结尾 → 标签
  _is_answer_structural_label('压带轮')   = False   # 主体 → 可配图
  _is_answer_structural_label('预铣刀')   = False   # 主体 → 可配图
```

### A.6 端到端：`action_prefixes` 驱动图匹配

```
== 6. iqr_terms._figure_matches_query_object ==
  query='检查压带轮残胶'
  _figure_matches_query_object(q, '图注：压带轮残胶清理示意') = True   # 对象词命中
  _figure_matches_query_object(q, '图注：吸尘风机安装位置')   = False  # 不命中
```

> 因为 A.4 把「检查」剥成「压带轮残胶」，图注含该词才判 True；若 schema 无「检查」，
> 对象会带动词，两个图注都会判 False——**一个词的有无直接决定选图结果**。

### A.7 复现方式

在项目根目录运行（把 `scripts/` 加入 `sys.path` 后直接 import 各模块）：

```powershell
$env:PYTHONIOENCODING="utf-8"
python -c "import sys; sys.path.insert(0,'scripts');
from iqr_domain_schema import schema;
from iqr_terms import _action_object_cjk, _figure_matches_query_object;
from iqr_figure_target import _is_answer_structural_label;
print(schema.truncate_filename('NB680维护保养.pdf'));
print(_action_object_cjk('检查压带轮残胶'));
print(_is_answer_structural_label('保养周期'));
print(_figure_matches_query_object('检查压带轮残胶','图注：压带轮残胶清理示意'))"
```

预期输出：`NB680` / `压带轮残胶` / `True` / `True`，与上文 A.2–A.6 一致。
