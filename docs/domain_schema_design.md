# 领域 Schema 外置方案：行业定制 RAG 的通用化架构

## 1. 背景与问题

本系统（RAG-Anything）当前服务于**工业设备维护手册**场景（南兴封边机系列）。
配图管线、锚定逻辑、部件提取等核心模块中嵌入了大量领域关键词：

- 手册结构词：`保养步骤`、`保养内容`、`保养周期`
- 维护动作词：`清理`、`检查`、`更换`、`调整`、`清洁`
- 结构标签词：`周期`、`步骤`、`内容`、`方式`、`部位`……

这些词汇使系统在目标领域达到了 17/17 全 PASS 的高质量，但若更换手册类型
（操作手册、安装手册、其他行业文档），这些 hardcode 将全部失效。

## 2. 行业现状：领域 RAG 的定制化是必然

| 领域 | 典型 hardcode | 代表系统 |
|------|--------------|----------|
| 医疗 | diagnosis / treatment / contraindication | PubMedQA, MedRAG |
| 法律 | jurisdiction / precedent / statute | Harvey AI, CaseText |
| 金融 | revenue / EBITDA / guidance | Bloomberg GPT |
| 工业维护 | 保养步骤 / 保养周期 / 清理 / 检查 | 本系统 |

**结论**：高质量 RAG 系统必然包含领域知识。区别在于嵌入方式——
是散落在代码逻辑中，还是集中为可替换的配置。

## 3. 设计方案：Domain Schema 外置

### 3.1 核心思路

```
代码逻辑（不变）  ←读取←  Domain Schema（可替换）
```

- 代码只引用 schema 对象（如 `schema.section_markers`），不写死词汇
- 切换领域时，只需替换 `config/domain_schema.json`，代码零改动
- Schema 支持环境变量 `RAG_DOMAIN_SCHEMA` 指定路径（默认 `config/domain_schema.json`）

### 3.2 Schema 结构

```json
{
  "domain": "industrial_equipment_maintenance",
  "version": "1.0",

  "section_markers": ["保养步骤", "保养内容", "保养周期"],

  "structural_field_keys": [
    "周期", "步骤", "内容", "方式", "部位",
    "部件", "工具", "标准", "依据", "要求", "方法", "说明"
  ],

  "action_prefixes": ["清理", "检查", "更换", "调整", "清洁"],

  "footnote_labels": ["注", "备注"],

  "paragraph_connectors": ["此外", "另外", "同时", "除此之外"],

  "filename_truncate_markers": ["维护保养"],

  "special_query_patterns": ["开机前"],

  "catalog_page_marker": "本手册适用产品型号",

  "image_block_fields": ["图片路径", "页码", "关联正文", "图注", "脚注"]
}
```

### 3.3 字段说明

| 字段 | 用途 | 消费方 |
|------|------|--------|
| `section_markers` | 手册章节模板字段名，用于锚定和 topic 截断 | iqr_align, iqr_figure_target, iqr_terms |
| `structural_field_keys` | 判断 bullet 是否为"部件列举"的结构信号 | iqr_figure_target (`_COMPONENT_FIELD_KEYS`) |
| `action_prefixes` | 从"清理残胶"中剥离动作词提取部件名 | iqr_terms |
| `footnote_labels` | 排除手册脚注行 | iqr_figure_target |
| `paragraph_connectors` | 段落连接词检测（补充说明判断） | iqr_figure_target |
| `filename_truncate_markers` | PDF 文件名截断标记（取机型名） | iqr_figure_target |
| `special_query_patterns` | 需要特殊处理路径的查询模式 | iqr_anchor, iqr_figure_target |
| `catalog_page_marker` | 目录页标识行（所有手册第一页固定文本） | iqr_figure_target |
| `image_block_fields` | 图片块元数据字段名（灌库模板协议） | iqr_protocol, iqr_align |

### 3.4 加载器 API

```python
# scripts/iqr_domain_schema.py
from iqr_domain_schema import schema

schema.section_markers        # -> ["保养步骤", "保养内容", "保养周期"]
schema.action_prefixes        # -> ["清理", "检查", "更换", "调整", "清洁"]
schema.is_section_marker(x)   # -> bool
schema.is_action_prefix(x)    # -> bool
schema.truncate_filename(x)   # -> 截断后的机型名
```

### 3.5 适配新领域示例

假设要适配**数控机床操作手册**：

```json
{
  "domain": "cnc_machine_operation",
  "section_markers": ["操作步骤", "参数设置", "故障排除", "安全注意事项"],
  "structural_field_keys": ["参数", "步骤", "模式", "轴", "刀具", "转速", "进给"],
  "action_prefixes": ["设置", "输入", "确认", "启动", "校准"],
  "footnote_labels": ["注意", "警告"],
  "paragraph_connectors": ["然后", "接着", "随后", "最后"],
  "filename_truncate_markers": ["操作手册", "使用说明书"],
  "special_query_patterns": ["开机前", "关机后", "急停"],
  "catalog_page_marker": "适用机型",
  "image_block_fields": ["图片路径", "页码", "关联正文", "图注", "脚注"]
}
```

代码零改动，只需 `RAG_DOMAIN_SCHEMA=config/cnc_schema.json`。

## 4. 不变的部分（A 类 / 协议层）

以下属于**系统内部协议**，不随领域变化：

- 图片块 Markdown 模板格式（`[图片]\n图片路径：…\n页码：…`）——这是灌库管线写死的
- `_BOLD_RE`、`_WS_RE` 等正则工具——纯文本处理
- 向量检索 / rerank / LLM 调用逻辑——通用 RAG 基础设施

## 5. 实施计划

1. 创建 `config/domain_schema.json`（当前领域默认值）
2. 创建 `scripts/iqr_domain_schema.py`（加载器 + 缓存 + 便捷方法）
3. 重构 `iqr_figure_target.py`、`iqr_align.py`、`iqr_terms.py` 引用 schema
4. 修复 D 类 hardcode（`"封边机"` → 动态词表；`"开机前"` → schema.special_query_patterns）
5. pytest 全量验证

## 6. 向后兼容

- 若 `config/domain_schema.json` 不存在，加载器 fallback 到内置默认值（= 当前 hardcode）
- 现有部署无需任何配置变更即可正常运行
- 环境变量 `RAG_DOMAIN_SCHEMA` 为可选覆盖

## 7. 换客户操作指南

### 7.1 三步切换

**Step 1 — 复制模板，填写新领域词汇**

```bash
cp config/domain_schema.json config/schema_<customer>.json
```

按字段填入新领域词汇（参见 7.2 字段填写指引）。

**Step 2 — 设置环境变量**

```bash
# Linux / Mac
export RAG_DOMAIN_SCHEMA=/path/to/config/schema_<customer>.json

# Windows PowerShell
$env:RAG_DOMAIN_SCHEMA = "D:\projects\config\schema_<customer>.json"
```

**Step 3 — 正常启动，代码零改动**

加载器 `iqr_domain_schema.py` 自动读取指定文件。
优先级：环境变量 > `config/domain_schema.json` > 内置 fallback。

### 7.2 字段填写指引

| 字段 | 怎么找 |
|------|--------|
| `section_markers` | 翻开新手册，看每节开头的固定格式词（如"保养步骤：""操作步骤："） |
| `structural_field_keys` | 手册中 KV 对 / 表格的 key 集合（"周期""工具""参数"等） |
| `action_prefixes` | 步骤描述开头的操作动词（"清理""消毒""校准"等） |
| `footnote_labels` | 手册页脚 / 表尾的注释标签（"注""警告""注意"） |
| `paragraph_connectors` | 多段描述中的过渡词（"此外""然后""接着"） |
| `filename_truncate_markers` | PDF 文件名中要截掉的后缀（如"XX设备**维护保养**.pdf"→截掉得到机型名） |
| `special_query_patterns` | 需要触发特殊处理逻辑的查询关键词（如"开机前""急停"） |
| `machine_class_suffixes` | 设备类别后缀词（用于判断文本是否提及某类设备） |
| `catalog_page_marker` | 手册中列出适用型号的那页的固定文字 |
| `image_block_fields` | 灌库时图片块的字段结构（一般不变，除非改灌库模板） |

### 7.3 自动填充

项目中已有辅助工具：

- `scripts/_induce_field_schema.py` — 从已灌库的 content_list 中自动归纳字段 schema

推荐流程：

1. **先灌一批新手册**（用现有 pipeline，灌库不依赖 domain schema）
2. **跑归纳脚本**扫描灌库结果，自动提取高频章节标记、字段名、动词前缀
3. **人工审核微调**（~5 分钟），输出为新 JSON
4. **设环境变量，上线**

`machine_class_suffixes` 和 `special_query_patterns` 建议从客户 FAQ / 历史查询日志中抽取。

### 7.4 多客户并存（多租户扩展）

当前架构为 per-process 单 schema（`lru_cache` 单例）。若未来需多租户并存：

- 方案 A：每租户独立进程，各设自己的 `RAG_DOMAIN_SCHEMA`
- 方案 B：将 `_load_schema()` 改为按 path 参数缓存多实例，查询入口按 tenant 选择

当前设计已预留扩展点（环境变量即 per-process 切换开关），无需重构核心逻辑。
