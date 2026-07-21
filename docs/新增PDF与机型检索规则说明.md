# 新增 PDF 与机型检索规则说明

本文说明：向现有知识库**追加新 PDF** 时的灌库步骤，以及何时需要维护 **`scripts/query_doc_steering.py`** 中的机型过滤规则。

相关脚本与配置：

- 灌库 / 问答：[`scripts/rag_pipeline_parse_graph_chat.py`](../scripts/rag_pipeline_parse_graph_chat.py)
- 机型过滤：[`scripts/query_doc_steering.py`](../scripts/query_doc_steering.py)（Web/工具链合入后可用）
- 检索自检：主脚本 `--query-only`（可选本地调试脚本 `dump_query_context.py`，未必在核心包中）
- Web 灌库 / 问答：[`scripts/rag_web_server.py`](../scripts/rag_web_server.py)（Web PR 合入后可用）
- 环境变量示例：[`env.example`](../env.example)

更完整的脚本参数说明见：[文档解析灌库与图谱问答脚本说明](./文档解析灌库与图谱问答脚本说明.md)。

---

## 核心结论

| 事项 | 做法 |
|------|------|
| 每次新增 PDF | **灌库到与查询相同的 `-w` 工作目录** |
| 机型过滤开关 | `.env` 中 `RAG_QUERY_DOC_FILTER=true`（默认开启即可） |
| 是否为每台机器配 `.env` | **不需要**；机型规则写在 `MACHINE_PROFILES` |
| 何时改规则 | 仅当新手册代表**新机型**且会与旧手册在检索中「串台」时 |

---

## 一、常规操作（每次加 PDF 都要做）

### 1. 放入待灌库目录

将新 PDF 放入例如 `my_documents/`，或通过 Web 界面上传灌库。

### 2. 灌库（必须与查询使用同一个 `-w`）

当前示例工作目录为 `rag_storage_run`（与 `.env` 中 `RAG_WEB_WORKING_DIR` 一致时请保持相同路径）。

在**仓库根目录**执行：

```powershell
cd D:\dev\RAGdemo\RAG-Anything

uv run python scripts/rag_pipeline_parse_graph_chat.py `
  --input-folder ./my_documents `
  -w ./rag_storage_run `
  --ingest-only
```

**只灌新文件、避免重复处理旧库：** 可将**新 PDF 单独放在一个文件夹**，对该文件夹执行上述命令。若同名文件曾灌入同一 `-w`，需确认是否应改名、换目录或新建 `-w`，以免文档状态混乱。

**Web 方式：** 启动 `rag_web_server.py` 后，在左侧「上传灌库」拖入 PDF；`RAG_WEB_WORKING_DIR` 须与 CLI 的 `-w` 指向同一目录。

### 3. 环境与模型保持一致

- **LLM / Embedding** 须与首次灌库时一致（`.env` 中 `EMBEDDING_MODEL`、`EMBEDDING_DIM` 等勿随意更改）。
- 若更换 embedding 模型，应对**新的 `-w` 全量重灌**，否则向量空间不一致，检索会失真。

### 4. 验证是否入库成功

```powershell
uv run python scripts/rag_pipeline_parse_graph_chat.py --query-only -w rag_storage_run `
  --query "针对新手册内容的一个测试问题"
```

确认答案能引用新 PDF，或在 Web「检索范围」中看到对应手册路径。

### 5. 重启 Web 服务（若使用浏览器问答）

灌库完成后重启 `rag_web_server.py`，再提问验证答案与界面上的「检索范围」提示。

---

## 二、什么时候要改 `query_doc_steering.py`？

**不是每加一本 PDF 都要改代码。**

| 情况 | 是否需要改 `MACHINE_PROFILES` |
|------|------------------------------|
| 新 PDF 属于**已有型号**的补充说明 | 一般**不需要** |
| 新 PDF 为**全新机型**，且与现有手册共用「输送链条」等通用词 | **需要**新增或调整一条 profile |
| 新 PDF 为**通用资料**（报警、电气等），用户问题**不写机型** | **不需要**（未识别机型时不做过滤） |
| 新产品线、与封边机无关 | 建议**新建 `-w` 独立知识库**，优于堆叠 deny 规则 |

### 机型过滤如何工作（简述）

1. 从用户问题中匹配 `query_phrases`（较长短语优先）。
2. 命中某机型 profile 后，在 rerank 之后丢弃 `file_path` 含 `deny_path_substrings` 的 chunk。
3. Web 界面通过 SSE 事件 `retrieval_scope` 展示「检索范围」与已排除的手册文件名（非暗箱操作）。

总开关：`.env` 中 `RAG_QUERY_DOC_FILTER=false` 可关闭过滤。

### 当前内置机型（节选）

规则定义见 [`scripts/query_doc_steering.py`](../scripts/query_doc_steering.py) 中的 `MACHINE_PROFILES`，主要包括：

- 高速智能封边机
- 高速自动封边机
- 双端封边机
- 自动封边机

用户问题里写「高速自动封边机」时，会自动排除自动封边机、封边机连线项目等其它手册片段，**无需**在 `.env` 里为每个机型单独配置。

---

## 三、新增机型时如何改规则

在 `MACHINE_PROFILES` 列表末尾追加一项，例如：

```python
{
    "id": "new_machine_x",
    "label": "某某封边机",           # 界面「检索范围」显示用
    "query_phrases": [              # 用户问题中出现即匹配（越长越优先）
        "某某封边机",
        "型号 ABC123",
    ],
    "query_exclude_if_contains": [  # 可选：避免与其它机型短语冲突
        "高速智能",
    ],
    "deny_path_substrings": [        # 要排除的 PDF 文件名子串（与灌库后 file_path 一致）
        "自动封边机维护保养手册",
        "高速自动封边机维护保养手册",
        # …其它不应混入的手册文件名关键词
    ],
},
```

**同时**检查其它已有机型的 `deny_path_substrings`，必要时加入**新 PDF 文件名**中的可识别子串；否则用户问「高速智能」时仍可能召回新机型手册。

修改后重启 Web 服务或重新运行 CLI 查询即可生效，**不必**为每台机器修改 `.env`。

### 不改代码的临时方式（运维）

在 `.env` 中设置 `RAG_QUERY_DOC_FILTER_RULES_JSON`（JSON 数组，结构与 `MACHINE_PROFILES` 相同），见 [`env.example`](../env.example) 中 `### Pipeline script only: query steering` 段。适合试验；长期仍建议写入 `query_doc_steering.py` 便于版本管理。

可选：通过 `RAG_QUERY_DOC_DENY_SUBSTRINGS` 追加全局排除子串（英文逗号分隔）。

---

## 四、推荐策略（按规模选择）

```
少量 PDF、仍属封边机系列
  → 继续使用同一 rag_storage_run
  → 新机型时再补 MACHINE_PROFILES

全新产品线 / 与现有手册差异大
  → 新建 -w（例如 rag_storage_new_line）
  → Web 的 RAG_WEB_WORKING_DIR 指向新目录
  → 不必维护一长串 deny 列表

用户经常在问题里不写机型
  → 优先考虑拆库；或后续在灌库时写入 metadata 再按字段过滤（可扩展性更好）
```

---

## 五、加 PDF 后的自检清单

1. `--query-only`（或 Web 问答）能召回到新 PDF 的内容。
2. 使用**带机型**的问题提问 → Web 界面应出现「检索范围：xxx」及参考/已排除列表。
3. 确认不会错误引用应排除的手册（例：问高速智能时不应出现《自动封边机…》中的「每天长城导轨油 68#」）。
4. 答案末尾 References 仅列出实际引用的手册。

---

## 六、问答仍出现已排除内容（如「每天长城导轨油」）？

1. **检索 chunk 已正确，但答案仍错**：`mix` 模式还会把**知识图谱**里跨手册合并的实体（如 `Conveyor Chain` 日润滑）送进 LLM。本项目在 `query_doc_steering.py` 中对「高速智能 + 输送链条」做了 **KG 上下文清洗** 与 **按问题注入 user_prompt**（无需写进 `.env`）。
2. **LLM 问答缓存**：若 `.env` 中 `ENABLE_LLM_CACHE=true`，可能直接返回旧答案。调试时建议 `ENABLE_LLM_CACHE=false`，并执行：
   ```powershell
   uv run python scripts/clear_query_llm_cache.py -w rag_storage_run
   ```
   然后**重启 Web 服务**再提问。
3. 用 `--query-only` / Web 确认答案与检索范围是否仍含 `长城导轨油`；若检索已干净而答案仍有，即属 KG 或缓存问题。

---

## 七、常见问题

**Q：只加了 PDF，问答变差了？**
A：检查 embedding 是否与灌库时一致；用 `--query-only` 看新文档是否进入答案/检索范围；适当调整 `CHUNK_TOP_K`、`MIN_RERANK_SCORE`（见 `env.example`）。

**Q：新 PDF 灌了但检索不到？**
A：确认 `-w` 与 Web `RAG_WEB_WORKING_DIR` 一致；解析是否成功（查看 `parser_output_dir`）；cosine 阈值是否过高。

**Q：过滤太狠，上下文为空？**
A：实现上若过滤后无 chunk 会回退到过滤前结果；若仍异常，检查 `deny_path_substrings` 是否误伤了应保留的手册（例如「封边机连线」手册内含高速智能章节，不应被高速智能 profile 排除）。

---

## 相关环境变量

| 变量 | 说明 |
|------|------|
| `RAG_QUERY_DOC_FILTER` | `true`（默认）开启按机型过滤；`false` 关闭 |
| `RAG_QUERY_DOC_FILTER_RULES_JSON` | 可选，追加 JSON 规则 |
| `RAG_QUERY_DOC_DENY_SUBSTRINGS` | 可选，全局额外排除的 file_path 子串 |
| `RAG_QUERY_USER_PROMPT` | 可选，注入 LightRAG `user_prompt`，非按机型必填 |
| `RAG_WEB_WORKING_DIR` | Web 使用的 LightRAG 工作目录，须与灌库 `-w` 一致 |
