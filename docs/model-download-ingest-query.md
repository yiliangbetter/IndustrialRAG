# 模型下载、灌库与问答（独立操作说明）

本文档从 **「只拉模型 → 只灌库 → 只问答」** 拆步说明，并解释各步在做什么、常见坑与排错。主脚本仍为 **[`scripts/rag_pipeline_parse_graph_chat.py`](../scripts/rag_pipeline_parse_graph_chat.py)**。

更偏「参数一览与示例」的说明见：**[文档解析灌库与图谱问答脚本说明.md](./文档解析灌库与图谱问答脚本说明.md)**。环境变量完整示例可参考仓库根目录 **[`env.example`](../env.example)**。

---

## 0. 通用约定

- **工作目录**：以下命令均在 **仓库根目录**（含 `pyproject.toml`、`scripts/`）下执行。
- **包管理**：推荐 **`uv run python ...`**，与项目 `.venv` 一致。
- **配置**：根目录 **`.env`**。脚本会从 **仓库根** 加载 `.env`（与你在终端里的 `cd` 无关，但 **必须在仓库根执行脚本** 才能稳定找到 `.env`）。
- **Windows PowerShell**：续行用 **反引号 `` ` ``**，不要用 bash 的反斜杠 `\`。

---

## 0.5 终端代理（可选，建议在拉模型前）

若 **Hugging Face、`uv`、`huggingface-cli`、MinerU 权重下载** 等直连很慢或超时，可在 **当前 PowerShell 窗口** 设置代理环境变量，让**该窗口内**后续命令走本机 VPN 客户端开放的本地端口（**关掉窗口后失效**，不写进 `.env` 也行）。

**常见写法**（端口以你 VPN 软件为准；下例为常见的 HTTP 代理 **`7890`**，SOCKS 常为 **`7891`**）：

```powershell
$env:HTTP_PROXY  = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
# 若客户端说明使用 SOCKS5，可改用：
# $env:ALL_PROXY = "socks5://127.0.0.1:7891"
```

不少工具会读**小写**变量，可与上面一并设置：

```powershell
$env:http_proxy  = $env:HTTP_PROXY
$env:https_proxy = $env:HTTPS_PROXY
```

**说明**：

- **`http://` 与 `socks5://`、端口号** 以 Clash、v2rayN 等界面里的「本地监听 / 混合代理」为准，不要照搬示例端口。
- 若 VPN 已开 **TUN 模式** 或 **系统代理**，有时终端可不设变量也能走代理。
- **Git** 若仍直连：`git config --global http.proxy http://127.0.0.1:7890`；取消：`git config --global --unset http.proxy`。

---

## 1. 单独拉模型（预下载）

灌库或问答前，按需下载；**不必每次灌库都拉**，缓存齐即可。

### 1.1 MinerU 解析用 pipeline 权重

**作用**：MinerU 解析 PDF 等时依赖的 **pipeline 模型包**（体积大，首次或换机时需要）。

**方式 A：只下载、不跑灌库（推荐）**

与主脚本内部调用一致，在仓库根目录执行（`MINERU_MODEL_SOURCE` 可选 **`huggingface`** / **`modelscope`**，默认前者）：

```powershell
cd D:\dev\RAGdemo\RAG-Anything
uv run mineru-models-download -s huggingface -m pipeline
```

**方式 B：灌库前顺带下载**

在未使用 **`--query-only`** 的命令里加 **`--mineru-download-models`**，会在解析前执行与上相同的下载。**注意**：若使用 **`--query-only`**，脚本 **不会** 执行该下载（见 [`rag_pipeline_parse_graph_chat.py`](../scripts/rag_pipeline_parse_graph_chat.py) 中条件判断）。

**说明**：

- 若解析时报 **`PDF-Extract-Kit` / `.pth` 不存在**，多为缓存不完整：删除对应 **`HF_HOME`** 下 `models--opendatalab--PDF-Extract-Kit-1.0` 目录后重下，或单独用下节命令拉仓库。

### 1.2 Hugging Face：`opendatalab/PDF-Extract-Kit-1.0`（OCR 等子权重）

**作用**：MinerU 链路上引用 **`ch_PP-OCRv4_rec_server_doc_infer.pth`** 等文件；断网或半包会导致 `FileNotFoundError`。

**命令**（与 `.env` 中 **`HF_HOME`** 一致，建议 **绝对路径**）：

```powershell
cd D:\dev\RAGdemo\RAG-Anything
$env:HF_HOME = "D:\dev\RAGdemo\RAG-Anything\.hf_cache"
uv run huggingface-cli download opendatalab/PDF-Extract-Kit-1.0
```

### 1.3 Hugging Face：本地向量模型 `BAAI/bge-m3`

**作用**：当 **`EMBEDDING_BACKEND=hf`** 时，`sentence-transformers` 从缓存加载该模型；预先下载可避免灌库/首次查询时长时间拉 **`pytorch_model.bin`**。

```powershell
$env:HF_HOME = "D:\dev\RAGdemo\RAG-Anything\.hf_cache"
uv run huggingface-cli download BAAI/bge-m3
```

**说明**：

- **`HF_HOME`** 与灌库、问答、预下载应 **保持一致**（建议写在 `.env` 里用绝对路径）。
- 查询阶段若不想访问 **`huggingface.co`**：在 `.env` 中设置 **`HF_EMBED_OFFLINE=1`**（项目内会对本地 snapshot 路径加载），详见另一篇脚本说明文档。

### 1.4 网络与镜像

- 访问 Hub 慢：除上文 **「0.5 终端代理」** 外，可换网络，或在合规前提下配置 **`HF_ENDPOINT`** 等镜像（以你环境为准）。
- **`huggingface-cli login`** 仅在 **门控/私有模型** 时需要；公开模型如 `BAAI/bge-m3` 一般不必。

---

## 2. 单独灌库（`--ingest-only`）

**作用**：对 **`--input-folder`** 下文件做 **解析（如 MinerU）→ 写入 `-w` 指向的 LightRAG 工作区**（向量 + 图谱等），**不**进入问答。

### 2.1 基本命令

```powershell
cd D:\dev\RAGdemo\RAG-Anything
uv run python .\scripts\rag_pipeline_parse_graph_chat.py `
  --input-folder .\my_pdfs `
  -w .\rag_storage_run `
  --ingest-only
```

**说明**：

- **`-w`**：知识库根目录；以后 **仅查询** 必须用 **同一 `-w`**。
- **`--parser-output-dir`**：解析落盘目录，默认 **`output/pipeline_parse`**，可按需指定。
- **`--limit N`**：只处理前 N 个文件，调试用。
- **`--recursive` / `--no-recursive`**：是否递归子文件夹。

### 2.2 改分块大小（`CHUNK_SIZE`）后重灌

**作用**：LightRAG 按 **`CHUNK_SIZE`**（token 级近似）切正文再建索引；调大有利于 **表格、附表** 少被切断，但 **单块更长 → 图谱抽取时 LLM 更慢**，易触发超时。

在 **`.env`** 中设置（示例）：

```env
CHUNK_SIZE=1800
CHUNK_OVERLAP_SIZE=120
```

**重要**：改 **`CHUNK_SIZE` 后应视为新索引策略**，建议 **删除原 `-w` 整个目录** 或 **换新的 `-w` 路径** 再全量灌库；否则旧块与新块混在同一工作区，行为难预期。

### 2.3 灌库阶段 LLM 超时

图谱抽取会对每个 chunk 调 LLM。若出现 **`LLM func: Worker execution timeout after 360s`**：

- 在 `.env` 提高 **`LLM_TIMEOUT`**（如 **`600`** / **`900`**）；
- 或略降 **`CHUNK_SIZE`**、降低 **`MAX_ASYNC`** 减轻排队。

### 2.5 多模态灌库（表格 / 图片 / 公式）

默认 **`--ingest-only`** 等价于 **`--skip-multimodal`**：入库快；表格的 **`table_body` / HTML** 仍会拼进正文参与向量索引（见 **`raganything/processor.py`** 中 `insert_content_list`）。

若要对 **表格 / 图片 / 公式** 走 **RAGAnything 多模态处理器**（如 **`TableModalProcessor`** 生成表描述、单独多模态 chunk 等），灌库时加上 **`--no-skip-multimodal`**：

```powershell
cd D:\dev\RAGdemo\RAG-Anything
uv run python .\scripts\rag_pipeline_parse_graph_chat.py `
  --input-folder .\my_pdfs `
  -w .\rag_storage_run `
  --ingest-only `
  --no-skip-multimodal
```

**`.env` 中与多模态相关的开关**（在 **`--no-skip-multimodal`** 时由 **`_build_rag`** 读取；未写时默认视为 **true**）：

- **`ENABLE_TABLE_PROCESSING`**：表格处理器  
- **`ENABLE_IMAGE_PROCESSING`**：图片（可能使用 **`VISION_MODEL`**）  
- **`ENABLE_EQUATION_PROCESSING`**：公式  

**仅表格、关闭图片与公式** 示例：

```env
ENABLE_TABLE_PROCESSING=true
ENABLE_IMAGE_PROCESSING=false
ENABLE_EQUATION_PROCESSING=false
```

**注意**：多模态灌库会 **增加 LLM 调用** 与耗时，需 **`LLM_BINDING_HOST` 可达**；切换默认路径与多模态路径后建议 **新 `-w` 或清空后重灌**。更完整的参数说明见 **[文档解析灌库与图谱问答脚本说明.md](./文档解析灌库与图谱问答脚本说明.md)**。

---

## 3. 单独问答（`--query-only`）

**作用**：**不解析、不灌库**，从 **`-w`** 加载已有 LightRAG 数据，执行 **`--query` 一次** 或 **交互问答**。

### 3.1 交互问答

```powershell
cd D:\dev\RAGdemo\RAG-Anything
uv run python .\scripts\rag_pipeline_parse_graph_chat.py --query-only -w .\rag_storage_run
```

**退出方式**：

- 输入 **`exit` / `quit` / `bye` / `/exit` / `/quit` / `:q` / `!q` / `退出` / `再见`**
- **空行**：不退出，仅再显示一行 **`Q>`**（与常见终端行为类似）
- **`Ctrl+C`**：结束（部分环境下可试 **`Ctrl+Break`**）

### 3.2 单次提问后退出

```powershell
uv run python .\scripts\rag_pipeline_parse_graph_chat.py `
  --query-only `
  -w .\rag_storage_run `
  --query "你的问题全文"
```

### 3.3 查询模式与检索参数

- **`--query-mode`**：如 **`mix`**（图+向量）、**`naive`**（仅向量 chunk）等；也可在 `.env` 设 **`RAG_QUERY_MODE`**。
- **LightRAG 常用 `.env` 项**（与「图 vs chunk 预算」相关，详见 LightRAG 文档）：**`TOP_K`**、**`CHUNK_TOP_K`**、**`COSINE_THRESHOLD`**、**`MAX_ENTITY_TOKENS`**、**`MAX_RELATION_TOKENS`**、**`MAX_TOTAL_TOKENS`**、**`RERANK_BY_DEFAULT`**。
- **重排序（rerank）**：LightRAG 在 **`RERANK_BY_DEFAULT=true`** 时会对检索到的 chunk 做重排，但必须在构建 **`LightRAG`** 时提供 **`rerank_model_func`**。本仓库的 **`scripts/rag_pipeline_parse_graph_chat.py`**（及依赖其 **`_build_rag`** 的 **`dump_query_context.py`**）已按环境变量注入该函数。请在 `.env` 中配置（详见根目录 **`env.example`** 中 `### Rerank` 段）：
  - **`RERANK_BINDING=none`**（或不设）：不重排（与「关掉 rerank」一致）。
  - **`RERANK_BINDING=jina` / `cohere` / `aliyun`**：使用对应云端 API，需 **`RERANK_BINDING_API_KEY`**（或各厂商同名变量），**`RERANK_MODEL`** 填该 API 支持的模型名。
  - **`RERANK_BINDING=hf`**：本地 **`sentence_transformers.CrossEncoder`**，需 **`uv sync --extra local-embed`**，并用 **`huggingface-cli download`** 把 **`RERANK_MODEL`**（默认 **`BAAI/bge-reranker-base`**）下载到与 **`HF_HOME`** 一致的 Hub 缓存。加载时会自动解析 **`HF_HOME/hub/models--…--…/snapshots/<hash>`**（以目录内 **`config.json`** 为准；CrossEncoder 仓库**没有** embedding 用的 **`modules.json`**，属正常现象）。若仍尝试连 Hub，请设 **`HF_EMBED_OFFLINE=1`** 或 **`RERANK_HF_OFFLINE=1`**（与脚本里已设的 **`HF_HUB_OFFLINE`** 一致），并确认 **`HF_HOME`** 指向已含该模型的缓存目录。
  - 打开重排时请将 **`RERANK_BY_DEFAULT=true`**（LightRAG 默认即为 true；若曾为 false 请改回）。

**注意**：灌库与查询时 **Embedding 模型、维度、`HF_HOME`** 等应 **一致**，否则向量空间对不齐。

---

## 4. 检索自检（可选）

**脚本**：[**`scripts/dump_query_context.py`**](../scripts/dump_query_context.py)

**作用**：对某条问题调用 **`aquery_data`**，把 **最终要送给 LLM 的实体/关系/chunk 原文** 写入 UTF-8 文件，**不调大模型**，用于判断「表里内容有没有进 chunk」。

```powershell
# 未写问题时使用脚本内置默认问题；输出默认 docs/query_context_dump.txt
uv run python .\scripts\dump_query_context.py -w .\rag_storage_run --query-mode mix

# 指定问题（位置参数，放在选项之后）；自定义输出文件
uv run python .\scripts\dump_query_context.py -w .\rag_storage_run --query-mode naive --out .\docs\query_context_dump_naive.txt "你的问题全文"
```

---

## 5. 流程串联速查

| 目标 | 核心命令或动作 |
|------|----------------|
| 只下 MinerU pipeline | `uv run mineru-models-download -s huggingface -m pipeline`；或灌库命令加 `--mineru-download-models`（不可与 `--query-only` 同用） |
| 只下 HF 模型 | `HF_HOME=...` + `huggingface-cli download <repo>` |
| 只灌库 | `--input-folder ... -w ... --ingest-only` |
| 多模态灌库 | 同上并加 **`--no-skip-multimodal`**；按需配置 **`ENABLE_TABLE_PROCESSING`** 等（见上文 **「2.5 多模态灌库」**） |
| 只问答 | `--query-only -w ...`（可加 `--query "..."`） |
| 干净重灌 | 删除 **`-w`** 目录或换 **新 `-w`**，再 `--ingest-only` |

---

## 6. 与主说明文档的关系

- **参数表、更多示例**：见 **[文档解析灌库与图谱问答脚本说明.md](./文档解析灌库与图谱问答脚本说明.md)**。
- **环境变量模板**：见 **[`env.example`](../env.example)**。
