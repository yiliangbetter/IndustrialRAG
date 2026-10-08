# chunk-0d30 检索调试记录与结论

本文记录针对目标块 **`chunk-0d30e1b587eb2728343d0f3b5f31ae1c`**（《封边机连线项目维护保养手册》中含「3.14.5 电控板检查」「保养周期：每季度一次」及变频器等段落）的排查过程与最终结论，便于日后复现或对比参数。

---

## 1. 目标分两层

| 阶段 | 含义 | 目标 |
|------|------|------|
| A | LightRAG `_merge_all_chunks` 之后、rerank 之前，合并列表约 30 条 | 合并列表中含 `chunk-0d30…` |
| B | `aquery_data` / `dump_query_context` 看到的**最终** chunk 列表 | 最终列表中含 `chunk-0d30…` |

阶段 A 已通过脚本 `scripts/capture_merged_chunk_ids.py` 验证：输出 `docs/merged_pre_rerank_chunk_ids.txt` 中第 8 行为该 chunk。

阶段 B 最初失败：`MAX_TOTAL_TOKENS=40000` 时最终约 **14** 条 chunk，其中**没有** `chunk-0d30…`。

---

## 2. 使用到的命令与脚本

- **合并后、rerank 前的 chunk id**  
  `uv run python scripts/capture_merged_chunk_ids.py -w .\rag_storage_run "四种封边机的电控板的保养周期分别是多久"`  
  输出：`docs/merged_pre_rerank_chunk_ids.txt`（及 `.meta.txt` 中的条数）。

- **最终进入上下文的 chunk（不调 LLM）**  
  `uv run python scripts/dump_query_context.py -w .\rag_storage_run "四种封边机的电控板的保养周期分别是多久" [--out 某文件]`  
  查看 `metadata.processing_info`、`chunks=` 与各条 `chunk_id`。

- **对照实验（仓库外可在一小段内联脚本中完成）**  
  同一进程内 `await rag.lightrag.aquery_data(..., QueryParam(mode="mix", max_total_tokens=...))` 扫描 `max_total_tokens`，避免反复 `import`/重建模型导致 GPU OOM。

---

## 3. 已排除的原因

- **纯向量召回**：单独对 chunks VDB 做 top-k 查询时，`chunk-0d30…` 可出现在 top 30 内，说明「根本没进向量候选」不是主因。
- **合并阶段丢 id**：`merged_pre_rerank_chunk_ids.txt` 已包含该 id，说明**不是** `_merge_all_chunks` 的 round-robin/dedup 把它挡在 30 条外。

---

## 4. 根因定位（阶段 B）

LightRAG 在构建 mix 模式上下文时（`operate.py` 中 `_build_context_str` 一带逻辑）：

1. 用 **总预算** `max_total_tokens`（来自 `QueryParam` / `LightRAG` 配置，与 `.env` 中 `MAX_TOTAL_TOKENS` 一致）减去 **系统模板、KG 实体与关系 JSON、用户 query、buffer** 等，得到 **`available_chunk_tokens`**。
2. 对合并后的 chunk 调用 `utils.py` 中的 **`process_chunks_unified`**：先 rerank（若开启），再按 `MIN_RERANK_SCORE` 过滤，再按 **`truncate_list_by_token_size`** 按**列表顺序**累加 token，超出则**整段丢弃后续 chunk**。

因此：

- 日志里常见 **「Rerank filtering: 19 chunks remained」** 再 **「Final context: … 14 chunks」**：`0d30` **仍在 rerank 过滤后的集合里**，但在 **chunk 级 token 截断** 时排在较后，**前 14 段累计已超过 `available_chunk_tokens`**，故被裁掉。
- 将 `MAX_TOTAL_TOKENS` 提到 **200000** 且保持 `MIN_RERANK_SCORE=0.22` 时，最终可保留 **19** 条且含 `chunk-0d30…`，进一步说明 **瓶颈是总预算下的 chunk 截断**，而非「从未进入 rerank 后集合」。
- 将 **`MIN_RERANK_SCORE=0`** 且 **`MAX_TOTAL_TOKENS=40000`** 时，若仍只有 **14** 条且不含 `0d30`，说明在 **不过滤 rerank 分数** 的情况下，**30 条仍被 token 前缀截成 14**，`0d30` 在 rerank 排序中偏后，同样被截断——**再次印证主因是 chunk 可用 token 不足**。

---

## 5. 参数层面的结论与当前配置

- **有效手段**：提高 **`MAX_TOTAL_TOKENS`**，使 `available_chunk_tokens` 能覆盖 rerank 后仍保留的 chunk 序列中更靠后的条目（含 `chunk-0d30…`）。
- **扫描结果（单次进程内改 `QueryParam(max_total_tokens=…)`）**：在 **`MAX_ENTITY_TOKENS=5000`、`MAX_RELATION_TOKENS=6500`** 不变的前提下，**`max_total_tokens=48000`** 起即可在最终列表中出现 `chunk-0d30…`（当次最终 chunk 条数约 18，随检索与 rerank 结果略有变化）。
- **当前 `.env`（摘录，无密钥）**：为留余量，将 **`MAX_TOTAL_TOKENS=52000`**，并保留 **`MIN_RERANK_SCORE=0.22`**、`TOP_K=20`、`CHUNK_TOP_K=30` 等原有检索侧设置；`COSINE_THRESHOLD` 以当时 `.env` 为准（例如 **0.33**）。

验证示例（生成 UTF-8 报告）：

```text
uv run python scripts/dump_query_context.py -w .\rag_storage_run "四种封边机的电控板的保养周期分别是多久" --out docs/query_context_verify_0d30.txt
```

在调高 `MAX_TOTAL_TOKENS` 后的报告中应能看到 **`chunk_id='chunk-0d30e1b587eb2728343d0f3b5f31ae1c'`**；`final_chunks_count` 可能大于此前的 14（取决于 rerank 是否再滤掉部分 chunk），**目标应以「含目标 chunk」为准，而非固定条数**。

---

## 6. 若需「更少 chunk 但仍保留 0d30」

在总 token 不变时，可适当**压低** `MAX_ENTITY_TOKENS` / `MAX_RELATION_TOKENS`，让 KG 占用变小，从而**相对提高** `available_chunk_tokens`；会削弱图谱侧信息量，需按业务权衡。

---

## 7. 相关代码位置（依赖版本：当前 venv 内 `lightrag`）

- **`lightrag/operate.py`**：`_build_context_str` 中计算 `available_chunk_tokens` 并调用 `process_chunks_unified`。
- **`lightrag/utils.py`**：`process_chunks_unified`、`truncate_list_by_token_size`、`apply_rerank_if_enabled`。

升级 LightRAG 后若行为变化，应以当时源码为准。

---

## 8. 文档与产物索引

| 文件 | 说明 |
|------|------|
| `docs/merged_pre_rerank_chunk_ids.txt` | 合并后、rerank 前的 chunk id 列表（由 capture 脚本生成） |
| `docs/query_context_verify_0d30.txt` | 调高 `MAX_TOTAL_TOKENS` 后的一次 dump 验证样例 |
| `scripts/capture_merged_chunk_ids.py` |  monkeypatch `_merge_all_chunks` 写 id |
| `scripts/dump_query_context.py` | 走 `aquery_data`，导出最终上下文中的 chunk |

---

*记录整理自本仓库上的调试会话；日期以仓库变更或本文件创建时为准。*
