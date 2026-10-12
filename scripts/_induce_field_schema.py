"""临时分析：验证"字段模式归纳"能否零 hardcode 地发现各手册的字段键与角色。

对 data/pipeline_parse 下每份 content_list JSON：
1. 收集所有 "键：值" 形式的行首键（键长 2-8，无空白/标点）
2. 统计每个键的：出现次数 / 后随编号步骤行占比 / 值短占比
3. 输出推断的角色，与现有 hardcode（保养周期/保养内容/保养步骤/操作步骤）对照
"""

from __future__ import annotations

import collections
import io
import json
import re
from pathlib import Path

KEY_RE = re.compile(r"^([^\s：:，。、,.]{2,8})：")
ENUM_RE = re.compile(r"^\s*(\d+[\.、．)）]|[①-⑳]|[(（]\d+[)）])")
SENT_PUNCT_RE = re.compile(r"[。！？]")

root = Path("data/pipeline_parse")
out = io.open("logs/_tmp_schema_induce.txt", "w", encoding="utf-8")

seen_docs = set()
for cl in sorted(root.rglob("*content_list*.json")):
    doc = cl.parts[2].rsplit("_", 1)[0]
    if doc in seen_docs:
        continue
    seen_docs.add(doc)
    try:
        items = json.loads(cl.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        out.write(f"== {doc}: LOAD FAIL {exc}\n")
        continue
    texts = [str(it.get("text") or "") for it in items if it.get("type") == "text"]
    keys: collections.Counter[str] = collections.Counter()
    follow_enum: collections.Counter[str] = collections.Counter()
    short_val: collections.Counter[str] = collections.Counter()
    for i, t in enumerate(texts):
        stripped = t.strip()
        first = stripped.splitlines()[0] if stripped else ""
        m = KEY_RE.match(first)
        if not m:
            continue
        k = m.group(1)
        keys[k] += 1
        val = first[m.end() :].strip()
        rest = "\n".join(stripped.splitlines()[1:]).strip()
        nxt = texts[i + 1].strip() if i + 1 < len(texts) else ""
        if ENUM_RE.match(rest or nxt) or ENUM_RE.search(val):
            follow_enum[k] += 1
        if len(val) <= 30 and not SENT_PUNCT_RE.search(val):
            short_val[k] += 1
    out.write(f"== {doc} (text items={len(texts)}, path={cl})\n")
    for k, c in keys.most_common(15):
        enum_ratio = follow_enum[k] / c
        short_ratio = short_val[k] / c
        # 推断角色：重复>=3 才算 schema 字段；enum 占比高 => procedure
        if c < 3:
            role = "(noise: n<3)"
        elif enum_ratio >= 0.5:
            role = "PROCEDURE"
        else:
            role = "METADATA"
        out.write(
            f"  {k}: n={c} enum_follow={follow_enum[k]}({enum_ratio:.0%}) "
            f"short_val={short_val[k]}({short_ratio:.0%}) -> {role}\n"
        )
out.close()
print("done, docs:", len(seen_docs))
