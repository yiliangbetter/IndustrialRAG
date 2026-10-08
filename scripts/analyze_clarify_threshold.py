#!/usr/bin/env python3
"""Analyze shili17 vs green8 naive scores for clarification gate calibration."""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, label: str) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        s = o["scores"]
        c = o["case"]
        q = float(s["query"]["max_cosine_similarity"])
        h = float(s["high_level"]["max_cosine_similarity"])
        l = float(s["low_level"]["max_cosine_similarity"])
        rows.append(
            {
                "set": label,
                "id": c.get("id"),
                "query": (c.get("query") or "")[:60],
                "q": q,
                "h": h,
                "l": l,
                "dh": h - q,
                "dl": l - q,
                "min_qlh": min(q, h, l),
                "max_qlh": max(q, h, l),
                "spread": max(q, h, l) - min(q, h, l),
                "avg_qlh": (q + h + l) / 3,
            }
        )
    return rows


def _stats(vals: list[float]) -> dict[str, float]:
    s = sorted(vals)
    n = len(s)
    return {
        "min": s[0],
        "max": s[-1],
        "avg": sum(s) / n,
        "med": s[n // 2],
    }


def _clarify_rule(r: dict, *, qt: float, spread_t: float, distort: bool) -> bool:
    if r["q"] < qt:
        return True
    if r["min_qlh"] < qt:
        return True
    if r["spread"] > spread_t:
        return True
    if distort and (abs(r["dh"]) > 0.10 or abs(r["dl"]) > 0.08):
        return True
    return False


def main() -> None:
    shili = _load(_ROOT / "logs" / "relevance_shili17.jsonl", "shili17")
    green = _load(_ROOT / "logs" / "relevance_voice_green8.jsonl", "green8")

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("澄清门控校准分析 — shili17（不澄清）vs green8（需澄清）")
    lines.append("=" * 72)
    lines.append("")

    lines.append("一、分布对比")
    lines.append("-" * 72)
    for key in ("q", "h", "l", "min_qlh", "spread", "dh", "dl"):
        ss = _stats([r[key] for r in shili])
        gs = _stats([r[key] for r in green])
        lines.append(
            f"{key:10s}  shili17 min={ss['min']:.3f} avg={ss['avg']:.3f} max={ss['max']:.3f}"
            f"  |  green8 min={gs['min']:.3f} avg={gs['avg']:.3f} max={gs['max']:.3f}"
        )
    lines.append("")
    lines.append("重叠区（query）：")
    lines.append(
        f"  shili17 [{min(r['q'] for r in shili):.3f}, {max(r['q'] for r in shili):.3f}]"
    )
    lines.append(
        f"  green8  [{min(r['q'] for r in green):.3f}, {max(r['q'] for r in green):.3f}]"
    )
    lo = max(min(r["q"] for r in shili), min(r["q"] for r in green))
    hi = min(max(r["q"] for r in shili), max(r["q"] for r in green))
    lines.append(f"  交叠     [{lo:.3f}, {hi:.3f}]  ← 单靠 query 无法完全分开")
    lines.append("")

    lines.append("二、逐条分数")
    lines.append("-" * 72)
    lines.append("shili17（目标：不澄清）")
    for r in sorted(shili, key=lambda x: x["q"]):
        lines.append(
            f"  Q{r['id']:>2} q={r['q']:.3f} h={r['h']:.3f} l={r['l']:.3f}"
            f"  spread={r['spread']:.3f}  dh={r['dh']:+.3f} dl={r['dl']:+.3f}"
        )
    lines.append("green8（目标：澄清）")
    for r in green:
        lines.append(
            f"  #{r['id']:>2} q={r['q']:.3f} h={r['h']:.3f} l={r['l']:.3f}"
            f"  spread={r['spread']:.3f}  dh={r['dh']:+.3f} dl={r['dl']:+.3f}"
        )
    lines.append("")

    lines.append("三、单阈值能否分开？")
    lines.append("-" * 72)
    lines.append("规则：query < T → 需澄清")
    for t in [x / 100 for x in range(58, 78)]:
        sc = sum(1 for r in shili if r["q"] < t)
        gc = sum(1 for r in green if r["q"] < t)
        mark = ""
        if sc == 0 and gc == 8:
            mark = "  ← 完美"
        elif sc == 0:
            mark = f"  ← shili 全过，green 仅 {gc}/8"
        lines.append(f"  T={t:.2f}  shili误澄清={sc}/17  green命中={gc}/8{mark}")
    lines.append("")
    lines.append("结论：不存在 query 阈值使 shili=0 且 green=8（绿标含 0.854 气压报警，shili 最高 0.902）")
    lines.append("")

    lines.append("四、组合规则网格搜索")
    lines.append("-" * 72)
    lines.append("候选：澄清 if (q<T) OR (min(q,h,l)<T) OR (spread>S) OR (关键词失真)")
    perfect: list[str] = []
    near: list[str] = []
    for qt in [0.62, 0.65, 0.68, 0.70, 0.72]:
        for spread_t in [0.06, 0.08, 0.10, 0.12, 0.15]:
            for distort in (False, True):
                sc = sum(
                    _clarify_rule(r, qt=qt, spread_t=spread_t, distort=distort)
                    for r in shili
                )
                gc = sum(
                    _clarify_rule(r, qt=qt, spread_t=spread_t, distort=distort)
                    for r in green
                )
                rule = f"q<{qt:.2f}|min<{qt:.2f}|spread>{spread_t:.2f}|distort={distort}"
                if sc == 0 and gc == 8:
                    perfect.append(f"  {rule}  → shili=0 green=8")
                elif sc <= 1 and gc >= 7:
                    near.append(f"  {rule}  → shili={sc} green={gc}")
    lines.append("完美分离（0/17 + 8/8）：")
    lines.extend(perfect[:20] if perfect else ["  （无）"])
    lines.append("近似（shili≤1, green≥7）：")
    lines.extend(near[:12] if near else ["  （无）"])
    lines.append("")

    lines.append("五、推荐澄清策略（基于当前 25 条标注样本）")
    lines.append("-" * 72)
    lines.append(
        "1) 不能单靠 query cosine 与 COSINE_THRESHOLD(0.33) — 两批全过线，无区分度。"
    )
    lines.append(
        "2) 不能单靠 query 高分阈值 — green8 含 0.854，与 shili17 大量重叠在 0.65–0.85。"
    )
    lines.append("3) 可行方向（需组合 + 问法形态，而非纯向量）：")
    lines.append("")
    lines.append("   A. 问法形态门控（优先，与 Excel 绿标一致）")
    lines.append("      · 完整问句 + 明确设备/部件/动作 → 倾向不澄清（shili17 模式）")
    lines.append("      · 短句/报警码/口语碎片（≤N 字、无问号结构）→ 倾向澄清（green8 模式）")
    lines.append("      · green8 平均题干远短于 shili17，这是最强先验")
    lines.append("")
    lines.append("   B. naive 向量辅助信号（在形态门控之后或叠加）")
    lines.append("      · primary = query 分；辅助看 spread=max(q,h,l)-min(q,h,l)")
    lines.append("      · spread > 0.10 或 |high-query|>0.10 或 |low-query|>0.08 → 澄清候选")
    lines.append("        shili 误伤：Q8(high失真), Q7/Q2(low失真) 共 3 条；green 命中约 5/8")
    lines.append("      · query < 0.70 且 min(q,h,l) < 0.65 → 澄清候选")
    lines.append("        可抓到 green 漏胶/三相电等，但 shili Q16(0.658) 会误澄清")
    lines.append("")
    lines.append("   C. 业务白名单 / 绿标名单（最稳）")
    lines.append("      · voice_script_green8.json 标准问法 → 强制走澄清流程")
    lines.append("      · 完整测试例/手册问句模式 → 跳过澄清")
    lines.append("")
    lines.append("   D. 推荐默认 pipeline（澄清前）")
    lines.append("      Step1: 若在 green8/口语短句库 → clarify")
    lines.append("      Step2: elif 问句长度≥20 且含「怎么/哪些/多久/步骤」等完整问法 → pass")
    lines.append("      Step3: elif query<0.70 OR spread>0.12 OR keyword_distortion → clarify")
    lines.append("      Step4: else pass")
    lines.append("")

    # evaluate recommended pipeline D on our data
    def pipeline_d(r: dict, *, green_ids: set[str]) -> bool:
        if r["set"] == "green8":
            return True
        if len(r.get("query", "")) >= 20 and any(
            w in r.get("query", "") for w in ("怎么", "哪些", "多久", "步骤", "适用于", "要不要")
        ):
            return False
        if r["q"] < 0.70 or r["spread"] > 0.12:
            return True
        if abs(r["dh"]) > 0.10 or abs(r["dl"]) > 0.08:
            return True
        return False

    sc = sum(pipeline_d(r, green_ids=set()) for r in shili)
    gc = sum(pipeline_d(r, green_ids=set()) for r in green)
    lines.append("六、示例 pipeline D 回放（green8 全强制澄清）")
    lines.append(f"  shili误澄清: {sc}/17  green命中: {gc}/8")
    for r in shili:
        if pipeline_d(r, green_ids=set()):
            lines.append(f"    shili误伤 Q{r['id']} q={r['q']:.3f} {r['query'][:40]}")
    lines.append("")

    out = _ROOT / "logs" / "clarify_threshold_analysis.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.as_posix())
    print("\n".join(lines[:30]))
    print("...")


if __name__ == "__main__":
    main()
