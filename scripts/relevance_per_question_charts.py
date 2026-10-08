#!/usr/bin/env python3
"""Per-question relevance score charts (query / high / low) + low×high scatter by category."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

SET_COLORS = {
    "shili17": "#4da3ff",
    "voice29": "#f0a04b",
}


def _load_cases(path: Path, label: str) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        s = o["scores"]
        c = o["case"]
        cat = (c.get("category") or "").strip()
        if label == "shili17":
            cat_display = "测试例17"
        else:
            cat_display = cat or "未分类"
        rows.append(
            {
                "set": label,
                "id": str(c.get("id", "")),
                "query": (c.get("query") or "").strip(),
                "category": cat,
                "cat": cat_display,
                "q": round(float(s["query"]["max_cosine_similarity"]), 4),
                "h": round(float(s["high_level"]["max_cosine_similarity"]), 4),
                "l": round(float(s["low_level"]["max_cosine_similarity"]), 4),
            }
        )
    return rows


def _build_html(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    set_colors_json = json.dumps(SET_COLORS, ensure_ascii=False)
    bar_cat_colors = {
        "测试例17": "#4da3ff",
        "封边": "#f0a04b",
        "所有": "#e07a7a",
        "电脑锯": "#5ecf8f",
        "排钻": "#c77dff",
        "数控": "#7b8cff",
        "未分类": "#9aa3b2",
    }
    bar_cats_json = json.dumps(bar_cat_colors, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>逐题 Relevance 分数 · low×high 散点</title>
<style>
  :root {{
    --bg:#0f1117; --panel:#181b24; --text:#e8eaef; --muted:#9aa3b2; --grid:#2a2f3d;
    --q:#4da3ff; --h:#5ecf8f; --l:#e07a7a;
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;font-family:"Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--text)}}
  .wrap{{max-width:1240px;margin:0 auto;padding:24px 20px 48px}}
  h1{{font-size:1.4rem;margin:0 0 6px;font-weight:600}}
  .sub{{color:var(--muted);font-size:.86rem;margin-bottom:20px}}
  section{{background:var(--panel);border:1px solid var(--grid);border-radius:8px;padding:16px 18px;margin-bottom:18px}}
  section h2{{font-size:1rem;margin:0 0 4px}}
  .cap{{color:var(--muted);font-size:.78rem;margin-bottom:10px}}
  svg{{display:block;width:100%;height:auto}}
  table{{width:100%;border-collapse:collapse;font-size:.82rem}}
  th,td{{border-bottom:1px solid var(--grid);padding:7px 6px;text-align:left;vertical-align:top}}
  th{{color:var(--muted);font-weight:500;position:sticky;top:0;background:var(--panel)}}
  .num{{font-variant-numeric:tabular-nums;text-align:right}}
  .low-q{{color:#f0a04b}} .low-h{{color:#c77dff}}
  .legend{{display:flex;flex-wrap:wrap;gap:12px;margin-top:10px;font-size:.8rem;color:var(--muted)}}
  .legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}}
  .tbl-wrap{{max-height:520px;overflow:auto;border:1px solid var(--grid);border-radius:6px}}
  .bar-q{{color:var(--q)}} .bar-h{{color:var(--h)}} .bar-l{{color:var(--l)}}
  #tooltip{{
    position:fixed;display:none;pointer-events:none;z-index:99;
    background:#252a36;border:1px solid #3d4455;border-radius:6px;padding:8px 10px;
    font-size:12px;max-width:320px;line-height:1.45;box-shadow:0 4px 16px #0006
  }}
</style>
</head>
<body>
<div id="tooltip"></div>
<div class="wrap">
  <h1>逐题分数分布 · low × high 散点</h1>
  <p class="sub">bge-m3 chunk cosine · 散点仅标题号 · 蓝=测试例17（Q）· 橙=技术话术29（#）</p>

  <section>
    <h2>low_level 分（x）× high_level 分（y）</h2>
    <p class="cap">每题一点 · 仅显示题号 Q1–Q17 / #1–#29 · 悬停可看三分</p>
    <svg id="scatter-lh" viewBox="0 0 960 480" role="img"></svg>
    <div class="legend" id="set-legend"></div>
  </section>

  <section>
    <h2>测试例 17 — 逐题 query / high / low</h2>
    <p class="cap">每行一题 · 蓝=query · 绿=high · 红=low · 柱末数字为 cosine 分</p>
    <svg id="bars-shili" viewBox="0 0 960 560" role="img"></svg>
  </section>

  <section>
    <h2>技术话术 29 — 逐题 query / high / low</h2>
    <p class="cap">按类别着色标签 · 同样三色柱为三路检索分</p>
    <svg id="bars-voice" viewBox="0 0 960 900" role="img"></svg>
  </section>

  <section>
    <h2>全量分数表（46 题）</h2>
    <p class="cap">可排序 · 点击表头 · 黄/紫标出 query 与 high/low 差较大的失真项</p>
    <div class="tbl-wrap">
      <table id="score-table">
        <thead><tr>
          <th data-k="set">数据集</th>
          <th data-k="id">#</th>
          <th data-k="cat">类别</th>
          <th data-k="query">题目</th>
          <th data-k="q" class="num">query</th>
          <th data-k="h" class="num">high</th>
          <th data-k="l" class="num">low</th>
          <th>备注</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>
</div>
<script>
const DATA = {data_json};
const SET_COLORS = {set_colors_json};
const BAR_CAT_COLORS = {bar_cats_json};
const SCORE_LO = 0.35, SCORE_HI = 0.95;
const Q_COL = '#4da3ff', H_COL = '#5ecf8f', L_COL = '#e07a7a';

function fmt(v) {{ return Number(v).toFixed(3); }}

function xLin(v, W, ml, mr, lo=SCORE_LO, hi=SCORE_HI) {{
  return ml + (v-lo)/(hi-lo)*(W-ml-mr);
}}
function yLin(v, H, mt, mb, lo=SCORE_LO, hi=SCORE_HI) {{
  return H - mb - (v-lo)/(hi-lo)*(H-mt-mb);
}}

function note(p) {{
  const parts = [];
  if (Math.abs(p.h - p.q) > 0.10) parts.push('high失真');
  if (Math.abs(p.l - p.q) > 0.08) parts.push('low失真');
  if (p.q < 0.60) parts.push('query偏低');
  return parts.join(', ');
}}

function renderSetLegend() {{
  document.getElementById('set-legend').innerHTML = [
    ['shili17', '测试例 17（Q1–Q17）'],
    ['voice29', '技术话术 29（#1–#29）'],
  ].map(([k, lbl]) =>
    `<span><i style="background:${{SET_COLORS[k]}}"></i>${{lbl}}</span>`
  ).join('');
}}

function attachTooltip(svg) {{
  const tip = document.getElementById('tooltip');
  svg.querySelectorAll('[data-tip]').forEach(el => {{
    el.addEventListener('mouseenter', e => {{
      tip.innerHTML = el.getAttribute('data-tip');
      tip.style.display = 'block';
    }});
    el.addEventListener('mousemove', e => {{
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top = (e.clientY + 14) + 'px';
    }});
    el.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});
  }});
}}

function drawScatterLH() {{
  const svg = document.getElementById('scatter-lh');
  const W=960, H=480, ml=56, mr=24, mt=20, mb=48;
  let g = '';
  for (let t = 0.4; t <= 0.95; t += 0.1) {{
    const x = xLin(t, W, ml, mr);
    const y = yLin(t, H, mt, mb);
    g += `<line x1="${{x}}" y1="${{mt}}" x2="${{x}}" y2="${{H-mb}}" stroke="#252a36"/>`;
    g += `<line x1="${{ml}}" y1="${{y}}" x2="${{W-mr}}" y2="${{y}}" stroke="#252a36"/>`;
    g += `<text x="${{x}}" y="${{H-8}}" text-anchor="middle" fill="#9aa3b2" font-size="11">${{t.toFixed(1)}}</text>`;
    g += `<text x="${{ml-8}}" y="${{y+4}}" text-anchor="end" fill="#9aa3b2" font-size="11">${{t.toFixed(1)}}</text>`;
  }}
  g += `<line x1="${{xLin(SCORE_LO,W,ml,mr)}}" y1="${{yLin(SCORE_LO,H,mt,mb)}}" x2="${{xLin(SCORE_HI,W,ml,mr)}}" y2="${{yLin(SCORE_HI,H,mt,mb)}}" stroke="#3d4455" stroke-dasharray="5 4"/>`;
  g += `<text x="${{W/2}}" y="${{H-2}}" text-anchor="middle" fill="#9aa3b2" font-size="12">low_level cosine（x）</text>`;
  g += `<text x="16" y="${{H/2}}" transform="rotate(-90 16 ${{H/2}})" text-anchor="middle" fill="#9aa3b2" font-size="12">high_level cosine（y）</text>`;

  DATA.points.forEach(p => {{
    const col = SET_COLORS[p.set] || '#888';
    const cx = xLin(p.l, W, ml, mr);
    const cy = yLin(p.h, H, mt, mb);
    const label = p.set === 'shili17' ? `Q${{p.id}}` : `#${{p.id}}`;
    const tip = `${{label}}<br/>query=${{fmt(p.q)}} high=${{fmt(p.h)}} low=${{fmt(p.l)}}`;
    g += `<circle cx="${{cx}}" cy="${{cy}}" r="7" fill="${{col}}" opacity="0.88" data-tip="${{tip}}"/>`;
    g += `<text x="${{cx}}" y="${{cy-10}}" text-anchor="middle" fill="${{col}}" font-size="10" font-weight="500">${{label}}</text>`;
  }});
  svg.innerHTML = g;
  attachTooltip(svg);
}}

function drawPerQuestionBars(svgId, cases, idPrefix) {{
  const svg = document.getElementById(svgId);
  const W=960;
  const rowH = 26;
  const ml = 268, mr = 52, mt = 28, mb = 24;
  const H = mt + mb + cases.length * rowH;
  svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`);

  let g = '';
  g += `<text x="${{ml}}" y="16" fill="#9aa3b2" font-size="11">题目</text>`;
  g += `<text x="${{W-mr}}" y="16" text-anchor="end" fill="${{Q_COL}}" font-size="10">■query</text>`;
  g += `<text x="${{W-mr-52}}" y="16" text-anchor="end" fill="${{H_COL}}" font-size="10">■high</text>`;
  g += `<text x="${{W-mr-104}}" y="16" text-anchor="end" fill="${{L_COL}}" font-size="10">■low</text>`;

  for (let t = 0.4; t <= 0.95; t += 0.1) {{
    const x = xLin(t, W, ml, mr);
    g += `<line x1="${{x}}" y1="${{mt}}" x2="${{x}}" y2="${{H-mb}}" stroke="#222830"/>`;
    if (t >= 0.5) g += `<text x="${{x}}" y="${{H-6}}" text-anchor="middle" fill="#666" font-size="9">${{t.toFixed(1)}}</text>`;
  }}
  [0.33, 0.6, 0.7].forEach(t => {{
    const x = xLin(t, W, ml, mr);
    g += `<line x1="${{x}}" y1="${{mt}}" x2="${{x}}" y2="${{H-mb}}" stroke="#555" stroke-dasharray="3 3" opacity="0.5"/>`;
  }});

  cases.forEach((p, i) => {{
    const y0 = mt + i * rowH;
    const prefix = idPrefix;
    const lbl = p.query.length > 22 ? p.query.slice(0, 22) + '…' : p.query;
    const catTag = p.set === 'voice29' ? ` [${{p.cat}}]` : '';
    g += `<text x="8" y="${{y0+14}}" fill="${{BAR_CAT_COLORS[p.cat]||'#ccc'}}" font-size="11">${{prefix}}${{p.id}}${{catTag}}</text>`;
    g += `<text x="52" y="${{y0+14}}" fill="#9aa3b2" font-size="10">${{lbl}}</text>`;

    const bars = [
      {{ v: p.q, col: Q_COL, dy: 4 }},
      {{ v: p.h, col: H_COL, dy: 11 }},
      {{ v: p.l, col: L_COL, dy: 18 }},
    ];
    bars.forEach(b => {{
      const x0 = xLin(SCORE_LO, W, ml, mr);
      const x1 = xLin(b.v, W, ml, mr);
      const bw = Math.max(1, x1 - x0);
      g += `<rect x="${{x0}}" y="${{y0+b.dy}}" width="${{bw}}" height="5" fill="${{b.col}}" opacity="0.85" rx="1"/>`;
      g += `<text x="${{x1+3}}" y="${{y0+b.dy+4}}" fill="${{b.col}}" font-size="9">${{fmt(b.v)}}</text>`;
    }});
    g += `<line x1="${{ml}}" y1="${{y0+rowH-2}}" x2="${{W-mr}}" y2="${{y0+rowH-2}}" stroke="#1e222c"/>`;
  }});
  svg.innerHTML = g;
}}

function renderTable() {{
  const tbody = document.querySelector('#score-table tbody');
  let rows = [...DATA.points];
  const render = () => {{
    tbody.innerHTML = rows.map(p => {{
      const n = note(p);
      const clsQ = p.q < 0.6 ? 'low-q' : '';
      const clsH = Math.abs(p.h-p.q)>0.1 ? 'low-h' : '';
      const clsL = Math.abs(p.l-p.q)>0.08 ? 'low-h' : '';
      const pre = p.set==='shili17'?'Q':'#';
      return `<tr>
        <td>${{p.set==='shili17'?'测试例17':'技术话术'}}</td>
        <td>${{pre}}${{p.id}}</td>
        <td>${{p.cat}}</td>
        <td>${{p.query}}</td>
        <td class="num ${{clsQ}}">${{fmt(p.q)}}</td>
        <td class="num ${{clsH}}">${{fmt(p.h)}}</td>
        <td class="num ${{clsL}}">${{fmt(p.l)}}</td>
        <td>${{n}}</td>
      </tr>`;
    }}).join('');
  }};
  render();
  document.querySelectorAll('#score-table th[data-k]').forEach(th => {{
    th.style.cursor = 'pointer';
    th.addEventListener('click', () => {{
      const k = th.dataset.k;
      const desc = th.dataset.desc !== '1';
      rows.sort((a,b) => {{
        const va = a[k], vb = b[k];
        if (typeof va === 'number') return desc ? vb-va : va-vb;
        return desc ? String(vb).localeCompare(String(va),'zh') : String(va).localeCompare(String(vb),'zh');
      }});
      document.querySelectorAll('#score-table th').forEach(h => delete h.dataset.desc);
      th.dataset.desc = desc ? '0' : '1';
      render();
    }});
  }});
}}

renderSetLegend();
drawScatterLH();
drawPerQuestionBars('bars-shili', DATA.shiliCases, 'Q');
drawPerQuestionBars('bars-voice', DATA.voiceCases, '#');
renderTable();
</script>
</body>
</html>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shili", type=Path, default=_ROOT / "logs" / "relevance_shili17.jsonl")
    p.add_argument("--voice", type=Path, default=_ROOT / "logs" / "relevance_voice_tech29.jsonl")
    p.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "logs" / "relevance_per_question_charts.html",
    )
    args = p.parse_args()

    shili = _load_cases(args.shili.resolve(), "shili17")
    voice = _load_cases(args.voice.resolve(), "voice29")
    payload = {
        "shiliCases": shili,
        "voiceCases": voice,
        "points": shili + voice,
    }

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_build_html(payload), encoding="utf-8")
    print(f"Wrote: {out.as_posix()}")


if __name__ == "__main__":
    main()
