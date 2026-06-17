#!/usr/bin/env python3
"""Generate a self-contained HTML chart report for naive relevance scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_cases(path: Path, label: str) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        s = o["scores"]
        c = o["case"]
        rows.append(
            {
                "set": label,
                "id": str(c.get("id", "")),
                "query": (c.get("query") or "").strip(),
                "category": (c.get("category") or "").strip(),
                "q": float(s["query"]["max_cosine_similarity"]),
                "h": float(s["high_level"]["max_cosine_similarity"]),
                "l": float(s["low_level"]["max_cosine_similarity"]),
            }
        )
    return rows


def _stats(vals: list[float]) -> dict:
    s = sorted(vals)
    n = len(s)
    return {
        "min": s[0],
        "max": s[-1],
        "avg": sum(s) / n,
        "med": s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2,
        "p25": s[max(0, n // 4 - 1)],
        "p75": s[min(n - 1, (3 * n) // 4)],
    }


def _build_html(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Naive Relevance 分数分布与交叠</title>
<style>
  :root {{
    --bg: #0f1117; --panel: #181b24; --text: #e8eaef; --muted: #9aa3b2;
    --shili: #4da3ff; --voice: #f0a04b; --high: #5ecf8f; --low: #e07a7a;
    --line33: #666; --line60: #c9a227; --line70: #7b8cff;
    --grid: #2a2f3d;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 24px 20px 48px; }}
  h1 {{ font-size: 1.45rem; font-weight: 600; margin: 0 0 6px; }}
  .sub {{ color: var(--muted); font-size: 0.88rem; margin-bottom: 22px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 8px; padding: 14px 16px; }}
  .card .v {{ font-size: 1.35rem; font-weight: 600; }}
  .card .k {{ color: var(--muted); font-size: 0.82rem; }}
  section {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 8px; padding: 16px 18px 12px; margin-bottom: 18px; }}
  section h2 {{ font-size: 1rem; margin: 0 0 4px; font-weight: 600; }}
  .cap {{ color: var(--muted); font-size: 0.78rem; margin-bottom: 10px; }}
  svg {{ display: block; width: 100%; height: auto; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 8px; font-size: 0.82rem; color: var(--muted); }}
  .legend span::before {{ content: ""; display: inline-block; width: 10px; height: 10px; margin-right: 6px; border-radius: 2px; vertical-align: -1px; }}
  .lg-shili::before {{ background: var(--shili); }}
  .lg-voice::before {{ background: var(--voice); }}
  .lg-high::before {{ background: var(--high); }}
  .lg-low::before {{ background: var(--low); }}
  .lg-33::before {{ background: var(--line33); }}
  .lg-60::before {{ background: var(--line60); }}
  .lg-70::before {{ background: var(--line70); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
  th, td {{ border-bottom: 1px solid var(--grid); padding: 8px 6px; text-align: left; }}
  th {{ color: var(--muted); font-weight: 500; }}
  .tip {{ fill: var(--text); font-size: 11px; pointer-events: none; opacity: 0; transition: opacity .12s; }}
  .dot:hover + .tip, .dot:focus + .tip {{ opacity: 1; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Naive Relevance — 分数分布与交叠</h1>
  <p class="sub">Source: relevance_shili17.jsonl · relevance_voice_tech29.jsonl · bge-m3 chunk cosine · probe_top_k=20</p>
  <div class="cards" id="stats"></div>

  <section>
    <h2>query 分分布（重叠直方图）</h2>
    <p class="cap">横轴：query cosine · 纵轴：条数 · 半透明柱展示两套数据在同一分数区间的交叠</p>
    <svg id="hist-query" viewBox="0 0 900 300" role="img" aria-label="query score histogram"></svg>
    <div class="legend">
      <span class="lg-shili">测试例 17</span>
      <span class="lg-voice">技术话术 29</span>
      <span class="lg-33">阈值 0.33</span>
      <span class="lg-60">阈值 0.60</span>
      <span class="lg-70">阈值 0.70</span>
      <span style="color:#9aa3b2">▮ 淡紫区域 = 两套 query 分共同覆盖区间</span>
    </div>
  </section>

  <section>
    <h2>累积分布（ECDF）— 交叠程度</h2>
    <p class="cap">纵轴：≤该分数的用例占比 · 两曲线越近表示分布越相似；竖线标阈值下通过率差异</p>
    <svg id="ecdf-query" viewBox="0 0 900 300" role="img" aria-label="ECDF"></svg>
  </section>

  <section>
    <h2>query / high / low 箱线对比</h2>
    <p class="cap">每数据集三根箱线（min–p25–med–p75–max）· 直观看三路检索分是否分离</p>
    <svg id="boxplot" viewBox="0 0 900 320" role="img" aria-label="box plot"></svg>
  </section>

  <section>
    <h2>散点交叠：query vs high_level</h2>
    <p class="cap">点落在对角线附近 = high 与 query 一致 · 偏离对角线 = 关键词失真 · 悬停看用例</p>
    <svg id="scatter-qh" viewBox="0 0 900 340" role="img" aria-label="query vs high scatter"></svg>
  </section>

  <section>
    <h2>散点交叠：query vs low_level</h2>
    <p class="cap">同上，观察 low 关键词是否带偏检索</p>
    <svg id="scatter-ql" viewBox="0 0 900 340" role="img" aria-label="query vs low scatter"></svg>
  </section>

  <section>
    <h2>密度脊线（query 分）</h2>
    <p class="cap">核密度估计 · 峰值位置与宽度反映分数集中区间及两套分布重叠区</p>
    <svg id="ridge" viewBox="0 0 900 220" role="img" aria-label="density ridge"></svg>
  </section>

  <section>
    <h2>阈值下通过数对比</h2>
    <table id="thresh-table"><thead><tr><th>阈值</th><th>测试例17 通过</th><th>技术话术29 通过</th><th>合计</th></tr></thead><tbody></tbody></table>
  </section>
</div>
<script>
const DATA = {data_json};

const COLORS = {{ shili17: '#4da3ff', voice29: '#f0a04b', high: '#5ecf8f', low: '#e07a7a' }};
const THRESH = [0.33, 0.55, 0.6, 0.7];
const MARGIN = {{ l: 52, r: 18, t: 16, b: 42 }};

function fmt(x) {{ return x.toFixed(3); }}

function stats(vals) {{
  const s = [...vals].sort((a,b)=>a-b);
  const n = s.length;
  const q = (p) => s[Math.min(n-1, Math.max(0, Math.floor(p*(n-1))))];
  return {{ min:s[0], p25:q(0.25), med:q(0.5), p75:q(0.75), max:s[n-1], avg:s.reduce((a,b)=>a+b,0)/n }};
}}

function xScale(v, w, lo=0.48, hi=0.96) {{
  return MARGIN.l + (v-lo)/(hi-lo)*(w - MARGIN.l - MARGIN.r);
}}
function yScale(v, h, ymax) {{
  return h - MARGIN.b - (v/ymax)*(h - MARGIN.t - MARGIN.b);
}}

function renderStats() {{
  const el = document.getElementById('stats');
  const items = [
    ['总用例', DATA.points.length],
    ['测试例17 query 均值', fmt(stats(DATA.shili17.q).avg)],
    ['技术话术29 query 均值', fmt(stats(DATA.voice29.q).avg)],
    ['query 分数交叠区间', fmt(DATA.overlap.q_min) + ' – ' + fmt(DATA.overlap.q_max)],
  ];
  el.innerHTML = items.map(([k,v]) => `<div class="card"><div class="v">${{v}}</div><div class="k">${{k}}</div></div>`).join('');
}}

function histBins(vals, lo=0.52, hi=0.92, bins=20) {{
  const w = (hi-lo)/bins; const c = Array(bins).fill(0);
  vals.forEach(v => {{ let i = Math.floor((v-lo)/w); i = Math.max(0, Math.min(bins-1, i)); c[i]++; }});
  return c.map((n,i) => ({{ x: lo+(i+0.5)*w, n }}));
}}

function drawHist(svgId, series, ymax) {{
  const svg = document.getElementById(svgId);
  const W=900, H=300; const lo=0.52, hi=0.92; const binCount=20;
  const all = series.flatMap(s => histBins(s.vals, lo, hi, binCount));
  const maxN = Math.max(...all.map(b=>b.n), 1);
  ymax = ymax || maxN;
  let g = '';
  if (DATA.overlap) {{
    const x1=xScale(DATA.overlap.q_min,W,lo,hi);
    const x2=xScale(DATA.overlap.q_max,W,lo,hi);
    g += `<rect x="${{x1}}" y="${{MARGIN.t}}" width="${{x2-x1}}" height="${{H-MARGIN.t-MARGIN.b}}" fill="#7b8cff" opacity="0.12"/>`;
    g += `<text x="${{(x1+x2)/2}}" y="${{MARGIN.t+12}}" text-anchor="middle" fill="#9aa3b2" font-size="10">交叠区 ${{fmt(DATA.overlap.q_min)}}–${{fmt(DATA.overlap.q_max)}}</text>`;
  }}
  // grid + x axis
  for (let t=0.55; t<=0.9; t+=0.05) {{
    const x = xScale(t,W,lo,hi);
    g += `<line x1="${{x}}" y1="${{MARGIN.t}}" x2="${{x}}" y2="${{H-MARGIN.b}}" stroke="#2a2f3d"/>`;
    g += `<text x="${{x}}" y="${{H-12}}" text-anchor="middle" fill="#9aa3b2" font-size="11">${{t.toFixed(2)}}</text>`;
  }}
  g += `<text x="${{W/2}}" y="${{H-2}}" text-anchor="middle" fill="#9aa3b2" font-size="12">query cosine similarity</text>`;
  g += `<text x="14" y="${{H/2}}" transform="rotate(-90 14 ${{H/2}})" text-anchor="middle" fill="#9aa3b2" font-size="12">count</text>`;
  THRESH.forEach((t,i) => {{
    const cols=['#666','#888','#c9a227','#7b8cff'];
    const x=xScale(t,W,lo,hi);
    g+=`<line x1="${{x}}" y1="${{MARGIN.t}}" x2="${{x}}" y2="${{H-MARGIN.b}}" stroke="${{cols[i]}}" stroke-dasharray="4 3" stroke-width="1.2"/>`;
  }});
  const barW = (W - MARGIN.l - MARGIN.r) / binCount * 0.38;
  series.forEach((s, si) => {{
    const bins = histBins(s.vals, lo, hi, binCount);
    bins.forEach((b) => {{
      const cx = xScale(b.x, W, lo, hi);
      const x = cx - barW/2 + (si===0?-barW*0.15:barW*0.15);
      const h = (b.n/ymax)*(H-MARGIN.t-MARGIN.b);
      const y = yScale(b.n, H, ymax);
      g += `<rect x="${{x}}" y="${{y}}" width="${{barW}}" height="${{h}}" fill="${{s.color}}" opacity="0.72" rx="2"/>`;
    }});
  }});
  svg.innerHTML = g;
}}

function drawEcdf(svgId) {{
  const svg = document.getElementById(svgId);
  const W=900, H=300; const lo=0.52, hi=0.92;
  const mk = (vals, color, label) => {{
    const s=[...vals].sort((a,b)=>a-b);
    let d = `M ${{xScale(s[0],W,lo,hi)}} ${{yScale(1/s.length,H,1)}}`;
    s.forEach((v,i) => {{ d += ` L ${{xScale(v,W,lo,hi)}} ${{yScale((i+1)/s.length,H,1)}}`; }});
    return `<path d="${{d}}" fill="none" stroke="${{color}}" stroke-width="2.2"/><text x="${{xScale(s[s.length-1],W,lo,hi)-4}}" y="${{yScale(1,H,1)-6}}" text-anchor="end" fill="${{color}}" font-size="12">${{label}}</text>`;
  }};
  let g='';
  for (let p=0; p<=1; p+=0.25) {{
    const y=yScale(p,H,1);
    g+=`<line x1="${{MARGIN.l}}" y1="${{y}}" x2="${{W-MARGIN.r}}" y2="${{y}}" stroke="#2a2f3d"/>`;
    g+=`<text x="${{MARGIN.l-8}}" y="${{y+4}}" text-anchor="end" fill="#9aa3b2" font-size="11">${{(p*100).toFixed(0)}}%</text>`;
  }}
  for (let t=0.55; t<=0.9; t+=0.05) {{
    const x=xScale(t,W,lo,hi);
    g+=`<text x="${{x}}" y="${{H-12}}" text-anchor="middle" fill="#9aa3b2" font-size="11">${{t.toFixed(2)}}</text>`;
  }}
  THRESH.forEach(t => {{
    const x=xScale(t,W,lo,hi);
    g+=`<line x1="${{x}}" y1="${{MARGIN.t}}" x2="${{x}}" y2="${{H-MARGIN.b}}" stroke="#c9a227" stroke-dasharray="4 3"/>`;
  }});
  g += mk(DATA.shili17.q, COLORS.shili17, '测试例17');
  g += mk(DATA.voice29.q, COLORS.voice29, '技术话术29');
  g += `<text x="${{W/2}}" y="${{H-2}}" text-anchor="middle" fill="#9aa3b2" font-size="12">query cosine</text>`;
  g += `<text x="14" y="${{H/2}}" transform="rotate(-90 14 ${{H/2}})" text-anchor="middle" fill="#9aa3b2" font-size="12">cumulative fraction</text>`;
  svg.innerHTML = g;
}}

function drawBox(svgId) {{
  const svg = document.getElementById(svgId);
  const W=900, H=320;
  const groups = [
    {{ label: '测试例17', q: DATA.shili17.q, h: DATA.shili17.h, l: DATA.shili17.l }},
    {{ label: '技术话术29', q: DATA.voice29.q, h: DATA.voice29.h, l: DATA.voice29.l }},
  ];
  const lo=0.4, hi=0.95;
  let g='';
  for (let t=0.4; t<=0.95; t+=0.1) {{
    const y = MARGIN.t + (hi-t)/(hi-lo)*(H-MARGIN.t-MARGIN.b);
    g+=`<line x1="${{MARGIN.l}}" y1="${{y}}" x2="${{W-MARGIN.r}}" y2="${{y}}" stroke="#2a2f3d"/>`;
    g+=`<text x="${{MARGIN.l-8}}" y="${{y+4}}" text-anchor="end" fill="#9aa3b2" font-size="11">${{t.toFixed(1)}}</text>`;
  }}
  const yv = (v) => MARGIN.t + (hi-v)/(hi-lo)*(H-MARGIN.t-MARGIN.b);
  const slot = (W - MARGIN.l - MARGIN.r) / groups.length;
  groups.forEach((grp, gi) => {{
    const cx = MARGIN.l + slot*gi + slot/2;
    g += `<text x="${{cx}}" y="${{H-10}}" text-anchor="middle" fill="#e8eaef" font-size="12">${{grp.label}}</text>`;
    [['q', COLORS.shili17], ['h', COLORS.high], ['l', COLORS.low]].forEach(([key, col], ki) => {{
      const st = stats(grp[key]);
      const x = cx - 36 + ki*24;
      const y1=yv(st.p75), y2=yv(st.p25), ym=yv(st.med);
      g += `<line x1="${{x}}" y1="${{yv(st.min)}}" x2="${{x}}" y2="${{yv(st.max)}}" stroke="${{col}}" stroke-width="2"/>`;
      g += `<rect x="${{x-7}}" y="${{y1}}" width="14" height="${{Math.max(2,y2-y1)}}" fill="${{col}}" opacity="0.35" stroke="${{col}}"/>`;
      g += `<line x1="${{x-9}}" y1="${{ym}}" x2="${{x+9}}" y2="${{ym}}" stroke="${{col}}" stroke-width="2.5"/>`;
    }});
  }});
  g += `<text x="14" y="${{H/2}}" transform="rotate(-90 14 ${{H/2}})" text-anchor="middle" fill="#9aa3b2" font-size="12">cosine</text>`;
  g += `<text x="${{MARGIN.l+30}}" y="${{MARGIN.t+10}}" fill="#9aa3b2" font-size="11">■ query  ■ high  ■ low</text>`;
  svg.innerHTML = g;
}}

function drawScatter(svgId, yKey) {{
  const svg = document.getElementById(svgId);
  const W=900, H=340; const lo=0.4, hi=0.95;
  let g='';
  // diagonal y=x
  g += `<line x1="${{xScale(lo,W,lo,hi)}}" y1="${{H-MARGIN.b}}" x2="${{xScale(hi,W,lo,hi)}}" y2="${{MARGIN.t}}" stroke="#3d4455" stroke-dasharray="5 4"/>`;
  g += `<text x="${{xScale(hi,W,lo,hi)-4}}" y="${{MARGIN.t+14}}" text-anchor="end" fill="#666" font-size="10">y=x 一致线</text>`;
  for (let t=0.4; t<=0.95; t+=0.1) {{
    const x=xScale(t,W,lo,hi); const y=H-MARGIN.b-(t-lo)/(hi-lo)*(H-MARGIN.t-MARGIN.b);
    g+=`<line x1="${{x}}" y1="${{MARGIN.t}}" x2="${{x}}" y2="${{H-MARGIN.b}}" stroke="#252a36"/>`;
    g+=`<line x1="${{MARGIN.l}}" y1="${{y}}" x2="${{W-MARGIN.r}}" y2="${{y}}" stroke="#252a36"/>`;
    if (t>=0.5) g+=`<text x="${{x}}" y="${{H-10}}" text-anchor="middle" fill="#9aa3b2" font-size="10">${{t.toFixed(1)}}</text>`;
    g+=`<text x="${{MARGIN.l-6}}" y="${{y+3}}" text-anchor="end" fill="#9aa3b2" font-size="10">${{t.toFixed(1)}}</text>`;
  }}
  DATA.points.forEach(p => {{
    const col = p.set==='shili17'?COLORS.shili17:COLORS.voice29;
    const x = xScale(p.q,W,lo,hi);
    const y = H-MARGIN.b - (p[yKey]-lo)/(hi-lo)*(H-MARGIN.t-MARGIN.b);
    const id = p.set[0]+p.id;
    g += `<circle class="dot" cx="${{x}}" cy="${{y}}" r="5.5" fill="${{col}}" opacity="0.82" tabindex="0"><title>${{p.set}} ${{p.id}}: ${{p.label}}\\nq=${{fmt(p.q)}} ${{yKey}}=${{fmt(p[yKey])}}</title></circle>`;
  }});
  g += `<text x="${{W/2}}" y="${{H-2}}" text-anchor="middle" fill="#9aa3b2" font-size="12">query cosine</text>`;
  g += `<text x="12" y="${{H/2}}" transform="rotate(-90 12 ${{H/2}})" text-anchor="middle" fill="#9aa3b2" font-size="12">${{yKey}} cosine</text>`;
  svg.innerHTML = g;
}}

function kde(vals, lo=0.52, hi=0.92, points=80) {{
  const n = vals.length; if (!n) return [];
  const bw = 0.025;
  const out = [];
  for (let i=0; i<points; i++) {{
    const x = lo + (hi-lo)*i/(points-1);
    let s = 0;
    vals.forEach(v => {{ const u=(x-v)/bw; s += Math.exp(-0.5*u*u); }});
    out.push({{ x, y: s/(n*bw*Math.sqrt(2*Math.PI)) }});
  }}
  const maxY = Math.max(...out.map(p=>p.y), 1e-9);
  return out.map(p => ({{ x:p.x, y:p.y/maxY }}));
}}

function drawRidge(svgId) {{
  const svg = document.getElementById(svgId);
  const W=900, H=220; const lo=0.52, hi=0.92;
  const series = [
    {{ label:'测试例17', vals: DATA.shili17.q, color: COLORS.shili17, y: 70 }},
    {{ label:'技术话术29', vals: DATA.voice29.q, color: COLORS.voice29, y: 150 }},
  ];
  let g='';
  for (let t=0.55; t<=0.9; t+=0.05) {{
    const x=xScale(t,W,lo,hi);
    g+=`<line x1="${{x}}" y1="30" x2="${{x}}" y2="${{H-20}}" stroke="#2a2f3d"/>`;
    g+=`<text x="${{x}}" y="${{H-6}}" text-anchor="middle" fill="#9aa3b2" font-size="10">${{t.toFixed(2)}}</text>`;
  }}
  series.forEach(s => {{
    const pts = kde(s.vals, lo, hi);
    let d = `M ${{xScale(pts[0].x,W,lo,hi)}} ${{s.y}}`;
    pts.forEach(p => {{ d += ` L ${{xScale(p.x,W,lo,hi)}} ${{s.y - p.y*55}}`; }});
    d += ` L ${{xScale(pts[pts.length-1].x,W,lo,hi)}} ${{s.y}} Z`;
    g += `<path d="${{d}}" fill="${{s.color}}" opacity="0.45" stroke="${{s.color}}" stroke-width="1.5"/>`;
    g += `<text x="${{MARGIN.l}}" y="${{s.y+4}}" fill="${{s.color}}" font-size="12">${{s.label}}</text>`;
  }});
  g += `<text x="${{W/2}}" y="${{H-2}}" text-anchor="middle" fill="#9aa3b2" font-size="12">query cosine (KDE 归一化)</text>`;
  svg.innerHTML = g;
}}

function renderThresh() {{
  const tb = document.querySelector('#thresh-table tbody');
  THRESH.forEach(t => {{
    const a = DATA.shili17.q.filter(v=>v>=t).length;
    const b = DATA.voice29.q.filter(v=>v>=t).length;
    tb.innerHTML += `<tr><td>≥ ${{t}}</td><td>${{a}} / 17</td><td>${{b}} / 29</td><td>${{a+b}} / 46</td></tr>`;
  }});
}}

renderStats();
drawHist('hist-query', [
  {{ name:'shili17', vals: DATA.shili17.q, color: COLORS.shili17 }},
  {{ name:'voice29', vals: DATA.voice29.q, color: COLORS.voice29 }},
]);
drawEcdf('ecdf-query');
drawBox('boxplot');
drawScatter('scatter-qh', 'h');
drawScatter('scatter-ql', 'l');
drawRidge('ridge');
renderThresh();
</script>
</body>
</html>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shili",
        type=Path,
        default=_ROOT / "logs" / "relevance_shili17.jsonl",
    )
    p.add_argument(
        "--voice",
        type=Path,
        default=_ROOT / "logs" / "relevance_voice_tech29.jsonl",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "logs" / "relevance_distribution_charts.html",
    )
    args = p.parse_args()

    shili = _load_cases(args.shili.resolve(), "shili17")
    voice = _load_cases(args.voice.resolve(), "voice29")
    payload = {
        "shili17": {
            "n": len(shili),
            "q": [r["q"] for r in shili],
            "h": [r["h"] for r in shili],
            "l": [r["l"] for r in shili],
            "stats": {k: _stats([r[k] for r in shili]) for k in ("q", "h", "l")},
        },
        "voice29": {
            "n": len(voice),
            "q": [r["q"] for r in voice],
            "h": [r["h"] for r in voice],
            "l": [r["l"] for r in voice],
            "stats": {k: _stats([r[k] for r in voice]) for k in ("q", "h", "l")},
        },
        "points": [
            {
                "set": r["set"],
                "id": r["id"],
                "label": r["query"][:48],
                "q": r["q"],
                "h": r["h"],
                "l": r["l"],
            }
            for r in shili + voice
        ],
    }
    payload["overlap"] = {
        "q_min": max(min(payload["shili17"]["q"]), min(payload["voice29"]["q"])),
        "q_max": min(max(payload["shili17"]["q"]), max(payload["voice29"]["q"])),
    }

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_build_html(payload), encoding="utf-8")
    print(f"Wrote: {out.as_posix()}")


if __name__ == "__main__":
    main()
