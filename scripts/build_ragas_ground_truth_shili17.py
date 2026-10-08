#!/usr/bin/env python3
"""Build RAGAS ground-truth JSON for shili Q1–Q17 from docs/测试例参考答案.md.

The output is consumed by ``eval_ragas_webpath.py`` (read-only Web-path RAGAS eval).

Example::

  uv run python scripts/build_ragas_ground_truth_shili17.py
  uv run python scripts/build_ragas_ground_truth_shili17.py -o data/ragas_shili17_ground_truth.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))

spec = importlib.util.spec_from_file_location("run_web_path_q1_17", _SCRIPTS / "run_web_path_q1_17.py")
_web = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(_web)
REF = _web.REF

# Reference answer text synthesized from docs/测试例参考答案.md (for RAGAS context_recall).
GROUND_TRUTH: dict[int, str] = {
    1: (
        "高速智能封边机维护保养手册适用于 NB9-Smart 和 NB10-Smart 两种产品型号。"
    ),
    2: (
        "开机前应检查电源开关的外观和作用是否良好，并检查接地装置是否完整。"
    ),
    3: (
        "机床床身（机床外部）清洁应每天保养一次。"
    ),
    4: (
        "机床内部清洁步骤：工作结束后，需用吸尘机或碎布清洁封边机内外各功能单元"
        "储藏的刮削、灰尘等杂物，并擦拭干净表面的油污。"
    ),
    5: (
        "清理压带轮残胶应使用刮刀，操作时注意不要划伤压带轮表面涂层。"
    ),
    6: (
        "输送链条保养时应加注润滑脂2#（润滑脂 2#），可使用手动黄油枪从黄油杯处加注，"
        "直至链条侧面有黄油溢出。"
    ),
    7: (
        "进料部分保养时需使用百分表，表针跳动读数须小于 0.15mm。"
    ),
    8: (
        "更换预铣刀时，第一把预铣刀刀刃为顺铣方向，第二把预铣刀刀刃为逆铣方向。"
    ),
    9: (
        "注油泵油位过低时，应注入美孚品牌的长效液压油（美孚长效液压油）。"
    ),
    10: (
        "需要使用美孚长效液压油润滑的部件共九项：自动注油泵；辅助进料导轨/滑块；"
        "进料靠板部分；预铣机构导轨/滑块；平切机构导轨/滑块；精修导轨/滑块；"
        "仿形机构导轨/滑块；开槽机构丝杆/法兰；刮边机构导轨/滑块。"
    ),
    11: (
        "需要清理残胶的部件包括：压带轮、仿形靠板、涂胶轴（胶轴）；"
        "涂胶电机及电机区域掉落的残胶也可一并清理。"
    ),
    12: (
        "需要清理粉尘、碎屑的部件包括：机床表面及内外、输送台、光电感应器、风管（长丝/刮丝）、"
        "辅助进料机构链条、传动丝杆及滑块、平切/仿形齿条齿轮、各类圆形导杆/导轨"
        "（粗修、精修、仿形、刮边、开槽、抛光等）、各类电机散热风扇、电控柜冷却风扇、"
        "滤尘箱、油压缓冲器，以及涂胶轴相关粉尘/残胶区域等。"
    ),
    13: (
        "季度保养需准备两种润滑脂：润滑脂2#（广泛用于导杆/导轨、齿条齿轮、辅助进料链条等）；"
        "高温润滑脂（专用于涂胶轴轴承，不可用普通润滑脂代替）。"
    ),
    14: (
        "南兴封边机跨多份手册汇总，产品型号包括：双端封边机（如 NB6S2II、NB7HS2P、"
        "NB7CS2IIP、NB8CS2IIPT 等）；高速自动封边机（如 NB6PG、NB7PCG、NB7PCGM、"
        "NB8PCHGM 等）；自动封边机（如 NBC332、NB5J、NB6J、NB6CJ、NB7CJ、NB7CJM、"
        "NB557D 等）；高速智能封边机 NB9-Smart、NB10-Smart。去重后约 17 个独立型号。"
    ),
    15: (
        "四种封边机电控板保养周期分别为：高速智能封边机每季度一次；"
        "自动封边机、双端封边机、高速自动封边机均为每半年一次。"
        "检查时应关注电控板元器件是否有电弧、异味或异常，控制箱温度不宜超过 55℃。"
    ),
    16: (
        "1#透平油（气动油，符合 ISOVG32）用于气源三联件油雾器/油杯的手动加油保养，"
        "约每 6–7 个工作循环加一滴，每周检查油雾器油量并清理水杯污水。"
        "适用机型为双端封边机、自动封边机、高速自动封边机（分别对应各自维护保养手册）。"
    ),
    17: (
        "各机型需清除残胶的部件如下。"
        "高速智能封边机：仿形靠模/仿形靠板、压带轮、涂胶轴（老化胶水/残胶）、"
        "涂胶电机及电机区域残胶。"
        "高速自动封边机、自动封边机、双端封边机：激光发生器/激光器出光口胶带残胶须彻底清理。"
    ),
}


def build_cases() -> list[dict]:
    rows: list[dict] = []
    for cid in sorted(REF.keys()):
        spec = REF[cid]
        gt = (GROUND_TRUTH.get(cid) or "").strip()
        if not gt:
            raise ValueError(f"Missing ground_truth for Q{cid}")
        rows.append(
            {
                "id": cid,
                "question": spec["query"],
                "ground_truth": gt,
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=_ROOT / "data" / "ragas_shili17_ground_truth.json",
        help="Output JSON path (default: data/ragas_shili17_ground_truth.json)",
    )
    args = p.parse_args()
    cases = build_cases()
    payload = {
        "schema": "ragas_shili17_v1",
        "source": "docs/测试例参考答案.md",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(cases),
        "test_cases": cases,
    }
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} cases -> {out}", flush=True)


if __name__ == "__main__":
    main()
