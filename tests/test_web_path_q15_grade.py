"""Q15 batch grade: machine + cycle must co-occur in the same bullet."""

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
spec = importlib.util.spec_from_file_location(
    "run_web_path_q1_17", _SCRIPTS / "run_web_path_q1_17.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["run_web_path_q1_17"] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)

grade_text = mod.grade_text
REF = mod.REF


def test_q15_grade_rejects_half_year_on_smart_line_with_quarter_footnote():
    wrong = (
        "* **高速智能封边机**：针对电控板的检查要求**每半年严格执行一次** [1]。"
        "（*注：该设备手册中“每季度一次”的电气保养项目主要是指检查电控箱内部…*）"
    )
    ok, misses = grade_text(wrong, REF[15])
    assert not ok
    assert any("machine_cycle" in m for m in misses)


def test_q15_grade_accepts_per_machine_cycles():
    good = (
        "* **高速智能封边机**：电控板保养周期为**每季度一次** [1]。\n"
        "* **自动封边机**：电控板**每半年保养一次** [2]。\n"
        "* **双端封边机**：电控板**每半年保养一次** [3]。\n"
        "* **高速自动封边机**：电控板**每半年保养一次** [4]。"
    )
    ok, misses = grade_text(good, REF[15])
    assert ok, misses
