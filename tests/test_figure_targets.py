import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from iqr_figure_target import (  # noqa: E402
    _answer_bullet_lines_for_figure_targets,
    _figure_ref_matches_target_topic,
    _resolve_manual_hint_for_figure_targets,
    _should_use_unified_figure_targets,
    extract_figure_targets,
)

Q15_QUERY = "这四种封边机的电控板的保养周期分别是多久"

Q12_QUERY = "高速智能封边机的保养中，需要清理粉尘、碎屑的部件有哪些？"

Q12_ANSWER = """\
高速智能封边机保养中需清理粉尘、碎屑的主要部件如下：

- **仿形齿条与齿轮**：清理齿条、齿轮上的粉尘与碎屑
- **平切齿条与齿轮**：清理平切部位齿条齿轮粉尘
- **滚珠丝杆副**：清理导轨与丝杆粉尘
- **辅助进料链条**：清除灰尘、木屑
- **涂胶电机及散热风扇**：清理电机散热片粉尘
- **滤尘箱**：内刮丝碎屑清理
- **机床床身**、输送台、光电感应器：清理表面木屑及粉尘

### References
[1] 高速智能封边机使用手册维护与保养.pdf
"""

Q15_ANSWER = """\
- **高速智能封边机**：检查电控板设备
- **双端封边机**：检查电控板设备
- **斜切封边机**：检查电控板设备
- **自动封边机**：检查电控板设备
"""

Q17_QUERY = "这四种封边机的熔胶盒部件有哪些？"


def test_answer_bullet_lines_extracts_q12_bullets():
    lines = _answer_bullet_lines_for_figure_targets(Q12_ANSWER)
    assert len(lines) >= 6
    joined = "\n".join(lines)
    assert "仿形齿条" in joined
    assert "滤尘箱" in joined
    assert "涂胶电机" in joined


def test_extract_figure_targets_q12_is_answer_bullet():
    targets = extract_figure_targets(Q12_QUERY, Q12_ANSWER)
    assert len(targets) >= 6
    assert all(t.kind == "answer_bullet" for t in targets)
    assert targets[0].manual_hint
    assert "仿形齿条" in targets[0].anchor_text


def test_should_use_unified_for_q12_answer_bullet(monkeypatch):
    monkeypatch.setenv("RAG_IMAGE_CHUNK_LOCALITY", "1")
    assert _should_use_unified_figure_targets(Q12_QUERY, Q12_ANSWER) is True


def test_resolve_manual_hint_from_query_machine():
    hint = _resolve_manual_hint_for_figure_targets(Q12_ANSWER, Q12_QUERY)
    assert "封边机" in hint


def test_machine_bullet_priority_over_answer_bullet():
    targets = extract_figure_targets("四种封边机电控板保养", Q15_ANSWER)
    assert len(targets) == 4
    assert targets[0].kind == "machine_bullet"


def test_pair_priority_over_answer_bullet(monkeypatch):
    monkeypatch.setenv("RAG_IMAGE_CHUNK_LOCALITY", "1")
    answer = """\
## 高速智能封边机
- **熔胶盒**：定期清理
## 双端封边机
- **熔胶盒**：定期清理
"""
    targets = extract_figure_targets(Q17_QUERY, answer)
    if len(targets) >= 2 and targets[0].kind == "machine_component_pair":
        assert targets[0].kind == "machine_component_pair"


def test_target_topic_ref_only_ignores_unrelated_caption():
    """_figure_ref_matches_target_topic rejects refs whose content is unrelated."""
    ref = {
        "caption": "清空胶锅",
        "context": "清空胶锅，胶锅内倒入5公斤PUR 清胶剂",
    }
    anchor = "* **双端封边机**：电控板检查的保养周期为每半年一次"
    assert not _figure_ref_matches_target_topic(
        anchor,
        Q15_QUERY,
        ref,
        kind="machine_bullet",
    )


def test_target_topic_requires_query_theme_in_ref():
    motor_ref = {
        "caption": "电机检查清理",
        "context": "电机检查清理：每年例行保养一次",
    }
    board_ref = {
        "caption": "检查电控板设备",
        "context": "保养内容： 检查电控板设备",
    }
    anchor = "* **自动封边机**：电控板检查的保养周期为每半年一次"
    assert not _figure_ref_matches_target_topic(
        anchor,
        Q15_QUERY,
        motor_ref,
        kind="machine_bullet",
    )
    assert _figure_ref_matches_target_topic(
        anchor,
        Q15_QUERY,
        board_ref,
        kind="machine_bullet",
    )


def test_target_topic_listing_head_matches_q12_item():
    ref = {
        "caption": "滤尘箱内刮丝碎屑清理",
        "context": "保养步骤：工作结束后需把滤尘箱内的刮丝碎屑清理干净",
    }
    anchor = "* **滤尘箱**：工作结束后必须清理干净"
    assert _figure_ref_matches_target_topic(
        anchor, Q12_QUERY, ref, kind="answer_bullet"
    )


def test_target_topic_q12_rejects_needle_only_without_listing_head():
    safety_ref = {
        "caption": "安全罩门",
        "context": "保养内容：防止碎屑飞出伤人，检查安全罩门",
    }
    anchor = "* **仿形齿条与齿轮**：需清理粉尘与碎屑"
    assert not _figure_ref_matches_target_topic(
        anchor, Q12_QUERY, safety_ref, kind="answer_bullet"
    )


def test_target_topic_q15_rejects_unrelated_machine_bullet_figure():
    safety_ref = {
        "caption": "",
        "context": "保养步骤：检查封边机各安全罩门是否能关严，以防碎屑飞出伤人",
    }
    pur_ref = {
        "caption": "",
        "context": "清空胶锅，胶锅内倒入5公斤PUR 清胶剂",
    }
    anchor = "* **高速智能封边机**：电控板检查的保养周期为**每季度一次**"
    assert not _figure_ref_matches_target_topic(
        anchor, Q15_QUERY, safety_ref, kind="machine_bullet"
    )
    assert not _figure_ref_matches_target_topic(
        anchor, Q15_QUERY, pur_ref, kind="machine_bullet"
    )


def test_target_topic_q15_rejects_chunk_only_theme():
    pur_ref = {
        "caption": "",
        "context": "清空胶锅，胶锅内倒入5公斤PUR 清胶剂",
    }
    anchor = "* **双端封边机**：电控板（电控板元件检测）的保养周期为每半年一次"
    chunk = "3.2 电控板\n\n电控板检查：每半年保养一次"
    assert not _figure_ref_matches_target_topic(
        anchor,
        Q15_QUERY,
        pur_ref,
        kind="machine_bullet",
        doc_content=chunk,
    )


def test_target_topic_q15_control_box_ref_matches_subject():
    control_ref = {
        "caption": "电控板检查：每半年保养一次",
        "context": "检查控制箱内部各元器件",
    }
    anchor = "* **双端封边机**：电控板（电控板元件检测）的保养周期为每半年一次"
    assert _figure_ref_matches_target_topic(
        anchor, Q15_QUERY, control_ref, kind="machine_bullet"
    )


def test_target_topic_q15_anchor_section_aligns_control_box():
    control_ref = {
        "caption": "",
        "context": "检查控制箱内部各元器件",
    }
    anchor = "* **双端封边机**：电控板（电控板元件检测）的保养周期为每半年一次"
    section = (
        "3.2 电控板\n\n电控板检查：每半年保养一次\n\n保养内容：检查控制箱内部各元器件"
    )
    assert _figure_ref_matches_target_topic(
        anchor,
        Q15_QUERY,
        control_ref,
        kind="machine_bullet",
        anchor_content=section,
    )
    pur_ref = {"caption": "", "context": "清空胶锅，胶锅内倒入5公斤PUR 清胶剂"}
    assert not _figure_ref_matches_target_topic(
        anchor,
        Q15_QUERY,
        pur_ref,
        kind="machine_bullet",
        anchor_content=section,
    )


def test_target_topic_pair_uses_component():
    melter_ref = {
        "caption": "熔胶盒清理",
        "context": "保养步骤：清空熔胶盒内残胶",
    }
    assert _figure_ref_matches_target_topic(
        "高速智能封边机 熔胶盒",
        Q17_QUERY,
        melter_ref,
        kind="machine_component_pair",
        component="熔胶盒",
    )
    pur_ref = {
        "caption": "清空胶锅",
        "context": "清空胶锅，胶锅内倒入5公斤PUR 清胶剂",
    }
    assert not _figure_ref_matches_target_topic(
        "高速智能封边机 电控板",
        Q15_QUERY,
        pur_ref,
        kind="machine_component_pair",
        component="电控板",
    )
