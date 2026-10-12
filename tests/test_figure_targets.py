import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from iqr_figure_target import (
    _answer_abstains_from_images,
    _answer_bullet_lines_for_figure_targets,
    _figure_ref_matches_target_topic,
    _machine_component_listing_pair_targets,
    _manual_figure_refs_for_structured_targets,
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


def test_explicit_q17_pairs_are_not_expanded_by_candidate_pool():
    answer = """\
### 自动封边机
* **激光器出光口**：清除残胶。
### 高速智能封边机
* **仿形靠模**：清除残胶。
* **压带轮**：清除残胶。
* **涂胶轴**：清除残胶。
* **涂胶电机**：清除残胶。
### 双端封边机
* **激光器出光口**：清除残胶。
### 高速自动封边机
* **激光器出光口**：清除残胶。
"""
    unrelated_candidate = {
        "file_path": "高速智能封边机维护保养手册.pdf",
        "content": (
            "3.1 机床外部清洁保养\n保养内容：机床外部清洁\n"
            "[图片]\n图片路径：/tmp/exterior.jpg\n"
            "关联正文：保养内容：机床外部清洁"
        ),
    }

    targets = extract_figure_targets(
        "所有机型中，各自哪些部件需要清除残胶？",
        answer,
        cite_pool=[unrelated_candidate],
    )

    assert len(targets) == 7
    assert all(target.kind == "machine_component_pair" for target in targets)
    assert not any(target.component == "机床外部清洁" for target in targets)


def test_refusal_answers_abstain_from_figure_targets():
    answer = "根据当前知识库，没有足够的信息回答这个问题。"
    assert _answer_abstains_from_images(answer)
    assert extract_figure_targets("电控板如何保养？", answer) == []
    assert not _should_use_unified_figure_targets("电控板如何保养？", answer)


def test_bulleted_refusal_still_abstains_from_figure_targets():
    answer = "- 根据当前知识库，没有足够的信息回答这个问题。"
    assert _answer_abstains_from_images(answer)
    assert extract_figure_targets("电控板如何保养？", answer) == []


@pytest.mark.parametrize(
    "answer",
    [
        "未检索到与该问题相关的信息。",
        "当前检索结果未发现相关内容，建议补充资料。",
        "知识库中暂无相关内容，无法作答。",
        "I could not find relevant information in the provided documents.",
        "- 建议补充设备型号。\n- 当前资料不足，无法回答。",
        "- 问题范围：电控板保养。\n- 根据当前知识库，没有足够的信息回答这个问题。",
        "| 部件 | 周期 |\n| --- | --- |\n当前资料不足以确定相关内容。",
        "关于电控板保养，当前资料不足，无法回答。",
        "对于电控板的保养周期，知识库中暂无相关信息。",
        "您询问的是电控板如何保养，但当前知识库没有足够的信息回答。",
        "电控板每多久保养一次？当前资料不足，无法判断。",
        "关于1#透平油的用途，当前资料不足，无法回答。",
    ],
)
def test_common_refusal_and_advice_variants_abstain(answer):
    assert _answer_abstains_from_images(answer, "电控板如何保养？")
    assert extract_figure_targets("电控板如何保养？", answer) == []


def test_intro_limitation_keeps_following_supported_steps():
    answer = (
        "仅凭当前资料无法判断具体故障，但可以按以下步骤检查：\n"
        "1. **电控板**：检查是否存在电弧、异味或异常。"
    )
    assert not _answer_abstains_from_images(answer)


def test_supported_prose_before_scope_caveat_keeps_images():
    answer = "电控板每半年检查一次；但资料不足以确定其他部件的周期。"
    assert not _answer_abstains_from_images(answer, "电控板多久检查一次？")


def test_supported_table_before_scope_caveat_keeps_images():
    answer = (
        "| 部件 | 周期 |\n"
        "| --- | --- |\n"
        "| 电控板 | 每半年一次 |\n"
        "当前资料不足以确定其他机型。"
    )
    assert not _answer_abstains_from_images(answer, "电控板的保养周期是什么？")


def test_partial_scope_note_is_not_mistaken_for_answer_refusal():
    answer = (
        "* **自动封边机**：用于保养**气源三联件中的油杯**。\n"
        "* **双端封边机**：用于保养**气源三联件中的油杯**。\n"
        "注：其他机型的手册中未出现「1#透平油」字样，故不列入。"
    )
    assert not _answer_abstains_from_images(answer)
    assert len(extract_figure_targets("1#透平油用于哪些机型和部件？", answer)) == 2


def test_explicit_missing_other_manuals_note_keeps_prior_answer_targets():
    answer = (
        "* **自动封边机**：用于保养**气源三联件中的油杯**。\n"
        "* **双端封边机**：用于保养**气源三联件中的油杯**。\n\n"
        "注：文档未提供其他机型相关信息。"
    )
    assert not _answer_abstains_from_images(answer)
    assert len(extract_figure_targets("1#透平油用于哪些机型和部件？", answer)) == 2


def test_explicit_pair_count_uses_configured_image_limit(monkeypatch):
    pairs = [(f"机型{index}", f"部件{index}") for index in range(17)]
    monkeypatch.setattr(
        "iqr_figure_target._machine_component_targets_from_answer",
        lambda *_args, **_kwargs: pairs,
    )
    monkeypatch.setattr(
        "iqr_figure_target._filter_cross_listing_pair_targets",
        lambda _query, _answer, rows: rows,
    )
    monkeypatch.setattr("iqr_figure_target._multi_figure_image_limit", lambda: 24)

    assert (
        len(_machine_component_listing_pair_targets("query", "answer", pool=[])) == 17
    )


def test_structured_manual_fallback_returns_strict_selected_refs(monkeypatch):
    monkeypatch.setenv("RAG_IMAGE_CHUNK_LOCALITY", "1")
    answer = "* **自动封边机**：检查电控板设备。\n* **双端封边机**：检查电控板设备。"
    selected = [
        {
            "path": "/tmp/auto.jpg",
            "locality_machine": "自动封边机",
        },
        {
            "path": "/tmp/double.jpg",
            "locality_machine": "双端封边机",
        },
    ]
    monkeypatch.setattr(
        "iqr_figure_target._refs_from_unified_figure_targets",
        lambda *args, **kwargs: (selected, {"picked": 2}),
    )

    refs, meta = _manual_figure_refs_for_structured_targets(
        "两种封边机电控板如何保养？", answer
    )
    assert refs == selected
    assert meta["picked"] == 2
    assert meta["coverage"] == 1.0

    monkeypatch.setattr(
        "iqr_figure_target._refs_from_unified_figure_targets",
        lambda *args, **kwargs: (selected[:1], {"picked": 1}),
    )
    refs, meta = _manual_figure_refs_for_structured_targets(
        "两种封边机电控板如何保养？", answer
    )
    assert refs == []
    assert meta["coverage"] == 0.5
    assert meta["reason"] == "incomplete_target_coverage"


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
