"""Per-document record-field schema induction (run-signature) unit tests.

Synthetic cases run everywhere (CI included).  The optional real-manual cases
at the bottom only run when ``data/pipeline_parse`` is present locally.
"""

from pathlib import Path

import pytest

from raganything.utils import (
    DocFieldSchema,
    _line_field_key,
    _segment_field_keys,
    _segment_has_closer,
    _segment_has_initiator,
    induce_field_schema,
)


def _records(n, keys=("保养周期", "保养内容", "保养步骤"), heading="2.1"):
    """Build ``n`` numbered sections each holding one ``键：值`` record run."""
    segs = []
    for i in range(n):
        segs.append(f"{heading}.{i + 1} 保养项目{i + 1}")
        for key in keys:
            segs.append(f"{key}：第{i + 1}条内容")
    return segs


def _image(path="images/a.jpg") -> str:
    return f"[图片]\n图片路径：{path}\n页码：5"


# --- basic induction -------------------------------------------------------


def test_maintenance_shape_induces_three_roles():
    schema = induce_field_schema(_records(5))
    assert schema is not None
    assert schema.initiator == "保养周期"
    assert schema.closer == "保养步骤"
    assert schema.metadata_keys == frozenset({"保养内容"})


def test_operation_step_shape_closer_is_operation():
    # 加工中心 / 数控六面钻 形态：closer 为 操作步骤
    schema = induce_field_schema(_records(5, keys=("保养周期", "保养内容", "操作步骤")))
    assert schema is not None
    assert schema.initiator == "保养周期"
    assert schema.closer == "操作步骤"
    assert schema.metadata_keys == frozenset({"保养内容"})


def test_prose_closer_is_position_based():
    # 数控六面钻：操作步骤 值为整段散文（无编号子步骤），按位置仍判为 closer
    segs = []
    for i in range(4):
        segs.append(f"3.{i + 1} 保养项目")
        segs.append("保养周期：两周一次")
        segs.append("保养内容：清理刀排上的灰尘及油污，检查刀夹的卡槽磨损情况。")
        segs.append(
            "操作步骤：使用气枪和抹布将刀排附着在上面的粉尘和油污进行清理，"
            "再使用油枪给刀库支架上的滑块进行加油。检查刀夹的卡槽是否破损或变形。"
        )
    schema = induce_field_schema(segs)
    assert schema is not None
    assert schema.closer == "操作步骤"


def test_image_segments_do_not_break_run():
    segs = []
    for i in range(4):
        segs.append(f"2.1.{i + 1} 项目")
        segs.append("保养周期：每天一次")
        segs.append("保养内容：清洁外观")
        segs.append(_image(f"images/{i}.jpg"))  # image must not break the run
        segs.append("保养步骤：擦拭干净")
    schema = induce_field_schema(segs)
    assert schema is not None
    assert (schema.initiator, schema.closer) == ("保养周期", "保养步骤")
    assert schema.metadata_keys == frozenset({"保养内容"})


def test_interference_signature_filtered_by_dominant_family():
    # 加工中心存在干扰签名 第一步>第二步 ×3；支配家族按 initiator 过滤后排除
    segs = _records(6)  # 保养家族 ×6（initiator=保养周期）
    segs += ["准备", "第一步：开机", "第二步：复位", "结束"] * 3  # 干扰家族 ×3
    schema = induce_field_schema(segs)
    assert schema is not None
    assert schema.initiator == "保养周期"
    assert schema.closer == "保养步骤"
    assert "第一步" not in schema.field_keys
    assert "第二步" not in schema.field_keys


# --- degenerate inputs -----------------------------------------------------


def test_empty_input_returns_none():
    assert induce_field_schema([]) is None


def test_blank_and_image_only_returns_none():
    assert induce_field_schema(["", "  ", _image(), _image("images/b.jpg")]) is None


def test_single_key_repeat_returns_none():
    # 仅 1 个不同键 < _SCHEMA_MIN_KEYS(2)，不构成记录 schema
    segs = []
    for i in range(5):
        segs.append(f"2.1.{i} 标题")
        segs.append("保养周期：每天一次")
    assert induce_field_schema(segs) is None


def test_no_repeated_signature_returns_none():
    # 无记录式字段行（双端/自动/高速自动/PC 形态）：签名不重复 ⇒ None
    segs = [
        "1.1 安装前检查",
        "设备就位后检查外观是否完好，随机附件是否齐全。",
        "1.2 运行中检查",
        "运行过程中注意观察各机构动作是否平稳。",
        _image(),
        "1.3 关机操作",
        "工作结束后依次关闭各功能单元电源。",
    ]
    assert induce_field_schema(segs) is None


def test_below_min_repeat_returns_none():
    # 签名仅重复 2 次 < _SCHEMA_MIN_REPEAT(3)
    assert induce_field_schema(_records(2)) is None


# --- DocFieldSchema helpers ------------------------------------------------


def _schema():
    return DocFieldSchema(
        initiator="保养周期", closer="保养步骤", metadata_keys=frozenset({"保养内容"})
    )


def test_field_keys_union():
    assert _schema().field_keys == frozenset({"保养周期", "保养内容", "保养步骤"})


def test_field_key_recognises_only_schema_keys():
    schema = _schema()
    assert schema.field_key("保养周期：每天一次") == "保养周期"
    assert schema.field_key("保养内容：清洁") == "保养内容"
    assert schema.field_key("保养步骤：擦拭") == "保养步骤"
    assert schema.field_key("操作步骤：使用气枪") is None  # not in this schema
    assert schema.field_key("2.1.1 机床床身清洁") is None
    assert schema.field_key("") is None


def test_role_predicates():
    schema = _schema()
    assert schema.is_initiator("保养周期")
    assert schema.is_closer("保养步骤")
    assert schema.is_metadata("保养内容")
    assert not schema.is_metadata("保养周期")
    assert not schema.is_closer("保养内容")


def test_line_field_key_schema_independent():
    assert _line_field_key("保养周期：每天一次") == "保养周期"
    assert _line_field_key("操作步骤：使用气枪") == "操作步骤"
    assert _line_field_key("2.1.1 标题") is None
    assert _line_field_key("纯散文没有冒号") is None


def test_segment_field_keys_and_role_scanners():
    schema = _schema()
    seg = "2.1.1 机床床身清洁\n\n保养周期：每天一次\n\n保养内容：机床外部清洁"
    assert _segment_field_keys(seg, schema) == ["保养周期", "保养内容"]
    assert _segment_has_initiator(seg, schema)
    assert not _segment_has_closer(seg, schema)
    closed = seg + "\n\n保养步骤：擦拭干净"
    assert _segment_has_closer(closed, schema)


def test_segment_helpers_none_schema_are_safe():
    assert _segment_field_keys("保养周期：x", None) == []
    assert not _segment_has_initiator("保养周期：x", None)
    assert not _segment_has_closer("保养步骤：x", None)


# --- optional real-manual validation (local data only) ---------------------

_DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "pipeline_parse"

_SCHEMA_EXPECTED = {
    "高速智能封边机": ("保养周期", "保养步骤"),
    "加工中心": ("保养周期", "操作步骤"),
    "数控六面钻": ("保养周期", "操作步骤"),
}
_NONE_DOCS = ("双端封边机", "自动封边机", "高速自动封边机", "PC封边机")


def _load_doc_segments(key):
    import json

    from raganything.processor import ProcessorMixin

    class _Stub(ProcessorMixin):
        pass

    stub = _Stub()
    seen = set()
    for cl_path in sorted(_DATA_ROOT.rglob("*content_list*.json")):
        doc = cl_path.relative_to(_DATA_ROOT).parts[0].rsplit("_", 1)[0]
        if doc in seen:
            continue
        seen.add(doc)
        if key not in doc:
            continue
        cl = json.loads(cl_path.read_text(encoding="utf-8"))
        return stub._build_document_parts_for_ingest(cl)
    return None


@pytest.mark.skipif(not _DATA_ROOT.exists(), reason="data/pipeline_parse not present")
@pytest.mark.parametrize("key", list(_SCHEMA_EXPECTED))
def test_real_manual_induces_expected_schema(key):
    parts = _load_doc_segments(key)
    assert parts is not None, f"manual not found for {key}"
    schema = induce_field_schema([p for p in parts if (p or "").strip()])
    assert schema is not None
    initiator, closer = _SCHEMA_EXPECTED[key]
    assert schema.initiator == initiator
    assert schema.closer == closer


@pytest.mark.skipif(not _DATA_ROOT.exists(), reason="data/pipeline_parse not present")
@pytest.mark.parametrize("key", list(_NONE_DOCS))
def test_real_manual_without_record_fields_returns_none(key):
    parts = _load_doc_segments(key)
    assert parts is not None, f"manual not found for {key}"
    schema = induce_field_schema([p for p in parts if (p or "").strip()])
    assert schema is None
