"""Schema-driven coalesce core tests on synthetic content (CI-runnable).

Locks down the document-level field-schema threading: single-record section
groups must still merge their record fields because the schema is induced once
over the whole document and passed down (run-signature needs >=3 repetitions,
which a lone tiny section group cannot provide).
"""

from raganything.utils import (
    build_image_ref_block,
    build_table_aware_ingest_segments,
    coalesce_text_image_segments,
    induce_field_schema,
)


def _img(path: str) -> str:
    return build_image_ref_block(img_path=path, page_idx=5, context="保养内容：清洁外观")


def _record(i: int, closer: str = "保养步骤") -> list:
    return [
        f"2.1.{i} 保养项目{i}",
        "保养周期：每天一次",
        "保养内容：清洁外观",
        _img(f"images/{i}.jpg"),
        f"{closer}：擦拭干净",
    ]


def _maintenance_parts(n: int = 4, closer: str = "保养步骤") -> list:
    parts: list = []
    for i in range(1, n + 1):
        parts.extend(_record(i, closer=closer))
    return parts


def test_doc_level_schema_merges_single_record_sections():
    """Each numbered subsection is its own (tiny) section group, yet still merges."""
    parts = _maintenance_parts(4)
    schema = induce_field_schema(parts)
    assert schema is not None
    assert (schema.initiator, schema.closer) == ("保养周期", "保养步骤")

    out = build_table_aware_ingest_segments(parts)
    # one fully-merged segment per record: heading + 周期 + 内容 + 图片 + 步骤
    assert len(out) == 4
    for i, seg in enumerate(out, start=1):
        assert f"2.1.{i} 保养项目{i}" in seg
        assert "保养周期：每天一次" in seg
        assert "保养内容：清洁外观" in seg
        assert "保养步骤：擦拭干净" in seg
        assert f"images/{i}.jpg" in seg


def test_tiny_group_fragments_locally_but_merges_with_doc_schema():
    """Direct caller on a lone record gets schema=None; doc schema fixes it."""
    parts = _maintenance_parts(4)
    doc_schema = induce_field_schema(parts)
    tiny = _record(1)

    # locally induced schema is None (single run < _SCHEMA_MIN_REPEAT)
    assert induce_field_schema(tiny) is None
    fragmented = coalesce_text_image_segments(tiny)
    assert len(fragmented) > 1  # the historical bug: fields split apart

    merged = coalesce_text_image_segments(tiny, schema=doc_schema)
    assert len(merged) == 1
    assert "保养周期：每天一次" in merged[0]
    assert "保养步骤：擦拭干净" in merged[0]


def test_overmerged_segment_splits_at_repeated_initiator():
    """A part bundling two records splits when the initiator reappears."""
    overmerged = (
        "保养周期：A\n\n保养内容：B\n\n保养步骤：C\n\n"
        "保养周期：D\n\n保养内容：E\n\n保养步骤：F"
    )
    parts = [overmerged] + _maintenance_parts(3)
    assert induce_field_schema(parts) is not None

    out = build_table_aware_ingest_segments(parts)
    # 2 (split from the bundled part) + 3 normal records
    assert len(out) == 5
    assert any("保养周期：A" in s and "保养步骤：C" in s for s in out)
    assert any("保养周期：D" in s and "保养步骤：F" in s for s in out)
    # the two records must not stay glued together
    assert not any("保养周期：A" in s and "保养周期：D" in s for s in out)


def test_operation_step_closer_participates_in_merge():
    """加工中心/数控六面钻 shape: 操作步骤 becomes the closer and merges in."""
    parts = _maintenance_parts(4, closer="操作步骤")
    schema = induce_field_schema(parts)
    assert schema is not None
    assert schema.closer == "操作步骤"

    out = build_table_aware_ingest_segments(parts)
    assert len(out) == 4
    for seg in out:
        assert "操作步骤：擦拭干净" in seg
        assert "保养周期：每天一次" in seg


def test_schema_none_path_is_noop_for_plain_manual():
    """Manuals without record fields induce None and keep simple coalesce output."""
    parts = [
        "1.1 安装前检查",
        "设备就位后检查外观是否完好，随机附件是否齐全。",
        "1.2 运行中检查",
        "运行过程中注意观察各机构动作是否平稳。",
    ]
    assert induce_field_schema(parts) is None
    out = build_table_aware_ingest_segments(parts)
    # no field fragmentation: text is preserved verbatim across output segments
    joined = "\n\n".join(out)
    for part in parts:
        assert part in joined


def test_empty_input_returns_empty():
    assert build_table_aware_ingest_segments([]) == []
    assert coalesce_text_image_segments([]) == []
