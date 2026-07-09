"""Phase 6 ingest coalesce: section heading + figure + trailing 保养 fields."""

from __future__ import annotations

from raganything.utils import coalesce_text_image_segments

_FEED_IMG = (
    "[图片]\n"
    "图片路径：D:/data/images/feed_section.jpg\n"
    "页码：9\n"
    "脚注：进料"
)
_CHAIN_IMG = (
    "[图片]\n"
    "图片路径：D:/data/images/chain_section.jpg\n"
    "页码：9\n"
    "脚注：输送链条"
)


def _q7_like_segments() -> list[str]:
    return [
        "3.1.3 进料部分保养",
        _FEED_IMG,
        "保养周期：每半年保养一次",
        "保养内容：进料靠板无倾斜，进料丝杆加润滑脂。",
        (
            "保养步骤：用3.6米标准胶木板从前端贴紧进料靠板，百分表吸在机架上，"
            "指针对准胶板侧面，启动履带走板，观察表针读数，表针跳动如超过0.15mm，"
            "需松开进料靠板固定螺丝，调整靠板位置，直至表针跳动＜0.15mm。"
        ),
        "3.1.4 输送电机保养",
        "保养周期：每半年保养一次",
    ]


def test_coalesce_merges_heading_figure_and_procedure_body() -> None:
    merged = coalesce_text_image_segments(_q7_like_segments())
    feed_blocks = [seg for seg in merged if "3.1.3" in seg]
    assert len(feed_blocks) == 1
    block = feed_blocks[0]
    assert "百分表" in block
    assert "0.15mm" in block
    assert _FEED_IMG.splitlines()[0] in block
    assert "保养周期：" in block
    assert "保养内容：" in block
    assert "保养步骤：" in block
    motor_blocks = [seg for seg in merged if "3.1.4" in seg]
    assert len(motor_blocks) == 1
    assert "百分表" not in motor_blocks[0]


def test_coalesce_does_not_cross_numbered_sections() -> None:
    segments = [
        "3.1.2 输送链条保养",
        _CHAIN_IMG,
        "保养周期：每半年保养一次",
        "保养内容：输送链条打润滑脂",
        "保养步骤：拆开输送链条侧面的不锈钢护板加注润滑脂。",
        "3.1.3 进料部分保养",
        _FEED_IMG,
        "保养周期：每半年保养一次",
        "保养内容：进料靠板无倾斜，进料丝杆加润滑脂。",
        "保养步骤：百分表吸在机架上，表针跳动如超过0.15mm需调整。",
    ]
    merged = coalesce_text_image_segments(segments)
    chain = [seg for seg in merged if "3.1.2" in seg]
    feed = [seg for seg in merged if "3.1.3" in seg]
    assert len(chain) == 1
    assert len(feed) == 1
    assert "百分表" in feed[0]
    assert "百分表" not in chain[0]
    assert "输送链条打润滑脂" in chain[0]
