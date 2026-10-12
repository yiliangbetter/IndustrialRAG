"""Real parser paths keep document provenance through API image placement."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from iqr_media import resolve_media_path
from iqr_placement import (
    _api_images_from_figure_refs,
    build_inline_placements,
)
from iqr_protocol import extract_image_refs_from_context
from iqr_store import _source_key_from_path

MANUALS = {
    "自动封边机": "自动封边机维护保养手册8-25",
    "高速智能封边机": "高速智能封边机维护保养手册",
    "双端封边机": "双端封边机维护保养手册 新版8-24最终版",
    "高速自动封边机": "高速自动封边机维护保养手册8-24",
}


def _figure_ref(
    root: Path,
    machine: str,
    component: str,
    order: int,
) -> dict[str, Any]:
    manual = MANUALS[machine]
    path = root / f"{manual}_deadbeef" / manual / "txt" / "images" / f"f{order}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test image")
    return {
        "path": str(path),
        "caption": component,
        "context": f"保养内容：{component}",
        "locality_machine": machine,
        "locality_source": "component_pool",
        "figure_target_kind": "machine_component_pair",
        "target_id": f"target-{order}",
        "target_order": order,
        "target_anchor": component,
        "target_component": component,
        "figure_id": f"figure-{order}",
        "document_id": f"document-{machine}",
        "source_document_id": f"source-document-{machine}",
        "source_document": f"/private/parser/{manual}.pdf",
        "chunk_id": f"chunk-{order}",
        "evidence_chunk_id": f"evidence-chunk-{order}",
    }


def _api_images(root: Path, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    refs = [
        _figure_ref(root, machine, component, order)
        for order, (machine, component) in enumerate(pairs)
    ]
    return _api_images_from_figure_refs(refs, [root])


def test_source_key_uses_document_folder_for_real_parser_paths():
    linux_path = (
        "/data/pipeline_parse/自动封边机维护保养手册8-25_bddf3258/"
        "自动封边机维护保养手册8-25/txt/images/a.jpg"
    )
    windows_path = (
        "D:\\data\\pipeline_parse\\高速智能封边机维护保养手册_9c9fd6db\\"
        r"高速智能封边机维护保养手册\txt\images\b.jpg"
    )

    assert _source_key_from_path(linux_path) == "自动封边机维护保养手册8-25"
    assert _source_key_from_path(windows_path) == "高速智能封边机维护保养手册"
    assert (
        _source_key_from_path("/parser/manual-a/txt/images/section-2/figure/shared.jpg")
        == "manual-a"
    )
    assert _source_key_from_path("images/c.jpg") == "c"


def test_api_converter_preserves_target_and_source_provenance(tmp_path: Path):
    ref = _figure_ref(tmp_path, "高速智能封边机", "电控板", 3)

    images = _api_images_from_figure_refs([ref], [tmp_path])

    assert len(images) == 1
    image = images[0]
    assert image["source_key"] == "高速智能封边机维护保养手册"
    assert image["source_label"] == "高速智能封边机"
    assert image["manual_id"] == "source-document-高速智能封边机"
    assert image["target_machine"] == "高速智能封边机"
    assert image["display_caption"] == "高速智能封边机 — 电控板"
    assert image["caption"] == "电控板"
    assert image["raw_caption"] == "电控板"
    assert image["context"] == "保养内容：电控板"
    assert image["raw_context"] == "保养内容：电控板"
    assert image["locality_source"] == "component_pool"
    assert image["figure_target_kind"] == "machine_component_pair"
    assert image["target_id"] == "target-3"
    assert image["target_order"] == 3
    assert image["target_anchor"] == "电控板"
    assert image["target_component"] == "电控板"
    assert image["figure_id"] == "figure-3"
    assert image["document_id"] == "document-高速智能封边机"
    assert image["source_document_id"] == "source-document-高速智能封边机"
    assert image["source_document"] == "高速智能封边机维护保养手册.pdf"
    assert image["chunk_id"] == "chunk-3"
    assert image["evidence_chunk_id"] == "evidence-chunk-3"


def test_api_converter_does_not_collapse_same_basename_across_manuals(
    tmp_path: Path,
):
    refs = [
        _figure_ref(tmp_path, "自动封边机", "激光器出光口", 1),
        _figure_ref(tmp_path, "双端封边机", "激光器出光口", 2),
    ]
    for ref in refs:
        original = Path(ref["path"])
        shared_name = original.with_name("shared-image.jpg")
        shared_name.write_bytes(b"same visual bytes")
        ref["path"] = str(shared_name)

    images = _api_images_from_figure_refs(refs, [tmp_path])

    assert len(images) == 2
    assert {image["source_key"] for image in images} == {
        MANUALS["自动封边机"],
        MANUALS["双端封边机"],
    }


def test_context_parser_keeps_same_basename_from_different_manuals():
    context = """[图片]
图片路径：/parser/manual-a/txt/images/shared.jpg
关联正文：部件 A

[图片]
图片路径：/parser/manual-b/txt/images/shared.jpg
关联正文：部件 B
"""

    refs = extract_image_refs_from_context(context)

    assert [ref["path"] for ref in refs] == [
        "/parser/manual-a/txt/images/shared.jpg",
        "/parser/manual-b/txt/images/shared.jpg",
    ]


def test_stale_path_resolution_never_crosses_manual_provenance(tmp_path: Path):
    actual = tmp_path / "manual-a" / "txt" / "images" / "shared.jpg"
    actual.parent.mkdir(parents=True)
    actual.write_bytes(b"manual-a")

    resolved = resolve_media_path(
        "/stale/manual-b/txt/images/shared.jpg",
        [tmp_path],
        source_hint="manual-b",
    )

    assert resolved is None


def test_stale_path_resolution_recovers_only_the_same_manual(tmp_path: Path):
    actual = tmp_path / "manual-a" / "txt" / "images" / "shared.jpg"
    actual.parent.mkdir(parents=True)
    actual.write_bytes(b"manual-a")

    resolved = resolve_media_path(
        "/stale/manual-a/txt/images/shared.jpg",
        [tmp_path],
        source_hint="manual-a",
    )

    assert resolved == actual.resolve()


def test_ambiguous_basename_fallback_fails_closed(tmp_path: Path):
    for manual, content in (("manual-a", b"a"), ("manual-b", b"b")):
        path = tmp_path / manual / "txt" / "images" / "shared.jpg"
        path.parent.mkdir(parents=True)
        path.write_bytes(content)

    assert resolve_media_path("shared.jpg", [tmp_path]) is None


def test_identical_cross_manual_fallback_is_still_ambiguous(tmp_path: Path):
    for manual in ("manual-a", "manual-b"):
        path = tmp_path / manual / "txt" / "images" / "shared.jpg"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"same bytes")

    assert resolve_media_path("shared.jpg", [tmp_path]) is None


def test_identical_duplicate_parse_trees_resolve_with_source_hint(tmp_path: Path):
    candidates = []
    for parse_run in ("run-a", "run-b"):
        path = tmp_path / parse_run / "manual-a" / "txt" / "images" / "shared.jpg"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"same bytes")
        candidates.append(path.resolve())

    resolved = resolve_media_path(
        "/stale/manual-a/txt/images/shared.jpg",
        [tmp_path],
        source_hint="manual-a",
    )

    assert resolved == min(candidates, key=str)


def test_api_converter_drops_stale_cross_manual_match(tmp_path: Path):
    actual = tmp_path / "manual-a" / "txt" / "images" / "shared.jpg"
    actual.parent.mkdir(parents=True)
    actual.write_bytes(b"manual-a")
    ref = {
        "path": "/stale/manual-b/txt/images/shared.jpg",
        "locality_machine": "manual-b",
        "source_document": "manual-b.pdf",
        "source_document_id": "doc-b",
        "target_id": "target-1",
    }

    assert _api_images_from_figure_refs([ref], [tmp_path]) == []


@pytest.mark.parametrize(
    ("case_id", "query", "answer", "pairs"),
    [
        (
            "Q15",
            "这四种封边机的电控板的保养周期分别是多久",
            """* **自动封边机**：电控板检查的保养周期为**每半年一次**。
* **高速智能封边机**：电控板检查的保养周期为**每季度一次**。
* **双端封边机**：电控板检查的保养周期为**每半年一次**。
* **高速自动封边机**：电控板检查的保养周期为**每半年一次**。""",
            [(machine, "电控板") for machine in MANUALS],
        ),
        (
            "Q16",
            "1#透平油（气动油）是哪些机型的保养所需要的，是用来保养哪个部件？",
            """* **自动封边机**：用于保养**气源三联件中的油杯**。
* **双端封边机**：用于保养**气源三联件中的油杯**。
* **高速自动封边机**：用于保养**气源三联件中的油杯**。""",
            [
                ("自动封边机", "气源三联件中的油杯"),
                ("双端封边机", "气源三联件中的油杯"),
                ("高速自动封边机", "气源三联件中的油杯"),
            ],
        ),
        (
            "Q17",
            "所有机型中，各自哪些部件需要清除残胶",
            """### 自动封边机
* **激光器出光口**：清理残胶。
### 高速智能封边机
* **仿形靠模**：清理残胶。
* **压带轮**：清理残胶。
* **涂胶轴**：清理老化胶水。
* **涂胶电机**：清理残胶。
### 双端封边机
* **激光器出光口**：清理残胶。
### 高速自动封边机
* **激光器出光口**：清理残胶。""",
            [
                ("自动封边机", "激光器出光口"),
                ("高速智能封边机", "仿形靠模"),
                ("高速智能封边机", "压带轮"),
                ("高速智能封边机", "涂胶轴"),
                ("高速智能封边机", "涂胶电机"),
                ("双端封边机", "激光器出光口"),
                ("高速自动封边机", "激光器出光口"),
            ],
        ),
    ],
    ids=lambda value: (
        value if isinstance(value, str) and value.startswith("Q") else None
    ),
)
def test_real_path_provenance_places_every_expected_image(
    tmp_path: Path,
    case_id: str,
    query: str,
    answer: str,
    pairs: list[tuple[str, str]],
):
    images = _api_images(tmp_path / case_id, pairs)

    placements = build_inline_placements(answer, images, query=query)

    assert len(placements) == len(pairs)
    assert {placement["image_index"] for placement in placements} == set(
        range(len(pairs))
    )
    assert len({image["source_key"] for image in images}) == len(
        {machine for machine, _component in pairs}
    )
