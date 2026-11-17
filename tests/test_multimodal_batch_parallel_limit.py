"""Concurrency cap for type-aware multimodal description generation."""

import asyncio

import pytest

from raganything.processor import ProcessorMixin


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class _Captioner:
    def __init__(self, describe):
        self.generate_description_only = describe


def _description(content_type):
    return (
        f"desc-{content_type}",
        {
            "entity_name": content_type,
            "entity_type": "component",
            "summary": content_type,
        },
    )


def _items():
    return [
        {"type": "image", "page_idx": 1},
        {"type": "table", "page_idx": 2},
        {"type": "equation", "page_idx": 3},
    ]


def _processor(max_parallel_insert):
    processor = ProcessorMixin.__new__(ProcessorMixin)
    processor.logger = _Logger()
    processor.modal_processors = {item["type"]: _Captioner(None) for item in _items()}
    processor.captured = None
    processor.stages = []

    class _LightRAG:
        pass

    lightrag = _LightRAG()
    if max_parallel_insert is not None:
        lightrag.max_parallel_insert = max_parallel_insert

    async def get_by_id(_doc_id):
        return {"chunks_count": 4}

    lightrag.doc_status = type("Status", (), {"get_by_id": staticmethod(get_by_id)})()
    processor.lightrag = lightrag

    def convert(multimodal_data_list, _file_path, _doc_id):
        processor.captured = list(multimodal_data_list)
        return {f"chunk-{item['index']}": {} for item in multimodal_data_list}

    async def store(chunks):
        processor.stages.append(("store", list(chunks)))

    async def entities(_data, _chunks, _file_path, _doc_id):
        processor.stages.append("entities")

    async def extract(chunks):
        processor.stages.append(("extract", list(chunks)))
        return []

    async def belongs(chunk_results, _data):
        processor.stages.append("belongs")
        return chunk_results

    async def merge(_results, _file_path, _doc_id):
        processor.stages.append("merge")

    async def update(doc_id, chunk_ids):
        processor.stages.append((doc_id, list(chunk_ids)))

    processor._convert_to_lightrag_chunks_type_aware = convert
    processor._store_chunks_to_lightrag_storage_type_aware = store
    processor._store_multimodal_main_entities = entities
    processor._batch_extract_entities_lightrag_style_type_aware = extract
    processor._batch_add_belongs_to_relations_type_aware = belongs
    processor._batch_merge_lightrag_style_type_aware = merge
    processor._update_doc_status_with_chunks_type_aware = update
    return processor


async def _run(processor, describe):
    for proc in processor.modal_processors.values():
        proc.generate_description_only = describe

    await processor._process_multimodal_content_batch_type_aware(
        _items(),
        "manuals/pump.pdf",
        "doc-pump",
    )


@pytest.mark.asyncio
async def test_parallel_insert_one_never_overlaps_descriptions():
    processor = _processor(1)
    state = {"n": 0, "peak": 0}

    async def describe(**kwargs):
        state["n"] += 1
        state["peak"] = max(state["peak"], state["n"])
        try:
            await asyncio.sleep(0)
            return _description(kwargs["content_type"])
        finally:
            state["n"] -= 1

    await _run(processor, describe)

    assert state["peak"] == 1
    assert [item["content_type"] for item in processor.captured] == [
        "image",
        "table",
        "equation",
    ]
    assert [item["chunk_order_index"] for item in processor.captured] == [4, 5, 6]
    assert processor.stages[0][0] == "store"


@pytest.mark.asyncio
@pytest.mark.parametrize("max_parallel_insert", [2, None])
async def test_two_descriptions_overlap_and_the_third_waits(max_parallel_insert):
    processor = _processor(max_parallel_insert)
    state = {
        "n": 0,
        "peak": 0,
        "entered": [],
        "both_in": asyncio.Event(),
        "release": asyncio.Event(),
        "third_while_held": False,
    }

    async def describe(**kwargs):
        state["n"] += 1
        state["peak"] = max(state["peak"], state["n"])
        slot = state["n"]
        state["entered"].append(kwargs["content_type"])
        try:
            if slot <= 2:
                if slot == 2:
                    state["both_in"].set()
                await state["release"].wait()
            else:
                state["third_while_held"] = True
            return _description(kwargs["content_type"])
        finally:
            state["n"] -= 1

    task = asyncio.create_task(_run(processor, describe))
    await asyncio.wait_for(state["both_in"].wait(), timeout=2)
    assert state["n"] == 2
    assert state["peak"] == 2
    assert len(state["entered"]) == 2
    assert state["third_while_held"] is False

    state["release"].set()
    await asyncio.wait_for(task, timeout=2)

    assert state["peak"] == 2
    assert state["third_while_held"] is False
    assert sorted(state["entered"]) == ["equation", "image", "table"]
    assert [item["chunk_order_index"] for item in processor.captured] == [4, 5, 6]
    assert [item["index"] for item in processor.captured] == [0, 1, 2]


@pytest.mark.asyncio
async def test_failed_description_releases_the_single_slot():
    processor = _processor(1)
    state = {"n": 0, "peak": 0}

    async def describe(**kwargs):
        state["n"] += 1
        state["peak"] = max(state["peak"], state["n"])
        try:
            await asyncio.sleep(0)
            if kwargs["content_type"] == "image":
                raise RuntimeError("vision down")
            return _description(kwargs["content_type"])
        finally:
            state["n"] -= 1

    await asyncio.wait_for(_run(processor, describe), timeout=2)

    assert state["peak"] == 1
    assert [item["content_type"] for item in processor.captured] == [
        "table",
        "equation",
    ]
    assert [item["chunk_order_index"] for item in processor.captured] == [5, 6]
    assert processor.stages[-1][0] == "doc-pump"
