"""Multimodal query preprocessing latency and ordering tests."""

from __future__ import annotations

import asyncio
import logging
import time
from types import MethodType, SimpleNamespace

import pytest

from raganything.query import QueryMixin


def _query_mixin(*, concurrency: int) -> QueryMixin:
    query = QueryMixin()
    query.logger = logging.getLogger(__name__)
    query.config = SimpleNamespace(max_concurrent_query_content=concurrency)
    query.modal_processors = {"generic": object()}
    return query


@pytest.mark.asyncio
async def test_multimodal_descriptions_run_concurrently_in_input_order() -> None:
    query = _query_mixin(concurrency=4)
    active = 0
    max_active = 0

    async def describe(self, _processor, content, _content_type):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return content["value"]

    query._generate_query_content_description = MethodType(describe, query)
    content = [{"type": "note", "value": f"value-{i}"} for i in range(4)]

    started = time.perf_counter()
    result = await query._process_multimodal_query_content("question", content)
    elapsed = time.perf_counter() - started

    assert max_active == 4
    assert elapsed < 0.14
    positions = [result.index(f"value-{i}") for i in range(4)]
    assert positions == sorted(positions)


@pytest.mark.asyncio
async def test_multimodal_concurrency_can_be_limited_to_one() -> None:
    query = _query_mixin(concurrency=1)
    active = 0
    max_active = 0

    async def describe(self, _processor, content, _content_type):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
        return content["value"]

    query._generate_query_content_description = MethodType(describe, query)
    await query._process_multimodal_query_content(
        "question",
        [{"type": "note", "value": str(i)} for i in range(3)],
    )

    assert max_active == 1
