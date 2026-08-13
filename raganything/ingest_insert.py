"""Doc-scoped text insertion entry points for ingest.

Split from utils.py (stage-1 refactor, see
docs/utils_refactor_schema_induction.md).
"""

import asyncio
import inspect
from datetime import datetime, timezone
from typing import Any, Dict, List

from lightrag.utils import compute_mdhash_id, logger

from .ingest_coalesce import (
    _INGEST_SEGMENT_DELIMITER,
    _ingest_chunk_token_size,
    prepare_table_aware_ingest_segments,
    table_aware_ingest_enabled,
)
from .machine_derive import derive_machine_from_docname


def compute_ingest_chunk_id(
    full_doc_id: str, chunk_order_index: int, content: str
) -> str:
    """Scope chunk ids by document so identical text in different manuals stays distinct."""
    return compute_mdhash_id(
        f"{full_doc_id}:{chunk_order_index}:{content}", prefix="chunk-"
    )


def _status_field(status_doc: Any, name: str, default: Any = "") -> Any:
    if status_doc is None:
        return default
    if isinstance(status_doc, dict):
        return status_doc.get(name, default)
    return getattr(status_doc, name, default)


async def _resolve_ingest_segments(
    lightrag,
    *,
    input_text: str,
    document_parts: List[str] | None,
    split_by_character: str | None,
    split_by_character_only: bool,
) -> List[str]:
    if table_aware_ingest_enabled() and document_parts:
        tokenizer = getattr(lightrag, "tokenizer", None)
        if tokenizer is not None:
            max_tokens = _ingest_chunk_token_size(lightrag)
            segments = prepare_table_aware_ingest_segments(
                document_parts, tokenizer, max_tokens
            )
            if segments:
                return segments

    chunking_result = lightrag.chunking_func(
        lightrag.tokenizer,
        input_text,
        split_by_character,
        split_by_character_only,
        lightrag.chunk_overlap_token_size,
        lightrag.chunk_token_size,
    )
    if inspect.isawaitable(chunking_result):
        chunking_result = await chunking_result
    if not isinstance(chunking_result, (list, tuple)):
        raise TypeError(
            f"chunking_func must return a list or tuple of dicts, got {type(chunking_result)}"
        )
    return [
        str(row["content"])
        for row in chunking_result
        if isinstance(row, dict) and str(row.get("content") or "").strip()
    ]


async def insert_doc_scoped_text_content(
    lightrag,
    *,
    enqueue_input: str,
    doc_id: str,
    file_path: str,
    segments: List[str],
    ids: str | list[str] | None,
    file_paths: str | list[str] | None,
) -> None:
    """Insert text chunks with per-document chunk ids; runs KG extract + merge like ``ainsert``."""
    from lightrag.base import DocStatus
    from lightrag.kg.shared_storage import get_namespace_data, get_pipeline_status_lock
    from lightrag.operate import merge_nodes_and_edges

    # Fail loud when LightRAG internals/storages are missing. Do NOT silently
    # fall back to ``ainsert`` here: callers chose doc-scoped ids for image
    # locality / order_index / table linking; a content-only ainsert would look
    # successful while producing the wrong chunk shape for later query PRs.
    required = (
        "apipeline_enqueue_documents",
        "_process_extract_entities",
        "_insert_done",
        "doc_status",
        "chunks_vdb",
        "text_chunks",
        "tokenizer",
        "chunk_entity_relation_graph",
        "entities_vdb",
        "relationships_vdb",
        "full_entities",
        "full_relations",
        "llm_response_cache",
        "entity_chunks",
        "relation_chunks",
    )
    missing = [name for name in required if not hasattr(lightrag, name)]
    if missing:
        raise RuntimeError(
            "Doc-scoped ingest requires LightRAG APIs that are missing on this "
            f"build: {', '.join(missing)}. Upgrade `lightrag-hku`, or call "
            "standard `ainsert` from the non-doc-scoped path instead."
        )

    pipeline_status = await get_namespace_data("pipeline_status")
    pipeline_status_lock = get_pipeline_status_lock()

    await lightrag.apipeline_enqueue_documents(
        enqueue_input, ids=ids, file_paths=file_paths
    )

    status_doc = await lightrag.doc_status.get_by_id(doc_id)
    processing_start_time = int(datetime.now(timezone.utc).timestamp())

    chunks: Dict[str, Dict[str, Any]] = {}
    order = 0
    for seg in segments:
        content = (seg or "").strip()
        if not content:
            continue
        chunk_id = compute_ingest_chunk_id(doc_id, order, content)
        try:
            tokens = len(lightrag.tokenizer.encode(content))
        except Exception:
            tokens = len(content.split())
        chunks[chunk_id] = {
            "content": content,
            "full_doc_id": doc_id,
            "file_path": file_path,
            "machine": derive_machine_from_docname(file_path),
            "chunk_order_index": order,
            "tokens": tokens,
            "llm_cache_list": [],
        }
        order += 1

    if not chunks:
        logger.warning("Doc-scoped ingest: no segments for doc_id=%s", doc_id)
        return

    await lightrag.doc_status.upsert(
        {
            doc_id: {
                "status": DocStatus.PROCESSING,
                "chunks_count": len(chunks),
                "chunks_list": list(chunks.keys()),
                "content_summary": _status_field(status_doc, "content_summary"),
                "content_length": _status_field(status_doc, "content_length"),
                "created_at": _status_field(status_doc, "created_at"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": file_path,
                "track_id": _status_field(status_doc, "track_id"),
                "metadata": {"processing_start_time": processing_start_time},
            }
        }
    )

    await asyncio.gather(
        lightrag.chunks_vdb.upsert(chunks),
        lightrag.text_chunks.upsert(chunks),
    )

    chunk_results = await lightrag._process_extract_entities(
        chunks, pipeline_status, pipeline_status_lock
    )

    await merge_nodes_and_edges(
        chunk_results=chunk_results,
        knowledge_graph_inst=lightrag.chunk_entity_relation_graph,
        entity_vdb=lightrag.entities_vdb,
        relationships_vdb=lightrag.relationships_vdb,
        global_config=lightrag.__dict__,
        full_entities_storage=lightrag.full_entities,
        full_relations_storage=lightrag.full_relations,
        doc_id=doc_id,
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        llm_response_cache=lightrag.llm_response_cache,
        entity_chunks_storage=lightrag.entity_chunks,
        relation_chunks_storage=lightrag.relation_chunks,
        current_file_number=1,
        total_files=1,
        file_path=file_path,
    )

    processing_end_time = int(datetime.now(timezone.utc).timestamp())
    await lightrag.doc_status.upsert(
        {
            doc_id: {
                "status": DocStatus.PROCESSED,
                "chunks_count": len(chunks),
                "chunks_list": list(chunks.keys()),
                "content_summary": _status_field(status_doc, "content_summary"),
                "content_length": _status_field(status_doc, "content_length"),
                "created_at": _status_field(status_doc, "created_at"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": file_path,
                "track_id": _status_field(status_doc, "track_id"),
                "metadata": {
                    "processing_start_time": processing_start_time,
                    "processing_end_time": processing_end_time,
                },
            }
        }
    )
    await lightrag._insert_done()


async def insert_text_content(
    lightrag,
    input: str | list[str],
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
    *,
    document_parts: List[str] | None = None,
):
    """
    Insert pure text content into LightRAG

    Args:
        lightrag: LightRAG instance
        input: Single document string or list of document strings
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
        document_parts: optional pre-split parts (``[Table]`` blocks stay atomic when table-aware ingest is on)
    """
    logger.info("Starting text content insertion into LightRAG...")

    if isinstance(ids, list) and not ids:
        ids = None
    if isinstance(file_paths, list) and not file_paths:
        file_paths = None
    if isinstance(ids, str) and not ids.strip():
        ids = None
    if isinstance(file_paths, str) and not file_paths.strip():
        file_paths = None

    if isinstance(input, str) and ids is not None:
        if isinstance(ids, list) and len(ids) != 1:
            raise ValueError(
                "Doc-scoped ingest for a single input string requires "
                f"ids to be a str or a one-element list, got len={len(ids)}"
            )
        if isinstance(file_paths, list) and len(file_paths) != 1:
            raise ValueError(
                "Doc-scoped ingest for a single input string requires "
                "file_paths to be a str, None, or a one-element list, "
                f"got len={len(file_paths)}"
            )
        doc_id = ids if isinstance(ids, str) else ids[0]
        file_path = ""
        if file_paths:
            file_path = file_paths if isinstance(file_paths, str) else file_paths[0]
        file_path = file_path or "unknown_source"

        segments = await _resolve_ingest_segments(
            lightrag,
            input_text=input,
            document_parts=document_parts,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
        )
        if segments:
            enqueue_input = input
            if (
                table_aware_ingest_enabled()
                and document_parts
                and _INGEST_SEGMENT_DELIMITER.join(segments) != input.strip()
            ):
                enqueue_input = _INGEST_SEGMENT_DELIMITER.join(segments)
            logger.info(
                "Doc-scoped chunk ingest: %d segment(s) for %s",
                len(segments),
                file_path,
            )
            await insert_doc_scoped_text_content(
                lightrag,
                enqueue_input=enqueue_input,
                doc_id=doc_id,
                file_path=file_path,
                segments=segments,
                ids=ids,
                file_paths=file_paths,
            )
            logger.info("Text content insertion complete")
            return

    # Fallback: legacy LightRAG path (content-only chunk ids)
    await lightrag.ainsert(
        input=input,
        file_paths=file_paths,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        ids=ids,
    )

    logger.info("Text content insertion complete")


async def insert_text_content_with_multimodal_content(
    lightrag,
    input: str | list[str],
    multimodal_content: list[dict[str, any]] | None = None,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
    scheme_name: str | None = None,
):
    """
    Insert pure text content into LightRAG

    Args:
        lightrag: LightRAG instance
        input: Single document string or list of document strings
        multimodal_content: Multimodal content list (optional)
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
        scheme_name: scheme name (optional)
    """
    logger.info("Starting text content insertion into LightRAG...")

    # Use LightRAG's insert method with all parameters
    try:
        await lightrag.ainsert(
            input=input,
            multimodal_content=multimodal_content,
            file_paths=file_paths,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            ids=ids,
            scheme_name=scheme_name,
        )
    except Exception as e:
        logger.info(f"Error: {e}")
        logger.info(
            "If the error is caused by the ainsert function not having a multimodal content parameter, please update the raganything branch of lightrag"
        )

    logger.info("Text content insertion complete")


def get_processor_for_type(modal_processors: Dict[str, Any], content_type: str):
    """
    Get appropriate processor based on content type

    Args:
        modal_processors: Dictionary of available processors
        content_type: Content type

    Returns:
        Corresponding processor instance
    """
    # Direct mapping to corresponding processor
    if content_type == "image":
        return modal_processors.get("image")
    elif content_type == "table":
        return modal_processors.get("table")
    elif content_type == "equation":
        return modal_processors.get("equation")
    else:
        # For other types, use generic processor
        return modal_processors.get("generic")


def get_processor_supports(proc_type: str) -> List[str]:
    """Get processor supported features"""
    supports_map = {
        "image": [
            "Image content analysis",
            "Visual understanding",
            "Image description generation",
            "Image entity extraction",
        ],
        "table": [
            "Table structure analysis",
            "Data statistics",
            "Trend identification",
            "Table entity extraction",
        ],
        "equation": [
            "Mathematical formula parsing",
            "Variable identification",
            "Formula meaning explanation",
            "Formula entity extraction",
        ],
        "generic": [
            "General content analysis",
            "Structured processing",
            "Entity extraction",
        ],
    }
    return supports_map.get(proc_type, ["Basic processing"])
