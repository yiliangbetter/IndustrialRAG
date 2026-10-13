"""MinerU v2 content lists must not collapse to one document id or drop plaintext."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from raganything.processor import ProcessorMixin


class _Logger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


class _ContentListProcessor(ProcessorMixin):
    def __init__(self):
        self.logger = _Logger()
        self.callback_manager = None
        self.lightrag = object()
        self.inserted = []
        self.config = SimpleNamespace(
            use_full_path=False,
            display_content_stats=False,
            allow_embedding_only_ingestion=False,
            content_format="minerU",
        )
        self._ensure_lightrag_initialized = AsyncMock(return_value={"success": True})
        self._process_multimodal_content = AsyncMock()
        self._mark_multimodal_processing_complete = AsyncMock()

    async def _record_insert(self, lightrag, input, file_paths=None, **kwargs):
        self.inserted.append(
            {
                "lightrag": lightrag,
                "input": input,
                "file_paths": file_paths,
                "ids": kwargs.get("ids"),
            }
        )


def _paragraph(text):
    return {
        "type": "paragraph",
        "content": {
            "paragraph_content": [
                {"type": "text", "content": "Tighten"},
                {"type": "text", "content": text},
            ]
        },
    }


@pytest.fixture
def processor():
    return _ContentListProcessor()


@pytest.mark.asyncio
async def test_nested_v2_wrapper_keeps_distinct_ids_and_harvested_text(processor):
    alpha = [
        [
            {"type": "title", "content": {"title_content": "  Torque spec  "}},
            _paragraph("to 40 Nm."),
            {
                "type": "list",
                "content": {
                    "list_items": [
                        {
                            "prefix": "1.",
                            "item_content": [
                                {"type": "text", "content": "Loosen the bolt"}
                            ],
                        },
                        {"prefix": "2.", "item_content": "   "},
                        {"item_content": "Retorque"},
                        "skip",
                    ]
                },
            },
            {
                "type": "table",
                "content": {"html": "  <table><tr><td>40</td></tr></table>  "},
            },
            {"type": "image", "content": {"image_caption": ["Fig 1", "", None]}},
            {"type": "text", "text": "   "},
            "skip",
            None,
        ]
    ]
    beta = [[{"type": "paragraph", "content": {"paragraph_content": "Other spec"}}]]

    with patch(
        "raganything.processor.insert_text_content",
        new=processor._record_insert,
    ):
        await processor.insert_content_list(alpha, file_path="manuals/press.pdf")
        await processor.insert_content_list(beta, file_path="manuals/other.pdf")
        await processor.insert_content_list(
            beta,
            file_path="manuals/other.pdf",
            doc_id="doc-pinned",
        )

    alpha_insert, beta_insert, pinned = processor.inserted
    assert alpha_insert["input"] == (
        "Torque spec\n\n"
        "Tighten to 40 Nm.\n\n"
        "1. Loosen the bolt\n"
        "Retorque\n\n"
        "<table><tr><td>40</td></tr></table>\n\n"
        "Fig 1"
    )
    assert alpha_insert["file_paths"] == "press.pdf"
    assert alpha_insert["lightrag"] is processor.lightrag
    assert beta_insert["input"] == "Other spec"
    assert beta_insert["file_paths"] == "other.pdf"

    flat_alpha = processor._normalize_nested_content_list(alpha)
    flat_beta = processor._normalize_nested_content_list(beta)
    assert alpha_insert["ids"] == processor._generate_content_based_doc_id(flat_alpha)
    assert beta_insert["ids"] == processor._generate_content_based_doc_id(flat_beta)
    assert alpha_insert["ids"] != beta_insert["ids"]
    assert alpha_insert["ids"].startswith("doc-")
    # Un-normalized wrappers are not dicts, so both raw lists hash as empty.
    assert processor._generate_content_based_doc_id(
        alpha
    ) == processor._generate_content_based_doc_id(beta)
    assert pinned["ids"] == "doc-pinned"
    assert pinned["input"] == "Other spec"


@pytest.mark.asyncio
async def test_text_block_is_inserted_without_appending_v2_plaintext(processor):
    content = [
        {"type": "text", "text": "Keep this sentence."},
        {
            "type": "paragraph",
            "content": {"paragraph_content": "Do not append"},
        },
    ]

    with patch(
        "raganything.processor.insert_text_content",
        new=processor._record_insert,
    ):
        await processor.insert_content_list(content, file_path="notes.txt")

    assert processor.inserted == [
        {
            "lightrag": processor.lightrag,
            "input": "Keep this sentence.",
            "file_paths": "notes.txt",
            "ids": processor._generate_content_based_doc_id(content),
        }
    ]


@pytest.mark.asyncio
async def test_non_v2_text_fields_are_stripped_when_harvest_is_empty(processor):
    content = [
        {"type": "header", "text": "  Section A  "},
        {"type": "caption", "text": " Detail "},
        {"type": "note", "text": "   "},
        {"type": "note", "text": 12},
        {"type": "note"},
    ]

    with patch(
        "raganything.processor.insert_text_content",
        new=processor._record_insert,
    ):
        await processor.insert_content_list(content, file_path="section.txt")

    assert processor.inserted[0]["input"] == "Section A\n\nDetail"
    assert processor.inserted[0]["ids"] == processor._generate_content_based_doc_id(
        content
    )


def test_content_doc_id_uses_type_specific_signatures(processor):
    hello = processor._generate_content_based_doc_id(
        [{"type": "text", "text": "  Hello  "}]
    )
    assert hello.startswith("doc-")
    assert hello == processor._generate_content_based_doc_id(
        [{"type": "text", "text": "Hello"}]
    )
    assert hello != processor._generate_content_based_doc_id(
        [{"type": "text", "text": "Hello!"}]
    )
    ordered = processor._generate_content_based_doc_id(
        [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]
    )
    assert ordered != processor._generate_content_based_doc_id(
        [{"type": "text", "text": "B"}, {"type": "text", "text": "A"}]
    )

    image_a = {"type": "image", "img_path": "/figs/a.png", "image_caption": ["A"]}
    image_b = {"type": "image", "img_path": "/figs/a.png", "image_caption": ["B"]}
    assert processor._generate_content_based_doc_id(
        [image_a]
    ) == processor._generate_content_based_doc_id([image_b])
    assert processor._generate_content_based_doc_id([image_a]) != (
        processor._generate_content_based_doc_id(
            [{"type": "image", "img_path": "/figs/b.png", "image_caption": ["A"]}]
        )
    )
    missing_path_a = {"type": "image", "img_path": "", "image_caption": ["A"]}
    missing_path_b = {"type": "image", "img_path": "", "image_caption": ["B"]}
    assert processor._generate_content_based_doc_id(
        [missing_path_a]
    ) != processor._generate_content_based_doc_id([missing_path_b])

    assert processor._generate_content_based_doc_id(
        [{"type": "table", "table_body": "|a|", "table_caption": ["c1"]}]
    ) == processor._generate_content_based_doc_id(
        [{"type": "table", "table_body": "|a|", "table_caption": ["c2"]}]
    )
    assert processor._generate_content_based_doc_id(
        [{"type": "table", "table_body": "|a|"}]
    ) != processor._generate_content_based_doc_id(
        [{"type": "table", "table_body": "|b|"}]
    )

    assert processor._generate_content_based_doc_id(
        [{"type": "equation", "text": "E=mc^2", "latex": "E=mc^2"}]
    ) == processor._generate_content_based_doc_id(
        [{"type": "equation", "text": "E=mc^2", "latex": "E"}]
    )
    assert processor._generate_content_based_doc_id(
        [{"type": "equation", "text": "E=mc^2", "latex": "E=mc^2"}]
    ) != processor._generate_content_based_doc_id(
        [{"type": "equation", "text": "F=ma", "latex": "E=mc^2"}]
    )

    paragraph = {"type": "paragraph", "content": {"paragraph_content": "Hi"}}
    assert processor._generate_content_based_doc_id(
        [paragraph]
    ) != processor._generate_content_based_doc_id([{"type": "text", "text": "Hi"}])
    assert processor._generate_content_based_doc_id(
        [paragraph, "not-a-block"]
    ) == processor._generate_content_based_doc_id([paragraph])


def test_span_list_and_caption_edges(processor):
    assert processor._mineru_span_text(None) == ""
    assert processor._mineru_span_text("  x  ") == "x"
    assert (
        processor._mineru_span_text(
            [
                " A ",
                None,
                "",
                {"type": "text", "content": "B"},
                {"text": " C "},
                {"content": ["D", "E"]},
            ]
        )
        == "A B C D E"
    )
    assert (
        processor._mineru_span_text({"type": "text", "content": "", "text": "fallback"})
        == ""
    )
    assert (
        processor._mineru_span_text({"type": "text", "content": ["A", {"text": "B"}]})
        == "A B"
    )
    assert processor._mineru_list_items_text("nope") == ""
    assert (
        processor._plaintext_from_mineru_blocks(
            [
                {"type": "image", "content": {"image_caption": "  Side view  "}},
                {"type": "image", "content": {"image_caption": None}},
                {"type": "image", "content": {"image_caption": ["", None, "Fig"]}},
                {"type": "paragraph", "content": "not-a-dict"},
                {"type": "mystery", "content": {"paragraph_content": "hidden"}},
                "skip",
            ]
        )
        == "Side view\n\nFig"
    )
