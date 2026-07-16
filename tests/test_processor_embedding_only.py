from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from raganything.processor import ProcessorMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_process_document_embedding_only_recovers_mineru_v2_text():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.config = SimpleNamespace(
        allow_embedding_only_ingestion=True,
        display_content_stats=False,
        parse_method="auto",
        parser_output_dir=None,
        use_full_path=False,
    )
    processor.logger = FakeLogger()
    processor.callback_manager = None
    processor._ensure_lightrag_initialized = AsyncMock(
        return_value={"success": True}
    )
    processor.parse_document = AsyncMock(
        return_value=(
            [
                {
                    "type": "title",
                    "content": {
                        "title_content": [
                            {"type": "text", "content": "Safety Manual"}
                        ]
                    },
                },
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "Disconnect power first."}
                        ]
                    },
                },
            ],
            "doc-123",
        )
    )
    processor._insert_text_content_embedding_only = AsyncMock()

    await processor.process_document_complete("manual.pdf")

    processor._insert_text_content_embedding_only.assert_awaited_once_with(
        text_content="Safety Manual\n\nDisconnect power first.",
        file_ref="manual.pdf",
        doc_id="doc-123",
    )
