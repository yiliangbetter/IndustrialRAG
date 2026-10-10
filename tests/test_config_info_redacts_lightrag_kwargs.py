"""Config snapshots must not publish credentials, and unknown updates must not stick.

``get_config_info`` is included in processor/debug dumps. ``llm_model_kwargs``
and ``vector_db_storage_cls_kwargs`` can hold API keys and database passwords,
and model callables are not safe to copy into that snapshot. ``update_config``
is the live setter for the same object: a typo must not become a new attribute.
"""

from types import SimpleNamespace

from raganything.config import RAGAnythingConfig
from raganything.raganything import RAGAnything


LLM_SECRET = "llm-kw-secret-9f3a"
VECTOR_SECRET = "vector-db-secret-9f3a"


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.debugs = []

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg) % args if args else str(msg))

    def debug(self, msg, *args, **kwargs):
        self.debugs.append(str(msg) % args if args else str(msg))

    def info(self, msg, *args, **kwargs):
        return None


def _config():
    return RAGAnythingConfig(
        working_dir="/tmp/plant-rag",
        parser_output_dir="/tmp/plant-out",
        parser="docling",
        parse_method="ocr",
        display_content_stats=False,
        enable_image_processing=False,
        enable_table_processing=True,
        enable_equation_processing=False,
        context_window=2,
        context_mode="chunk",
        max_context_tokens=100,
        include_headers=False,
        include_captions=False,
        context_filter_content_types=["text", "table"],
        max_concurrent_files=3,
        supported_file_extensions=[".pdf", ".docx"],
        recursive_folder_processing=False,
    )


def _host(config, lightrag_kwargs=None):
    host = SimpleNamespace(
        config=config,
        lightrag_kwargs=lightrag_kwargs if lightrag_kwargs is not None else {},
        logger=FakeLogger(),
    )
    host.get_config_info = RAGAnything.get_config_info.__get__(host)
    host.update_config = RAGAnything.update_config.__get__(host)
    return host


def _embed():
    return None


def test_snapshot_omits_callables_and_secret_storage_kwargs():
    host = _host(
        _config(),
        {
            "llm_model_func": _embed,
            "embedding_func": _embed,
            "llm_model_kwargs": {"api_key": LLM_SECRET},
            "vector_db_storage_cls_kwargs": {"password": VECTOR_SECRET},
            "workspace": "plant-a",
            "top_k": 8,
        },
    )

    info = host.get_config_info()
    custom = info["lightrag_config"]["custom_parameters"]

    assert custom == {"workspace": "plant-a", "top_k": 8}
    assert all(not callable(value) for value in custom.values())
    assert "llm_model_kwargs" not in custom
    assert "vector_db_storage_cls_kwargs" not in custom
    assert "llm_model_func" not in custom
    dumped = repr(info)
    assert LLM_SECRET not in dumped
    assert VECTOR_SECRET not in dumped
    assert info["lightrag_config"]["note"].startswith("LightRAG will be initialized")

    assert info["directory"] == {
        "working_dir": "/tmp/plant-rag",
        "parser_output_dir": "/tmp/plant-out",
    }
    assert info["parsing"] == {
        "parser": "docling",
        "parse_method": "ocr",
        "display_content_stats": False,
    }
    assert info["multimodal_processing"] == {
        "enable_image_processing": False,
        "enable_table_processing": True,
        "enable_equation_processing": False,
    }
    assert info["context_extraction"]["context_window"] == 2
    assert info["context_extraction"]["context_mode"] == "chunk"
    assert info["context_extraction"]["max_context_tokens"] == 100
    assert info["context_extraction"]["include_headers"] is False
    assert info["context_extraction"]["include_captions"] is False
    assert info["context_extraction"]["filter_content_types"] == ["text", "table"]
    assert info["batch_processing"] == {
        "max_concurrent_files": 3,
        "supported_file_extensions": [".pdf", ".docx"],
        "recursive_folder_processing": False,
    }


def test_empty_lightrag_kwargs_reports_defaults():
    info = _host(_config(), {}).get_config_info()

    assert info["lightrag_config"]["custom_parameters"] == {}
    assert info["lightrag_config"]["note"] == "Using default LightRAG parameters"


def test_update_config_applies_known_fields_and_drops_unknown_names():
    host = _host(_config())

    host.update_config(
        parse_method="txt",
        parser="mineru",
        enable_image_processing=True,
        not_a_real_key="do-not-attach",
    )

    assert host.config.parse_method == "txt"
    assert host.config.parser == "mineru"
    assert host.config.enable_image_processing is True
    assert not hasattr(host.config, "not_a_real_key")
    assert any("Unknown config parameter" in msg for msg in host.logger.warnings)
    assert host.get_config_info()["parsing"]["parse_method"] == "txt"
    assert host.get_config_info()["parsing"]["parser"] == "mineru"
