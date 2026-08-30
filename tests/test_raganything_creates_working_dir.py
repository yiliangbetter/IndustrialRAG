"""RAGAnything must create a missing working_dir during init.

Storage writes and parse cache live under working_dir. A missing directory
turns the first ingest into an opaque filesystem error instead of a usable
instance.
"""

import pytest


def test_missing_working_dir_is_created(monkeypatch, tmp_path):
    pytest.importorskip("lightrag")

    import raganything.raganything as rag_module
    from raganything.config import RAGAnythingConfig

    class StubParser:
        def check_installation(self):
            return True

    monkeypatch.setattr(rag_module, "get_parser", lambda _name: StubParser())
    monkeypatch.setattr(rag_module.atexit, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(rag_module.atexit, "unregister", lambda *args, **kwargs: None)

    workdir = tmp_path / "fresh_rag_storage"
    assert not workdir.exists()

    rag = rag_module.RAGAnything(
        config=RAGAnythingConfig(working_dir=str(workdir), parser="mineru")
    )

    assert workdir.is_dir()
    assert rag.working_dir == str(workdir)
