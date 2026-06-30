import sys
from pathlib import Path

from wiki_chat.config import Config, parse_args


def test_default_config_values():
    cfg = Config()
    assert cfg.index_path.name == "wiki.index"
    assert cfg.chunks_path.name == "chunks.parquet"
    assert cfg.meta_path.name == "build_meta.json"
    assert cfg.llama_url == "http://localhost:8080"
    assert cfg.topk_fetch == 50
    assert cfg.topk_rerank == 5
    assert cfg.history_turns == 3


def test_parse_args_overrides():
    argv_save = sys.argv
    try:
        sys.argv = [
            "rag_cli.py",
            "--llama-url", "http://gpu-7:9090",
            "--topk-fetch", "30",
            "--topk-rerank", "8",
            "--history-turns", "5",
        ]
        cfg = parse_args()
        assert cfg.llama_url == "http://gpu-7:9090"
        assert cfg.topk_fetch == 30
        assert cfg.topk_rerank == 8
        assert cfg.history_turns == 5
    finally:
        sys.argv = argv_save


def test_chunks_and_meta_resolve_relative_to_index_dir():
    cfg = Config(index_path=Path("/tmp/x/wiki.index"))
    # Compare Path objects, not str (WindowsPath vs PosixPath stringification differs)
    assert cfg.chunks_path == Path("/tmp/x/chunks.parquet")
    assert cfg.meta_path == Path("/tmp/x/build_meta.json")


def test_config_paths_have_exists_method():
    """PurePosixPath has no .exists(); ensure Config stores real Path objects."""
    cfg = Config()
    assert hasattr(cfg.index_path, "exists")
    assert hasattr(cfg.chunks_path, "exists")
    assert hasattr(cfg.meta_path, "exists")
