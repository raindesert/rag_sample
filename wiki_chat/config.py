"""CLI 配置：默认值 + argparse 解析。"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """所有可配置项及默认值。"""

    index_path: Path = field(default_factory=lambda: Path("data/wiki.index"))
    chunks_path: Path = field(default_factory=lambda: Path("data/chunks.parquet"))
    meta_path: Path = field(default_factory=lambda: Path("data/build_meta.json"))
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    llama_url: str = "http://localhost:8080"
    llama_model: str = "qwen3.5-4b-q4"
    topk_fetch: int = 50
    topk_rerank: int = 5
    history_turns: int = 3
    from_modelscope: bool = False

    def __post_init__(self):
        # Always coerce to Path (so .exists() works on all platforms)
        self.index_path = Path(self.index_path)
        self.chunks_path = Path(self.chunks_path)
        self.meta_path = Path(self.meta_path)
        if not self.chunks_path.is_absolute():
            self.chunks_path = self.index_path.parent / self.chunks_path.name
        if not self.meta_path.is_absolute():
            self.meta_path = self.index_path.parent / self.meta_path.name


def parse_args(argv: list[str] | None = None) -> Config:
    """CLI 参数解析，返回 Config。"""
    p = argparse.ArgumentParser(description="Wiki RAG CLI")
    p.add_argument("--index", dest="index_path", default="data/wiki.index")
    p.add_argument("--chunks", dest="chunks_path", default="data/chunks.parquet")
    p.add_argument("--meta", dest="meta_path", default="data/build_meta.json")
    p.add_argument("--llama-url", default="http://localhost:8080")
    p.add_argument("--llama-model", default="qwen3.5-4b-q4")
    p.add_argument("--topk-fetch", type=int, default=50)
    p.add_argument("--topk-rerank", type=int, default=5)
    p.add_argument("--history-turns", type=int, default=3)
    p.add_argument(
        "--from-modelscope",
        dest="from_modelscope",
        action="store_true",
        help="从 ModelScope 下载 embedding 模型（默认 HuggingFace）",
    )
    ns = p.parse_args(argv)
    return Config(
        index_path=ns.index_path,
        chunks_path=ns.chunks_path,
        meta_path=ns.meta_path,
        llama_url=ns.llama_url,
        llama_model=ns.llama_model,
        topk_fetch=ns.topk_fetch,
        topk_rerank=ns.topk_rerank,
        history_turns=ns.history_turns,
        from_modelscope=ns.from_modelscope,
    )
