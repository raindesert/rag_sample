"""本地 CLI：加载索引、检索、流式生成、多轮 REPL。"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer

from wiki_chat.config import Config, parse_args
from wiki_chat.prompts import build_messages


@dataclass
class Resources:
    cfg: Config
    index: faiss.Index
    chunks: object  # pa.Table
    embedder: SentenceTransformer
    reranker: CrossEncoder
    # 运行时态（不是构造参数）
    ask_count: int = 0
    last_retrieved: list[dict] = field(default_factory=list)


def load_resources(cfg: Config) -> Resources:
    """一次性加载索引、embedding、rerank 模型。"""
    if not cfg.index_path.exists():
        raise FileNotFoundError(
            f"索引文件不存在: {cfg.index_path}\n"
            "请先在远程跑 build_index.py，并把产物放到 data/ 目录。"
        )
    index = faiss.read_index(str(cfg.index_path))
    chunks = pq.read_table(str(cfg.chunks_path))
    print(f"[*] 加载 embedding 模型 {cfg.embedding_model}...")
    embedder = SentenceTransformer(cfg.embedding_model, device="cpu")
    print(f"[*] 加载 reranker {cfg.rerank_model}...")
    reranker = CrossEncoder(cfg.rerank_model, device="cpu")
    return Resources(
        cfg=cfg, index=index, chunks=chunks,
        embedder=embedder, reranker=reranker,
    )


def retrieve(res: Resources, query: str) -> list[dict]:
    """query → embed → faiss.search → 取元数据 → rerank → top-n。"""
    q_vec = res.embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    scores, idxs = res.index.search(q_vec, res.cfg.topk_fetch)

    candidates = []
    for idx, score in zip(idxs[0], scores[0]):
        if idx < 0:
            continue
        row = res.chunks.slice(idx, 1).to_pylist()[0]
        candidates.append({
            "doc_title": row["doc_title"],
            "chunk_text": row["chunk_text"],
            "chunk_id": row["chunk_id"],
            "faiss_score": float(score),
        })

    if not candidates:
        return []

    pairs = [(query, c["chunk_text"]) for c in candidates]
    rerank_scores = res.reranker.predict(pairs, show_progress_bar=False)
    for c, rs in zip(candidates, rerank_scores):
        c["rerank_score"] = float(rs)
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[: res.cfg.topk_rerank]


def main() -> int:
    cfg = parse_args()
    try:
        res = load_resources(cfg)
    except FileNotFoundError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    print(f"[*] 索引 {res.index.ntotal} chunks，准备好（输入 /quit 退出，/help 查看命令）")
    # 完整 REPL 在 Task 9 实现
    return 0


if __name__ == "__main__":
    sys.exit(main())