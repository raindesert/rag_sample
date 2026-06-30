"""远程一次性索引构建脚本。

读 wikipedia-zh-cn-20240820.json（前 N 行），切片，用 bge-small-zh-v1.5 编码，
构建 FAISS IndexFlatIP，保存 wiki.index + chunks.parquet + build_meta.json。
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
from sentence_transformers import SentenceTransformer

from wiki_chat.chunker import chunk_doc, chunk_id_for


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build wiki RAG index")
    p.add_argument("--input", required=True, help="Path to wikipedia-zh-cn JSONL")
    p.add_argument("--output-dir", required=True, help="Output dir for index/chunks/meta")
    p.add_argument("--max-lines", type=int, default=100_000, help="Number of JSONL lines to read")
    p.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--chunk-overlap", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    return p.parse_args()


def read_jsonl(path: Path, max_lines: int) -> list[dict]:
    """流式读 JSONL，最多 max_lines 条。"""
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def chunk_all(docs: list[dict], size: int, overlap: int) -> list[dict]:
    """对所有文档切片，返回 chunk 字典列表。"""
    out: list[dict] = []
    for doc in docs:
        text = doc.get("text", "")
        chunks = chunk_doc(doc.get("title", ""), text, size=size, overlap=overlap)
        for idx, ch in enumerate(chunks):
            out.append({
                "chunk_id": chunk_id_for(doc["id"], idx),
                "doc_id": doc["id"],
                "doc_title": doc.get("title", ""),
                "chunk_idx": idx,
                "chunk_text": ch,
            })
    return out


def embed_all(chunks: list[dict], model_name: str, batch_size: int):
    """对所有 chunk 文本做 embedding，返回 numpy 数组。"""
    model = SentenceTransformer(model_name)
    texts = [c["chunk_text"] for c in chunks]
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # 余弦等价于内积
    )
    return vecs


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Reading up to {args.max_lines} lines from {args.input}...")
    t0 = time.time()
    docs = read_jsonl(Path(args.input), args.max_lines)
    print(f"      Read {len(docs)} docs in {time.time() - t0:.1f}s")

    print(f"[2/4] Chunking (size={args.chunk_size}, overlap={args.chunk_overlap})...")
    t0 = time.time()
    chunks = chunk_all(docs, args.chunk_size, args.chunk_overlap)
    print(f"      Produced {len(chunks)} chunks in {time.time() - t0:.1f}s")

    print(f"[3/4] Embedding with {args.embedding_model} (batch={args.batch_size})...")
    t0 = time.time()
    vecs = embed_all(chunks, args.embedding_model, args.batch_size)
    print(f"      Embedded {len(vecs)} chunks in {time.time() - t0:.1f}s")

    print(f"[4/4] Saving to {out_dir}...")
    save(chunks, vecs, args, out_dir)
    print("Done.")
    return 0


def save(chunks: list[dict], vecs, args: argparse.Namespace, out_dir: Path) -> None:
    """Save FAISS index + chunks parquet + build_meta.json."""
    arr = np.asarray(vecs, dtype="float32")
    dim = arr.shape[1]

    # FAISS IndexFlatIP：向量已 normalize_embeddings=True，内积=余弦
    index = faiss.IndexFlatIP(dim)
    index.add(arr)

    index_path = out_dir / "wiki.index"
    faiss.write_index(index, str(index_path))

    # chunks.parquet
    table = pa.Table.from_pylist(chunks)
    pq.write_table(table, out_dir / "chunks.parquet")

    # build_meta.json
    meta = {
        "embedding_model": args.embedding_model,
        "dim": dim,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "title_prefix": True,
        "index_type": "IndexFlatIP",
        "total_chunks": len(chunks),
        "total_docs": len({c["doc_id"] for c in chunks}),
        "source_file": str(args.input),
        "source_lines_kept": args.max_lines,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
    }
    (out_dir / "build_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"      Saved: {index_path} ({index_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
