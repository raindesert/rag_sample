"""本地 CLI：加载索引、检索、流式生成、多轮 REPL。"""

import json
import sys
from dataclasses import dataclass, field

import faiss
import pyarrow.parquet as pq
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer

from wiki_chat.config import Config, parse_args
from wiki_chat.models import load_cross_encoder, load_sentence_transformer
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


# rerank 分数低于此阈值的 chunk 直接丢弃，避免无关内容进 prompt。
# spec §8：top-50 全低于阈值时 LLM 应答 "未找到相关信息"。
RERANK_THRESHOLD = 0.99


def load_resources(cfg: Config) -> Resources:
    """一次性加载索引、embedding、rerank 模型。"""
    if not cfg.index_path.exists():
        raise FileNotFoundError(
            f"索引文件不存在: {cfg.index_path}\n"
            "请先在远程跑 build_index.py，并把产物放到 data/ 目录。"
        )
    index = faiss.read_index(str(cfg.index_path))
    chunks = pq.read_table(str(cfg.chunks_path))
    source = "ModelScope" if cfg.from_modelscope else "HuggingFace"
    print(f"[*] 加载 embedding 模型 {cfg.embedding_model} ({source})...")
    embedder = load_sentence_transformer(cfg.embedding_model, cfg.from_modelscope, device="cpu")
    source = "ModelScope" if cfg.from_modelscope else "HuggingFace"
    print(f"[*] 加载 reranker {cfg.rerank_model} ({source})...")
    reranker = load_cross_encoder(cfg.rerank_model, cfg.from_modelscope, device="cpu")
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

    kept = [c for c in candidates if c["rerank_score"] >= RERANK_THRESHOLD]
    return kept[: res.cfg.topk_rerank]


def stream_answer(cfg: Config, messages: list[dict]) -> str:
    """调用 llama.cpp OpenAI 兼容 chat/completions 流式接口，逐 token 打印。"""
    url = f"{cfg.llama_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": cfg.llama_model,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
    }
    full = ""
    with requests.post(url, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        # 强制使用 UTF-8，避免流式响应编码推断错误导致乱码
        r.encoding = "utf-8"
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta:
                print(delta, end="", flush=True)
                full += delta
    print()  # 末尾换行
    return full


def main() -> int:
    cfg = parse_args()
    try:
        res = load_resources(cfg)
    except FileNotFoundError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    print(f"[*] 索引 {res.index.ntotal} chunks，准备好。输入 /quit 退出，/help 查看命令。")

    history: list[dict] = []
    max_history = cfg.history_turns * 2

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/quit", "exit", "/exit"):
            break
        if q == "/clear":
            history = []
            print("[已清空历史]")
            continue
        if q == "/help":
            print("命令：/quit 退出 /clear 清空历史 /stats 统计 /show <n> 显示第 n 个引用")
            continue
        if q.startswith("/show "):
            try:
                n = int(q.split()[1]) - 1
                item = res.last_retrieved[n]
                print(f"[{n+1}] 《{item['doc_title']}》\n{item['chunk_text']}")
            except (ValueError, IndexError):
                print("[错误] 用法: /show <编号>")
            continue
        if q == "/stats":
            print(f"已提问 {res.ask_count} 次")
            continue

        # 主流程：检索 + 生成
        try:
            retrieved = retrieve(res, q)
        except Exception as e:
            print(f"[检索失败] {e}", file=sys.stderr)
            continue

        if not retrieved:
            print("[未找到相关内容]")
            continue

        print()
        for i, c in enumerate(retrieved, start=1):
            print(f"[{i}] 《{c['doc_title']}》 (rerank: {c['rerank_score']:.3f})")

        messages = build_messages(q, retrieved, history=history)
        try:
            answer = stream_answer(cfg, messages)
        except requests.RequestException as e:
            print(f"\n[生成失败] {e}", file=sys.stderr)
            continue

        titles = "、".join(f"[{i+1}] {c['doc_title']}" for i, c in enumerate(retrieved))
        print(f"\n引用：{titles}")

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
        if len(history) > max_history:
            history = history[-max_history:]

        res.ask_count += 1
        res.last_retrieved = retrieved

    print("再见。")
    return 0


if __name__ == "__main__":
    sys.exit(main())