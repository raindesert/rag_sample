# 中文维基百科 RAG · 设计文档

**日期**：2026-06-30
**作者**：Billy + Claude
**状态**：已批准，待写实施计划

---

## 1. 目标与范围

**目标**：在本地（8-16 GB CPU 笔记本）以 CLI 形式，对中文维基百科（保留前 10 万行的 `wikipedia-zh-cn-20240820.json`）做语义检索 + 本地 llama.cpp 部署的 Qwen3.5-4B-Q4_K_M 增强问答。

**范围内**：
- 远程（AMD GPU 192 GB）一次性构建 FAISS 索引
- 本地 CLI 问答：检索 → 重排 → 拼 prompt → 流式调用 llama.cpp → 打印引用
- 多轮对话、流式输出、引用来源展示

**范围外**：
- Web UI / HTTP API 服务
- 增量更新（一次性构建）
- 自动评估、模型微调
- 远程建索引以外的 GPU 运行时（远程只跑 `build_index.py` 一次）

---

## 2. 架构

混合拓扑：远程一次性建索引，下载到本地后所有检索与生成都在本地完成。

```
┌────────────────────────┐         ┌────────────────────────────┐
│  远程 AMD GPU          │         │  本地 8-16GB CPU            │
│  (ModelScope Notebook) │         │  (你的笔记本)               │
│                        │         │                            │
│  build_index.py        │  产物   │  rag_cli.py                │
│  - 读 JSONL            │──下载──▶│  - 加载 bge-small INT8     │
│  - 切 chunk            │  data/  │  - 加载 faiss IndexFlatIP  │
│  - bge-small 编码       │         │  - 加载 reranker INT8      │
│  - FAISS 索引          │         │  - 流式调 llama.cpp        │
│                        │         │            │               │
│                        │         │            ▼               │
│                        │         │  llama.cpp (localhost:8080)│
│                        │         │  Qwen3.5-4B-Q4_K_M        │
└────────────────────────┘         └────────────────────────────┘
```

---

## 3. 关键决策

| 维度 | 选择 | 理由 |
|------|------|------|
| 形态 | CLI REPL | 用户选择 |
| 拓扑 | 混合（远程建索引、本地检索+生成） | 用户选择 |
| 切片粒度 | 512 token / 64 overlap | bge-small 原生 512 上下文，粒度与速度均衡 |
| 规模 | 全量 10 万篇（保留前 10 万行） | 用户调整后文件规模 |
| title 策略 | 拼进 chunk text（"标题：xxx\n\n..."） | 提升同名词歧义检索 |
| Embedding 模型 | `BAAI/bge-small-zh-v1.5` | 512 维、~100 MB、中文 SOTA 之一、CPU 可跑 |
| Rerank 模型 | `BAAI/bge-reranker-base` + dynamic INT8 | 显著提升 top-k 质量、CPU 内存减半 |
| 向量库 | FAISS `IndexFlatIP`（不量化） | 35 万 chunk × 512 维 × 4 B ≈ 700 MB，无需量化 |
| 索引类型 | Flat（精确余弦） | 规模小到不需要 IVF；零训练成本 |
| 交互 | 流式输出 + 多轮对话（保留 3 轮） | 用户选择 |
| 引用 | LLM 回答末尾列出条目名 | 用户选择 |
| LLM | llama.cpp OpenAI 兼容 HTTP API（默认 `http://localhost:8080`） | 用户已部署 |
| 切片按自然边界 | 是（在最近 `\n` 处断开） | 避免从句子中间切 |

---

## 4. 存储布局

远程产物（下载到本地 `data/`）：

```
data/
├── wiki.index              # FAISS IndexFlatIP，~700 MB
├── chunks.parquet          # 全部 chunk 元数据 + 文本，~100 MB
└── build_meta.json         # 构建参数（KB 级）
```

**chunks.parquet schema**：

| 列 | 类型 | 说明 |
|---|---|---|
| `chunk_id` | string | `"{doc_id}#{chunk_idx}"`，唯一主键 |
| `doc_id` | string | 原始文章 id |
| `doc_title` | string | 文章标题（用于展示引用） |
| `chunk_idx` | int | 该文章内的 chunk 序号 |
| `chunk_text` | string | 切片后实际文本（含 "标题：..." 前缀） |

**build_meta.json**：

```json
{
  "embedding_model": "BAAI/bge-small-zh-v1.5",
  "dim": 512,
  "chunk_size": 512,
  "chunk_overlap": 64,
  "title_prefix": true,
  "index_type": "IndexFlatIP",
  "total_chunks": 350000,
  "total_docs": 100000,
  "source_file": "wikipedia-zh-cn-20240820.json",
  "source_lines_kept": 100000,
  "created_at": "2026-06-30T..."
}
```

---

## 5. 切片算法（远程本地共用 `chunker.py`）

```python
def chunk_doc(title: str, text: str, size: int = 512, overlap: int = 64):
    full = f"标题：{title}\n\n{text}"
    step_chars = int(size * 1.5)        # ~768 字符
    overlap_chars = int(overlap * 1.5)  # ~96 字符
    stride = step_chars - overlap_chars
    chunks = []
    i = 0
    while i < len(full):
        end = min(i + step_chars, len(full))
        chunk = full[i:end]
        if end < len(full):
            last_nl = chunk.rfind('\n')
            if last_nl > step_chars // 2:
                chunk = chunk[:last_nl]
        chunks.append(chunk)
        i += stride
    return chunks
```

**预估产出**：
- 10 万篇 × 平均 3.5 chunk ≈ **35 万 chunks**
- 索引 700 MB、parquet 100 MB、meta KB 级

---

## 6. 数据流（CLI 单轮问答）

```
用户输入 q
  → embed(q) → vec[512]
  → faiss.search(vec, k=50) → [(idx, score), ...]
  → chunks.parquet 取 50 行 → [(chunk_id, doc_title, chunk_text), ...]
  → rerank(q, [text1..text50]) INT8 → top-5
  → 拼 prompt：system + 历史(≤3轮) + context[top-5] + q
  → llama.cpp /v1/chat/completions stream=true，逐 token 打印
  → 打印"引用：[1] doc_title_1, [2] doc_title_2, ..."
```

---

## 7. CLI 行为

**启动**：

```bash
python rag_cli.py
# 默认配置：./data/、http://localhost:8080、topk_fetch=50、topk_rerank=5、history_turns=3

# 自定义
python rag_cli.py \
  --index data/wiki.index \
  --chunks data/chunks.parquet \
  --meta data/build_meta.json \
  --llama-url http://localhost:8080 \
  --llama-model qwen3.5-4b-q4 \
  --topk-fetch 50 \
  --topk-rerank 5 \
  --history-turns 3
```

**REPL 内置命令**：

| 命令 | 作用 |
|------|------|
| `/clear` | 清空多轮历史 |
| `/quit` / `exit` | 退出 |
| `/stats` | 显示本次会话：提问次数、平均检索+rerank+生成耗时 |
| `/show <n>` | 显示第 n 个引用的完整 chunk 文本 |

**多轮对话**：
- 维护 `history: list[dict]`，只保留最近 N 轮（默认 3 轮 = 6 条消息）
- 不做 query 改写（4B 模型改写质量不稳），直接把"当前问题 + 前 N 轮历史"交给 LLM

**Prompt 模板**（Qwen ChatML 格式）：

```
<|im_start|>system
你是一个基于维基百科的中文知识助手。请仅根据"参考资料"回答问题，不要编造。
回答中引用的事实需要标注来源编号，如 [1][2]。<|im_end|>
<|im_start|>user
参考资料：
[1] {doc_title_1}
{chunk_text_1}

[2] {doc_title_2}
{chunk_text_2}

...

[5] {doc_title_5}
{chunk_text_5}

问题：{query}<|im_end|>
<|im_start|>assistant
```

---

## 8. 错误处理

| 场景 | 处理 |
|------|------|
| llama.cpp 服务未起 | 启动时 health check，失败给明确提示 + 启动命令建议 |
| 索引/元数据文件缺失 | 提示先在远程跑 `build_index.py` |
| 检索 top-50 全低于阈值 | LLM 回答"未找到相关信息"，不强行编造 |
| 单个 chunk 文本超 6000 字符 | 截断到 6000（Qwen 4B 8K 上下文留余量给问题+历史） |
| 流式中断（Ctrl+C） | 优雅退出、保留历史、清空显存句柄 |
| parquet / faiss 读取异常 | 明确报错 + 提示重新下载产物 |

---

## 9. 测试策略

**单元测试**（`tests/test_chunker.py`）：
- title 拼接正确
- overlap 大小正确
- 长文能切到自然边界（不在句子中间断）
- 空 text / 极短 text / 极长 text 边界
- chunk_id 格式正确

**冒烟测试**（`scripts/smoke_test.py`，远程跑完索引后跑）：
- 加载索引，随机抽 5 个 chunk，用其内容查回自己
- 跑 3 个固定问题（"牛顿第二定律""爱因斯坦是谁""北京是哪个国家的首都"），人工看 top-5 是否合理

**人工端到端**（CLI 启动后）：
- 5-10 个真实问题跑一遍，重点测：
  - 同名词歧义（"苹果"应召回苹果公司 + 水果苹果）
  - 长尾知识（极冷门条目）
  - 多轮指代（"它和第三定律有什么关系？"）

**不写**：
- embedding 数值精度测试
- 性能基准（除非发现实际慢到不能用）

---

## 10. 依赖

**远程 `build_index.py`**：

```
torch>=2.0
sentence-transformers>=2.7
faiss-gpu>=1.7
tqdm
pyarrow>=12
```

**本地 `rag_cli.py`**：

```
faiss-cpu>=1.7
sentence-transformers>=2.7
pyarrow>=12
requests>=2.31
tiktoken        # 切片 token 估算（可选，远程用）
```

> `sentence-transformers` 会拉 PyTorch。本地可选 `pip install --no-deps sentence-transformers` + 手动装 CPU 版 torch 减小体积。

---

## 11. 文件结构（最终）

```
D:\work\python\rag_sample\
├── build_index.py             # 远程跑一次
├── rag_cli.py                 # 本地 CLI
├── wiki_chat/
│   ├── __init__.py
│   ├── config.py              # 默认参数、CLI 参数解析
│   ├── chunker.py             # 远程本地共用
│   └── prompts.py             # system prompt 模板
├── tests/
│   ├── __init__.py
│   └── test_chunker.py
├── scripts/
│   └── smoke_test.py
├── data/                      # 远程产物，下载到此
│   ├── wiki.index
│   ├── chunks.parquet
│   └── build_meta.json
├── docs/superpowers/specs/
│   └── 2026-06-30-wiki-rag-design.md  # 本文档
└── requirements.txt
```

---

## 12. 风险与权衡

| 风险 | 缓解 |
|------|------|
| llama.cpp Qwen3.5-4B Q4 CPU 流式估 5-15 tok/s，体感偏慢 | 接受，CLI 不强求实时；模型本来就走 CPU |
| rerank 模型 INT8 量化后精度损失 | bge-reranker-base 量化容忍度高，预期损失 < 1% |
| 10 万篇里有大量短条目（多义词、消歧页、列表） | 接受；冷门条目检索质量靠 rerank 兜底 |
| 远程 → 本地索引文件下载 ~800 MB | 一次性，可接受 |
| 多轮对话不做 query 改写 | 4B 模型改写质量不稳；当前轮直接喂 LLM 让它自己处理指代 |

---

## 13. 后续待办（不在本次实施范围）

- 接 Web UI（Gradio）
- 增量更新索引
- 多语言（en/zh 混合）
- 自动评估（Recall@k 用 held-out 问答对）