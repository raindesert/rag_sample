# wiki_chat · 中文维基百科 RAG

基于中文维基百科（10 万篇）+ bge-small-zh-v1.5 embedding + bge-reranker-base + Qwen3.5-4B-Q4_K_M (llama.cpp) 的本地 CLI 问答。

## 架构

远程 AMD GPU 一次性建索引 → 下载到本地 → 本地 CLI 检索 + 生成。

详见 `docs/superpowers/specs/2026-06-30-wiki-rag-design.md`。

## 使用

### 1. 远程构建索引（在 ModelScope Notebook 上）

```bash
pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com python build_index.py \
  --input wikipedia-zh-cn.fixed.json \
  --output-dir data \
  --max-lines 100000 \
  --batch-size 128
```

> ⚠️ 源数据文件 `wikipedia-zh-cn.json` 是损坏的 UTF-8（被错误地当作 Latin-1 重新标记）。先用 `python scripts/fix_encoding.py` 生成 `wikipedia-zh-cn.fixed.json`，再构建索引。
>
> ⚠️ HuggingFace 在中国大陆访问受限，设置 `HF_ENDPOINT=https://hf-mirror.com` 用镜像。

### 2. 下载产物到本地

`data/wiki.index`、`data/chunks.parquet`、`data/build_meta.json` 三个文件。

### 3. 启动 llama.cpp 服务

```bash
# 假设已用 llama.cpp 部署 Qwen3.5-4B-Q4_K_M，监听 8080
curl http://localhost:8080/v1/models
```

### 4. 启动 CLI

```bash
pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com python rag_cli.py
```

## 命令

- `/help` 查看内置命令
- `/quit` 退出
- `/clear` 清空多轮历史
- `/stats` 显示统计
- `/show <n>` 显示第 n 个引用完整文本

## 测试

```bash
python -m pytest tests/ -v
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/smoke_test.py
```

## 故障排查

- **`ModuleNotFoundError: No module named 'faiss'`**：在远程装 `faiss-gpu`，本地装 `faiss-cpu`。
- **`Connection refused` llama.cpp**：先启动 llama.cpp 服务，确认 `curl http://localhost:8080/v1/models` 有响应。
- **索引文件不存在**：先跑远程 `build_index.py`，把产物放到 `data/`。
- **HuggingFace 下载慢/失败**：设置 `HF_ENDPOINT=https://hf-mirror.com`。
- **`datetime.utcnow()` deprecation warning**：Python 3.12+ 显示，无害。
- **rerank INT8 量化**：本项目因 transformers 5.x 兼容性已禁用 INT8 量化。原始 plan 中提到但兼容性不好。
- **数据含 U+FFFD 乱码**：`wikipedia-zh-cn.fixed.json` 中 96.86% 的行含不可恢复字符（源文件本身损坏），RAG 整体可用但部分条目结尾有 �。