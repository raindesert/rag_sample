"""文档切片：title 拼接 + 固定字符步长 + 自然边界。

远程 build_index.py 与（潜在）re-chunk 共用，保证切片一致。
"""


def chunk_doc(title: str, text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """把单篇文章切成 chunk。

    每个 chunk 开头为 "标题：{title}\n\n{text_fragment}"。
    按字符步长滑动，size 默认 512 token（≈ 768 字符）。
    接近末尾时若 chunk 中部之后有 \n，截断到该处（自然边界）。
    """
    if not text:
        return []
    full = f"标题：{title}\n\n{text}"
    header_len = len(full) - len(text)  # "标题：{title}\n\n" 长度
    step_chars = int(size * 1.5)
    overlap_chars = int(overlap * 1.5)
    stride = max(step_chars - overlap_chars, 1)
    chunks: list[str] = []
    i = 0
    while i < len(full):
        end = min(i + step_chars, len(full))
        chunk = full[i:end]
        if end < len(full):
            last_nl = chunk.rfind("\n", max(0, header_len - i))
            if last_nl > 0:
                chunk = chunk[:last_nl]
        chunks.append(chunk)
        if end >= len(full):
            break
        i += stride
    return chunks


def chunk_id_for(doc_id: str, chunk_idx: int) -> str:
    """生成唯一 chunk_id。"""
    return f"{doc_id}#{chunk_idx}"
