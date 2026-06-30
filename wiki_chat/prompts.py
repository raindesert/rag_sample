"""Prompt 模板（Qwen ChatML）。"""

SYSTEM_PROMPT = (
    "你是一个基于维基百科的中文知识助手。"
    "请仅根据下面的\"参考资料\"回答用户的问题，不要使用你自己的先验知识，不要编造。"
    "如果参考资料不足，请直接说\"未找到相关信息\"。"
    "回答中引用的事实请用来源编号标注，如 [1][2]。"
)


def build_user_prompt(query: str, retrieved: list[dict]) -> str:
    """拼 user 段：参考资料 + 问题。"""
    lines = ["参考资料："]
    for i, item in enumerate(retrieved, start=1):
        lines.append(f"[{i}] {item['doc_title']}")
        lines.append(item["chunk_text"])
        lines.append("")
    lines.append(f"问题：{query}")
    return "\n".join(lines)


def build_messages(query: str, retrieved: list[dict], history: list[dict] | None = None) -> list[dict]:
    """构造完整 messages：system + 历史 + user(query with context)。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": build_user_prompt(query, retrieved)})
    return messages