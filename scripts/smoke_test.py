"""索引 + 检索冒烟测试。

用法：
    python scripts/smoke_test.py
"""

import sys
from pathlib import Path

# 让脚本能从仓库根目录 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_cli import load_resources, retrieve
from wiki_chat.config import parse_args


FIXED_QUERIES = [
    "牛顿第二定律",
    "爱因斯坦是谁",
    "北京的首都",
]


def main() -> int:
    cfg = parse_args([])
    res = load_resources(cfg)
    print(f"[*] 索引 ntotal={res.index.ntotal}")

    failures = 0
    for q in FIXED_QUERIES:
        print(f"\n[Q] {q}")
        hits = retrieve(res, q)
        if not hits:
            print("  [FAIL] 无结果")
            failures += 1
            continue
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] 《{h['doc_title']}》 (rerank: {h['rerank_score']:.3f})")

    print(f"\n完成。{failures} 个查询无结果。")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
