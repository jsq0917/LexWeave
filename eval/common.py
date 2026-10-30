"""评测公用件：数据加载、文本重合度、Cohen's κ、LLM 结果缓存。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score


def bigrams(text: str) -> set[str]:
    text = re.sub(r"[\s，。；、：？！“”‘’（）《》]", "", text)
    return {text[i : i + 2] for i in range(len(text) - 1)}


def bigram_overlap(question: str, passage: str) -> float:
    """题面与法条原文的二元组重合率：用于检测「抄原文」式泄漏。"""
    q = bigrams(question)
    p = bigrams(passage)
    if not q or not p:
        return 0.0
    return len(q & p) / len(q)


def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    if len(labels_a) != len(labels_b) or len(set(labels_a) | set(labels_b)) < 2:
        return float("nan")
    return float(cohen_kappa_score(labels_a, labels_b))


class JsonCache:
    """LLM 结果缓存：同一个 prompt 只花一次钱，重跑评测结果稳定。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def key(*parts: str) -> str:
        raw = "\u0000".join(parts)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
