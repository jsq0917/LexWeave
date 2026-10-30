"""中文文本处理小工具：分词、停用词、轻量归一化。"""

from __future__ import annotations

import re
from functools import lru_cache

import jieba

STOPWORDS = set(
    """
的 了 和 与 及 或 在 对 由 向 从 到 为 以 因 其 该 这 那 是 有 无 不 得 应当 可以 依照 根据 本法
之 一 二 三 四 五 六 七 八 九 十 条 款 项 第 规定 情形 下列 其他 以及 但是 如果 或者 并 就 都 也
""".split()
)

_PUNCT = re.compile(r"[\s，。；、：？！“”‘’（）《》【】\-—…·.,;:!?()\[\]{}<>\"'`~|/\\@#$%^&*+=_]")
_CHAR_STOP = set("的了和与及或是在对由向从到为以因其该这那")


@lru_cache(maxsize=1)
def _jieba_init() -> bool:
    jieba.setLogLevel(20)
    return True


def tokenize(text: str) -> list[str]:
    _jieba_init()
    text = _PUNCT.sub(" ", text or "")
    toks = []
    for tok in jieba.lcut(text):
        tok = tok.strip()
        if len(tok) < 2 or tok in STOPWORDS:
            if len(tok) == 1 and tok not in _CHAR_STOP and tok.strip():
                toks.append(tok)
            continue
        toks.append(tok)
    return toks


def keyword_terms(text: str, top: int = 12) -> list[str]:
    """按词频 + 长度粗排的关键词，用于查询特征与日志展示。"""
    counts: dict[str, int] = {}
    for t in tokenize(text):
        counts[t] = counts.get(t, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (kv[1] * (1 + len(kv[0]) * 0.2)), reverse=True)
    return [k for k, _ in ranked[:top]]


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", _PUNCT.sub("", text or ""))


def char_bigrams(text: str) -> set[str]:
    t = normalize(text)
    return {t[i : i + 2] for i in range(len(t) - 1)}
