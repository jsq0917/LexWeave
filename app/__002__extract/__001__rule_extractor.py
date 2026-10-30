"""规则抽取：别名词典扫描（概念提及）+ 法条互引正则（CITES）。

设计取舍
* 规则层是**主干**：无 LLM 时也能建出完整图，评测可复现；
* 别名词典用「首字索引 + 最长优先」扫描，1465 条语料毫秒级；
* 提及置信度 = 别名特异性（1 - 文档频率）+ 长度因子，通用词（工资、期间）自然降权；
* CITES 只在条号真实存在时才落边，避免「第X条」误配到不存在的条文。
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.law_utils import LAW_CODE_TO_FULL, LAW_FULL_TO_CODE, article_no_to_int  # noqa: E402
from common.path_utils import read_json  # noqa: E402

MIN_ALIAS_LEN = 2
# P3 扩词典后概念数从 218 增至 291，12 的上限会让新概念挤掉既有金标概念；
# 实测提到 16 时新旧边零丢失（旧边 3621 条全部保留），故上调。
MAX_CONCEPTS_PER_ARTICLE = 16
MIN_CONCEPT_CONFIDENCE = 0.5

_CITE_RE = re.compile(r"第([零一二三四五六七八九十百千]+|\d+)条")
_EXPLICIT_LAW_RE = re.compile(r"(《?中华人民共和国(?:劳动合同法|劳动法|民法典)》?)")
_LAW_ALIAS = {
    "劳动合同法": "中华人民共和国劳动合同法",
    "劳动法": "中华人民共和国劳动法",
    "民法典": "中华人民共和国民法典",
}


@dataclass
class Mention:
    alias: str
    start: int
    end: int
    concept_ids: list[str]


class AliasMatcher:
    def __init__(self, alias_index: dict[str, list[dict]]) -> None:
        self.by_first: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
        for alias, entries in alias_index.items():
            if len(alias) < MIN_ALIAS_LEN:
                continue
            ids = sorted({e["id"] for e in entries if e["kind"] == "concept"})
            if not ids:
                continue
            self.by_first[alias[0]].append((alias, ids))
        for bucket in self.by_first.values():
            bucket.sort(key=lambda x: len(x[0]), reverse=True)
        self.max_len = max((len(a) for b in self.by_first.values() for a, _ in b), default=1)

    def find(self, text: str) -> list[Mention]:
        """返回所有命中（允许重叠）。

        早期版本做「最长优先 + 跳过长词」，会让「享受基本养老保险待遇」吞掉其中的
        「基本养老保险待遇」，导致概念漏抽。改为逐字推进、收集全部匹配，
        长词与短词各算各的，文档频率与置信度会自然抑制泛词。
        """
        out: list[Mention] = []
        n = len(text)
        for i in range(n):
            bucket = self.by_first.get(text[i])
            if not bucket:
                continue
            for alias, ids in bucket:
                if text.startswith(alias, i):
                    out.append(Mention(alias, i, i + len(alias), ids))
        return out


def alias_document_frequency(articles: Iterable[dict], alias_index: dict[str, list[dict]]) -> dict[str, int]:
    matcher = AliasMatcher(alias_index)
    df: Counter[str] = Counter()
    for art in articles:
        for m in matcher.find(art["content"]):
            df[m.alias] += 1
    return dict(df)


def mention_confidence(alias: str, df: int, n_articles: int, is_primary: bool, chapter_hit: bool) -> float:
    specificity = 1.0 - min(df / max(n_articles, 1), 1.0)
    length_factor = min(len(alias), 6) / 6
    conf = 0.30 + 0.35 * specificity + 0.25 * length_factor + (0.05 if is_primary else 0.0) + (
        0.05 if chapter_hit else 0.0
    )
    return round(min(conf, 0.99), 3)


def extract_mentions(
    articles: list[dict],
    concepts: dict[str, dict],
    alias_index: dict[str, list[dict]],
    df: dict[str, int],
) -> list[dict]:
    """每条文章输出若干 (article_id, concept_id) 提及记录。"""
    matcher = AliasMatcher(alias_index)
    n = len(articles)
    rows: list[dict] = []
    for art in articles:
        text = art["content"]
        chapter = art.get("chapter_path", "") or ""
        counts: Counter[str] = Counter()
        aliases: dict[str, set[str]] = defaultdict(set)
        primary: dict[str, bool] = defaultdict(bool)
        for m in matcher.find(text):
            for cid in m.concept_ids:
                counts[cid] += 1
                aliases[cid].add(m.alias)
                is_primary = m.alias == concepts.get(cid, {}).get("name")
                primary[cid] = primary[cid] or is_primary
        scored = []
        for cid, cnt in counts.items():
            best_alias = max(aliases[cid], key=len)
            d = df.get(best_alias, cnt)
            chapter_hit = any(a in chapter for a in aliases[cid]) or concepts.get(cid, {}).get("name", "") in chapter
            conf = mention_confidence(best_alias, d, n, primary[cid], chapter_hit)
            if conf < MIN_CONCEPT_CONFIDENCE:
                continue
            scored.append((conf, cnt, cid, sorted(aliases[cid])))
        scored.sort(reverse=True)
        for conf, cnt, cid, als in scored[:MAX_CONCEPTS_PER_ARTICLE]:
            rows.append(
                {
                    "article_id": art["article_id"],
                    "concept_id": cid,
                    "aliases": als,
                    "count": cnt,
                    "confidence": conf,
                    "provenance": "rule",
                }
            )
    return rows


def _resolve_target_law(text: str, pos: int, default_law: str) -> tuple[str, float]:
    window = text[max(0, pos - 18) : pos]
    m = _EXPLICIT_LAW_RE.search(window)
    if m:
        raw = m.group(1)
        for alias, full in _LAW_ALIAS.items():
            if alias in raw:
                return full, 0.95
    if "本法" in window:
        return default_law, 0.9
    return default_law, 0.75


def extract_citations(articles: list[dict], valid_ids: set[str]) -> list[dict]:
    rows: list[dict] = []
    for art in articles:
        text = art["content"]
        src = art["article_id"]
        seen: set[str] = set()
        for m in _CITE_RE.finditer(text):
            try:
                num = article_no_to_int(m.group(0))
            except ValueError:
                continue
            law, conf = _resolve_target_law(text, m.start(), art["law_name"])
            code = LAW_FULL_TO_CODE.get(law)
            if not code:
                continue
            target = f"{code}#{num}"
            if target == src or target not in valid_ids or target in seen:
                continue
            seen.add(target)
            rows.append(
                {
                    "source": src,
                    "target": target,
                    "article_no": m.group(0),
                    "confidence": conf,
                    "provenance": "regex",
                }
            )
    return rows


def load_lexicon(lexicon_dir: Path | None = None) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    from config import settings
    from common.path_utils import read_jsonl

    d = lexicon_dir or settings.lexicon_dir
    concepts = {r["id"]: r for r in read_jsonl(d / "concepts.jsonl")}
    alias_index = read_json(d / "alias_index.json")
    return concepts, alias_index


__all__ = [
    "AliasMatcher",
    "extract_mentions",
    "extract_citations",
    "alias_document_frequency",
    "load_lexicon",
    "LAW_CODE_TO_FULL",
]
