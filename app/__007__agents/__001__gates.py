""""五道闸"：把 LexSynergy 里那套「生成后再校验」的机制迁到图问答路径。

五道闸全部是**确定性规则**，不额外调用 LLM —— 这是刻意的：
B6 要回答的是「校验编排本身有没有价值」，如果闸门自己也要花 LLM，
成本表就说不清增量是来自闸门还是来自又一次生成。

1. ``cite_exists``      —— 答案里的条号必须真的存在于 1,465 条语料里（拦住编造条号）；
2. ``cite_in_retrieval``—— 引用的条文必须在本轮检索结果内（拦住「自由发挥」式引用）；
3. ``path_support``     —— 被引条文必须在图上与本次查询的概念相连（拦住「检索到了但与问题概念无关」的条文）；
4. ``evidence_span``    —— 依据部分每个引用都要给出该条原文里的关键片段（拦住空引用）；
5. ``low_conf_handoff`` —— 低置信（top1 < 0.62 且覆盖 < 0.60）时必须转人工，不许硬下结论。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import settings


@dataclass
class GateResult:
    name: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"gate": self.name, "ok": self.ok, "detail": self.detail}


_ARTICLE_TAIL = re.compile(r"第[一二三四五六七八九十百千零〇0-9]+条")
# 「第N条」往左取法名时优先认这三个法的尾名，长名在前，避免「劳动法」吃掉「劳动合同法」
_LAW_TAILS = ("劳动合同法", "劳动法", "民法典")
_GENERIC_LAW = re.compile(r"[\u4e00-\u9fa5]{2,6}法$")


def _known_citations(article_index: dict[str, dict]) -> set[str]:
    return {a["citation"] for a in article_index.values()}


def cited_patterns(answer: str) -> list[str]:
    """从答案里抠出所有「XX法第N条」形态的引用（含模型自由生成的假条号）。

    不能用单条正则从左往右贪婪匹配：像「根据中华人民共和国劳动法第五十条」会把
    「根据」一起吞进法名，归一化后反被判成编造条号。改成「先定位第N条、再向左取法名尾」。
    """
    out: set[str] = set()
    for m in _ARTICLE_TAIL.finditer(answer):
        head = answer[max(0, m.start() - 14) : m.start()].rstrip("》\"」』】〕 \u3000")
        for name in _LAW_TAILS:
            if head.endswith(name):
                out.add(name + m.group())
                break
        else:
            mm = _GENERIC_LAW.search(head)
            out.add((mm.group() if mm else "") + m.group())
    return sorted(out)


def cited_citations(answer: str, known: set[str] | None = None) -> list[str]:
    """归一化到语料里的规范引用名。

    正则抓到的是「劳动合同法第八十五条」这种片段，语料里存的是
    「中华人民共和国劳动合同法第八十五条」；不归一化的话，明明引对了也会被判成越界。
    """
    pats = cited_patterns(answer)
    if known is None:
        return pats
    norm: set[str] = set()
    for p in pats:
        hits = [k for k in known if p in k]
        norm.update(hits or [p])
    return sorted(norm)


def build_article_concepts(store: Any, concepts_path: str = "data/lexicon/concepts.jsonl") -> dict[str, set[str]]:
    """反查表：条文 -> 它定义过的概念集合（闸门 3 的判据，不依赖走哪条检索路）。"""
    from common.path_utils import read_jsonl

    out: dict[str, set[str]] = {}
    for c in read_jsonl(concepts_path):
        for row in store.concept_articles([c["id"]], limit=500):
            out.setdefault(row["article_id"], set()).add(c["id"])
    return out


def gate_cite_exists(answer: str, items: list[dict], article_index: dict[str, dict]) -> GateResult:
    known = _known_citations(article_index)
    bad = [p for p in cited_patterns(answer) if not any(p in k for k in known)]
    return GateResult("cite_exists", not bad, "全部条号可查" if not bad else f"不存在的条号: {bad[:3]}")


def gate_cite_in_retrieval(answer: str, items: list[dict], article_index: dict[str, dict]) -> GateResult:
    allowed = {i["citation"] for i in items}
    cites = cited_citations(answer, _known_citations(article_index))
    bad = [c for c in cites if c not in allowed]
    return GateResult("cite_in_retrieval", not bad, "引用均在检索结果内" if not bad else f"越界引用: {bad[:3]}")


def gate_path_support(
    answer: str,
    items: list[dict],
    coverage: float,
    linked: set[str],
    art_concepts: dict[str, set[str]],
    known: set[str],
) -> GateResult:
    cites = set(cited_citations(answer, known))
    cited = [i for i in items if i["citation"] in cites] or items[:1]
    supported = [i["article_id"] for i in cited if art_concepts.get(i["article_id"], set()) & linked]
    ok = coverage >= settings.path_coverage_threshold or bool(supported)
    detail = f"coverage={coverage:.2f}, 有图连接的被引条文={len(supported)}/{len(cited)}"
    return GateResult("path_support", ok, detail)


def gate_evidence_span(answer: str, items: list[dict], known: set[str], min_span: int = 8) -> GateResult:
    """每个「- 引用（法名）：原文」式依据行，必须真的含该条原文的一个 8 字以上片段。"""
    lines = [ln for ln in answer.splitlines() if ln.strip().startswith("-")]
    if not lines:
        return GateResult("evidence_span", False, "答案没有依据段落")
    by_cite = {i["citation"]: i.get("content", "") for i in items}
    bad = []
    for ln in lines:
        cites = [c for c in cited_citations(ln, known) if c in by_cite]
        if not cites:
            bad.append(ln[:24])
            continue
        content = by_cite[cites[0]]
        if not any(content[i : i + min_span] in ln for i in range(0, max(len(content) - min_span, 0), 4)):
            bad.append(cites[0])
    return GateResult("evidence_span", not bad, "依据均有原文片段" if not bad else f"缺原文片段: {bad[:3]}")


def gate_low_conf_handoff(answer: str, retrieval: dict[str, Any]) -> GateResult:
    low = bool(retrieval.get("low_confidence"))
    handoff = ("人工" in answer) or ("补充" in answer) or ("无法确定" in answer) or ("无法给出" in answer)
    ok = (not low) or handoff
    return GateResult("low_conf_handoff", ok, f"low_confidence={low}, handoff={handoff}")


def run_gates(
    answer: str, retrieval: dict[str, Any], article_index: dict[str, dict], art_concepts: dict[str, set[str]]
) -> list[dict[str, Any]]:
    items = retrieval.get("items", [])
    linked = {c["id"] for c in retrieval.get("linked", {}).get("concepts", [])}
    known = _known_citations(article_index)
    results = [
        gate_cite_exists(answer, items, article_index),
        gate_cite_in_retrieval(answer, items, article_index),
        gate_path_support(answer, items, float(retrieval.get("coverage", 0.0)), linked, art_concepts, known),
        gate_evidence_span(answer, items, known),
        gate_low_conf_handoff(answer, retrieval),
    ]
    return [r.as_dict() for r in results]
