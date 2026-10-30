"""图路检索：实体链接 -> 概念扩展 -> 受限图查询 -> 路径覆盖度打分。

一次查询的完整路径（也是 UI/评测里展示的 explainability 链条）：
    query
      └─ entity linking → Concept / Scenario
            └─ APPLIES_TO（场景补全） + REQUIRES/SPECIAL_OF（一跳扩展）
                  └─ concept_articles（DEFINES 反向索引）
                        └─ CITES（多跳时沿法条互引再走一步）

打分不是简单叠加：文章分 = Σ(概念权重 × 定义边置信度)，再按「本次查询命中的概念覆盖率」排序，
这样多跳问题里「只命中一个概念」的高置信条文不会压过「覆盖两个概念」的条文。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.__003__graph.__003__graph_store import get_graph_store  # noqa: E402
from app.__004__retrieval.__001__entity_linker import EntityLinker, LinkingResult  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import read_jsonl  # noqa: E402

log = get_logger("retrieval_graph")


@dataclass
class GraphHit:
    article_id: str
    score: float
    concept_ids: list[str]
    coverage: float
    path: list[str]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphResult:
    hits: list[GraphHit]
    linking: LinkingResult
    coverage: float
    expanded_concepts: dict[str, float]
    plans: list[str]
    scenarios: list[dict]


def diversify(scored: list[GraphHit], k: int, penalty: float = 0.25) -> list[GraphHit]:
    """概念集去重挑选：已选条文的「概念集」与候选重叠越大，惩罚越重。

    纯按分数取 top-k 时，多跳问题会被「同一条高置信条文 + 一堆它的近义条文」占满，
    第二个跳点的条文挤不进来。这里用 Jaccard 惩罚做轻量 MMR，保证每条被选中的条文
    **带来新的概念**，而不是重复同一个概念的不同表述。
    """
    remaining = list(scored)
    selected: list[GraphHit] = []
    while remaining and len(selected) < k:
        best, best_val = None, float("-inf")
        for h in remaining:
            overlap = 0.0
            if selected:
                hset = set(h.concept_ids)
                overlap = max(
                    len(hset & set(s.concept_ids)) / max(len(hset | set(s.concept_ids)), 1) for s in selected
                )
            val = h.score * (1.0 - penalty * overlap)
            if val > best_val:
                best, best_val = h, val
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
    return selected


class GraphRetriever:
    def __init__(self, linker: EntityLinker | None = None) -> None:
        self.linker = linker or EntityLinker()
        self.store = get_graph_store()
        self.articles = {a["article_id"]: a for a in read_jsonl("data/extracted/articles.jsonl")}
        self._names = {c["id"]: [c["name"], *c["aliases"]] for c in read_jsonl("data/lexicon/concepts.jsonl")}

    def _canonicality(self, article_id: str, concept_ids: list[str]) -> float:
        """条文对概念的「代表性」：概念名出现在章节名或首句，说明这就是该概念的正式条款。

        没有这一步，同一概念在十几条法条里都出现时，图路会随机挑中一条，
        和金标锁定的「首选条款」对不上 —— 实测这也是图路 R@5 上不去的主因。
        """
        art = self.articles.get(article_id) or {}
        text = art.get("content", "")
        chapter = art.get("chapter_path", "")
        best = 0.0
        for cid in concept_ids:
            for name in self._names.get(cid, []):
                if not name:
                    continue
                if name in chapter:
                    return 1.0
                if name in text[:80]:
                    best = max(best, 0.7)
        return best

    def _expand(self, seed: dict[str, float], hops: int) -> dict[str, float]:
        expanded = dict(seed)
        frontier = dict(seed)
        for _ in range(max(hops - 1, 0)):
            nxt: dict[str, float] = {}
            for cid, w in frontier.items():
                for rel in self.store.concept_neighbors([cid], ["REQUIRES"], "out", limit=20):
                    nxt[rel["target"]] = max(nxt.get(rel["target"], 0), w * 0.62 * rel.get("confidence", 0.9))
                for rel in self.store.concept_neighbors([cid], ["SPECIAL_OF"], "out", limit=10):
                    nxt[rel["target"]] = max(nxt.get(rel["target"], 0), w * 0.48)
            for cid, w in nxt.items():
                expanded[cid] = max(expanded.get(cid, 0), w)
            frontier = nxt
        return expanded

    def retrieve(self, query: str, top_k: int = 5, hops: int = 2, use_cites: bool = False) -> GraphResult:
        plans: list[str] = []
        linking = self.linker.link(query)

        seed = {c.id: c.score for c in linking.concepts}
        scenario_rows: list[dict] = []
        if linking.scenarios:
            sids = [s.id for s in linking.scenarios]
            scenario_rows = self.store.scenario_concepts(sids)
            plans.append(f"scenario_concepts[{','.join(sids)}]")
            for row in scenario_rows:
                w = 0.80 * row.get("weight", 0.8) * linking.scenarios[0].score
                seed[row["concept_id"]] = max(seed.get(row["concept_id"], 0.0), w)

        expanded = self._expand(seed, hops)
        if expanded:
            plans.append(f"concept_neighbors[hops={hops}]")

        if not expanded:
            return GraphResult([], linking, 0.0, {}, plans, scenario_rows)

        # 限定参与查询的概念数量，避免长尾概念把候选池冲爆
        top_concepts = sorted(expanded.items(), key=lambda kv: -kv[1])[:8]
        cids = [c for c, _ in top_concepts]
        rows = self.store.concept_articles(cids, limit=60)
        plans.append(f"concept_articles[{len(cids)} concepts]")

        hit_concepts = {c.id for c in linking.concepts}
        if not hit_concepts:
            hit_concepts = set(seed)

        scored: list[GraphHit] = []
        scenario_prefix = [f"sc:{linking.scenarios[0].id}"] if linking.scenarios else []
        for row in rows:
            cset = set(row["concepts"])
            weights = [expanded.get(c, 0.0) for c in cset]
            base = sum(w * row.get("confidence", 0.8) for w in weights)
            cov = len(cset & hit_concepts) / max(len(hit_concepts), 1)
            canon = self._canonicality(row["article_id"], sorted(cset & set(expanded)))
            score = base * (1.0 + 0.35 * cov) * (1.0 + 0.50 * canon)
            path = [*scenario_prefix, *sorted(cset), row["article_id"]]
            scored.append(
                GraphHit(
                    article_id=row["article_id"],
                    score=round(score, 4),
                    concept_ids=sorted(cset),
                    coverage=round(cov, 4),
                    path=path,
                    meta={
                        "law_name": row["law_name"],
                        "article_no": row["article_no"],
                        "chapter_path": row["chapter_path"],
                    },
                )
            )
        scored.sort(key=lambda h: (-h.score, -h.coverage, h.article_id))
        hits = diversify(scored, top_k)

        if use_cites and hits:
            cited = self.store.article_cites([h.article_id for h in hits])[:top_k]
            plans.append("article_cites")
            known = {h.article_id for h in hits}
            extra: list[GraphHit] = []
            for c in cited:
                if c["target"] in known:
                    continue
                src = next((h for h in hits if h.article_id == c["source"]), None)
                extra.append(
                    GraphHit(
                        article_id=c["target"],
                        score=round((src.score if src else 0.5) * 0.55, 4),
                        concept_ids=[],
                        coverage=0.0,
                        path=[c["source"], c["target"]],
                        meta={"law_name": c.get("law_name"), "article_no": c.get("article_no")},
                    )
                )
            hits = hits + extra

        total_cov = (
            len({c for h in hits for c in h.concept_ids} & hit_concepts) / max(len(hit_concepts), 1)
            if hit_concepts
            else 0.0
        )
        return GraphResult(hits, linking, round(total_cov, 4), expanded, plans, scenario_rows)
