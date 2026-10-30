"""检索门面：对上层（Agent / API / UI / 评测）只暴露一个 ``retrieve``。

三种模式的语义差异要说清楚，否则指标没法横向比：
* ``text``   只走 dense+BM25+rerank，作为「不用图谱」的基线；
* ``graph``  只走实体链接+概念扩展+受限查询，用来量化「纯图能到什么程度」；
* ``hybrid`` 双路 RRF + 统一精排 + 覆盖加成，是主路径。

路由（simple/medium/complex）只决定 ``top_k`` 与两路权重，不改变模式本身；
这样 B6 消融里「有路由 / 无路由」的控制变量才是干净的。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.__004__retrieval.__000__text_index import TextIndex  # noqa: E402
from app.__004__retrieval.__001__entity_linker import LinkingResult  # noqa: E402
from app.__004__retrieval.__002__graph_retrieval import GraphResult, GraphRetriever  # noqa: E402
from app.__004__retrieval.__003__hybrid import HybridRetriever  # noqa: E402
from app.__004__retrieval.__004__router import LEVELS, QueryRouter, RouteDecision  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402

log = get_logger("retriever")


@dataclass
class RetrievalItem:
    article_id: str
    score: float
    source: str
    coverage: float = 0.0
    path: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return self.meta.get("citation", self.article_id)

    @property
    def content(self) -> str:
        return self.meta.get("content", "")


@dataclass
class RetrievalOutput:
    query: str
    mode: str
    decision: RouteDecision
    items: list[RetrievalItem]
    linking: LinkingResult
    plans: list[str]
    top1_similarity: float
    coverage: float
    latency_ms: float
    path_coverage_threshold: float = 0.60
    refuse_threshold: float = 0.62

    @property
    def low_confidence(self) -> bool:
        """两路都没底：语义相似度低且图路径没拼起来 —— B5 人在环的触发条件。"""
        return self.top1_similarity < self.refuse_threshold and self.coverage < self.path_coverage_threshold

    @property
    def article_ids(self) -> list[str]:
        return [i.article_id for i in self.items]

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "level": self.decision.level,
            "reason": self.decision.reason,
            "top_k": self.decision.top_k,
            "weights": [self.decision.text_weight, self.decision.graph_weight],
            "top1_similarity": round(self.top1_similarity, 4),
            "coverage": round(self.coverage, 4),
            "low_confidence": self.low_confidence,
            "plans": self.plans,
            "latency_ms": round(self.latency_ms, 1),
            "items": [
                {
                    "article_id": i.article_id,
                    "citation": i.citation,
                    "law_name": i.meta.get("law_name"),
                    "score": i.score,
                    "source": i.source,
                    "coverage": i.coverage,
                    "path": i.path,
                    "content": i.content,
                }
                for i in self.items
            ],
            "linked": {
                "concepts": [{"id": c.id, "name": c.name, "score": c.score, "method": c.method} for c in self.linking.concepts],
                "scenarios": [{"id": s.id, "name": s.name, "score": s.score, "method": s.method} for s in self.linking.scenarios],
            },
        }


class Retriever:
    """懒加载三路组件，避免评测里每种模式都重新加载模型。"""

    def __init__(self, use_router: bool = True) -> None:
        from config import settings

        self.settings = settings
        self._text: TextIndex | None = None
        self._graph: GraphRetriever | None = None
        self._hybrid: HybridRetriever | None = None
        self.router = QueryRouter() if use_router else None

    # ---------------- 组件 ----------------
    @property
    def text(self) -> TextIndex:
        if self._text is None:
            self._text = TextIndex().load()
        return self._text

    @property
    def graph(self) -> GraphRetriever:
        if self._graph is None:
            self._graph = GraphRetriever()
        return self._graph

    @property
    def hybrid(self) -> HybridRetriever:
        if self._hybrid is None:
            self._hybrid = HybridRetriever(self.text, self.graph)
        return self._hybrid

    # ---------------- 路由 ----------------
    def decide(self, query: str, level: str | None = None) -> RouteDecision:
        if level is not None:
            return RouteDecision(level=level, reason="显式指定", features={}, **LEVELS[level])
        if self.router is None:
            return _plain("medium")
        linking = self.graph.linker.link(query)
        return self.router.route(query, len(linking.concepts), len(linking.scenarios))

    # ---------------- 主入口 ----------------
    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        level: str | None = None,
        top_k: int | None = None,
        hops: int | None = None,
        use_cites: bool = True,
    ) -> RetrievalOutput:
        t0 = time.perf_counter()
        decision = self.decide(query, level)
        k = top_k or decision.top_k

        if mode == "text":
            text_hits = self.text.search(query, top_k=k, use_rerank=True)
            items = [
                RetrievalItem(r.article_id, r.score, "text", 0.0, [], r.meta)
                for r in text_hits
            ]
            linking = LinkingResult()
            plans: list[str] = []
            top1 = self.text.top1_similarity(query)
            coverage = 0.0
        elif mode == "graph":
            graph_res = self.graph.retrieve(query, top_k=k, hops=hops or decision.hops, use_cites=use_cites)
            items = [
                RetrievalItem(
                    h.article_id,
                    h.score,
                    "graph",
                    h.coverage,
                    h.path,
                    {
                        **h.meta,
                        **(self.text.articles.get(h.article_id) or {}),
                    },
                )
                for h in graph_res.hits
            ]
            linking = graph_res.linking
            plans = graph_res.plans
            top1 = self.text.top1_similarity(query)
            coverage = graph_res.coverage
        elif mode == "hybrid":
            hybrid_res = self.hybrid.retrieve(
                query,
                top_k=k,
                hops=hops or decision.hops,
                text_weight=decision.text_weight,
                graph_weight=decision.graph_weight,
                use_cites=use_cites,
                rrf_k=self.settings.rrf_k,
            )
            items = [
                RetrievalItem(h.article_id, h.score, h.source, h.coverage, h.path, h.meta)
                for h in hybrid_res.hits
            ]
            linking = hybrid_res.graph.linking
            plans = hybrid_res.graph.plans
            top1 = hybrid_res.top1_similarity
            coverage = hybrid_res.graph.coverage
        else:
            raise ValueError(f"未知检索模式: {mode}")

        out = RetrievalOutput(
            query=query,
            mode=mode,
            decision=decision,
            items=items,
            linking=linking,
            plans=plans,
            top1_similarity=float(top1),
            coverage=float(coverage),
            latency_ms=(time.perf_counter() - t0) * 1000,
            path_coverage_threshold=self.settings.path_coverage_threshold,
            refuse_threshold=self.settings.refuse_threshold,
        )
        return out


def _plain(level: str) -> RouteDecision:
    cfg = LEVELS[level]
    return RouteDecision(level=level, reason="无路由", features={}, **cfg)


_retriever: Retriever | None = None


def get_retriever(use_router: bool = True) -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(use_router=use_router)
    return _retriever
