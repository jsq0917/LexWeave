"""双路混合检索：文本路（dense+BM25）与图路（概念扩展+受限查询）融合。

融合顺序刻意分成三步，而不是「把两路分数加权求和」：
1. **RRF 融合**：只看名次，用 ``text_weight`` / ``graph_weight`` 调制，规避 cosine 与图分数量纲不可比；
2. **统一精排**：融合后的前 25 条一起过 bge-reranker-v2-m3，让同一把尺子重新排序；
3. **覆盖加成（只做微调）**：图路给出「本条覆盖了本次查询的几个概念」，按 ``+0.02 × coverage``
   在精排 logits 上加一个很小的量，只用于打散「精排几乎同分」的近邻。

第 3 步的系数是**实测校准出来的**，不是拍脑袋：早期版本用 ``sigmoid(logit) + 0.15 × coverage``，
60 题上 hybrid R@5 只有 0.411，反而低于纯文本 0.444 —— 因为 bge-reranker 在法条域输出的 logits
集中在 0.0~0.5，sigmoid 会把它们全部压到 0.5±0.1，精排差异被抹平，而 coverage 项（0.15×0.5=0.075）
足以把图路挑错的条文整体抬到精排正确项之上（g13 的召回从 1.0 掉到 0.0）。
改回 raw logits、把覆盖系数降到 0.02 后，官方 60 题评测 hybrid R@5 = 0.461（纯文本 0.444，只有 +1.7pt；
参数网格里最好的一组是 0.478，但那是在同一批 60 题上搜出来的，有轻度过拟合，不作为交付口径）。
消融明细见 data/eval_out/hybrid_sweep.json。

另有一条重要的**边界**：60 题里 text30 的候选召回是 0.608，text30∪graph10 是 0.711，
但把「整个候选池」都交给精排、取 top5 的召回上限也只有 0.461 ——说明这一档的瓶颈是交叉编码器
对短法条的排序能力，而不是融合策略。因此「hybrid 比纯文本高 15pt」这个目标在当前模型下不成立，
README 里如实记为未达标，并附上这条天花板分析。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.__004__retrieval.__000__text_index import TextIndex, rrf_fuse  # noqa: E402
from app.__004__retrieval.__002__graph_retrieval import GraphRetriever, GraphResult  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402

log = get_logger("retrieval_hybrid")

RERANK_CANDIDATES = 25
# 覆盖项系数作用在**精排 logits 的原始量纲**上：它只该用来打散近邻，不该参与主排序。
COVERAGE_BONUS = 0.02


@dataclass
class HybridHit:
    article_id: str
    score: float
    source: str  # text | graph | both
    coverage: float
    path: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridResult:
    hits: list[HybridHit]
    text_hits: list[str]
    graph_hits: list[str]
    graph: GraphResult
    top1_similarity: float
    weights: tuple[float, float]


class HybridRetriever:
    def __init__(self, text_index: TextIndex | None = None, graph_retriever: GraphRetriever | None = None) -> None:
        self.text = text_index or TextIndex().load()
        self.graph = graph_retriever or GraphRetriever()

    @property
    def reranker(self) -> Any:
        return self.text.reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        hops: int = 2,
        text_weight: float = 0.6,
        graph_weight: float = 0.4,
        use_cites: bool = False,
        rrf_k: int = 60,
        use_rerank: bool = True,
    ) -> HybridResult:
        text_res = self.text.search(query, top_k=30, use_rerank=False, rrf_k=rrf_k)
        graph_res = self.graph.retrieve(query, top_k=10, hops=hops, use_cites=use_cites)
        text_ids = [r.article_id for r in text_res]
        graph_ids = [h.article_id for h in graph_res.hits]

        fused = rrf_fuse([text_ids, graph_ids], k=rrf_k, weights=[text_weight, graph_weight])
        order = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:RERANK_CANDIDATES]

        gmap = {h.article_id: h for h in graph_res.hits}
        in_text, in_graph = set(text_ids), set(graph_ids)

        hits: list[HybridHit] = []
        cand_ids = [a for a, _ in order]
        if use_rerank and cand_ids:
            docs = [self.text.articles[a]["content"] for a in cand_ids]
            raw = self.reranker.score(query, docs)
            # 保留原始 logits 作为主排序依据（同一次精排内量纲一致，归一化只会损失分辨率）。
            ranked = list(zip(cand_ids, [float(s) for s in raw]))
        else:
            ranked = list(order)

        for aid, base in ranked:
            gh = gmap.get(aid)
            coverage = gh.coverage if gh is not None else 0.0
            src = "both" if (aid in in_text and aid in in_graph) else ("graph" if aid in in_graph else "text")
            art = self.text.articles[aid]
            hits.append(
                HybridHit(
                    article_id=aid,
                    score=round(float(base) + COVERAGE_BONUS * coverage, 4),
                    source=src,
                    coverage=coverage,
                    path=gh.path if gh is not None else [],
                    meta={
                        "citation": art["citation"],
                        "content": art["content"],
                        "law_name": art["law_name"],
                        "chapter_path": art.get("chapter_path", ""),
                    },
                )
            )
        hits.sort(key=lambda h: -h.score)
        top1_sim = self.text.top1_similarity(query)
        return HybridResult(
            hits=hits[:top_k],
            text_hits=text_ids,
            graph_hits=graph_ids,
            graph=graph_res,
            top1_similarity=top1_sim,
            weights=(text_weight, graph_weight),
        )
