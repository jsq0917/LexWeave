"""文本路：bge-m3 dense + BM25 双召回 -> RRF -> 交叉编码器精排。

和 LawAnswer 的检索服务保持同一套工程口径：
* dense 用归一化向量 + 内积（等价余弦），相似度直接可比、便于设拒答阈值；
* BM25 补精确术语匹配（法条里的固定表述 dense 容易「抹平」）；
* RRF 只看名次不看分数量纲，避免两路分数不可比；
* rerank 只对融合后的前 30 条做，控制延迟。
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import read_jsonl  # noqa: E402
from common.text_utils import tokenize  # noqa: E402

log = get_logger("retrieval_text")

DENSE_K = 20
BM25_K = 20
FUSE_K = 30


@dataclass
class RetrievedArticle:
    article_id: str
    score: float
    source: str
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def rrf_fuse(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> dict[str, float]:
    """Reciprocal Rank Fusion：score(d) = Σ w_i / (k + rank_i(d))。"""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    return scores


class TextIndex:
    def __init__(self, index_dir: Path | None = None) -> None:
        from config import settings

        self.dir = index_dir or settings.index_dir
        self.articles: dict[str, dict] = {}
        self.ids: list[str] = []
        self.faiss_index: Any = None
        self.bm25: Any = None
        self.tokens: list[list[str]] = []
        self.dim: int = 0
        self._encoder: Any = None
        self._reranker: Any = None

    # ---------------- 构建 ----------------
    def build(self, articles: list[dict], batch_size: int = 32) -> dict[str, Any]:
        from common.embedding_model import get_dense_encoder

        self.dir.mkdir(parents=True, exist_ok=True)
        self.articles = {a["article_id"]: a for a in articles}
        self.ids = [a["article_id"] for a in articles]
        texts = [f"{a['citation']} {a['content']}" for a in articles]

        encoder = get_dense_encoder()
        vecs = encoder.encode(texts, batch_size=batch_size, show_progress=True)
        self.dim = vecs.shape[1]
        index = faiss.IndexFlatIP(self.dim)
        index.add(vecs)
        faiss.write_index(index, str(self.dir / "text.faiss"))
        self.faiss_index = index

        self.tokens = [tokenize(f"{a['content']}") for a in articles]
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi(self.tokens)
        with open(self.dir / "bm25.pkl", "wb") as f:
            pickle.dump({"bm25": self.bm25, "tokens": self.tokens, "ids": self.ids}, f)

        (self.dir / "text_meta.json").write_text(
            json.dumps({"ids": self.ids, "dim": self.dim, "n": len(self.ids)}, ensure_ascii=False),
            encoding="utf-8",
        )
        stats = {"articles": len(self.ids), "dim": self.dim}
        log.info("文本索引完成 %s", stats)
        return stats

    # ---------------- 加载 ----------------
    def load(self) -> "TextIndex":
        from config import settings

        meta_path = self.dir / "text_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"文本索引不存在，先运行 scripts/build_index.py（{meta_path}）")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.ids = meta["ids"]
        self.dim = meta["dim"]
        self.faiss_index = faiss.read_index(str(self.dir / "text.faiss"))
        with open(self.dir / "bm25.pkl", "rb") as f:
            blob = pickle.load(f)
        self.bm25, self.tokens = blob["bm25"], blob["tokens"]
        if not self.articles:
            self.articles = {a["article_id"]: a for a in read_jsonl(settings.extracted_dir / "articles.jsonl")}
        return self

    # ---------------- 检索 ----------------
    @property
    def encoder(self) -> Any:
        if self._encoder is None:
            from common.embedding_model import get_dense_encoder

            self._encoder = get_dense_encoder()
        return self._encoder

    @property
    def reranker(self) -> Any:
        if self._reranker is None:
            from common.embedding_model import get_reranker

            self._reranker = get_reranker()
        return self._reranker

    def dense_search(self, query: str, k: int = DENSE_K) -> list[tuple[str, float]]:
        vec = self.encoder.encode([query])
        scores, idx = self.faiss_index.search(vec, k)
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0 or i >= len(self.ids):
                continue
            out.append((self.ids[i], float(s)))
        return out

    def bm25_search(self, query: str, k: int = BM25_K) -> list[tuple[str, float]]:
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        order = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True,
        rrf_k: int = 60,
    ) -> list[RetrievedArticle]:
        dense = self.dense_search(query, DENSE_K)
        bm25 = self.bm25_search(query, BM25_K)
        fused = rrf_fuse([[a for a, _ in dense], [a for a, _ in bm25]], k=rrf_k)
        cand = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:FUSE_K]
        dmap = dict(dense)
        bmap = dict(bm25)
        if use_rerank and cand:
            docs = [self.articles[a]["content"] for a, _ in cand]
            rr = self.reranker.score(query, docs)
            cand = [((a, s), r) for (a, s), r in zip(cand, rr)]
            cand.sort(key=lambda x: x[1], reverse=True)
            results = [
                RetrievedArticle(
                    article_id=a,
                    score=float(r),
                    source="text",
                    dense_score=dmap.get(a),
                    bm25_score=bmap.get(a),
                    rerank_score=float(r),
                )
                for (a, _), r in cand[:top_k]
            ]
        else:
            results = [
                RetrievedArticle(article_id=a, score=s, source="text", dense_score=dmap.get(a), bm25_score=bmap.get(a))
                for a, s in cand[:top_k]
            ]
        for r in results:
            r.meta = {
                "citation": self.articles[r.article_id]["citation"],
                "content": self.articles[r.article_id]["content"],
                "law_name": self.articles[r.article_id]["law_name"],
            }
        return results

    def top1_similarity(self, query: str) -> float:
        d = self.dense_search(query, 1)
        return d[0][1] if d else 0.0


def build_index(articles: Iterable[dict], index_dir: Path | None = None) -> dict[str, Any]:
    return TextIndex(index_dir).build(list(articles))


if __name__ == "__main__":
    from config import settings

    build_index(read_jsonl(settings.extracted_dir / "articles.jsonl"))
