"""查询路由：判断简单/中等/复杂，决定 topK 与双路权重。

规则路由保证可解释、可回归；同时用 60 条金标题 + 20 条单概念问题做留一交叉验证，
把「规则 vs 学习型路由」的准确率都量出来（不达标就承认，而不是只报最好的那个）。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.logging_utils import get_logger  # noqa: E402

log = get_logger("router")

MULTI_PART_MARKERS = ("同时", "并且", "又", "还要", "另外", "以及", "分别", "一方面", "不但", "而且")

FEATURE_KEYS = ("n_concepts", "n_scenarios", "length", "markers", "clauses")

LEVELS = {
    "simple": {"top_k": 3, "text_weight": 0.65, "graph_weight": 0.35, "hops": 1},
    "medium": {"top_k": 5, "text_weight": 0.55, "graph_weight": 0.45, "hops": 2},
    "complex": {"top_k": 8, "text_weight": 0.45, "graph_weight": 0.55, "hops": 2},
}


@dataclass
class RouteDecision:
    level: str
    top_k: int
    text_weight: float
    graph_weight: float
    hops: int
    reason: str
    features: dict[str, float]


def features_of(query: str, n_concepts: int, n_scenarios: int) -> dict[str, float]:
    markers = sum(1 for m in MULTI_PART_MARKERS if m in query)
    question_parts = query.count("？") + query.count("?")
    clauses = sum(query.count(p) for p in ("，", "、", "；", "。"))
    return {
        "n_concepts": float(n_concepts),
        "n_scenarios": float(n_scenarios),
        "length": float(len(query)),
        "markers": float(markers),
        "questions": float(question_parts),
        "clauses": float(clauses),
    }


def rule_route(f: dict[str, float]) -> tuple[str, str]:
    if f["markers"] >= 1 or f["n_concepts"] >= 3 or (f["n_concepts"] >= 2 and f["length"] >= 28):
        return "complex", "多诉求/多概念"
    if f["n_concepts"] <= 1 and f["length"] < 28 and f["markers"] == 0:
        return "simple", "单概念短问"
    if f["n_concepts"] >= 2 or f["clauses"] >= 2:
        return "medium", "双概念"
    return "medium", "默认中等"


class QueryRouter:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or Path("data/index/router_lr.json")
        self.model = None
        self.meta: dict = {}
        if self.model_path.exists():
            try:
                self._load()
            except Exception:
                log.warning("路由模型加载失败，回落规则路由")

    def _load(self) -> None:
        from sklearn.linear_model import LogisticRegression

        blob = json.loads(self.model_path.read_text(encoding="utf-8"))
        clf = LogisticRegression(max_iter=500)
        import numpy as np

        clf.classes_ = np.array(blob["classes"])
        clf.coef_ = np.array(blob["coef"])
        clf.intercept_ = np.array(blob["intercept"])
        self.model = clf
        self.meta = blob.get("meta", {})

    def fit(self, X: list[list[float]], y: list[str]) -> dict:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        self.model_path.write_text(
            json.dumps(
                {
                    "classes": list(clf.classes_),
                    "coef": np.asarray(clf.coef_).tolist(),
                    "intercept": np.asarray(clf.intercept_).tolist(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.model = clf
        return {"classes": list(clf.classes_)}

    def route(self, query: str, n_concepts: int, n_scenarios: int, use_model: bool = True) -> RouteDecision:
        f = features_of(query, n_concepts, n_scenarios)
        level, reason = rule_route(f)
        if use_model and self.model is not None:
            import numpy as np

            x = np.array([[f[k] for k in FEATURE_KEYS]])
            pred = str(self.model.predict(x)[0])
            reason = f"LR 模型（规则={level}）"
            level = pred
        cfg = LEVELS[level]
        return RouteDecision(level=level, reason=reason, features=f, **cfg)


def feature_vector(query: str, n_concepts: int, n_scenarios: int) -> list[float]:
    f = features_of(query, n_concepts, n_scenarios)
    return [f[k] for k in FEATURE_KEYS]
