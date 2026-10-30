"""图查询门面：统一 Neo4j 与 NetworkX 两种后端，对上层只暴露「查询计划」。

上层（检索/Agent）永远不写 Cypher，只调用 ``concept_articles`` / ``scenario_concepts`` 等方法；
后端切换只影响执行层，评测口径因此可比。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.__003__graph.__000__schema import QueryPlan, scenario_node_id, to_cypher  # noqa: E402
from app.__003__graph.__001__networkx_store import NetworkXGraphStore, build_and_save  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.neo4j_manager import get_neo4j_manager  # noqa: E402

log = get_logger("graph_store")


class Neo4jGraphStore:
    backend = "neo4j"

    def __init__(self) -> None:
        self.mgr = get_neo4j_manager()
        if not self.mgr.connect():
            raise ConnectionError("Neo4j 不可用")
        self.query_log: list[dict[str, Any]] = []

    def execute(self, plan: QueryPlan) -> list[dict]:
        cypher, params = to_cypher(plan)
        ok, msg = self.mgr.validate_cypher(cypher, params)
        if not ok:
            raise ValueError(f"Cypher 预检失败: {msg}")
        t0 = time.perf_counter()
        rows = self.mgr.run_query(cypher, params)
        self.query_log.append({"plan": plan.name, "ms": round((time.perf_counter() - t0) * 1000, 2), "rows": len(rows)})
        return rows

    def concept_articles(self, concept_ids: list[str], limit: int = 20) -> list[dict]:
        return self.execute(QueryPlan("concept_articles", {"concept_ids": concept_ids, "limit": limit}))

    def scenario_concepts(self, scenario_ids: list[str]) -> list[dict]:
        return self.execute(
            QueryPlan("scenario_concepts", {"scenario_ids": [scenario_node_id(s) for s in scenario_ids]})
        )

    def concept_neighbors(
        self, concept_ids: list[str], relations: list[str] | None = None, direction: str = "out", limit: int = 50
    ) -> list[dict]:
        return self.execute(
            QueryPlan(
                "concept_neighbors",
                {"concept_ids": concept_ids, "relations": relations or ["REQUIRES", "SPECIAL_OF"], "direction": direction, "limit": limit},
            )
        )

    def article_cites(self, article_ids: list[str]) -> list[dict]:
        return self.execute(QueryPlan("article_cites", {"article_ids": article_ids}))

    def concept_scenarios(self, concept_ids: list[str], limit: int = 20) -> list[dict]:
        return self.execute(QueryPlan("concept_scenarios", {"concept_ids": concept_ids, "limit": limit}))

    def article_detail(self, ids: list[str]) -> list[dict]:
        return self.execute(QueryPlan("article_detail", {"ids": ids}))

    def stats(self) -> dict[str, Any]:
        return {"backend": "neo4j", **self.mgr.export_metadata_to_json()}

    def export_json(self, path: Path) -> None:  # 与 NetworkX 后端保持同一接口
        nodes = self.mgr.run_query("MATCH (n) RETURN labels(n) AS labels, properties(n) AS props")
        edges = self.mgr.run_query(
            "MATCH (a)-[r]->(b) RETURN a.id AS source, b.id AS target, type(r) AS relation, properties(r) AS props"
        )
        from common.path_utils import write_json

        write_json(path, {"nodes": nodes, "edges": edges})


class GraphStore:
    """后端选择器。``auto``：能连 Neo4j 就用，否则用离线镜像（并自动建镜像）。"""

    def __init__(self, backend: str | None = None) -> None:
        from config import settings

        want = backend or settings.graph_backend
        self.impl: Neo4jGraphStore | NetworkXGraphStore | None = None
        if want in ("auto", "neo4j"):
            try:
                self.impl = Neo4jGraphStore()
                log.info("图后端: neo4j")
            except Exception as exc:
                if want == "neo4j":
                    raise
                log.info("Neo4j 不可用（%s），切换 NetworkX 离线镜像", exc)
        if self.impl is None:
            pkl = settings.graph_dir / "lexweave_graph.pkl"
            if not pkl.exists():
                log.info("离线镜像不存在，现场构建")
                build_and_save(out_path=pkl)
            self.impl = NetworkXGraphStore(pkl)
            log.info("图后端: networkx (%s)", pkl)

    @property
    def backend(self) -> str:
        return self.impl.backend if self.impl else "none"

    def execute(self, plan: QueryPlan) -> list[dict]:
        plan.validate()
        return self.impl.execute(plan)

    # ---------------- 便捷方法 ----------------
    def concept_articles(self, concept_ids: list[str], limit: int = 20) -> list[dict]:
        if self.impl is None:
            return []
        return self.impl.concept_articles(concept_ids, limit)

    def scenario_concepts(self, scenario_ids: list[str]) -> list[dict]:
        if self.impl is None:
            return []
        return self.impl.scenario_concepts(scenario_ids)

    def concept_neighbors(
        self, concept_ids: list[str], relations: list[str] | None = None, direction: str = "out", limit: int = 50
    ) -> list[dict]:
        if self.impl is None:
            return []
        return self.impl.concept_neighbors(concept_ids, relations, direction, limit)

    def article_cites(self, article_ids: list[str]) -> list[dict]:
        if self.impl is None:
            return []
        return self.impl.article_cites(article_ids)

    def concept_scenarios(self, concept_ids: list[str], limit: int = 20) -> list[dict]:
        if self.impl is None:
            return []
        return self.impl.concept_scenarios(concept_ids, limit)

    def article_detail(self, ids: list[str]) -> list[dict]:
        if self.impl is None:
            return []
        return self.impl.article_detail(ids)

    def stats(self) -> dict[str, Any]:
        return self.impl.stats() if self.impl else {}

    def export_json(self, path: Path) -> None:
        if self.impl is not None:
            self.impl.export_json(path)


_store: GraphStore | None = None


def get_graph_store() -> GraphStore:
    global _store
    if _store is None:
        _store = GraphStore()
    return _store
