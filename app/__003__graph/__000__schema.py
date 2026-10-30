"""图 schema 与「受限查询计划」。

安全边界：图路不拼装用户输入生成的 Cypher —— 查询计划是枚举类型，参数走参数化绑定，
Neo4j 执行前再用 EXPLAIN 预检；NetworkX 后端实现同一批计划，保证两种后端语义一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

LABELS = ("Article", "Concept", "Scenario")
RELATIONS = ("APPLIES_TO", "DEFINES", "REQUIRES", "SPECIAL_OF", "CITES")

# 概念与场景存在同名 ID（如 labor_dispatch / annual_leave），必须做命名空间隔离，
# 否则同一图里两个不同实体共用节点键。Article 用「law#num」天然唯一，无需前缀。
SCENARIO_PREFIX = "sc:"


def scenario_node_id(sid: str) -> str:
    return sid if sid.startswith(SCENARIO_PREFIX) else f"{SCENARIO_PREFIX}{sid}"

PlanName = Literal[
    "concept_articles",
    "scenario_concepts",
    "concept_neighbors",
    "article_cites",
    "concept_scenarios",
    "article_detail",
]

ALLOWED_PLANS: tuple[str, ...] = (
    "concept_articles",
    "scenario_concepts",
    "concept_neighbors",
    "article_cites",
    "concept_scenarios",
    "article_detail",
)


@dataclass
class QueryPlan:
    name: PlanName
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "QueryPlan":
        if self.name not in ALLOWED_PLANS:
            raise ValueError(f"未授权的查询计划: {self.name}")
        rels = self.params.get("relations")
        if rels is not None:
            bad = [r for r in rels if r not in RELATIONS]
            if bad:
                raise ValueError(f"非法关系类型: {bad}")
        for key in ("concept_ids", "scenario_ids", "article_ids", "ids"):
            if key in self.params and not isinstance(self.params[key], list):
                raise TypeError(f"{key} 必须是列表")
        return self


CYPHER_TEMPLATES: dict[str, str] = {
    "concept_articles": """
        MATCH (a:Article)-[r:DEFINES]->(c:Concept)
        WHERE c.id IN $concept_ids
        RETURN a.id AS article_id, a.law_name AS law_name, a.article_no AS article_no,
               a.article_num AS article_num, a.chapter_path AS chapter_path,
               collect(DISTINCT c.id) AS concepts, avg(r.confidence) AS confidence,
               count(DISTINCT c.id) AS concept_hits
        ORDER BY confidence DESC, concept_hits DESC, a.id ASC
        LIMIT $limit
    """,
    "scenario_concepts": """
        MATCH (s:Scenario)-[r:APPLIES_TO]->(c:Concept)
        WHERE s.id IN $scenario_ids
        RETURN s.id AS scenario_id, s.name AS scenario_name, c.id AS concept_id,
               c.name AS concept_name, r.weight AS weight
    """,
    "concept_neighbors": """
        MATCH (c:Concept)-[r:REQUIRES|SPECIAL_OF]->(n:Concept)
        WHERE c.id IN $concept_ids AND type(r) IN $relations
        RETURN c.id AS source, n.id AS target, n.name AS target_name, type(r) AS relation,
               r.confidence AS confidence
        LIMIT $limit
    """,
    "concept_neighbors_in": """
        MATCH (n:Concept)-[r:REQUIRES|SPECIAL_OF]->(c:Concept)
        WHERE c.id IN $concept_ids AND type(r) IN $relations
        RETURN c.id AS source, n.id AS target, n.name AS target_name, type(r) AS relation,
               r.confidence AS confidence
        LIMIT $limit
    """,
    "article_cites": """
        MATCH (a:Article)-[r:CITES]->(b:Article)
        WHERE a.id IN $article_ids
        RETURN a.id AS source, b.id AS target, b.law_name AS law_name,
               b.article_no AS article_no, r.confidence AS confidence
    """,
    "article_detail": """
        MATCH (a:Article)
        WHERE a.id IN $ids
        RETURN a.id AS id, a.law_name AS law_name, a.article_no AS article_no,
               a.article_num AS article_num, a.chapter_path AS chapter_path,
               a.content AS content
    """,
    "concept_scenarios": """
        MATCH (s:Scenario)-[r:APPLIES_TO]->(c:Concept)
        WHERE c.id IN $concept_ids
        RETURN s.id AS scenario_id, s.name AS scenario_name, c.id AS concept_id,
               r.weight AS weight
        ORDER BY weight DESC LIMIT $limit
    """,
}


def to_cypher(plan: QueryPlan, database: str = "neo4j") -> tuple[str, dict[str, Any]]:
    """把查询计划翻译成参数化 Cypher（不含任何字符串拼接的用户输入）。"""
    plan.validate()
    name = plan.name
    if name == "concept_neighbors" and plan.params.get("direction") == "in":
        name = "concept_neighbors_in"
    template = CYPHER_TEMPLATES.get(name)
    if template is None:
        raise ValueError(f"计划 {plan.name} 无 Cypher 模板")
    params = {k: v for k, v in plan.params.items() if k != "direction"}
    return template.strip(), params


def plan_summary(plan: QueryPlan) -> str:
    keys = ", ".join(f"{k}={v}" for k, v in plan.params.items())
    return f"{plan.name}({keys})"
