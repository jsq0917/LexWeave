"""Neo4j 连接管理（对照 TCM ``common/neo4j_manager.py`` 精简而来）。

保留原项目里真正有用的四件事：
1. 单例 driver + 按需建连，连不上只警告不崩；
2. ``run_multiple_cypher`` 批量提交（建图时按批 MERGE，比逐条快一个量级）；
3. ``validate_cypher`` 用 EXPLAIN 做语法/权限预检，图查询路由执行前必须过这一关；
4. ``export_metadata`` 把节点/关系/属性导出成 JSON，供离线镜像与抽检使用。
"""

from __future__ import annotations

from typing import Any, Iterable

from config import settings
from common.logging_utils import get_logger

log = get_logger("neo4j")


class Neo4jManager:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database
        self._driver: Any | None = None
        self._available: bool | None = None

    # ----------------------------- 连接 -----------------------------
    def connect(self) -> bool:
        if self._driver is not None:
            return True
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._driver.verify_connectivity()
            self._available = True
            log.info("Neo4j 已连接: %s db=%s", self.uri, self.database)
            return True
        except Exception as exc:
            self._driver = None
            self._available = False
            log.warning("Neo4j 不可用（%s），将使用离线图镜像", exc)
            return False

    @property
    def available(self) -> bool:
        if self._available is None:
            self.connect()
        return bool(self._available)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            self._available = None

    # ----------------------------- 查询 -----------------------------
    def run_query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not self.available:
            raise ConnectionError("Neo4j 未连接")
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    def run_multiple_cypher(
        self, statements: Iterable[tuple[str, dict[str, Any]]], batch_log: bool = False
    ) -> int:
        """在一个事务里提交多条语句，返回语句条数。"""
        if not self.available:
            raise ConnectionError("Neo4j 未连接")
        statements = list(statements)
        if not statements:
            return 0
        with self._driver.session(database=self.database) as session:
            tx = session.begin_transaction()
            try:
                for cypher, params in statements:
                    tx.run(cypher, params)
                tx.commit()
            except Exception:
                tx.rollback()
                raise
        if batch_log:
            log.info("已提交 %d 条 Cypher", len(statements))
        return len(statements)

    def validate_cypher(self, cypher: str, params: dict[str, Any] | None = None) -> tuple[bool, str]:
        """EXPLAIN 预检：语法正确且不触发写操作才放行。"""
        forbidden = ("CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP", "CALL DB.")
        upper = cypher.upper()
        for word in forbidden:
            if word in upper:
                return False, f"查询包含禁止的子句: {word}"
        if not self.available:
            return True, "neo4j-unavailable (skipped)"
        try:
            with self._driver.session(database=self.database) as session:
                session.run(f"EXPLAIN {cypher}", params or {}).consume()
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    # ----------------------------- 元数据 -----------------------------
    def get_all_node_names(self, label: str) -> list[str]:
        rows = self.run_query(f"MATCH (n:`{label}`) RETURN n.name AS name")
        return [r["name"] for r in rows if r.get("name")]

    def export_metadata_to_json(self) -> dict[str, Any]:
        labels = self.run_query("CALL db.labels() YIELD label RETURN label")
        rels = self.run_query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
        counts = {row["label"]: self.run_query(f"MATCH (n:`{row['label']}`) RETURN count(n) AS c")[0]["c"] for row in labels}
        rel_counts = {
            row["relationshipType"]: self.run_query(
                f"MATCH ()-[r:`{row['relationshipType']}`]->() RETURN count(r) AS c"
            )[0]["c"]
            for row in rels
        }
        return {
            "labels": {row["label"]: counts[row["label"]] for row in labels},
            "relationships": {row["relationshipType"]: rel_counts[row["relationshipType"]] for row in rels},
        }

    def clear_database(self) -> None:
        if not self.available:
            return
        self.run_query("MATCH (n) DETACH DELETE n")
        log.info("Neo4j 图已清空")

    def ensure_schema(self) -> None:
        """唯一约束 + 索引；重复执行安全。"""
        stmts = [
            ("CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE", {}),
            ("CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE", {}),
            ("CREATE CONSTRAINT scenario_id IF NOT EXISTS FOR (s:Scenario) REQUIRE s.id IS UNIQUE", {}),
            ("CREATE INDEX article_law IF NOT EXISTS FOR (a:Article) ON (a.law_name)", {}),
            ("CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)", {}),
        ]
        self.run_multiple_cypher(stmts)
        log.info("Neo4j schema 已就绪")


_manager: Neo4jManager | None = None


def get_neo4j_manager() -> Neo4jManager:
    global _manager
    if _manager is None:
        _manager = Neo4jManager()
    return _manager
