"""把抽取产物导入 Neo4j（批量 MERGE，对照 TCM ``__001__graph_importer.py``）。

为什么不逐条 CREATE：1465 条文 + 约 4000 条边，逐条走事务会让导入从秒级变成分钟级；
这里按 500 行一批 UNWIND，MERGE 保证幂等，重复导入不会产生重复节点。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.__003__graph.__000__schema import scenario_node_id  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.neo4j_manager import get_neo4j_manager  # noqa: E402
from common.path_utils import read_jsonl  # noqa: E402

log = get_logger("graph_import")

BATCH = 500

ARTICLE_CYPHER = """
UNWIND $rows AS r
MERGE (a:Article {id: r.id})
SET a.law_name = r.law_name, a.law_code = r.law_code, a.article_no = r.article_no,
    a.article_num = r.article_num, a.chapter_path = r.chapter_path,
    a.citation = r.citation, a.content = r.content, a.chars = r.chars
"""

CONCEPT_CYPHER = """
UNWIND $rows AS r
MERGE (c:Concept {id: r.id})
SET c.name = r.name, c.aliases = r.aliases, c.type = r.type,
    c.laws = r.laws, c.parent = r.parent
"""

SCENARIO_CYPHER = """
UNWIND $rows AS r
MERGE (s:Scenario {id: r.id})
SET s.key = r.key, s.name = r.name, s.aliases = r.aliases, s.description = r.description
"""

MENTION_REL_CYPHER = """
UNWIND $rows AS r
MATCH (a:Article {id: r.source}) MATCH (c:Concept {id: r.target})
MERGE (a)-[e:%s]->(c)
SET e.confidence = r.confidence, e.score = r.score, e.cue = r.cue,
    e.provenance = r.provenance, e.aliases = r.aliases, e.count = r.count
"""

APPLIES_CYPHER = """
UNWIND $rows AS r
MATCH (s:Scenario {id: r.source}) MATCH (c:Concept {id: r.target})
MERGE (s)-[e:APPLIES_TO]->(c)
SET e.weight = r.weight, e.provenance = r.provenance
"""

CONCEPT_REL_CYPHER = """
UNWIND $rows AS r
MATCH (a:Concept {id: r.source}) MATCH (b:Concept {id: r.target})
MERGE (a)-[e:%s]->(b)
SET e.confidence = r.confidence, e.provenance = r.provenance
"""

CITES_CYPHER = """
UNWIND $rows AS r
MATCH (a:Article {id: r.source}) MATCH (b:Article {id: r.target})
MERGE (a)-[e:CITES]->(b)
SET e.confidence = r.confidence, e.provenance = r.provenance
"""


def _chunks(rows: list[dict], size: int = BATCH) -> list[list[dict]]:
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def import_all(clear: bool = False, extracted_dir: Path | None = None, lexicon_dir: Path | None = None) -> dict[str, Any]:
    from config import settings

    extracted_dir = extracted_dir or settings.extracted_dir
    lexicon_dir = lexicon_dir or settings.lexicon_dir

    mgr = get_neo4j_manager()
    if not mgr.connect():
        raise ConnectionError(f"无法连接 Neo4j: {settings.neo4j_uri}（先用 docker compose up -d）")
    mgr.ensure_schema()
    if clear:
        mgr.clear_database()

    articles = read_jsonl(extracted_dir / "articles.jsonl")
    concepts = read_jsonl(lexicon_dir / "concepts.jsonl")
    scenarios = read_jsonl(lexicon_dir / "scenarios.jsonl")
    # B7：优先读边类型化文件；缺失时回落旧 mentions 并全部标 DEFINES。
    from app.__002__extract.__003__edge_typing import load_typed

    mentions = load_typed(extracted_dir)
    cites = read_jsonl(extracted_dir / "cites.jsonl")
    requires = read_jsonl(lexicon_dir / "requires.jsonl")
    special_of = read_jsonl(lexicon_dir / "special_of.jsonl")

    def _run(cypher: str, rows: list[dict], label: str) -> int:
        total = 0
        for chunk in _chunks(rows):
            total += mgr.run_multiple_cypher([(cypher, {"rows": chunk})])
        log.info("%s: %d 行", label, total)
        return total

    _run(ARTICLE_CYPHER, [dict(a, id=a["article_id"]) for a in articles], "Article")
    _run(CONCEPT_CYPHER, concepts, "Concept")
    _run(SCENARIO_CYPHER, [dict(s, id=scenario_node_id(s["id"])) for s in scenarios], "Scenario")

    by_type: dict[str, list[dict]] = {}
    for m in mentions:
        edge_type = m.get("edge_type", "DEFINES")
        by_type.setdefault(edge_type, []).append(
            {
                "source": m["article_id"],
                "target": m["concept_id"],
                "confidence": m.get("confidence", 0.8),
                "score": round(
                    float(m.get("confidence", 0.8)) * float(m.get("defines_score", 1.0)), 4
                ),
                "cue": m.get("cue"),
                "provenance": m.get("provenance", "rule"),
                "aliases": m.get("aliases", []),
                "count": m.get("count", 1),
            }
        )
    for edge_type in ("DEFINES", "REFERS_TO"):
        _run(MENTION_REL_CYPHER % edge_type, by_type.get(edge_type, []), edge_type)

    applies = [
        {
            "source": scenario_node_id(s["id"]),
            "target": cid,
            "weight": round(1.0 - 0.08 * i, 3),
            "provenance": "manual",
        }
        for s in scenarios
        for i, cid in enumerate(s["concepts"])
    ]
    _run(APPLIES_CYPHER, applies, "APPLIES_TO")

    rels = [
        {"source": r["source"], "target": r["target"], "confidence": r.get("confidence", 0.9), "provenance": r.get("provenance", "manual")}
        for r in requires
    ]
    _run(CONCEPT_REL_CYPHER % "REQUIRES", rels, "REQUIRES")

    specs = [
        {"source": r["child"], "target": r["parent"], "confidence": r.get("confidence", 0.95), "provenance": r.get("provenance", "manual")}
        for r in special_of
    ]
    _run(CONCEPT_REL_CYPHER % "SPECIAL_OF", specs, "SPECIAL_OF")

    _run(CITES_CYPHER, cites, "CITES")

    stats = mgr.export_metadata_to_json()
    log.info("Neo4j 导入完成 %s", stats)
    return stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", action="store_true")
    a = ap.parse_args()
    import_all(clear=a.clear)
