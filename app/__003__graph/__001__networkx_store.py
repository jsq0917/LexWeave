"""NetworkX 离线图镜像：与 Neo4j 同 schema、同查询计划，零外部依赖可复现。

用途有两个，都很关键：
1. CI / 评测机没有 Docker 时照样能跑全链路（图查询语义一致）；
2. 建图产物可以 pickle + JSON 双份进 git-lite 目录，review 时能直接 diff 边集。
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.__003__graph.__000__schema import scenario_node_id  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import read_jsonl, write_json  # noqa: E402

log = get_logger("graph_nx")


def build_multidigraph(
    articles: list[dict],
    concepts: list[dict],
    scenarios: list[dict],
    requires: list[dict],
    special_of: list[dict],
    mentions: list[dict],
    cites: list[dict],
) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()

    for a in articles:
        g.add_node(
            a["article_id"],
            label="Article",
            law_name=a["law_name"],
            law_code=a["article_id"].split("#")[0],
            article_no=a["article_no"],
            article_num=a["article_num"],
            chapter_path=a["chapter_path"],
            content=a["content"],
            citation=a["citation"],
            chars=a["chars"],
        )

    for c in concepts:
        g.add_node(
            c["id"],
            label="Concept",
            name=c["name"],
            aliases=c["aliases"],
            type=c["type"],
            laws=c["laws"],
            parent=c.get("parent"),
        )

    for s in scenarios:
        node_id = scenario_node_id(s["id"])
        g.add_node(
            node_id,
            label="Scenario",
            name=s["name"],
            key=s["id"],
            aliases=s["aliases"],
            description=s["description"],
        )
        for i, cid in enumerate(s["concepts"]):
            weight = round(1.0 - 0.08 * i, 3)
            g.add_edge(node_id, cid, key="APPLIES_TO", relation="APPLIES_TO", weight=weight, provenance="manual")

    for r in requires:
        g.add_edge(
            r["source"],
            r["target"],
            key="REQUIRES",
            relation="REQUIRES",
            confidence=r.get("confidence", 0.9),
            provenance=r.get("provenance", "manual"),
        )

    for r in special_of:
        g.add_edge(
            r["child"],
            r["parent"],
            key="SPECIAL_OF",
            relation="SPECIAL_OF",
            confidence=r.get("confidence", 0.95),
            provenance=r.get("provenance", "manual"),
        )

    for m in mentions:
        edge_type = m.get("edge_type", "DEFINES")
        score = round(float(m.get("confidence", 0.8)) * float(m.get("defines_score", 1.0)), 4)
        g.add_edge(
            m["article_id"],
            m["concept_id"],
            key=edge_type,
            relation=edge_type,
            confidence=m.get("confidence", 0.8),
            score=score,
            cue=m.get("cue"),
            provenance=m.get("provenance", "rule"),
            aliases=m.get("aliases", []),
            count=m.get("count", 1),
        )

    for c in cites:
        g.add_edge(
            c["source"],
            c["target"],
            key="CITES",
            relation="CITES",
            confidence=c.get("confidence", 0.8),
            provenance=c.get("provenance", "regex"),
        )

    return g


class NetworkXGraphStore:
    """查询计划执行器；索引在加载时一次性建好，评测时单次查询是纯字典查找。"""

    backend = "networkx"

    def __init__(self, graph_path: Path) -> None:
        with open(graph_path, "rb") as f:
            self.g: nx.MultiDiGraph = pickle.load(f)
        self._articles: dict[str, dict] = {}
        self._concept_to_articles: dict[str, list[tuple[str, float]]] = {}
        self._scenario_to_concepts: dict[str, list[tuple[str, float]]] = {}
        self._concept_out: dict[str, list[tuple[str, str, float]]] = {}
        self._concept_in: dict[str, list[tuple[str, str, float]]] = {}
        self._cites: dict[str, list[tuple[str, float]]] = {}
        self._concept_scenarios: dict[str, list[tuple[str, float]]] = {}
        self._index()

    def _index(self) -> None:
        for nid, data in self.g.nodes(data=True):
            if data.get("label") == "Article":
                self._articles[nid] = dict(data, id=nid)
        for u, v, key, data in self.g.edges(keys=True, data=True):
            rel = data.get("relation", key)
            conf = float(data.get("confidence", data.get("weight", 1.0)) or 1.0)
            score = float(data.get("score", conf) or conf)
            if rel in ("DEFINES", "REFERS_TO"):
                # 候选池不变（两种边都进），但 REFERS_TO 在排序上被降权（0.45）。
                self._concept_to_articles.setdefault(v, []).append((u, score))
            elif rel == "APPLIES_TO":
                self._scenario_to_concepts.setdefault(u, []).append((v, conf))
                self._concept_scenarios.setdefault(v, []).append((u, conf))
            elif rel in ("REQUIRES", "SPECIAL_OF"):
                self._concept_out.setdefault(u, []).append((v, rel, conf))
                self._concept_in.setdefault(v, []).append((u, rel, conf))
            elif rel == "CITES":
                self._cites.setdefault(u, []).append((v, conf))
        for key in self._concept_to_articles:
            self._concept_to_articles[key].sort(key=lambda x: x[1], reverse=True)

    # ---------------- 查询计划 ----------------
    def concept_articles(self, concept_ids: list[str], limit: int = 20) -> list[dict]:
        agg: dict[str, dict[str, Any]] = {}
        for cid in concept_ids:
            for aid, conf in self._concept_to_articles.get(cid, []):
                row = agg.setdefault(
                    aid,
                    {
                        "article_id": aid,
                        "law_name": self._articles[aid]["law_name"],
                        "article_no": self._articles[aid]["article_no"],
                        "article_num": self._articles[aid]["article_num"],
                        "chapter_path": self._articles[aid]["chapter_path"],
                        "concepts": [],
                        "_conf": [],
                    },
                )
                if cid not in row["concepts"]:
                    row["concepts"].append(cid)
                    row["_conf"].append(conf)
        rows = []
        for row in agg.values():
            row["confidence"] = round(sum(row["_conf"]) / len(row["_conf"]), 4)
            row["concept_hits"] = len(row["concepts"])
            row.pop("_conf")
            rows.append(row)
        rows.sort(key=lambda r: (-r["confidence"], -r["concept_hits"], r["article_id"]))
        return rows[:limit]

    def scenario_concepts(self, scenario_ids: list[str]) -> list[dict]:
        out = []
        for sid in scenario_ids:
            nid = scenario_node_id(sid)
            for cid, w in self._scenario_to_concepts.get(nid, []):
                out.append(
                    {
                        "scenario_id": nid,
                        "scenario_key": self.g.nodes[nid].get("key", sid),
                        "scenario_name": self.g.nodes[nid].get("name", sid),
                        "concept_id": cid,
                        "concept_name": self.g.nodes[cid].get("name", cid),
                        "weight": w,
                    }
                )
        return out

    def concept_neighbors(
        self, concept_ids: list[str], relations: list[str] | None = None, direction: str = "out", limit: int = 50
    ) -> list[dict]:
        rels = set(relations or ["REQUIRES", "SPECIAL_OF"])
        table = self._concept_in if direction == "in" else self._concept_out
        out = []
        seen = set()
        for cid in concept_ids:
            for other, rel, conf in table.get(cid, []):
                if rel not in rels or (cid, other, rel) in seen:
                    continue
                seen.add((cid, other, rel))
                out.append(
                    {
                        "source": cid,
                        "target": other,
                        "target_name": self.g.nodes[other].get("name", other),
                        "relation": rel,
                        "confidence": conf,
                    }
                )
        out.sort(key=lambda r: r["confidence"], reverse=True)
        return out[:limit]

    def article_cites(self, article_ids: list[str]) -> list[dict]:
        out = []
        for aid in article_ids:
            for tid, conf in self._cites.get(aid, []):
                a = self._articles.get(tid, {})
                out.append(
                    {
                        "source": aid,
                        "target": tid,
                        "law_name": a.get("law_name"),
                        "article_no": a.get("article_no"),
                        "confidence": conf,
                    }
                )
        return out

    def concept_scenarios(self, concept_ids: list[str], limit: int = 20) -> list[dict]:
        out, seen = [], set()
        for cid in concept_ids:
            for sid, w in self._concept_scenarios.get(cid, []):
                if (cid, sid) in seen:
                    continue
                seen.add((cid, sid))
                out.append(
                    {
                        "scenario_id": sid,
                        "scenario_key": self.g.nodes[sid].get("key", sid),
                        "scenario_name": self.g.nodes[sid].get("name", sid),
                        "concept_id": cid,
                        "weight": w,
                    }
                )
        out.sort(key=lambda r: -r["weight"])
        return out[:limit]

    def article_detail(self, ids: list[str]) -> list[dict]:
        return [dict(self._articles[i], id=i) for i in ids if i in self._articles]

    # ---------------- 元信息 ----------------
    def stats(self) -> dict[str, Any]:
        labels: dict[str, int] = {}
        relations: dict[str, int] = {}
        for _, data in self.g.nodes(data=True):
            labels[data.get("label", "?")] = labels.get(data.get("label", "?"), 0) + 1
        for _, _, key, data in self.g.edges(keys=True, data=True):
            rel = data.get("relation", key)
            relations[rel] = relations.get(rel, 0) + 1
        return {
            "backend": "networkx",
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "labels": labels,
            "relationships": relations,
        }

    def export_json(self, path: Path) -> None:
        nodes = [{"id": n, **data} for n, data in self.g.nodes(data=True)]
        edges = [
            {"source": u, "target": v, "relation": d.get("relation", k), **{kk: vv for kk, vv in d.items() if kk != "relation"}}
            for u, v, k, d in self.g.edges(keys=True, data=True)
        ]
        write_json(path, {"nodes": nodes, "edges": edges})


def build_and_save(
    extracted_dir: Path | None = None,
    lexicon_dir: Path | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    from config import settings

    extracted_dir = extracted_dir or settings.extracted_dir
    lexicon_dir = lexicon_dir or settings.lexicon_dir
    out_path = out_path or (settings.graph_dir / "lexweave_graph.pkl")

    articles = read_jsonl(extracted_dir / "articles.jsonl")
    # B7：优先读边类型化文件（DEFINES / REFERS_TO）；缺失时回落旧文件并全部标 DEFINES。
    from app.__002__extract.__003__edge_typing import load_typed

    mentions = load_typed(extracted_dir)
    cites = read_jsonl(extracted_dir / "cites.jsonl")
    concepts = read_jsonl(lexicon_dir / "concepts.jsonl")
    scenarios = read_jsonl(lexicon_dir / "scenarios.jsonl")
    requires = read_jsonl(lexicon_dir / "requires.jsonl")
    special_of = read_jsonl(lexicon_dir / "special_of.jsonl")

    g = build_multidigraph(articles, concepts, scenarios, requires, special_of, mentions, cites)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)

    store = NetworkXGraphStore(out_path)
    store.export_json(out_path.parent / "graph.json")
    stats = store.stats()
    log.info("图镜像已保存 %s %s", out_path, stats)
    return stats


if __name__ == "__main__":
    build_and_save()
