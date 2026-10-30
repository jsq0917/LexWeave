"""图体检：规模 / 覆盖 / 结构 / 抽检精度。

抽检方法：随机抽 100 个概念，各取 1 条 DEFINES 边，回原文核对「概念名或别名确实出现」。
这是召回侧的精度代理指标（规则抽取理论上接近 1.0，LLM 补抽会拉低，所以必须量出来）。
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.__003__graph.__003__graph_store import get_graph_store  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import read_jsonl, write_json  # noqa: E402

log = get_logger("graph_stats")
AUDIT_N = 100
SEED = 20261029


def run_audit(store, concepts: dict[str, dict]) -> dict:
    rng = random.Random(SEED)
    ids = sorted(concepts)
    sample = rng.sample(ids, min(AUDIT_N, len(ids)))
    ok, checked, failures = 0, 0, []
    for cid in sample:
        arts = store.concept_articles([cid], limit=3)
        if not arts:
            continue
        aid = arts[0]["article_id"]
        detail = store.article_detail([aid])[0]
        text = detail["content"]
        needles = [concepts[cid]["name"], *concepts[cid]["aliases"]]
        if any(n and n in text for n in needles):
            ok += 1
        else:
            failures.append({"concept": cid, "article": aid, "text": text[:80]})
        checked += 1
    return {
        "sampled": checked,
        "pass": ok,
        "precision": round(ok / checked, 4) if checked else 0.0,
        "failures": failures[:10],
    }


def main() -> None:
    store = get_graph_store()
    concepts = {r["id"]: r for r in read_jsonl("data/lexicon/concepts.jsonl")}
    scenarios = read_jsonl("data/lexicon/scenarios.jsonl")
    mentions = read_jsonl("data/extracted/mentions.jsonl")

    stats = store.stats()
    defined = {m["concept_id"] for m in mentions}
    scen_concepts = {c for s in scenarios for c in s["concepts"]}
    audit = run_audit(store, concepts)

    by_prov = Counter(m.get("provenance", "rule") for m in mentions)
    confs = [m["confidence"] for m in mentions]
    out = {
        **stats,
        "concept_coverage": {
            "concepts": len(concepts),
            "covered_by_articles": len(defined),
            "coverage": round(len(defined) / len(concepts), 4),
            "covered_by_scenarios": len(scen_concepts),
        },
        "defines_by_provenance": dict(by_prov),
        "defines_confidence": {
            "mean": round(sum(confs) / len(confs), 4) if confs else 0.0,
            "min": round(min(confs), 4) if confs else 0.0,
            "max": round(max(confs), 4) if confs else 0.0,
        },
        "audit": audit,
    }
    write_json("data/graph/stats.json", out)
    log.info("图统计 %s", {k: out[k] for k in ("backend", "nodes", "edges", "labels", "relationships")})
    log.info("概念覆盖 %.1f%%，抽检精度 %.1f%%", out["concept_coverage"]["coverage"] * 100, audit["precision"] * 100)

    problems = []
    if out["labels"].get("Article", 0) != 1465:
        problems.append(f"Article 节点数 {out['labels'].get('Article')} != 1465")
    if out["labels"].get("Concept", 0) < 150:
        problems.append("Concept 少于 150")
    if out["relationships"].get("DEFINES", 0) < 800:
        problems.append("DEFINES 少于 800")
    if audit["precision"] < 0.85:
        problems.append(f"抽检精度 {audit['precision']} < 0.85")
    if problems:
        log.warning("B1 验收未过: %s", "; ".join(problems))
    else:
        log.info("B1 验收通过")


if __name__ == "__main__":
    main()
