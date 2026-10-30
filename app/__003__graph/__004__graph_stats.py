"""图体检：规模 / 覆盖 / 结构 / 抽检精度。

抽检方法：随机抽 100 个概念，各取 1 条 **DEFINES** 边，回原文核对「概念名或别名确实出现」。
这是召回侧的精度代理指标（规则抽取理论上接近 1.0，LLM 补抽会拉低，所以必须量出来）。
B7 起 Article→Concept 已拆成 DEFINES / REFERS_TO，本脚本分别报数；定义级抽检见
``scripts/audit_defines.py``（盲判落盘 ``data/eval_out/defines_blind_review.json``，
规则自洽性校验落盘 ``data/eval_out/defines_selfcheck.json``）。
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.__003__graph.__001__networkx_store import NetworkXGraphStore  # noqa: E402
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
        "scope": "literal_mention_proxy",
        "scope_note": (
            "只验概念名/别名是否出现在条文正文，是『召回侧字面命中』的代理指标，"
            "不等于『该条在定义该概念』。定义级精度见 data/eval_out/defines_blind_review.json（LLM 盲判，" 
            "不给别名与 cue），规则自洽性见 defines_selfcheck.json。"
        ),
        "failures": failures[:10],
    }


def main() -> None:
    # 统计口径绑定离线镜像（pkl），避免 Neo4j 刚清空/正在导入时把空库数字写进 stats.json。
    pkl = Path("data/graph/lexweave_graph.pkl")
    store = NetworkXGraphStore(pkl) if pkl.exists() else get_graph_store()
    concepts = {r["id"]: r for r in read_jsonl("data/lexicon/concepts.jsonl")}
    scenarios = read_jsonl("data/lexicon/scenarios.jsonl")
    from app.__002__extract.__003__edge_typing import load_typed

    mentions = load_typed(Path("data/extracted"))

    stats = store.stats()
    defines = [m for m in mentions if m.get("edge_type", "DEFINES") == "DEFINES"]
    refers = [m for m in mentions if m.get("edge_type") == "REFERS_TO"]
    defined = {m["concept_id"] for m in defines}
    scen_concepts = {c for s in scenarios for c in s["concepts"]}
    audit = run_audit(store, concepts)

    by_prov = Counter(m.get("provenance", "rule") for m in mentions)
    confs = [m["confidence"] for m in mentions]
    out = {
        **stats,
        "article_concept_edges": {
            "total": len(mentions),
            "DEFINES": len(defines),
            "REFERS_TO": len(refers),
            "defines_ratio": round(len(defines) / max(len(mentions), 1), 4),
        },
        "concept_coverage": {
            "concepts": len(concepts),
            "covered_by_defines": len(defined),
            "covered_by_articles": len({m["concept_id"] for m in mentions}),
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
    log.info(
        "Article→Concept %d 条（DEFINES %d / REFERS_TO %d），概念覆盖(DEFINES) %.1f%%",
        len(mentions),
        len(defines),
        len(refers),
        out["concept_coverage"]["coverage"] * 100,
    )

    problems = []
    if out["labels"].get("Article", 0) != 1465:
        problems.append(f"Article 节点数 {out['labels'].get('Article')} != 1465")
    if out["labels"].get("Concept", 0) < 150:
        problems.append("Concept 少于 150")
    if len(defines) < 800:
        problems.append(f"DEFINES {len(defines)} < 800（B7 拆边后口径变化，照实报告）")
    if audit["precision"] < 0.85:
        problems.append(f"字面命中精度 {audit['precision']} < 0.85")
    if problems:
        log.warning("B1 结构验收未过: %s", "; ".join(problems))
    else:
        log.info("B1 结构验收通过（规模 / provenance / 字面命中）")
    log.warning(
        "抽检口径：%.1f%% 是『字面命中』代理（只验概念名是否出现在正文），不等于定义级精度；"
        "定义级精度见 data/eval_out/defines_blind_review.json；defines_selfcheck.json 只是规则自洽性",
        audit["precision"] * 100,
    )


if __name__ == "__main__":
    main()
