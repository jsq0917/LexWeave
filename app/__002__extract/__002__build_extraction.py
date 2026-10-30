"""阶段 002 主流程：规则抽取 + 可选 LLM 补充 -> 结构化的图数据。

产出：
  data/extracted/articles.jsonl  带统一 article_id 的语料
  data/extracted/mentions.jsonl  (article, concept) 提及 -> DEFINES 边
  data/extracted/cites.jsonl     法条互引 -> CITES 边
  data/extracted/stats.json      抽取体检
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.__001__ingest.__000__load_corpus import load_corpus  # noqa: E402
from app.__002__extract.__000__extract_graph_data_utils import run_llm_extraction  # noqa: E402
from app.__002__extract.__001__rule_extractor import (  # noqa: E402
    alias_document_frequency,
    extract_citations,
    extract_mentions,
    load_lexicon,
)
from common.law_utils import article_id, citation  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import read_jsonl, write_json, write_jsonl  # noqa: E402

log = get_logger("extract")


def prepare_articles(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append(
            {
                "article_id": article_id(r["law_name"], r["article_num"]),
                "law_name": r["law_name"],
                "article_no": r["article_no"],
                "article_num": r["article_num"],
                "chapter_path": r["chapter_path"],
                "content": r["content"],
                "citation": citation(r["law_name"], r["article_no"]),
                "chars": len(r["content"]),
            }
        )
    return out


def merge_llm_mentions(rule_rows: list[dict], llm_rows: list[dict], articles: list[dict]) -> list[dict]:
    merged = {(r["article_id"], r["concept_id"]): dict(r) for r in rule_rows}
    for row in llm_rows:
        for cid in row.get("concepts", []):
            key = (row["article_id"], cid)
            if key in merged:
                merged[key]["llm_confirmed"] = True
                merged[key]["confidence"] = round(min(0.99, merged[key]["confidence"] + 0.03), 3)
            else:
                merged[key] = {
                    "article_id": row["article_id"],
                    "concept_id": cid,
                    "aliases": [],
                    "count": 1,
                    "confidence": 0.8,
                    "provenance": "llm",
                }
    return list(merged.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-llm", action="store_true", help="启用 LLM 补充抽取（需 LLM_API_KEY）")
    ap.add_argument("--llm-limit", type=int, default=200, help="最多用 LLM 处理多少条（控成本）")
    ap.add_argument("--llm-workers", type=int, default=4)
    args = ap.parse_args()

    articles = prepare_articles(load_corpus())
    write_jsonl(Path("data/extracted/articles.jsonl"), articles)

    concepts, alias_index = load_lexicon()
    concept_rows = read_jsonl(Path("data/lexicon/concepts.jsonl"))
    scenario_rows = read_jsonl(Path("data/lexicon/scenarios.jsonl"))

    df = alias_document_frequency(articles, alias_index)
    mentions = extract_mentions(articles, concepts, alias_index, df)

    llm_stats = None
    if args.use_llm:
        from config import settings

        if not settings.llm_enabled:
            raise SystemExit("--use-llm 需要配置 LLM_API_KEY")
        llm_stats = run_llm_extraction(
            articles,
            concept_rows,
            scenario_rows,
            Path("data/extracted/llm_mentions.jsonl"),
            max_workers=args.llm_workers,
            limit=args.llm_limit,
        )
        llm_rows = read_jsonl(Path("data/extracted/llm_mentions.jsonl"))
        mentions = merge_llm_mentions(mentions, llm_rows, articles)

    valid_ids = {a["article_id"] for a in articles}
    cites = extract_citations(articles, valid_ids)

    write_jsonl(Path("data/extracted/mentions.jsonl"), mentions)
    write_jsonl(Path("data/extracted/cites.jsonl"), cites)

    covered_articles = {m["article_id"] for m in mentions}
    covered_concepts = {m["concept_id"] for m in mentions}
    stats = {
        "articles": len(articles),
        "mentions": len(mentions),
        "defines_edges": len(mentions),
        "covered_articles": len(covered_articles),
        "article_coverage": round(len(covered_articles) / len(articles), 3),
        "covered_concepts": len(covered_concepts),
        "concept_coverage": round(len(covered_concepts) / len(concept_rows), 3),
        "uncovered_articles": len(articles) - len(covered_articles),
        "cites": len(cites),
        "cross_law_cites": sum(1 for c in cites if c["source"].split("#")[0] != c["target"].split("#")[0]),
        "top_concepts": Counter(m["concept_id"] for m in mentions).most_common(10),
        "llm": llm_stats,
    }
    write_json(Path("data/extracted/stats.json"), stats)
    log.info(
        "抽取完成: mentions=%d articles=%d(覆盖 %.1f%%) concepts=%d cites=%d",
        stats["mentions"],
        stats["covered_articles"],
        stats["article_coverage"] * 100,
        stats["covered_concepts"],
        stats["cites"],
    )
    if stats["defines_edges"] < 800:
        log.warning("DEFINES 边少于 800，检查别名表或降低置信度阈值")


if __name__ == "__main__":
    main()
