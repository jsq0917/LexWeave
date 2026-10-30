"""一键建图：抽取产物 -> NetworkX 镜像 -> 图体检（+ 可选 Neo4j 导入）。

注意：`data/graph/stats.json` 由本脚本末尾的 `graph_stats` 写入，不再单独维护。
早期版本需要手工再跑一次 `__004__graph_stats.py`，漏跑就会留下与 `lexweave_graph.pkl`
不一致的陈旧统计（曾出现 4,221 边 vs 4,259 边、REQUIRES 174 vs 209），已在 README §10 修正。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.__003__graph.__001__networkx_store import build_and_save  # noqa: E402
from app.__003__graph.__002__graph_importer import import_all  # noqa: E402
from app.__003__graph.__004__graph_stats import main as graph_stats_main  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402

log = get_logger("scripts.build_graph")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neo4j", action="store_true", help="同时导入 Neo4j")
    ap.add_argument("--clear", action="store_true", help="导入前清空 Neo4j")
    ap.add_argument("--skip-stats", action="store_true", help="跳过图体检（不刷新 data/graph/stats.json）")
    args = ap.parse_args()

    stats = build_and_save()
    log.info("NetworkX 镜像: %s", stats)
    if not args.skip_stats:
        graph_stats_main()
    else:
        log.warning("已跳过图体检：data/graph/stats.json 可能与本 pkl 不一致")
    if args.neo4j:
        nstats = import_all(clear=args.clear)
        log.info("Neo4j: %s", nstats)


if __name__ == "__main__":
    main()
