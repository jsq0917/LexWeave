"""一键建图：抽取产物 -> NetworkX 镜像（+ 可选 Neo4j 导入）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.__003__graph.__001__networkx_store import build_and_save  # noqa: E402
from app.__003__graph.__002__graph_importer import import_all  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402

log = get_logger("scripts.build_graph")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neo4j", action="store_true", help="同时导入 Neo4j")
    ap.add_argument("--clear", action="store_true", help="导入前清空 Neo4j")
    args = ap.parse_args()

    stats = build_and_save()
    log.info("NetworkX 镜像: %s", stats)
    if args.neo4j:
        nstats = import_all(clear=args.clear)
        log.info("Neo4j: %s", nstats)


if __name__ == "__main__":
    main()
