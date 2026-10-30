"""构建检索索引：文本路（bge-m3 + BM25）与图路（概念别名向量）。

两个索引分开落盘，评测里可以只重建其中一个：
* ``data/index/text.faiss`` / ``bm25.pkl`` / ``text_meta.json``
* ``data/index/concept.faiss`` / ``concept_meta.json``

用法：
    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --only text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.__004__retrieval.__000__text_index import TextIndex  # noqa: E402
from app.__004__retrieval.__001__entity_linker import ConceptIndex  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import read_jsonl, write_json  # noqa: E402

log = get_logger("build_index")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["text", "concept"], default=None)
    args = ap.parse_args()

    stats: dict = {}
    if args.only in (None, "text"):
        stats["text"] = TextIndex().build(read_jsonl("data/extracted/articles.jsonl"))
        log.info("文本索引 %s", stats["text"])
    if args.only in (None, "concept"):
        stats["concept"] = ConceptIndex().build(
            read_jsonl("data/lexicon/concepts.jsonl"),
            read_jsonl("data/lexicon/scenarios.jsonl"),
        )
        log.info("概念索引 %s", stats["concept"])
    write_json("data/index/build_stats.json", stats)


if __name__ == "__main__":
    main()
