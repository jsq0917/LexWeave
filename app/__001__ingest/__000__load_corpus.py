"""语料装载与体检：只做校验 + 生成 manifest，不改写原始数据。

语料直接复用 LexSynergy 的 1465 条法条（劳动合同法 98 / 劳动法 107 / 民法典 1260），
字段 ``law_name / doc_id / article_no / article_num / chapter_path / content``。
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import LAW_ORDER, settings  # noqa: E402
from common.law_utils import article_id, law_code  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import read_jsonl, write_json  # noqa: E402

log = get_logger("ingest")

REQUIRED_FIELDS = ("law_name", "doc_id", "article_no", "article_num", "chapter_path", "content")


def load_corpus(path: Path | None = None) -> list[dict]:
    path = path or settings.corpus_path
    rows = read_jsonl(path)
    if not rows:
        raise RuntimeError(f"语料为空: {path}")
    for i, row in enumerate(rows):
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            raise ValueError(f"第 {i} 行缺少字段 {missing}")
        if not isinstance(row["article_num"], int):
            raise TypeError(f"第 {i} 行 article_num 非整数")
    return rows


def build_manifest(rows: list[dict], path: Path | None = None) -> dict:
    path = path or settings.corpus_path
    raw = path.read_bytes()
    counts = Counter(r["law_name"] for r in rows)
    ids = [article_id(r["law_name"], r["article_num"]) for r in rows]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    manifest = {
        "source": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "total": len(rows),
        "by_law": {
            law_code(law): sum(1 for r in rows if r["law_name"] == law) for law in LAW_ORDER
        },
        "duplicate_ids": dupes,
        "chapter_coverage": len({r["chapter_path"] for r in rows}),
        "chars": sum(len(r["content"]) for r in rows),
    }
    write_json(settings.data_dir / "corpus" / "manifest.json", manifest)
    return manifest


def main() -> None:
    rows = load_corpus()
    manifest = build_manifest(rows)
    log.info("语料 %d 条 %s", manifest["total"], manifest["by_law"])
    if manifest["duplicate_ids"]:
        raise SystemExit(f"存在重复 ID: {manifest['duplicate_ids'][:5]}")
    for law in LAW_ORDER:
        n = sum(1 for r in rows if r["law_name"] == law)
        log.info("%s: %d 条", law, n)


if __name__ == "__main__":
    main()
