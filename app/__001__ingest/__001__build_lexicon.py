"""把人工种子编译成可被检索/建图消费的术语库（确定性、可 diff、可审计）。

产物：
  data/lexicon/concepts.jsonl   概念定义 + 别名 + 法律归属 + 上位概念
  data/lexicon/scenarios.jsonl  场景定义 + 关联概念
  data/lexicon/requires.jsonl   概念前置依赖边
  data/lexicon/special_of.jsonl 概念上下位边
  data/lexicon/alias_index.json 别名 -> 概念/场景（实体链接的词典层）
  data/lexicon/stats.json       术语库体检结果
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.__001__ingest.__000__lexicon_seed import (  # noqa: E402
    CONCEPTS,
    LAW_CODES,
    REQUIRES,
    SCENARIOS,
)
from config import settings  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import write_json, write_jsonl  # noqa: E402

log = get_logger("lexicon")


def build() -> dict:
    concept_ids = [c[0] for c in CONCEPTS]
    dup = [k for k, v in Counter(concept_ids).items() if v > 1]
    if dup:
        raise ValueError(f"概念 ID 重复: {dup}")

    known = set(concept_ids)
    concept_rows = []
    for cid, name, aliases, ctype, laws, parent in CONCEPTS:
        for law in laws:
            if law not in LAW_CODES:
                raise ValueError(f"{cid} 引用了未知法律代号 {law}")
        if parent and parent not in known:
            raise ValueError(f"{cid} 的 parent={parent} 不在概念表")
        alias_list = []
        for a in [name, *aliases]:
            a = a.strip()
            if a and a not in alias_list:
                alias_list.append(a)
        concept_rows.append(
            {
                "id": cid,
                "name": name,
                "aliases": alias_list,
                "type": ctype,
                "laws": [LAW_CODES[l] for l in laws],
                "law_codes": list(laws),
                "parent": parent,
            }
        )

    scenario_rows = []
    scenario_ids = [s[0] for s in SCENARIOS]
    for dup_s in [k for k, v in Counter(scenario_ids).items() if v > 1]:
        raise ValueError(f"场景 ID 重复: {dup_s}")
    for sid, name, aliases, desc, concepts in SCENARIOS:
        unknown = [c for c in concepts if c not in known]
        if unknown:
            raise ValueError(f"场景 {sid} 引用了未知概念 {unknown}")
        scenario_rows.append(
            {
                "id": sid,
                "name": name,
                "aliases": [name, *aliases],
                "description": desc,
                "concepts": concepts,
            }
        )

    requires_rows = []
    for src, dst in REQUIRES:
        if src not in known or dst not in known:
            raise ValueError(f"REQUIRES 边非法: {src} -> {dst}")
        requires_rows.append(
            {"source": src, "target": dst, "relation": "REQUIRES", "provenance": "manual", "confidence": 0.9}
        )

    special_rows = [
        {"child": r["id"], "parent": r["parent"], "relation": "SPECIAL_OF", "provenance": "manual", "confidence": 0.95}
        for r in concept_rows
        if r["parent"]
    ]

    # 别名词典：别名 -> [{id, kind, name}]；冲突时保留全部，由链接器按上下文打分
    alias_index: dict[str, list[dict]] = defaultdict(list)
    for r in concept_rows:
        for a in r["aliases"]:
            alias_index[a].append({"id": r["id"], "kind": "concept", "name": r["name"]})
    for r in scenario_rows:
        for a in r["aliases"]:
            alias_index[a].append({"id": r["id"], "kind": "scenario", "name": r["name"]})

    collisions = {a: v for a, v in alias_index.items() if len({x["id"] for x in v}) > 1}

    stats = {
        "concepts": len(concept_rows),
        "scenarios": len(scenario_rows),
        "requires_edges": len(requires_rows),
        "special_of_edges": len(special_rows),
        "scenario_concept_edges": sum(len(s["concepts"]) for s in scenario_rows),
        "aliases": len(alias_index),
        "alias_collisions": len(collisions),
        "concept_types": dict(Counter(r["type"] for r in concept_rows)),
        "concepts_per_law": {
            code: sum(1 for r in concept_rows if code in r["law_codes"]) for code in LAW_CODES
        },
    }

    out = settings.lexicon_dir
    write_jsonl(out / "concepts.jsonl", concept_rows)
    write_jsonl(out / "scenarios.jsonl", scenario_rows)
    write_jsonl(out / "requires.jsonl", requires_rows)
    write_jsonl(out / "special_of.jsonl", special_rows)
    write_json(out / "alias_index.json", {k: v for k, v in sorted(alias_index.items())})
    write_json(out / "alias_collisions.json", collisions)
    write_json(out / "stats.json", stats)
    return stats


def main() -> None:
    stats = build()
    log.info("术语库 %s", stats)
    if stats["concepts"] < 150:
        raise SystemExit("概念数不足 150")
    if stats["requires_edges"] < 120:
        raise SystemExit("REQUIRES 边不足 120")


if __name__ == "__main__":
    main()
