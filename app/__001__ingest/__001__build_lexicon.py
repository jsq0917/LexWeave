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
    GOLD_PATH_REQUIRES,
    LAW_CODES,
    REQUIRES,
    SCENARIOS,
)
from config import settings  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402

log = get_logger("lexicon")


def _compose() -> tuple[list[dict], list[dict], list[dict], list[dict], dict, dict, dict]:
    """按种子编译出全部产物（不落盘）。

    与 :func:`build` 分开，是为了让 ``--check`` 能在**不写文件**的前提下复算一遍，
    再把结果和磁盘上的 jsonl 逐条比对。这样「有人手改了 jsonl 却没改种子」会立刻暴露，
    而不是等到别人按 README 重建时数字对不上。
    """
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
    seen_requires: set[tuple[str, str]] = set()
    for src, dst in REQUIRES:
        if src not in known or dst not in known:
            raise ValueError(f"REQUIRES 边非法: {src} -> {dst}")
        if (src, dst) in seen_requires:
            continue
        seen_requires.add((src, dst))
        requires_rows.append(
            {"source": src, "target": dst, "relation": "REQUIRES", "provenance": "manual", "confidence": 0.9}
        )
    for src, dst in GOLD_PATH_REQUIRES:
        if src not in known or dst not in known:
            raise ValueError(f"金标补边非法: {src} -> {dst}")
        if (src, dst) in seen_requires:
            continue
        seen_requires.add((src, dst))
        requires_rows.append(
            {"source": src, "target": dst, "relation": "REQUIRES", "provenance": "manual", "confidence": 0.85}
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
        "gold_path_requires": len(GOLD_PATH_REQUIRES),
        "special_of_edges": len(special_rows),
        "scenario_concept_edges": sum(len(s["concepts"]) for s in scenario_rows),
        "aliases": len(alias_index),
        "alias_collisions": len(collisions),
        "concept_types": dict(Counter(r["type"] for r in concept_rows)),
        "concepts_per_law": {
            code: sum(1 for r in concept_rows if code in r["law_codes"]) for code in LAW_CODES
        },
    }

    return concept_rows, scenario_rows, requires_rows, special_rows, alias_index, collisions, stats


ARTIFACTS = (
    ("concepts.jsonl", 0),
    ("scenarios.jsonl", 1),
    ("requires.jsonl", 2),
    ("special_of.jsonl", 3),
    ("alias_index.json", 4),
    ("alias_collisions.json", 5),
    ("stats.json", 6),
)


def build() -> dict:
    """编译并落盘，返回体检结果。"""
    concept_rows, scenario_rows, requires_rows, special_rows, alias_index, collisions, stats = _compose()
    out = settings.lexicon_dir
    write_jsonl(out / "concepts.jsonl", concept_rows)
    write_jsonl(out / "scenarios.jsonl", scenario_rows)
    write_jsonl(out / "requires.jsonl", requires_rows)
    write_jsonl(out / "special_of.jsonl", special_rows)
    write_json(out / "alias_index.json", {k: v for k, v in sorted(alias_index.items())})
    write_json(out / "alias_collisions.json", collisions)
    write_json(out / "stats.json", stats)
    return stats


def check() -> list[str]:
    """不落盘地复算一遍，与磁盘上的产物逐条比对，返回漂移说明（空列表表示一致）。

    存在的意义：术语库是「种子 → jsonl」的单向编译产物，但历史上出现过
    直接改 jsonl、不回写种子的情况，于是重建一次就少 10 个概念、23 条边，
    而 README 里的节点/边数还停在旧构建上。这个检查把这类漂移变成一条可执行的断言。
    """
    composed = _compose()
    drift: list[str] = []
    out = settings.lexicon_dir
    for name, idx in ARTIFACTS:
        path = out / name
        if not path.exists():
            drift.append(f"{name}: 文件不存在")
            continue
        if name.endswith(".jsonl"):
            on_disk = read_jsonl(path)
            expected = composed[idx]
        else:
            on_disk = read_json(path)
            expected = composed[idx]
            if name == "alias_index.json":
                expected = {k: v for k, v in sorted(expected.items())}
        if on_disk != expected:
            if isinstance(expected, list):
                drift.append(
                    f"{name}: 磁盘 {len(on_disk)} 条 / 重建 {len(expected)} 条"
                    f"（{'内容不同，条数相同' if len(on_disk) == len(expected) else '条数已漂移'}）"
                )
            else:
                drift.append(f"{name}: 内容不一致")
    return drift


def main() -> None:
    if "--check" in sys.argv:
        drift = check()
        if drift:
            for d in drift:
                log.error("术语库漂移 %s", d)
            raise SystemExit(
                "术语库与种子不一致：请把改动回写 __000__lexicon_seed.py 后重跑 "
                "`python app/__001__ingest/__001__build_lexicon.py`"
            )
        log.info("术语库与种子一致（%s）", settings.lexicon_dir)
        return

    stats = build()
    log.info("术语库 %s", stats)
    if stats["concepts"] < 150:
        raise SystemExit("概念数不足 150")
    if stats["requires_edges"] < 120:
        raise SystemExit("REQUIRES 边不足 120")


if __name__ == "__main__":
    main()
