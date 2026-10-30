"""B1 抽检：从 DEFINES 边（mentions.jsonl）里等距抽 100 条，核对概念名/别名是否真的出现在条文正文。

这不是人工语义标注，只是可复现的机械抽检：
  * 命中名字只能说明「条文提到了这个词」，不能证明「该条就是在定义这个概念」；
  * 因此把抽检条目原样导出到 data/eval_out/defines_audit.json，供人逐条浏览并记录人工判定；
  * README 里分开写「机械命中率」与「人工抽检准确率」，不拿前者冒充后者。

用法：python scripts/audit_defines.py [--sample 100] [--seed 20261029]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.path_utils import read_jsonl, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()

    concepts = {c["id"]: c for c in read_jsonl("data/lexicon/concepts.jsonl")}
    articles = {a["article_id"]: a for a in read_jsonl("data/extracted/articles.jsonl")}
    mentions = read_jsonl("data/extracted/mentions.jsonl")

    rng = random.Random(args.seed)
    picked = rng.sample(mentions, min(args.sample, len(mentions)))

    rows, hit, prov = [], 0, {}
    for m in picked:
        c = concepts.get(m["concept_id"], {})
        a = articles.get(m["article_id"], {})
        needles = [c.get("name", ""), *(c.get("aliases") or [])]
        ok = any(n and n in a.get("content", "") for n in needles)
        hit += int(ok)
        prov[m.get("provenance") or "-"] = prov.get(m.get("provenance") or "-", 0) + 1
        rows.append(
            {
                "concept": m["concept_id"],
                "concept_name": c.get("name"),
                "article_id": m["article_id"],
                "citation": a.get("citation"),
                "snippet": (a.get("content") or "")[:90],
                "literal_hit": ok,
                "provenance": m.get("provenance"),
                "confidence": m.get("confidence"),
            }
        )

    report = {
        "sampled": len(rows),
        "literal_hit_rate": round(hit / max(len(rows), 1), 4),
        "provenance": prov,
        "note": "literal_hit 只判「条文正文出现了概念名或别名」，不等于人工确认「该条是在定义该概念」。",
        "items": rows,
    }
    write_json("data/eval_out/defines_audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, ensure_ascii=False, indent=1))
    print("\n| 概念 | 条文 | 字面命中 | provenance |")
    print("|---|---|---|---|")
    for r in rows[:25]:
        print(f"| {r['concept_name']} | {r['citation']} | {r['literal_hit']} | {r['provenance']} |")


if __name__ == "__main__":
    main()
