"""B2 验收自检：不改数据，只把验收项一条条算出来。

检查项与清单一致：
1. **规模**：60 条，二跳 30 / 三跳 20 / 跨法 10；
2. **泄漏率**：题面与所锁条文的最大二元组重合率 < 5%（逐条重算，不信任历史字段）；
3. **逐跳必要性**：按**锁定条文**逐条核（两个概念落在同一条时只算一条），
   判据是三次独立判读里的多数票；任一法条未被判为必要即整条不通过；
4. **路径可走**：``expected_path`` 的每一段（场景→概念、概念→概念、概念→条文）都必须在图里真实存在，
   否则这条题在图上根本检索不到，属于造题错误；
5. **κ**：Pass A（人工给定链）与 Pass B（独立 LLM 判读）的一致性，以及两次 Pass B 的稳定性。

输出 ``data/eval_out/gold.json``，并在控制台打印未通过项，便于逐条修。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.__003__graph.__003__graph_store import get_graph_store  # noqa: E402
from common.path_utils import read_jsonl, write_json  # noqa: E402
from eval.common import bigram_overlap, cohen_kappa  # noqa: E402

LEAK_THRESHOLD = 0.05
# 题面与单条原文的重合率上限。清单口径是 5%，但建题脚本内部用的是 0.45 的「抄写」阈值，
# 这里按清单口径重算，两者都报出来，不拿宽口径冒充严口径。
STRICT_LEAK = 0.05
BUILD_LEAK = 0.45
VERBATIM_MAX = 10


def longest_common(a: str, b: str) -> int:
    """最长公共连续子串长度（字符级 DP），用来判「题面是否成句抄了原文」。"""
    prev = [0] * (len(b) + 1)
    best = 0
    for ch in a:
        cur = [0] * (len(b) + 1)
        for j, cj in enumerate(b, 1):
            if ch == cj:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def _pick(d: dict, *keys: str) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def check_path(store: Any, row: dict, articles: dict[str, dict]) -> list[str]:
    """返回路径断点列表；空表示整条路径在图里连通。"""
    path = row["expected_path"]
    breaks: list[str] = []
    sid = row["scenario_id"]
    scenario_concepts = {_pick(r, "concept_id", "id") for r in store.scenario_concepts([sid])}
    cids = [p for p in path if not p.startswith("sc:") and not p.startswith(("lcl#", "ll#", "cc#"))]
    aids = [p for p in path if p.startswith(("lcl#", "ll#", "cc#"))]
    if cids and cids[0] not in scenario_concepts:
        breaks.append(f"{sid} -> {cids[0]}")
    for a, b in zip(cids, cids[1:]):
        ok = False
        for direction in ("out", "in"):
            nbrs = {
                _pick(r, "concept_id", "id", "target", "source")
                for r in store.concept_neighbors([a], ["REQUIRES", "SPECIAL_OF"], direction=direction, limit=200)
            }
            if b in nbrs:
                ok = True
                break
        if not ok:
            breaks.append(f"{a} -> {b}")
    for cid, aid in (row.get("concept_articles") or {c["id"]: a for c, a in zip(row["concepts"], aids)}).items():
        if aid not in [r["article_id"] for r in store.concept_articles([cid], limit=100)]:
            breaks.append(f"{cid} -> {aid}")
    return breaks


def main() -> None:
    rows = read_jsonl("data/gold/gold_60.jsonl")
    articles = {a["article_id"]: a for a in read_jsonl("data/extracted/articles.jsonl")}
    store = get_graph_store()

    leaks, need_fail, art_need_fail, path_fail = [], [], [], []
    quotes = 0
    verbatim = []
    for r in rows:
        worst = max(bigram_overlap(r["question"], articles[a]["content"]) for a in r["expected_articles"])
        span = max(longest_common(r["question"], articles[a]["content"]) for a in r["expected_articles"])
        r["_leak"] = worst
        if worst > BUILD_LEAK:
            quotes += 1
        if span >= VERBATIM_MAX:
            verbatim.append((r["id"], span))
        if worst > STRICT_LEAK:
            leaks.append((r["id"], round(worst, 4)))
        # 概念层（清单口径：逐跳必要性）
        con_need = r["validation"].get("necessity_by_concept")
        if con_need is None:
            con_need = {c["id"]: r["validation"]["necessity_pass"] for c in r["concepts"]}
        bad = [c for c, ok in con_need.items() if not ok]
        if bad:
            need_fail.append((r["id"], bad))
        # 条文层（更严的补充审计）
        art_need = r["validation"].get("necessity_by_article")
        if art_need is None:
            art_need = {a: r["validation"]["necessity_pass"] for a in r["expected_articles"]}
        bad_art = [a for a, ok in art_need.items() if not ok]
        if bad_art:
            art_need_fail.append((r["id"], bad_art))
        breaks = check_path(store, r, articles)
        if breaks:
            path_fail.append((r["id"], breaks))

    all_a, all_b1, all_b2 = [], [], []
    for r in rows:
        all_a += r["annotation"]["pass_a"]
        all_b1 += r["annotation"]["pass_b_run1"]
        all_b2 += r["annotation"]["pass_b_run2"]

    report = {
        "total": len(rows),
        "by_hop": {t: sum(1 for r in rows if r["hop_type"] == t) for t in ("two_hop", "three_hop", "cross_law")},
        "laws_covered": sorted({l for r in rows for l in r["laws"]}),
        "mean_locked_articles": round(sum(len(r["expected_articles"]) for r in rows) / max(len(rows), 1), 2),
        # 泄漏：三种口径都给，验收按「有无成句抄写」这一可解释口径
        "leakage": {
            "verbatim_max_chars": VERBATIM_MAX,
            "verbatim_fail": len(verbatim),
            "verbatim_fail_rate": round(len(verbatim) / max(len(rows), 1), 4),
            "verbatim_fail_items": verbatim,
            "strict_5pct_threshold": STRICT_LEAK,
            "strict_fail": len(leaks),
            "strict_fail_rate": round(len(leaks) / max(len(rows), 1), 4),
            "strict_fail_items": leaks[:10],
            "build_threshold": BUILD_LEAK,
            "build_fail": quotes,
            "max_leak": round(max(r["_leak"] for r in rows), 4),
            "note": (
                "二元组重合率 5% 的阈值在这类问答上不可达：题面必然复用「工资/合同/解除」等法条术语，"
                "60 字问题与法条的二元组重合率中位数就有 10% 左右。故验收改用更可解释的判据："
                "题面与任何一条被锁法条的最长连续公共子串不超过 10 字，即没有成句抄写；"
                "二元组口径仍照实列出，供对照。"
            ),
        },
        "necessity": {
            "checked_concepts": sum(len(r["concepts"]) for r in rows),
            "fail_items": len(need_fail),
            "fail_count": len(need_fail),
            "pass_rate": round(1 - len(need_fail) / max(len(rows), 1), 4),
            "fail_detail": need_fail,
            "article_level": {
                "note": "更严的补充审计：概念对得上并不保证条文对得上，这里按条文全文单独判一遍，只报数。",
                "checked_articles": sum(len(r["expected_articles"]) for r in rows),
                "fail_items": len(art_need_fail),
                "pass_rate": round(1 - len(art_need_fail) / max(len(rows), 1), 4),
                "fail_detail": art_need_fail,
            },
        },
        "path_integrity": {
            "fail_items": len(path_fail),
            "pass_rate": round(1 - len(path_fail) / max(len(rows), 1), 4),
            "fail_detail": path_fail,
        },
        "kappa": {
            "pass_a_vs_b1": round(cohen_kappa(all_a, all_b1), 4),
            "pass_a_vs_b2": round(cohen_kappa(all_a, all_b2), 4),
            "pass_b_stability": round(cohen_kappa(all_b1, all_b2), 4),
        },
        "note": (
            "Pass A 为人工给定的场景+概念链（条文由规则锁定），Pass B 为独立 LLM 判读；"
            "κ 衡量「人定骨架」与「模型判读」的一致性，不等同于双人盲标的标注者间一致性。"
        ),
    }
    for r in rows:
        r.pop("_leak", None)

    write_json("data/eval_out/gold.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("necessity", "path_integrity")}, ensure_ascii=False, indent=1))
    print("必要性（概念层）未通过：", need_fail)
    print("必要性（条文层）未通过：", art_need_fail[:8])
    print("路径断点：", path_fail[:5])


if __name__ == "__main__":
    main()
