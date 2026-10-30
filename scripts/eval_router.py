"""B3 路由评测：规则路由 vs 逻辑回归（留一交叉验证）。

标签来源（写清楚，避免「自造标签再自测」的循环论证）：
* ``simple``  —— 20 条单概念短问（模板生成，实体链接只应命中 1 个概念）；
* ``medium``  —— 金标 30 条 two_hop；
* ``complex`` —— 金标 20 条 three_hop + 10 条 cross_law。

规则路由是「先验可解释」的；LR 用同一批特征（概念数、场景数、长度、并列词、分句数）。
两边都报准确率，谁高谁低如实写进 README。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.__004__retrieval.__001__entity_linker import EntityLinker  # noqa: E402
from app.__004__retrieval.__004__router import QueryRouter, feature_vector, features_of, rule_route  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import read_jsonl, write_json  # noqa: E402

log = get_logger("eval_router")

SIMPLE_TEMPLATE = "{name}是怎么规定的？"


def build_dataset(linker: EntityLinker) -> list[dict]:
    gold = read_jsonl("data/gold/gold_60.jsonl")
    rows: list[dict] = []

    concepts = read_jsonl("data/lexicon/concepts.jsonl")
    # 均匀抽样 20 个概念（按 id 排序后等步长），避免只取到劳动合同法
    concepts = sorted(concepts, key=lambda c: c["id"])
    step = max(len(concepts) // 20, 1)
    for c in concepts[::step][:20]:
        rows.append({"query": SIMPLE_TEMPLATE.format(name=c["name"]), "label": "simple", "src": "synthetic"})

    for g in gold:
        label = "medium" if g["hop_type"] == "two_hop" else "complex"
        rows.append({"query": g["question"], "label": label, "src": g["id"]})

    for r in rows:
        linking = linker.link(r["query"])
        r["features"] = feature_vector(r["query"], len(linking.concepts), len(linking.scenarios))
        r["n_concepts"] = len(linking.concepts)
        r["rule"], r["rule_reason"] = rule_route(features_of(r["query"], len(linking.concepts), len(linking.scenarios)))
    return rows


def main() -> None:
    linker = EntityLinker()
    rows = build_dataset(linker)
    X = [r["features"] for r in rows]
    y = [r["label"] for r in rows]

    rule_acc = sum(1 for r in rows if r["rule"] == r["label"]) / len(rows)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    clf = LogisticRegression(max_iter=1000)
    preds = cross_val_predict(clf, X, y, cv=LeaveOneOut())
    lr_acc = sum(1 for p, t in zip(preds, y) if p == t) / len(y)

    # 混淆矩阵（规则路由），用来定位错在哪一档
    labels = ["simple", "medium", "complex"]
    confusion = {t: {p: 0 for p in labels} for t in labels}
    for r in rows:
        confusion[r["label"]][r["rule"]] += 1

    # 全量拟合一份 LR 落盘，供 Retriever 使用
    router = QueryRouter()
    router.fit(X, y)

    report = {
        "n": len(rows),
        "by_label": {k: sum(1 for r in rows if r["label"] == k) for k in labels},
        "rule_accuracy": round(rule_acc, 4),
        "lr_loo_accuracy": round(lr_acc, 4),
        "rule_confusion": confusion,
        "wrong_rule_examples": [
            {"query": r["query"], "label": r["label"], "rule": r["rule"], "n_concepts": r["n_concepts"]}
            for r in rows
            if r["rule"] != r["label"]
        ][:8],
        "wrong_lr_examples": [
            {"query": r["query"], "label": r["label"], "pred": str(p)}
            for r, p in zip(rows, preds)
            if p != r["label"]
        ][:8],
    }
    out = Path("data/eval_out/router.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    log.info(
        "规则 %.3f | LR 留一 %.3f | 样本 %s | 已写出 %s",
        rule_acc,
        lr_acc,
        report["by_label"],
        out,
    )


if __name__ == "__main__":
    main()
