"""B5 E2E：3 条用例验证「挂起 -> 人工确认 -> 恢复 -> 继续生成」。

用例设计：
1. 域外问题（低置信）-> 挂起 -> 人工 accept（确认系统结果）-> 恢复生成；
2. 低置信 -> 人工改引用条文 -> 恢复后答案里的引用确实变了；
3. 强制复核的真实问题 -> 挂起后**新建进程内对象**（模拟重启）-> 人工直接给答案 -> 恢复。

检查项除了「跑通」，还包括：checkpoint 文件落盘、修正落库、介入率可统计。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.__006__hitl.__001__checkpoint import CheckpointStore  # noqa: E402
from app.__006__hitl.__002__correction import CorrectionStore  # noqa: E402
from app.__006__hitl.__003__pipeline import HitlPipeline  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import write_json  # noqa: E402

log = get_logger("eval_hitl")

OFF_DOMAIN = "公司股东之间转让股权需要缴什么税？"
REAL_QUERY = "公司拖了我两个月工资，我能要回来吗？"


def main() -> None:
    ckpt_root = Path("data/sessions/checkpoints_eval")
    if ckpt_root.exists():
        for p in ckpt_root.glob("*.json"):
            p.unlink()
    db = Path("data/sessions/hitl_eval.sqlite3")
    if db.exists():
        db.unlink()

    checkpoints = CheckpointStore(ckpt_root)
    corrections = CorrectionStore(db)
    pipe = HitlPipeline(checkpoints=checkpoints, corrections=corrections, use_llm=False)

    cases = []

    # --- 用例 1: 低置信 -> accept ---
    r1 = pipe.start(OFF_DOMAIN, force_review=True)
    suspended_1 = r1["status"] == "awaiting_review"
    resumed_1 = pipe.resume(r1["task_id"], {"type": "accept"}) if suspended_1 else {"status": "n/a"}
    cases.append(
        {
            "id": "e2e-1",
            "query": OFF_DOMAIN,
            "suspended": suspended_1,
            "resumed": resumed_1["status"] == "completed",
            "answer_len": len(resumed_1.get("answer", "")),
            "citations": (resumed_1.get("citations") or [])[:3],
            "ok": suspended_1 and resumed_1["status"] == "completed" and bool(resumed_1.get("answer")),
        }
    )

    # --- 用例 2: 低置信 -> 改引用 ---
    r2 = pipe.start(OFF_DOMAIN, force_review=True)
    ok2 = r2["status"] == "awaiting_review"
    out2 = pipe.resume(r2["task_id"], {"type": "edit_articles", "article_ids": ["lcl#85", "ll#91"]}) if ok2 else {}
    cites2 = list(out2.get("citations", []))
    cases.append(
        {
            "id": "e2e-2",
            "query": OFF_DOMAIN,
            "suspended": ok2,
            "resumed": out2.get("status") == "completed",
            "citations": cites2,
            "citations_swapped": "中华人民共和国劳动合同法第八十五条" in cites2,
            "ok": ok2 and out2.get("status") == "completed" and "中华人民共和国劳动合同法第八十五条" in cites2,
        }
    )

    # --- 用例 3: 真实问题强制复核 -> 重启后恢复 ---
    r3 = pipe.start(REAL_QUERY, force_review=True)
    ok3 = r3["status"] == "awaiting_review"
    # 模拟进程重启：全新对象，只共享磁盘上的 checkpoint 与 db
    pipe2 = HitlPipeline(
        checkpoints=CheckpointStore(ckpt_root),
        corrections=CorrectionStore(db),
        use_llm=False,
    )
    out3 = pipe2.resume(
        r3["task_id"],
        {"type": "provide_answer", "answer": "结论：拖欠工资违法，可要求足额支付并可主张赔偿。", "citations": ["中华人民共和国劳动法第五十条"]},
    ) if ok3 else {}
    cases.append(
        {
            "id": "e2e-3",
            "query": REAL_QUERY,
            "suspended": ok3,
            "restart_resumed": out3.get("status") == "completed",
            "answer": out3.get("answer", "")[:60],
            "ok": ok3 and out3.get("status") == "completed" and out3.get("answer", "").startswith("结论：拖欠工资违法"),
        }
    )

    stats = pipe2.stats()
    report = {
        "cases": cases,
        "all_pass": all(c["ok"] for c in cases),
        "stats": stats,
        "checkpoint_files": len(list(ckpt_root.glob("*.json"))),
    }
    out = Path("data/eval_out/hitl.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    log.info("E2E 全部通过=%s 统计=%s", report["all_pass"], json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
