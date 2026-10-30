"""B5 挂起点与修正记录的单元测试：只测落盘与统计，不跑模型。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.__006__hitl.__001__checkpoint import Checkpoint, CheckpointStore  # noqa: E402
from app.__006__hitl.__002__correction import CorrectionStore  # noqa: E402


def test_checkpoint_roundtrip(tmp_path):
    store = CheckpointStore(tmp_path)
    ckpt = Checkpoint(
        task_id="t1",
        query="公司拖欠工资",
        mode="hybrid",
        status="awaiting_review",
        reason="low confidence",
        top1_similarity=0.31,
        coverage=0.2,
        retrieval={"items": [{"citation": "中华人民共和国劳动合同法第八十五条"}]},
    )
    store.save(ckpt)
    loaded = store.load("t1")
    assert loaded is not None
    assert loaded.query == "公司拖欠工资"
    assert loaded.status == "awaiting_review"


def test_checkpoint_list_by_status(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save(Checkpoint("a", "q1", "hybrid", "awaiting_review", "r", 0.3, 0.2))
    store.save(Checkpoint("b", "q2", "hybrid", "completed", "r", 0.9, 0.8))
    assert [c.task_id for c in store.list("awaiting_review")] == ["a"]
    assert len(store.list()) == 2


def test_correction_stats(tmp_path):
    db = tmp_path / "hitl.sqlite3"
    store = CorrectionStore(db)
    store.log_task("t1", "awaiting_review")
    store.log_task("t2", "awaiting_review")
    store.log_task("t2", "completed")
    store.log_correction("t2", "edit_articles", {"article_ids": ["lcl#85"]})
    stats = store.stats(2)
    assert stats["total_tasks"] == 2
    assert stats["reviewed_tasks"] == 2
    assert stats["correction_types"]["edit_articles"] == 1
