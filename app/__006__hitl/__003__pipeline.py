"""人在环流水线：低置信挂起 -> 人工修正 -> 恢复生成。

判定「低置信」用的是两个可解释信号（与配置项一致）：
* ``top1_similarity`` —— 稠密检索最高分，衡量「问题与法律语料是否对口」；
* ``coverage``         —— 图路路径覆盖度，衡量「概念链是否真的在图上拼起来」。
两者**同时**低于阈值才挂起：单看任何一个都会把「图没覆盖但文本很准」的正常问题误挂。

修正类型（会落库统计分布）：
* ``accept``          —— 人工确认系统结果，不改内容；
* ``edit_articles``   —— 换掉引用条文后重新生成答案；
* ``edit_answer``     —— 直接改答案文本；
* ``provide_answer``  —— 人工直接给最终答案（系统检索完全不可用时）。
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.__004__retrieval.__005__retriever import get_retriever  # noqa: E402
from app.__006__hitl.__000__generator import generate_answer  # noqa: E402
from app.__006__hitl.__001__checkpoint import Checkpoint, CheckpointStore  # noqa: E402
from app.__006__hitl.__002__correction import CorrectionStore  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402

log = get_logger("hitl")

VALID_ACTIONS = ("accept", "edit_articles", "edit_answer", "provide_answer")


class HitlPipeline:
    def __init__(
        self,
        retriever: Any | None = None,
        checkpoints: CheckpointStore | None = None,
        corrections: CorrectionStore | None = None,
        use_llm: bool | None = None,
    ) -> None:
        self._retriever = retriever
        self.checkpoints = checkpoints or CheckpointStore()
        self.corrections = corrections or CorrectionStore()
        self.use_llm = use_llm

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever

    # ---------------- 入口 ----------------
    def start(self, query: str, mode: str = "hybrid", force_review: bool = False) -> dict[str, Any]:
        task_id = uuid.uuid4().hex[:10]
        retrieval = self.retriever.retrieve(query, mode=mode).as_dict()
        low = retrieval["low_confidence"]

        if low or force_review:
            reason = (
                f"top1={retrieval['top1_similarity']:.3f} < {self.retriever.settings.refuse_threshold}"
                f" 且 coverage={retrieval['coverage']:.3f} < {self.retriever.settings.path_coverage_threshold}"
                if low
                else "人工强制复核"
            )
            ckpt = Checkpoint(
                task_id=task_id,
                query=query,
                mode=mode,
                status="awaiting_review",
                reason=reason,
                top1_similarity=retrieval["top1_similarity"],
                coverage=retrieval["coverage"],
                retrieval=retrieval,
            )
            self.checkpoints.save(ckpt)
            self.corrections.log_task(task_id, "awaiting_review")
            return {"task_id": task_id, "status": "awaiting_review", "reason": reason, "retrieval": retrieval}

        gen = generate_answer(query, retrieval, use_llm=self.use_llm)
        self.corrections.log_task(task_id, "completed")
        return {
            "task_id": task_id,
            "status": "completed",
            "answer": gen["answer"],
            "citations": gen["citations"],
            "used_llm": gen["used_llm"],
            "retrieval": retrieval,
        }

    def resume(self, task_id: str, action: dict[str, Any]) -> dict[str, Any]:
        ckpt = self.checkpoints.load(task_id)
        if ckpt is None:
            raise KeyError(f"找不到挂起任务 {task_id}")
        if ckpt.status == "completed" and ckpt.answer:
            return {
                "task_id": task_id,
                "status": "completed",
                "answer": ckpt.answer,
                "resolution": ckpt.resolution,
                "already_done": True,
            }

        ctype = action.get("type", "accept")
        if ctype not in VALID_ACTIONS:
            raise ValueError(f"未知修正类型: {ctype}")

        answer, citations = ckpt.answer, ckpt.retrieval.get("items", [])
        if ctype == "accept":
            gen = generate_answer(ckpt.query, ckpt.retrieval, use_llm=self.use_llm)
            answer, citations = gen["answer"], gen["citations"]
        elif ctype == "edit_articles":
            answer, citations = self._rebuild_with_articles(ckpt, action.get("article_ids") or [])
        elif ctype in ("edit_answer", "provide_answer"):
            answer = action.get("answer") or ""
            if not answer:
                raise ValueError("edit_answer/provide_answer 需要提供 answer")
            citations = action.get("citations") or citations

        ckpt.status = "completed"
        ckpt.answer = answer
        ckpt.resolution = {"type": ctype, "payload": action, "at": time.time()}
        self.checkpoints.save(ckpt)
        self.corrections.log_correction(task_id, ctype, action)
        self.corrections.log_task(task_id, "completed")
        return {
            "task_id": task_id,
            "status": "completed",
            "answer": answer,
            "citations": citations,
            "resolution": ckpt.resolution,
            "already_done": False,
        }

    def _rebuild_with_articles(self, ckpt: Checkpoint, article_ids: list[str]) -> tuple[str, list[dict]]:
        text = self.retriever.text
        items = []
        for aid in article_ids:
            art = text.articles.get(aid)
            if not art:
                continue
            items.append(
                {
                    "article_id": aid,
                    "citation": art["citation"],
                    "law_name": art["law_name"],
                    "content": art["content"],
                    "score": 1.0,
                    "source": "human",
                }
            )
        retrieval = {**ckpt.retrieval, "items": items}
        gen = generate_answer(ckpt.query, retrieval, use_llm=self.use_llm)
        return gen["answer"], gen["citations"]

    # ---------------- 统计 ----------------
    def pending(self) -> list[dict[str, Any]]:
        return [c.to_json() for c in self.checkpoints.list("awaiting_review")]

    def stats(self) -> dict[str, Any]:
        all_ckpts = self.checkpoints.list()
        stats = self.corrections.stats(len(all_ckpts))
        stats["pending"] = sum(1 for c in all_ckpts if c.status == "awaiting_review")
        stats["completed"] = sum(1 for c in all_ckpts if c.status == "completed")
        return stats
