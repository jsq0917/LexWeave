"""多轮对话管理：槽位抽取 -> 继承/改写 -> 缺失追问 -> 检索。

三个关键行为（B4 的验收点）：
1. **继承**：本轮没提到的槽位保留历史值，不会被空值冲掉；
2. **指代消解**：「那如果没签合同呢」这类接续问法，视为沿用上一轮事实，只改被明确改写的槽位；
3. **追问**：必需槽位缺失时返回 ``input_required``，不硬答。

状态每轮落 SQLite，进程重启后按 ``session_id`` 恢复 —— 这是「中断重连槽位不丢」的实现方式。
"""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.__005__dialog.__000__slots import (  # noqa: E402
    SLOT_QUESTIONS,
    SlotState,
    detect_intent,
    extract_slots,
    missing_slots,
)
from app.__005__dialog.__001__session_store import SessionStore  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402

log = get_logger("dialog")

COREF_PREFIXES = ("那", "那么", "如果", "要是", "假如", "同样", "换成", "这种")
COREF_MARKERS = ("这种情况", "像我这样", "同样的情况", "那要是", "那如果")
# 只有「真的在问」的轮次才允许切换意图；纯补槽句（「2024年5月离职了」）只是给事实，
# 里面的「离职」是契约事实词，不该把整段会话从 wage 意图改判成 contract。
ASK_MARKERS = ("吗", "呢", "?", "？", "怎么", "如何", "能不能", "能否", "是否", "多少", "多久", "哪", "算不算", "可不可以")


def is_followup(text: str) -> bool:
    t = text.strip()
    if any(t.startswith(p) for p in COREF_PREFIXES):
        return True
    return any(m in t for m in COREF_MARKERS)


@dataclass
class TurnResult:
    session_id: str
    turn_no: int
    status: str  # answered | input_required
    intent: str
    slots: dict[str, Any]
    missing: list[str]
    question: str | None
    coreference: bool
    inherited: list[str] = field(default_factory=list)
    retrieval: dict[str, Any] | None = None
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_no": self.turn_no,
            "status": self.status,
            "intent": self.intent,
            "slots": self.slots,
            "missing": self.missing,
            "question": self.question,
            "coreference": self.coreference,
            "inherited": self.inherited,
            "retrieval": self.retrieval,
            "latency_ms": round(self.latency_ms, 1),
        }


class DialogManager:
    def __init__(self, retriever: Any | None = None, store: SessionStore | None = None) -> None:
        self.store = store or SessionStore()
        self._retriever = retriever

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            from app.__004__retrieval.__005__retriever import get_retriever

            self._retriever = get_retriever()
        return self._retriever

    def new_session(self) -> str:
        return uuid.uuid4().hex[:12]

    def handle(self, session_id: str, user_text: str, mode: str = "hybrid") -> TurnResult:
        t0 = time.perf_counter()
        blob = self.store.load_state(session_id)
        state = SlotState.from_json(blob)
        prev_values = dict(state.values)

        followup = is_followup(user_text) and bool(prev_values)
        intent = detect_intent(user_text)
        # 纯陈述句（没有任何疑问标记）一律沿用在跑的意图：这类轮次是在回答上一轮的追问，
        # 里面的「离职 / 合同」只是事实词。否则 missing_slots 会按 other 或 contract 判错，
        # 要么提前判 answered，要么把已齐的 wage 槽位重新追问一遍。
        asking = any(m in user_text for m in ASK_MARKERS)
        if state.intent != "other" and not asking:
            intent = state.intent
        elif intent == "other" and state.intent != "other":
            intent = state.intent

        extracted = extract_slots(user_text)
        # 指代消解：接续问法下，没被改写的槽位一律沿用上一轮
        inherited = [k for k in prev_values if k not in extracted] if followup else []
        state.intent = intent
        state.merge(extracted)

        missing = missing_slots(state.values, intent)
        self.store.save_state(session_id, state.to_json())
        self.store.append_turn(
            session_id,
            "user",
            user_text,
            {"intent": intent, "extracted": extracted, "inherited": inherited},
        )

        turn_no = sum(1 for t in self.store.turns(session_id) if t["role"] == "user")
        if missing:
            question = SLOT_QUESTIONS[missing[0]]
            self.store.append_turn(session_id, "assistant", question, {"status": "input_required", "missing": missing})
            return TurnResult(
                session_id=session_id,
                turn_no=turn_no,
                status="input_required",
                intent=intent,
                slots=dict(state.values),
                missing=missing,
                question=question,
                coreference=followup,
                inherited=inherited,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        retrieval = self.retriever.retrieve(self.compose_query(user_text, state), mode=mode).as_dict()
        answer_stub = self.compose_answer(retrieval, state)
        self.store.append_turn(session_id, "assistant", answer_stub, {"status": "answered", "retrieval": retrieval})
        return TurnResult(
            session_id=session_id,
            turn_no=turn_no,
            status="answered",
            intent=intent,
            slots=dict(state.values),
            missing=[],
            question=None,
            coreference=followup,
            inherited=inherited,
            retrieval=retrieval,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    @staticmethod
    def compose_query(user_text: str, state: SlotState) -> str:
        """把槽位事实拼进检索查询：数值本身不是检索词，但「拖欠/未签」等事实能锚定法条。"""
        parts = [user_text]
        facts = state.describe()
        if facts:
            parts.append(f"（已知事实：{facts}）")
        return " ".join(parts)

    @staticmethod
    def compose_answer(retrieval: dict[str, Any], state: SlotState) -> str:
        items = retrieval.get("items", [])[:3]
        if not items:
            return "抱歉，没有检索到直接相关的法条，建议补充事实或咨询专业人士。"
        cites = "；".join(f"{i['citation']}（{i['law_name']}）" for i in items)
        return f"根据检索到的法条：{cites}。\n关键事实：{state.describe() or '暂无'}。"

    def history(self, session_id: str) -> list[dict[str, Any]]:
        return self.store.turns(session_id)
