"""LangGraph 编排：A（单链路）/ B（多 Agent）/ C（多 Agent + 五道闸）。

为什么用 LangGraph 而不是手写 if/else：
状态在节点间显式流转，C 档的「校验 -> 修复 -> 复检」天然是一个有环图，
用条件边表达比嵌套函数清楚，也便于把 ``repaired`` / ``attempts`` 落进评测。

三个档位的 LLM 调用结构（成本表的口径）：
* A: retrieve(0) + answer(1)                        = 1 次
* B: retrieve(0) + plan(1) + answer(1)               = 2 次
* C: retrieve(0) + plan(1) + answer(1) + repair(0)*  = 2 次（闸门是规则，不打 LLM）
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.__007__agents.__000__state import QAState  # noqa: E402
from app.__007__agents.__001__gates import run_gates  # noqa: E402
from common.llm import get_llm, llm_probe  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402

log = get_logger("agents")

PLAN_SYSTEM = "你是劳动法检索规划器。只根据给定概念与候选法条，用两三句话写出本案的推理链路（先看哪条、再看哪条、为什么）。不要编造条号。"
ANSWER_SYSTEM = (
    "你是劳动法咨询助手。只能使用提供的法条作答，不得引用未提供的法条，不得编造条号。"
    "回答分两部分：1) 结论（2-3 句）；2) 依据（逐行以「- 」开头，写出条号、法名与该条原文的关键片段）。"
)

MAX_ATTEMPTS = 2


def llm_enabled() -> bool:
    return llm_probe()


def _fmt_items(items: list[dict]) -> str:
    return "\n".join(f"{i['citation']}（{i['law_name']}）：{i.get('content', '')[:220]}" for i in items)


def _note_cites_ok(note: str, items: list[dict]) -> bool:
    """规划节点是 LLM 写的，可能顺手带出不存在的条号；确定性答案不能把它原样搬进正文。"""
    from app.__007__agents.__001__gates import cited_patterns

    cats = [i["citation"] for i in items]
    return all(any(p in c for c in cats) for p in cited_patterns(note))


def _deterministic_answer(query: str, retrieval: dict[str, Any], path_note: str = "") -> str:
    items = retrieval.get("items", [])[:4]
    if not items:
        return "没有检索到直接相关的法条，无法给出有依据的结论，建议补充事实或转人工。"
    head = items[0]
    lines = [f"结论：根据{head['citation']}（{head['law_name']}），相关行为应按该条规定处理。"]
    if path_note and _note_cites_ok(path_note, items):
        lines.append(f"推理链路：{path_note}")
    lines.append("依据：")
    lines += [f"- {i['citation']}（{i['law_name']}）：{i.get('content', '')[:160]}" for i in items]
    return "\n".join(lines)


def make_flow(
    retriever: Any,
    article_index: dict[str, dict],
    tier: str,
    use_llm: bool | None = None,
    art_concepts: dict[str, set[str]] | None = None,
) -> Callable[[str, str], dict]:
    """返回 ``run(query, mode) -> state``。图只编译一次，供 60 题复用。"""
    from langgraph.graph import END, StateGraph

    ask_llm = llm_enabled() if use_llm is None else use_llm
    art_concepts = art_concepts or {}

    # ---------------- 节点 ----------------
    def retrieve(state: QAState) -> dict:
        out = retriever.retrieve(state["query"], mode=state.get("mode", "hybrid")).as_dict()
        return {"retrieval": out, "llm_calls": 0}

    def plan(state: QAState) -> dict:
        ret = state["retrieval"]
        linked = [c["name"] for c in ret.get("linked", {}).get("concepts", [])]
        items = ret.get("items", [])[:4]
        if ask_llm:
            text = get_llm().try_chat(
                [
                    {"role": "system", "content": PLAN_SYSTEM},
                    {
                        "role": "user",
                        "content": f"问题：{state['query']}\n命中概念：{'、'.join(linked) or '无'}\n候选法条：\n{_fmt_items(items)}",
                    },
                ],
                default="",
                temperature=0.0,
                max_tokens=220,
            )
            if text.strip():
                return {"path_note": text.strip(), "llm_calls": state.get("llm_calls", 0) + 1}
        note = f"先看 {items[0]['citation']}" if items else "无候选条文"
        if len(items) > 1:
            note += f"，再核 {items[1]['citation']}"
        return {"path_note": note, "llm_calls": state.get("llm_calls", 0)}

    def answer(state: QAState) -> dict:
        ret = state["retrieval"]
        items = ret.get("items", [])[:4]
        if ask_llm and items:
            text = get_llm().try_chat(
                [
                    {"role": "system", "content": ANSWER_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"用户问题：{state['query']}\n推理链路参考：{state.get('path_note', '')}\n\n"
                            f"可用法条：\n{_fmt_items(items)}\n\n请作答。"
                        ),
                    },
                ],
                default="",
                temperature=0.0,
                max_tokens=480,
            )
            if text.strip():
                return {
                    "answer": text.strip(),
                    "citations": [i["citation"] for i in items],
                    "llm_calls": state.get("llm_calls", 0) + 1,
                }
        return {
            "answer": _deterministic_answer(state["query"], ret, state.get("path_note", "")),
            "citations": [i["citation"] for i in items],
            "llm_calls": state.get("llm_calls", 0),
        }

    def check(state: QAState) -> dict:
        checks = run_gates(state.get("answer", ""), state["retrieval"], article_index, art_concepts)
        return {"checks": checks}

    def repair(state: QAState) -> dict:
        """闸门失败时的确定性重建：剔除没有图连接/证据的条文，并显式给出原文片段。"""
        ret = state["retrieval"]
        gates_ok = {c["gate"]: c["ok"] for c in state.get("checks", [])}
        items = ret.get("items", [])
        if not gates_ok.get("path_support", True):
            linked = {c["id"] for c in ret.get("linked", {}).get("concepts", [])}
            kept = [i for i in items if art_concepts.get(i["article_id"], set()) & linked]
            items = kept or items
        sub = {**ret, "items": items}
        answer = _deterministic_answer(state["query"], sub, state.get("path_note", ""))
        if not gates_ok.get("low_conf_handoff", True):
            answer = "该问题检索证据不足（top1 相似度与图路径覆盖均偏低），无法确定结论，建议补充材料或转人工复核。\n" + answer
        return {
            "answer": answer,
            "citations": [i["citation"] for i in items[:4]],
            "attempts": state.get("attempts", 0) + 1,
            "repaired": True,
        }

    def after_check(state: QAState) -> str:
        if all(c["ok"] for c in state.get("checks", [])):
            return "done"
        if state.get("attempts", 0) < MAX_ATTEMPTS:
            return "repair"
        return "done"

    # ---------------- 建图 ----------------
    g = StateGraph(QAState)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.set_entry_point("retrieve")
    if tier == "A":
        g.add_edge("retrieve", "answer")
        g.add_edge("answer", END)
    else:
        g.add_node("plan", plan)
        g.add_edge("retrieve", "plan")
        g.add_edge("plan", "answer")
        if tier == "B":
            g.add_edge("answer", END)
        else:
            g.add_node("check", check)
            g.add_node("repair", repair)
            g.add_edge("answer", "check")
            g.add_conditional_edges("check", after_check, {"repair": "repair", "done": END})
            g.add_edge("repair", "check")
    compiled = g.compile()

    def run(query: str, mode: str = "hybrid") -> dict:
        init: QAState = {"query": query, "mode": mode, "attempts": 0, "repaired": False, "citations": [], "checks": []}
        out = compiled.invoke(init)
        out.setdefault("answer", "")
        out["refused"] = "无法确定" in out["answer"] or "无法给出" in out["answer"]
        return dict(out)

    def stream_steps(init: dict) -> Any:
        """SSE 用：按节点产出 ``{node_name: 该节点的增量}``，不改变 run 的语义。"""
        return compiled.stream(init, stream_mode="updates")

    run.stream_steps = stream_steps  # type: ignore[attr-defined]
    return run
