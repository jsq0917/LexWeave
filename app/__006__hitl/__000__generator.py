"""答案生成：有 LLM 用 LLM（严格约束引用条文），没有就退回确定性模板。

两条路径都**只允许使用检索到的条文**，不允许模型自由发挥：
LLM 的 prompt 明确要求「只能引用给定条文，缺依据就说不知道」，
并且在结果里返回 ``used_llm``，方便 B6 的成本表统计调用次数。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm import get_llm, llm_probe  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402

log = get_logger("answer")

SYSTEM = (
    "你是劳动法咨询助手。只能使用提供的法条作答，不得引用未提供的法条，不得编造条号。"
    "回答分两部分：1) 结论（2-3 句）；2) 依据（逐条列出引用条文的条号与关键句）。"
)

_LLM_OK: bool | None = None


def llm_available() -> bool:
    global _LLM_OK
    if _LLM_OK is None:
        from config import settings

        _LLM_OK = bool(settings.llm_enabled) and llm_probe()
    return _LLM_OK


def generate_answer(query: str, retrieval: dict[str, Any], use_llm: bool | None = None, max_items: int = 4) -> dict[str, Any]:
    items = retrieval.get("items", [])[:max_items]
    if not items:
        return {"answer": "没有检索到直接相关的法条，无法给出有依据的结论。", "used_llm": False, "citations": []}

    citations = [i["citation"] for i in items]
    context = "\n".join(f"{i['citation']}（{i['law_name']}）：{i['content']}" for i in items)
    if use_llm is None:
        use_llm = llm_available()

    if use_llm:
        text = get_llm().try_chat(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"用户问题：{query}\n\n可用法条：\n{context}\n\n请作答。"},
            ],
            default="",
            temperature=0.0,
            max_tokens=600,
        ).strip()
        if text:
            return {"answer": text, "used_llm": True, "citations": citations}

    head = items[0]
    answer = (
        f"结论：根据{head['citation']}（{head['law_name']}），相关行为需要依照该条规定处理。\n"
        "依据：\n" + "\n".join(f"- {i['citation']}（{i['law_name']}）：{i['content'][:160]}" for i in items)
    )
    return {"answer": answer, "used_llm": False, "citations": citations}
