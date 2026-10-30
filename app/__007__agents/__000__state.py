"""B6 消融用的图问答编排层（LangGraph）。

三档只改「编排与校验」，检索器、语料、模型全部相同：

* ``A`` 单链路：retrieve -> answer（1 个角色，1 次 LLM）；
* ``B`` 多 Agent：retrieve -> plan（路径解释角色）-> answer（作答角色），无校验；
* ``C`` 多 Agent + 五道闸：B 之后再过 ``check``，不通过就 ``repair`` 重建答案并复检。

把「角色分工」与「校验闸」拆成两档，才能在 B6 里单独回答
「图场景下多 Agent 的增益是否更大」以及「增益到底来自分工还是来自校验」。
"""

from __future__ import annotations

from typing import Any, TypedDict


class QAState(TypedDict, total=False):
    query: str
    mode: str  # text | graph | hybrid
    use_llm: bool
    retrieval: dict[str, Any]
    path_note: str
    answer: str
    citations: list[str]
    checks: list[dict[str, Any]]
    attempts: int
    repaired: bool
    refused: bool
    llm_calls: int
    errors: list[str]
