"""B6 五道闸的单元测试：全部是确定性规则，不需要模型与语料。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.__007__agents.__001__gates import (  # noqa: E402
    cited_citations,
    cited_patterns,
    gate_cite_exists,
    gate_cite_in_retrieval,
    gate_evidence_span,
    gate_low_conf_handoff,
    run_gates,
)

INDEX = {
    "lcl#85": {"article_id": "lcl#85", "citation": "中华人民共和国劳动合同法第八十五条", "content": "用人单位有下列情形之一的，由劳动行政部门责令限期支付劳动报酬、加班费或者经济补偿。"},
    "ll#50": {"article_id": "ll#50", "citation": "中华人民共和国劳动法第五十条", "content": "工资应当以货币形式按月支付给劳动者本人。不得克扣或者无故拖欠劳动者的工资。"},
    "lcl#10": {"article_id": "lcl#10", "citation": "中华人民共和国劳动合同法第十条", "content": "建立劳动关系，应当订立书面劳动合同。"},
}
ITEMS = [
    {"article_id": "ll#50", "citation": "中华人民共和国劳动法第五十条", "law_name": "中华人民共和国劳动法", "content": INDEX["ll#50"]["content"]},
    {"article_id": "lcl#85", "citation": "中华人民共和国劳动合同法第八十五条", "law_name": "中华人民共和国劳动合同法", "content": INDEX["lcl#85"]["content"]},
]


def test_cite_extraction_ignores_leading_preposition():
    # 「根据中华人民共和国劳动法第五十条」不能把「根据」并进法名（曾导致误判编造条号）
    pats = cited_patterns("根据中华人民共和国劳动法第五十条，公司应当支付工资。")
    assert pats == ["劳动法第五十条"]


def test_cite_extraction_handles_book_titles():
    pats = cited_patterns("依据《中华人民共和国劳动合同法》第八十五条，可主张加付赔偿金。")
    assert pats == ["劳动合同法第八十五条"]


def test_cite_normalization_to_corpus_names():
    known = {a["citation"] for a in INDEX.values()}
    cites = cited_citations("《劳动合同法》第八十五条与《劳动法》第五十条", known)
    assert sorted(cites) == sorted(["中华人民共和国劳动法第五十条", "中华人民共和国劳动合同法第八十五条"])


def test_cite_exists_rejects_fabricated_article():
    res = gate_cite_exists("根据劳动合同法第三百条，应当赔偿。", ITEMS, INDEX)
    assert not res.ok and "第三百条" in res.detail


def test_cite_in_retrieval_rejects_out_of_scope():
    # 第十条真实存在，但不在此次检索结果里 -> 越界引用
    res = gate_cite_in_retrieval("根据劳动合同法第十条，应签书面合同。", ITEMS, INDEX)
    assert not res.ok


def test_evidence_span_requires_verbatim_fragment():
    bad = "结论：应付工资。\n依据：\n- 中华人民共和国劳动法第五十条（中华人民共和国劳动法）：略。"
    res = gate_evidence_span(bad, ITEMS, {a["citation"] for a in INDEX.values()})
    assert not res.ok


def test_evidence_span_accepts_verbatim_fragment():
    good = (
        "结论：应付工资。\n依据：\n"
        "- 中华人民共和国劳动法第五十条（中华人民共和国劳动法）：工资应当以货币形式按月支付给劳动者本人。"
    )
    res = gate_evidence_span(good, ITEMS, {a["citation"] for a in INDEX.values()})
    assert res.ok


def test_low_conf_requires_handoff_language():
    retrieval = {"low_confidence": True, "items": ITEMS}
    assert not gate_low_conf_handoff("根据第五十条，公司应当支付。", retrieval).ok
    assert gate_low_conf_handoff("证据不足，建议补充材料后由人工复核。", retrieval).ok


def test_run_gates_all_pass_on_clean_answer():
    retrieval = {"items": ITEMS, "low_confidence": False, "coverage": 0.8, "linked": {"concepts": []}}
    answer = (
        "结论：应支付工资。\n依据：\n"
        "- 中华人民共和国劳动法第五十条（中华人民共和国劳动法）：工资应当以货币形式按月支付给劳动者本人。"
    )
    checks = run_gates(answer, retrieval, INDEX, {})
    assert len(checks) == 5
    assert all(c["ok"] for c in checks)
