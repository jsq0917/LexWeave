"""B7 边类型拆分：把原来的 ``mentions``（Article→Concept 一条边）拆成
``DEFINES``（该条在定义/正面规制这个概念）与 ``REFERS_TO``（只是顺带提到）。

为什么必须拆：原 ``mentions`` 把「提及」当「定义」，例如「合同履行」挂在民法典第 926 条
（委托合同第三人关系）、「诉讼时效中断」挂在第 199 条（撤销权期间）——概念名在正文出现，
但该条不是在定义它。B1 定义级抽检 75% 未达 85% 的根因就在这里。

判定规则（全部可复现，不调 LLM）：
1. ``definition_pattern``：概念名紧跟 ``是指 / 系指 / 即指 / 指的是 / 所称 / 包括 / 分为 / 统称``，
   或概念名前面 12 字内出现 ``本法所称 / 本条所称 / 本条例所称 / 以下称``；
2. ``copula_definition``：概念名出现在句首 6 字内，且紧跟裸系词 ``是``（排除「是否」）——
   中文法律的定义句常写成「买卖合同是……的合同」，不带「是指」；
3. ``normative_subject``：概念名出现在条文的首个分句里（句首 20 字内），且该分句含
   ``应当 / 不得 / 可以 / 有权 / 必须``——这类条文是「对概念的正面规制」，按 B7 口径也算 DEFINES；
   另加「分句主语位」：概念名位于任意分句开头、其后紧跟模态词，且后面不是「的 / 与 / 和 / 及 / 或 / 、」
   （避免把「劳动合同的解除应当……」误判成对「劳动合同」的规制）；
4. 其余一律 ``REFERS_TO``。

边界：一条 mention 可能在一段里出现多次，任一次命中规则 1/2 即判 DEFINES（取最强的那次），
并把命中的别名、字符偏移、±60 字窗口与 cue 落盘，供抽检复核。

用法：
    .venv\\Scripts\\python.exe app/__002__extract/__003__edge_typing.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.path_utils import read_jsonl, write_json, write_jsonl  # noqa: E402

WINDOW = 60
DEFINES = "DEFINES"
REFERS_TO = "REFERS_TO"

_SENT_END = re.compile(r"[。；\n]")
_MODAL = ("应当", "不得", "可以", "有权", "必须")
_AFTER_CUE = re.compile(r"\s*[（(，,：:]?\s*(是指|系指|即指|指的是|所称|包括|分为|统称)")
_BEFORE_CUE = re.compile(r"(本法|本条|本条例|本法例|以下|下)\s*所称\s*$")
_COPULA = re.compile(r"^[，,]?\s*是(?!否)")
_CLAUSE_END = re.compile(r"[，、；。：\n]")
_BAD_AFTER_ALIAS = ("的", "与", "和", "及", "或", "、")


def _sentence_start(text: str, pos: int) -> int:
    start = 0
    for m in _SENT_END.finditer(text, 0, pos):
        start = m.end()
    return start


def _first_clause(text: str, sent_start: int) -> str:
    """条文首句的第一个分句（到第一个逗号/顿号/分号/句号为止）。"""
    m = re.search(r"[，、；。\n]", text[sent_start:])
    end = sent_start + m.start() if m else len(text)
    return text[sent_start:end]


def _clauses(text: str, sent_start: int) -> list[tuple[int, str]]:
    """把首句切成 (起始偏移, 分句文本) 列表。"""
    sent_end = len(text)
    m = _SENT_END.search(text, sent_start)
    if m:
        sent_end = m.start()
    spans, start = [], sent_start
    for m in _CLAUSE_END.finditer(text, sent_start, sent_end):
        spans.append((start, text[start : m.start()]))
        start = m.end()
    spans.append((start, text[start:sent_end]))
    return spans


def classify_occurrence(text: str, start: int, end: int) -> tuple[str, str] | None:
    """判定单次出现；命中返回 (edge_type, cue)，否则 None。"""
    before = text[max(0, start - 12) : start]
    after = text[end : end + 12]
    if _BEFORE_CUE.search(before):
        return DEFINES, "所称号"
    m = _AFTER_CUE.match(after)
    if m:
        return DEFINES, m.group(1)
    sent_start = _sentence_start(text, start)
    if start - sent_start <= 6 and _COPULA.match(after):
        return DEFINES, "系词定义"
    clause = _first_clause(text, sent_start)
    in_first_clause = start - sent_start <= 20 and start < sent_start + len(clause)
    if in_first_clause:
        for modal in _MODAL:
            if modal in clause and text.find(modal, sent_start) >= end:
                return DEFINES, f"normative:{modal}"
    for cstart, ctext in _clauses(text, sent_start):
        if start != cstart:
            continue
        if text[end : end + 1] in _BAD_AFTER_ALIAS:
            continue
        tail = ctext[len(text[start:end]) :]
        for modal in _MODAL:
            if modal in tail:
                return DEFINES, f"normative_subject:{modal}"
    return None


def classify_mention(article: dict, mention: dict) -> dict:
    text = article["content"]
    aliases = [a for a in (mention.get("aliases") or []) if a]
    best: tuple[str, str, str, int] | None = None  # (type, cue, alias, offset)
    fallback: tuple[str, str, int] | None = None
    for alias in aliases:
        for m in re.finditer(re.escape(alias), text):
            hit = classify_occurrence(text, m.start(), m.end())
            if hit:
                if best is None or hit[0] == DEFINES:
                    best = (hit[0], hit[1], alias, m.start())
            elif fallback is None:
                fallback = (alias, m.start(), m.group(0))
    if best:
        edge_type, cue, alias, offset = best
    elif fallback:
        alias, offset, _ = fallback
        edge_type, cue = REFERS_TO, "no_cue"
    else:  # 别名在正文里找不到（理论上不会发生），保守判 REFERS_TO
        alias, offset = (aliases[0] if aliases else ""), 0
        edge_type, cue = REFERS_TO, "alias_not_found"
    end = offset + len(alias)
    return {
        **mention,
        "edge_type": edge_type,
        "cue": cue,
        "matched_alias": alias,
        "offset": offset,
        "defines_score": 1.0 if edge_type == DEFINES else 0.45,
        "window": text[max(0, offset - WINDOW) : min(len(text), end + WINDOW)],
    }


def type_mentions(articles: list[dict], mentions: list[dict]) -> list[dict]:
    by_id = {a["article_id"]: a for a in articles}
    rows = []
    for m in mentions:
        art = by_id.get(m["article_id"])
        if art is None:
            continue
        rows.append(classify_mention(art, m))
    return rows


MANUAL_MENTIONS = "manual_mentions.jsonl"


def load_manual_mentions(lexicon_dir: Path) -> list[dict]:
    """读取人工补的 Article→Concept 边（``provenance=manual``）。

    为什么需要人工补边：别名词典是字面匹配，若正确条文里没有概念名/别名（例如「休息日」
    在劳动法第三十八条写成了「每周至少休息一日」），这条本来正确的映射在图上就不存在。
    这类缺口按法律含义逐条补边、写清理由，不靠放宽匹配规则去凑数。
    """
    path = lexicon_dir / MANUAL_MENTIONS
    if not path.exists():
        return []
    rows = []
    for r in read_jsonl(path):
        edge_type = r.get("edge_type", DEFINES)
        rows.append(
            {
                "article_id": r["article_id"],
                "concept_id": r["concept_id"],
                "aliases": r.get("aliases", []),
                "count": r.get("count", 1),
                "confidence": float(r.get("confidence", 0.95)),
                "provenance": "manual",
                "edge_type": edge_type,
                "cue": "manual",
                "matched_alias": (r.get("aliases") or [""])[0],
                "offset": -1,
                "defines_score": 1.0 if edge_type == DEFINES else 0.45,
                "window": "",
                "note": r.get("note", ""),
            }
        )
    return rows


def merge_manual(rows: list[dict], manual: list[dict]) -> list[dict]:
    """人工边优先：同一 (article, concept) 若规则已抽出，则用人工边的类型与说明覆盖属性。"""
    index = {(r["article_id"], r["concept_id"]): i for i, r in enumerate(rows)}
    for m in manual:
        key = (m["article_id"], m["concept_id"])
        if key in index:
            rows[index[key]] = {**rows[index[key]], **m}
        else:
            index[key] = len(rows)
            rows.append(m)
    return rows


def load_typed(extracted_dir: Path) -> list[dict]:
    """优先读 mentions_typed.jsonl；缺失时回落旧 mentions.jsonl，并全部标 DEFINES。

    回落是为了让「只更新代码、不重跑抽取」的场景仍能建图，不会静默丢边。
    无论走哪条路径，都会合并 ``data/lexicon/manual_mentions.jsonl`` 里的人工补边。
    """
    typed = extracted_dir / "mentions_typed.jsonl"
    if typed.exists():
        rows = read_jsonl(typed)
    else:
        rows = read_jsonl(extracted_dir / "mentions.jsonl")
        for r in rows:
            r.setdefault("edge_type", DEFINES)
            r.setdefault("cue", "legacy")
            r.setdefault("defines_score", 1.0)
    return merge_manual(rows, load_manual_mentions(extracted_dir.parent / "lexicon"))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    extracted = root / "data" / "extracted"
    articles = read_jsonl(extracted / "articles.jsonl")
    mentions = read_jsonl(extracted / "mentions.jsonl")
    rows = type_mentions(articles, mentions)
    rows = merge_manual(rows, load_manual_mentions(root / "data" / "lexicon"))
    write_jsonl(extracted / "mentions_typed.jsonl", rows)

    types = Counter(r["edge_type"] for r in rows)
    cues = Counter(r["cue"] for r in rows)
    stats = {
        "total": len(rows),
        "rule_total": len(mentions),
        "manual_total": sum(1 for r in rows if r.get("provenance") == "manual"),
        "edge_type": dict(types),
        "defines_ratio": round(types[DEFINES] / max(len(rows), 1), 4),
        "cue_distribution": dict(cues.most_common()),
        "window_chars": WINDOW,
        "rule": (
            "定义式（是指/系指/即指/指的是/所称/包括/分为/统称，或前置『本法所称』）、"
            "句首裸系词『X 是……』、或分句主语位 + 应当/不得/可以/有权/必须 → DEFINES；其余 → REFERS_TO"
        ),
    }
    write_json(root / "data" / "extracted" / "mentions_typed_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
