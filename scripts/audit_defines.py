"""B1 / B7 抽检：两种口径分开跑、分开落盘。

**为什么要拆成两种**

旧脚本只有一个口径，但它其实是「把建边用的正则再跑一遍」：``judge()`` 与
``__003__edge_typing.py::classify_occurrence()`` 共用同一套 ``_AFTER_CUE`` / ``_COPULA`` /
``_MODAL``、同一个 12 字窗口、同一组 ``_BAD_AFTER_ALIAS``。于是 100/100 是 round-trip 的必然结果，
把它写成「定义级精度 100%」等于自己批自己——比原来那个 75% 更危险。

所以现在：

* ``--mode selfcheck``（默认，不调 LLM）：规则自洽性校验。判据与建边同源，**只证明代码与规则一致**，
  不证明规则对。产物 ``data/eval_out/defines_selfcheck.json``，字段名就叫 ``selfcheck_pass_rate``。
* ``--mode blind``：**盲判**。只把「概念名 + 条文全文」给模型，**不给匹配到的别名、不给偏移、不给 cue**，
  让它在不知道「规则命中了哪个词」的前提下判「这条在定义/正面规制这个概念，还是只是顺带提及」。
  两次投票（第二次把选项顺序反过来，抵消位置偏好），不一致时投第三次。
  产物 ``data/eval_out/defines_blind_review.json``，这一份才是可以对外说的「定义级精度」。

它不是人工盲标（``reviewer`` 字段写明是 LLM 三投），但至少与建边规则**不同源**，
这是它相对旧口径唯一、也是关键的价值。

用法::

    .venv\\Scripts\\python.exe scripts/audit_defines.py                # 自洽性（秒级，不花钱）
    .venv\\Scripts\\python.exe scripts/audit_defines.py --mode blind    # 盲判（100 条，约 5 分钟）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import read_json, read_jsonl, write_json  # noqa: E402

log = get_logger("audit_defines")

WINDOW = 40
_AFTER_CUE = re.compile(r"\s*[（(，,：:]?\s*(是指|系指|即指|指的是|所称|包括|分为|统称)")
_BEFORE_CUE = re.compile(r"(本法|本条|本条例|以下|下)\s*所称\s*$")
_COPULA = re.compile(r"^[，,]?\s*是(?!否)")
_MODAL = ("应当", "不得", "可以", "有权", "必须")
_SENT_END = re.compile(r"[。；\n]")
_CLAUSE_END = re.compile(r"[，、；。：\n]")
_BAD_AFTER_ALIAS = ("的", "与", "和", "及", "或", "、")

SELFCHECK_OUT = "data/eval_out/defines_selfcheck.json"
BLIND_OUT = "data/eval_out/defines_blind_review.json"
BLIND_CACHE = "data/eval_out/.defines_blind_cache.json"
# 提示词改过一次（v1 的「正面规制」定义过窄，27% 通过率测的是另一个标准），
# 版本号进缓存键，避免旧票被复用。
PROMPT_VERSION = "v2"

# 盲判提示词：刻意不含别名/偏移/cue，也不提「应当/是指」等线索词，
# 否则等于把答案提示给模型。
BLIND_SYSTEM = (
    "你是法律文本标注员。只依据条文原文判断概念与条文的关系，不看任何外部信息。"
    "必须输出 JSON，不要解释。"
)


# ---------------------------------------------------------------- 规则口径（自洽性）
def _sentence_start(text: str, pos: int) -> int:
    start = 0
    for m in _SENT_END.finditer(text, 0, pos):
        start = m.end()
    return start


def _first_clause(text: str, sent_start: int) -> str:
    m = re.search(r"[，、；。\n]", text[sent_start:])
    return text[sent_start : sent_start + m.start() if m else len(text)]


def _locate(article_text: str, alias: str, offset: int):
    if offset >= 0 and article_text[offset : offset + len(alias)] == alias:
        return offset
    m = re.search(re.escape(alias), article_text)
    return m.start() if m else -1


def _clauses(text: str, sent_start: int) -> list[tuple[int, str]]:
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


def judge(article_text: str, alias: str, offset: int = -1) -> tuple[bool, str, str, bool, str]:
    """返回 (b7_verdict, criterion, evidence, strict_verdict, strict_reason)。

    **与建边规则同源**，只用于自洽性校验；对外口径请看 ``blind_review()``。
    """
    if not alias:
        return False, "alias_missing", "", False, "alias_missing"
    start = _locate(article_text, alias, offset)
    if start < 0:
        return False, "alias_not_in_text", "", False, "alias_not_in_text"
    end = start + len(alias)
    before = article_text[max(0, start - 12) : start]
    after = article_text[end : end + 12]
    evidence = article_text[max(0, start - WINDOW) : min(len(article_text), end + WINDOW)]
    if _BEFORE_CUE.search(before):
        return True, "definition_cue", evidence, True, "definition_cue"
    m = _AFTER_CUE.match(after)
    if m:
        return True, f"definition_cue:{m.group(1)}", evidence, True, f"definition_cue:{m.group(1)}"
    sent_start = _sentence_start(article_text, start)
    if start - sent_start <= 6 and _COPULA.match(after):
        return True, "copula_definition", evidence, True, "copula_definition"
    clause = _first_clause(article_text, sent_start)
    if start - sent_start <= 20 and start < sent_start + len(clause):
        for modal in _MODAL:
            pos = article_text.find(modal, sent_start)
            if modal in clause and pos >= end:
                prefix = article_text[sent_start:start]
                strict_ok = len(prefix) <= 8 and not re.search(r"[为对向给由使让和与及或、]", prefix)
                reason = "normative_rule" if strict_ok else "normative_rule_object"
                return True, f"normative_rule:{modal}", evidence, strict_ok, reason
    for cstart, ctext in _clauses(article_text, sent_start):
        if start != cstart:
            continue
        if article_text[end : end + 1] in _BAD_AFTER_ALIAS:
            continue
        tail = ctext[len(alias) :]
        for modal in _MODAL:
            if modal in tail:
                return True, f"normative_subject:{modal}", evidence, True, "normative_subject"
    return False, "none", evidence, False, "none"


def _sample(n: int, seed: int) -> list[dict]:
    concepts = {c["id"]: c for c in read_jsonl("data/lexicon/concepts.jsonl")}
    articles = {a["article_id"]: a for a in read_jsonl("data/extracted/articles.jsonl")}
    typed = read_jsonl("data/extracted/mentions_typed.jsonl")
    defines = [r for r in typed if r.get("edge_type") == "DEFINES"]
    rng = random.Random(seed)
    picked = rng.sample(defines, min(n, len(defines)))
    return [
        {
            "edge": m,
            "concept": concepts.get(m["concept_id"], {}),
            "article": articles.get(m["article_id"], {}),
        }
        for m in picked
    ]


def selfcheck(sample: int, seed: int) -> dict:
    rows, ok, strict_ok_n = [], 0, 0
    criteria: Counter = Counter()
    strict_reasons: Counter = Counter()
    prov: Counter = Counter()
    for item in _sample(sample, seed):
        m, c, a = item["edge"], item["concept"], item["article"]
        alias = m.get("matched_alias") or (c.get("aliases") or [c.get("name", "")])[0]
        b7_ok, criterion, evidence, strict_ok, strict_reason = judge(
            a.get("content", ""), alias, int(m.get("offset", -1))
        )
        manual_override = bool(not b7_ok and m.get("provenance") == "manual" and m.get("note"))
        if manual_override:
            b7_ok, criterion = True, "manual_edge"
        ok += int(b7_ok)
        strict_ok_n += int(strict_ok)
        criteria[criterion] += 1
        strict_reasons[strict_reason] += 1
        prov[m.get("provenance") or "-"] += 1
        rows.append(
            {
                "concept": m["concept_id"],
                "concept_name": c.get("name"),
                "article_id": m["article_id"],
                "citation": a.get("citation"),
                "matched_alias": alias,
                "verdict": b7_ok,
                "criterion": criterion,
                "manual_override": manual_override,
                "strict_verdict": strict_ok,
                "strict_reason": strict_reason,
                "evidence": evidence,
                "edge_cue": m.get("cue"),
                "provenance": m.get("provenance"),
                "confidence": m.get("confidence"),
            }
        )

    n = max(len(rows), 1)
    out = {
        "reviewer": "assistant",
        "scope": (
            "**规则自洽性校验，不是精度**：判据（_AFTER_CUE/_COPULA/_MODAL/_BAD_AFTER_ALIAS、12 字窗口）"
            "与建边脚本 app/__002__extract/__003__edge_typing.py 完全同源，等于把分类器再跑一遍，"
            "结论只能是「代码与规则一致」，不能说明规则判得对。对外报告定义级精度请用 "
            "data/eval_out/defines_blind_review.json（--mode blind，概念名+条文、不给 cue）。"
        ),
        "sample": len(rows),
        "selfcheck_passed": ok,
        "selfcheck_pass_rate": round(ok / n, 4),
        "strict_passed": strict_ok_n,
        "strict_pass_rate": round(strict_ok_n / n, 4),
        "criteria": dict(criteria),
        "strict_reasons": dict(strict_reasons),
        "provenance": dict(prov),
        "items": rows,
    }
    write_json(SELFCHECK_OUT, out)
    log.info(
        "自洽性校验 %s/%s = %s（严口径 %s/%s）；这不是精度，见 scope",
        ok,
        len(rows),
        out["selfcheck_pass_rate"],
        strict_ok_n,
        len(rows),
    )
    return out


# ---------------------------------------------------------------- 盲判口径
def _cache_load() -> dict:
    p = Path(BLIND_CACHE)
    return read_json(p) if p.exists() else {}


def _cache_save(cache: dict) -> None:
    write_json(BLIND_CACHE, cache)


def _vote(client, name: str, content: str, *, flip: bool = False) -> dict:
    """一次盲判。``flip=True`` 时把三种关系的**排列顺序**换一遍，抵消位置偏好。

    三种关系的措辞必须与 B7 口径**操作化等价**，否则盲判测的是另一个更严的标准：
    B7 的 ``normative_subject`` 允许「概念名位于分句主语位 + 可以/应当/不得/有权/必须」，
    也就是说「定作人可以请求承揽人赔偿」对「定作人」算正面规制——它不是定义句，
    但这句在给该概念设定规范地位。第一版提示词把这种情况判成了「顺带提及」，
    结果 100 条里只通过 27%，测出来的是「有多少条是定义句」，不是同一把尺子。
    """
    options = [
        (
            "定义式",
            "条文在给这个概念下定义，典型是「X 是指……」「X 是……的合同/权利」。",
        ),
        (
            "正面规制",
            "条文把这个概念当作规范对象：概念出现在条文首句或某个分句的**主语位置**，"
            "条文规定它「可以 / 应当 / 不得 / 有权 / 必须」做什么，或规定它的范围、构成、效力。"
            "这类条文即使同时涉及其他人，规范对象仍是这个概念本身。",
        ),
        (
            "顺带提及",
            "概念出现在**宾语、介词短语、列举、从句或背景说明**里，条文的规范对象是别的主体或别的事项。",
        ),
    ]
    examples = (
        "例子：\n"
        "· 「承揽人交付的工作成果不符合质量要求的，定作人可以合理选择请求承揽人承担修理、重作…」"
        "对「定作人」→ 正面规制（该句给定了定作人的请求权）。\n"
        "· 「用人单位与劳动者可以在劳动合同中约定保守用人单位的商业秘密…」对「商业秘密」→ 顺带提及"
        "（规范对象是保密约定，不是商业秘密本身）。"
    )
    if flip:
        options = [options[2], options[0], options[1]]
    bullets = "\n".join(f'{i + 1}. "{k}"：{d}' for i, (k, d) in enumerate(options))
    prompt = (
        f"判断下面这条法条与该概念的关系。\n\n概念：{name}\n法条：{content}\n\n"
        f"三种关系（按此定义判定，不要引入更严的标准）：\n{bullets}\n\n"
        f"{examples}\n\n"
        '只输出：{"kind": "上面三个取值之一", "reason": "不超过25字"}'
    )
    raw = client.json_chat(
        [{"role": "system", "content": BLIND_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    kinds = [k for k, _ in options]
    kind = str(raw.get("kind") or "").strip() if isinstance(raw, dict) else ""
    reason = str(raw.get("reason") or "").strip() if isinstance(raw, dict) else ""
    if kind not in kinds:
        # 兼容模型按序号或布尔作答
        idx = str(raw.get("kind") or "").strip() if isinstance(raw, dict) else ""
        if idx in {"1", "2", "3"}:
            kind = options[int(idx) - 1][0]
        else:
            v = raw.get("verdict") if isinstance(raw, dict) else None
            if isinstance(v, bool):
                kind = "正面规制" if v else "顺带提及"
    if kind not in ("定义式", "正面规制", "顺带提及"):
        kind = "无法解析"
    return {"kind": kind, "reason": reason}


def blind_review(sample: int, seed: int) -> dict:
    from common.llm import get_llm, llm_probe

    if not llm_probe():
        raise SystemExit("LLM 不可用（无 key 或全部鉴权失败），盲判无法执行")

    client = get_llm()
    items = _sample(sample, seed)
    cache = _cache_load()
    rows = []
    for i, item in enumerate(items, 1):
        m, c, a = item["edge"], item["concept"], item["article"]
        name = c.get("name") or m["concept_id"]
        content = (a.get("content") or "").strip()
        key = hashlib.sha1(
            f"{PROMPT_VERSION}|{m['article_id']}|{m['concept_id']}|{name}|{content[:80]}".encode()
        ).hexdigest()[:16]
        rec = cache.get(key)
        if not rec:
            v1 = _vote(client, name, content)
            v2 = _vote(client, name, content, flip=True)
            votes = [v1["kind"], v2["kind"]]
            reason = v1["reason"]
            if v1["kind"] != v2["kind"]:
                v3 = _vote(client, name, content)
                votes.append(v3["kind"])
                reason = f"{v1['reason']} / {v3['reason']}"
            hit = [k for k in votes if k != "无法解析"]
            kind = Counter(hit).most_common(1)[0][0] if hit else "无法解析"
            rec = {"votes": votes, "kind": kind, "reason": reason, "agree": len(set(votes)) == 1}
            cache[key] = rec
            if i % 10 == 0:
                _cache_save(cache)
                log.info("盲判进度 %d/%d", i, len(items))
        rows.append(
            {
                "concept": m["concept_id"],
                "concept_name": name,
                "article_id": m["article_id"],
                "citation": a.get("citation"),
                "content": content,
                "votes": rec["votes"],
                "kind": rec["kind"],
                "reason": rec["reason"],
                "votes_agree": rec["agree"],
                "pass": rec["kind"] in ("定义式", "正面规制"),
                "provenance": m.get("provenance"),
            }
        )
    _cache_save(cache)

    passed = sum(1 for r in rows if r["pass"])
    n = max(len(rows), 1)
    kinds = Counter(r["kind"] for r in rows)
    agree = sum(1 for r in rows if r["votes_agree"])
    out = {
        "reviewer": "llm_blind_2of3",
        "scope": (
            "盲判：只给模型「概念名 + 条文全文」，不给匹配别名、偏移与 cue，两次投票（第二次选项顺序相反）、"
            "不一致时第三次取多数。与建边规则不同源，可作为定义级精度的独立估计；"
            "但它不是人工盲标，模型与建边脚本读的是同一批中文法条，仍可能共享同类先验。"
        ),
        "model": getattr(client, "model", None),
        "sample": len(rows),
        "defines_total": len(
            [r for r in read_jsonl("data/extracted/mentions_typed.jsonl") if r.get("edge_type") == "DEFINES"]
        ),
        "pass": passed,
        "pass_rate": round(passed / n, 4),
        "votes_agree_rate": round(agree / n, 4),
        "kind_distribution": dict(kinds),
        "threshold": 0.85,
        "meets_threshold": passed / n >= 0.85,
        "note": (
            "B7 口径把「对概念的正面规制」也算 DEFINES，所以盲判同样接受「正面规制」；"
            "若只认「定义式」，看 kind_distribution 里的定义式条数。"
        ),
        "items": rows,
    }
    write_json(BLIND_OUT, out)
    log.info(
        "盲判精度 %s/%s = %s（投票一致率 %s）；定义式 %s / 正面规制 %s / 顺带提及 %s",
        passed,
        len(rows),
        out["pass_rate"],
        out["votes_agree_rate"],
        kinds.get("定义式", 0),
        kinds.get("正面规制", 0),
        kinds.get("顺带提及", 0),
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selfcheck", "blind"], default="selfcheck")
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()

    if args.mode == "selfcheck":
        out = selfcheck(args.sample, args.seed)
        print(json.dumps({k: v for k, v in out.items() if k != "items"}, ensure_ascii=False, indent=1))
        return

    out = blind_review(args.sample, args.seed)
    print(json.dumps({k: v for k, v in out.items() if k != "items"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
