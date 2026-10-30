"""法条编号与引用解析（全项目唯一的 ID 口径）。

统一用 ``<law_code>#<article_num>`` 作为 Article 主键，例如 ``cc#584``：
* 比中文条号更适合做集合运算与评测匹配；
* 三部法律条号重复（都从第一条开始），必须带法律前缀；
* 中文条号只在展示层出现（``中华人民共和国劳动合同法 第二十三条``）。
"""

from __future__ import annotations

import re
from typing import Iterable

LAW_FULL_TO_CODE = {
    "中华人民共和国劳动合同法": "lcl",
    "中华人民共和国劳动法": "ll",
    "中华人民共和国民法典": "cc",
}
LAW_CODE_TO_FULL = {v: k for k, v in LAW_FULL_TO_CODE.items()}
LAW_SHORT = {"lcl": "劳动合同法", "ll": "劳动法", "cc": "民法典"}

_CN_DIGITS = "零一二三四五六七八九"
_CN_UNITS = ("", "十", "百", "千")


def law_code(law_name: str) -> str:
    if law_name in LAW_FULL_TO_CODE:
        return LAW_FULL_TO_CODE[law_name]
    for full, code in LAW_FULL_TO_CODE.items():
        if law_name and law_name in full:
            return code
    raise KeyError(f"未知法律: {law_name}")


def article_id(law_name: str, article_num: int) -> str:
    return f"{law_code(law_name)}#{article_num}"


def citation(law_name: str, article_no: str) -> str:
    return f"{law_name}{article_no}条" if article_no.endswith("条") is False else f"{law_name}{article_no}"


def parse_article_id(aid: str) -> tuple[str, int]:
    code, num = aid.split("#")
    return code, int(num)


def cn_to_int(text: str) -> int:
    """中文数字 -> int，支持「一百四十六」等。"""
    text = text.strip()
    if text.isdigit():
        return int(text)
    result, section = 0, 0
    for ch in text:
        if ch in _CN_DIGITS:
            section = _CN_DIGITS.index(ch)
        elif ch in _CN_UNITS:
            unit = 10 ** _CN_UNITS.index(ch) if _CN_UNITS.index(ch) else 1
            if unit == 1:
                result += section
                section = 0
            else:
                if section == 0:
                    section = 1
                result += section * unit
                section = 0
        elif ch == "零":
            continue
        else:
            raise ValueError(f"非法中文数字: {text}")
    return result + section


def int_to_cn(num: int) -> str:
    if num <= 0:
        return str(num)
    if num < 10:
        return _CN_DIGITS[num]
    out = []
    digits = [int(d) for d in str(num)]
    length = len(digits)
    for i, d in enumerate(digits):
        unit_pow = length - i - 1
        if d == 0:
            if out and out[-1] != "零":
                out.append("零")
            continue
        if not (d == 1 and unit_pow == 1 and not out):
            out.append(_CN_DIGITS[d])
        out.append(_CN_UNITS[unit_pow])
    res = "".join(out).rstrip("零")
    return res


_ARTICLE_NO_RE = re.compile(r"第([零一二三四五六七八九十百千]+|\d+)条")


def article_no_to_int(article_no: str) -> int:
    m = _ARTICLE_NO_RE.search(article_no)
    if not m:
        raise ValueError(f"非法条号: {article_no}")
    return cn_to_int(m.group(1))


def split_content_sentences(text: str) -> list[str]:
    """按中文句号/分号切句，用于抽取时的证据定位。"""
    parts = re.split(r"(?<=[。；])", text)
    return [p.strip() for p in parts if p.strip()]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def law_scope_of(ids: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for aid in ids:
        code, _ = parse_article_id(aid)
        if code not in seen:
            seen.append(code)
    return seen
