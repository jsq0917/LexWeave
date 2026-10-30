"""B4 槽位：定义、规则抽取、缺失判定。

为什么用规则抽取而不是让 LLM 抽：
* 槽位是**结构化数字/日期/布尔**，规则的正则边界清晰、可单测、零成本、零幻觉；
* 多轮场景里每轮都调用 LLM 抽槽位，成本与延迟都会压垮对话体验；
* LLM 只在「规则抽不到但语义上明显」的兜底场景才用（本实现没有开启，如实写在 README）。

槽位值统一成可序列化的基本类型，便于 SQLite 持久化与跨进程恢复。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

SLOT_ORDER = ("monthly_wage", "start_date", "end_date", "has_contract", "work_years", "city")

SLOT_QUESTIONS = {
    "monthly_wage": "方便说下你的月工资大概是多少吗？",
    "start_date": "这段劳动关系从什么时候开始的？",
    "end_date": "到什么时候结束（或准备什么时候离职）？",
    "has_contract": "你们签了书面劳动合同吗？",
    "work_years": "你在这家单位一共工作了多少年？",
    "city": "事情发生在哪个城市？",
}

WAGE_KEYWORDS = ("工资", "薪资", "报酬", "拖欠", "加班费", "月薪", "薪水", "赔偿金", "补偿")
CONTRACT_KEYWORDS = ("合同", "签", "续签", "试用期", "解除", "辞退", "离职", "竞业")

_CITIES = (
    "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "成都", "重庆", "武汉",
    "西安", "天津", "长沙", "郑州", "青岛", "宁波", "厦门", "合肥", "济南", "福州",
)

_CN_NUM = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def cn_to_int(text: str) -> int | None:
    """解析「三」「十二」「二十」「两千」这类中文数字（够用即可，不追求全覆盖）。"""
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text == "半":
        return 0
    total, section = 0, 0
    for ch in text:
        if ch in _CN_NUM:
            val = _CN_NUM[ch]
            if ch == "十":
                section = (section or 1) * 10
            else:
                section = section * 10 + val if section >= 10 else val
        elif ch == "百":
            total += (section or 1) * 100
            section = 0
        elif ch == "千":
            total += (section or 1) * 1000
            section = 0
        elif ch == "万":
            total += (section or 1) * 10000
            section = 0
        else:
            return None
    return total + section


def _wage(text: str) -> int | None:
    m = re.search(r"(?:月薪|每个月|每月|一个月|月工资|工资)\D{0,6}?(\d+(?:\.\d+)?)\s*(?:元|块|k|K|千)?", text)
    if m:
        val = float(m.group(1))
        if val < 100 and ("k" in m.group(0).lower() or "千" in m.group(0)):
            val *= 1000
        return int(val)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:k|K|千)", text)
    if m and any(k in text for k in ("工资", "月薪", "薪水", "薪资")):
        return int(float(m.group(1)) * 1000)
    return None


END_MARKERS = ("离职", "辞职", "结束", "解除", "截至", "截止", "干到", "做到", "到", "至")
START_MARKERS = ("入职", "开始", "起", "上班", "就职")


def _dates(text: str) -> list[str]:
    pat = r"(\d{4})\s*[年\-/]\s*(\d{1,2})(?:\s*[月\-/]\s*(\d{1,2}))?"
    hits = sorted({(int(y), int(mo), int(da) if da else 0) for y, mo, da in re.findall(pat, text)})
    return [f"{y:04d}-{mo:02d}" + (f"-{da:02d}" if da else "") for y, mo, da in hits]


def _has_contract(text: str) -> bool | None:
    # 否定在前：先判「没/未 + 插入语 + 签…合同」，再判「有…合同」。
    # 「签了书面合同」这类在 签了 与 合同 之间夹修饰语的写法也要认。
    if re.search(r"(?:没|未|没有)[\u4e00-\u9fa5]{0,5}?签(?:订|过|了)?(?:书面)?(?:劳动)?合同", text):
        return False
    if re.search(r"(?:签了|签订了|签过)(?:书面)?(?:劳动)?合同", text):
        return True
    # (?<!没) 防止「没有书面合同」被当成「有书面合同」
    if re.search(r"(?<!没)(?<!未)有(?:书面)?(?:劳动)?合同", text):
        return True
    if re.search(r"(?:没|未|没有)(?:书面)?(?:劳动)?合同", text):
        return False
    return None


def _work_years(text: str) -> float | None:
    m = re.search(r"(?:工作|干了|干了有|做了|一共)\s*([一二两三四五六七八九十\d]+)\s*年", text)
    if m:
        n = cn_to_int(m.group(1))
        return float(n) if n is not None else None
    m = re.search(r"([一二两三四五六七八九十\d]+)\s*年(?:工龄|了)", text)
    if m:
        n = cn_to_int(m.group(1))
        return float(n) if n is not None else None
    m = re.search(r"(\d+)\s*个月", text)
    if m and re.search(r"(工作|干了|做了)", text):
        return round(int(m.group(1)) / 12, 2)
    return None


def _city(text: str) -> str | None:
    for c in _CITIES:
        if c in text:
            return c
    m = re.search(r"([\u4e00-\u9fa5]{2,4})市", text)
    return m.group(1) if m else None


def extract_slots(text: str) -> dict[str, Any]:
    """只返回「本轮明确提到」的槽位，没提到的不下沉到结果里，避免覆盖历史值。"""
    out: dict[str, Any] = {}
    wage = _wage(text)
    if wage:
        out["monthly_wage"] = wage
    dates = _dates(text)
    if len(dates) >= 2:
        out["start_date"] = dates[0]
        out["end_date"] = dates[-1]
    elif dates:
        if any(k in text for k in END_MARKERS) and not any(k in text for k in START_MARKERS):
            out["end_date"] = dates[0]
        elif any(k in text for k in START_MARKERS) and not any(k in text for k in END_MARKERS):
            out["start_date"] = dates[0]
        elif any(k in text for k in END_MARKERS):
            out["end_date"] = dates[0]
        else:
            out["start_date"] = dates[0]
    hc = _has_contract(text)
    if hc is not None:
        out["has_contract"] = hc
    wy = _work_years(text)
    if wy is not None:
        out["work_years"] = wy
    city = _city(text)
    if city:
        out["city"] = city
    return out


def detect_intent(text: str) -> str:
    if any(k in text for k in WAGE_KEYWORDS):
        return "wage"
    if any(k in text for k in CONTRACT_KEYWORDS):
        return "contract"
    return "other"


REQUIRED_BY_INTENT = {
    "wage": ("monthly_wage", "start_date", "end_date"),
    "contract": ("has_contract",),
    "other": (),
}


def missing_slots(values: dict[str, Any], intent: str) -> list[str]:
    return [s for s in REQUIRED_BY_INTENT.get(intent, ()) if values.get(s) in (None, "")]


@dataclass
class SlotState:
    values: dict[str, Any] = field(default_factory=dict)
    intent: str = "other"

    def merge(self, new: dict[str, Any], inherit: bool = False) -> list[str]:
        """写入新槽位，返回本轮**变化**的槽位名；inherit=True 表示本轮是继承而非改写。"""
        changed = []
        for k, v in new.items():
            if v is None:
                continue
            if self.values.get(k) != v:
                self.values[k] = v
                changed.append(k)
        return changed

    def to_json(self) -> dict[str, Any]:
        return {"values": dict(self.values), "intent": self.intent}

    @classmethod
    def from_json(cls, blob: dict[str, Any] | None) -> "SlotState":
        blob = blob or {}
        return cls(values=dict(blob.get("values", {})), intent=blob.get("intent", "other"))

    def describe(self) -> str:
        if not self.values:
            return ""
        parts = []
        labels = {
            "monthly_wage": "月工资",
            "start_date": "起始",
            "end_date": "截止",
            "has_contract": "签合同",
            "work_years": "工龄",
            "city": "城市",
        }
        for k in SLOT_ORDER:
            if k in self.values:
                v = self.values[k]
                if k == "has_contract":
                    v = "是" if v else "否"
                parts.append(f"{labels[k]}={v}")
        return "；".join(parts)
