"""B4 槽位与继承的单元测试：不依赖模型，只测确定性逻辑。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.__005__dialog.__000__slots import SlotState, detect_intent, extract_slots, missing_slots  # noqa: E402


def test_extract_wage_and_dates():
    slots = extract_slots("我月薪8000，2024年5月离职了")
    assert slots["monthly_wage"] == 8000
    assert slots["end_date"].startswith("2024")


def test_has_contract_negative_with_infix():
    # 「没跟我签劳动合同」中间有插入语，早期正则漏判，这里回归
    assert extract_slots("公司一直没跟我签劳动合同")["has_contract"] is False
    assert extract_slots("签了书面合同")["has_contract"] is True


def test_intent_detection():
    assert detect_intent("公司拖欠工资怎么办") == "wage"
    assert detect_intent("公司要辞退我") == "contract"


def test_missing_slots_by_intent():
    assert "monthly_wage" in missing_slots({}, "wage")
    assert missing_slots({"monthly_wage": 8000, "start_date": "2023-01", "end_date": "2024-05"}, "wage") == []


def test_state_merge_keeps_untouched_values():
    state = SlotState(values={"monthly_wage": 9000, "city": "北京"})
    state.merge({"monthly_wage": None, "has_contract": False})
    assert state.values["monthly_wage"] == 9000  # None 不应覆盖旧值
    assert state.values["has_contract"] is False


def test_state_roundtrip_json():
    state = SlotState(values={"monthly_wage": 7000, "intent": "severance"})
    state.intent = "severance"
    restored = SlotState.from_json(state.to_json())
    assert restored.values == state.values
    assert restored.intent == "severance"
