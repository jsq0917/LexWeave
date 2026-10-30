"""B4 评测：20 条多轮脚本 -> 状态/槽位/继承/指代/重连五项指标。

指标口径：
* ``status_accuracy``      —— 该追问时追问、该回答时回答的比例；
* ``slot_accuracy``        —— 每轮结束后槽位字典与脚本期望**完全一致**的比例（最严格）；
* ``inheritance_accuracy`` —— 指代型轮次中，期望继承的槽位确实被继承且值未变的比例；
* ``coref_pass``           —— 5 条指代消解用例是否全部「识别为接续 + 槽位继承正确」；
* ``restart_ok``           —— 用全新进程内对象 + 同一 SQLite 文件恢复会话，槽位与中断前一致。

失败样例全部落盘，便于回看是哪一类抽取边界写错了，而不是只报一个总分。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.__005__dialog.__002__dialog_manager import DialogManager  # noqa: E402
from app.__005__dialog.__001__session_store import SessionStore  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import write_json  # noqa: E402

log = get_logger("eval_dialog")

DB_PATH = Path("data/sessions/eval_dialog.sqlite3")
SCRIPTS = Path("eval/dialog_scripts.json")


def main() -> None:
    scripts = json.loads(SCRIPTS.read_text(encoding="utf-8"))
    if DB_PATH.exists():
        DB_PATH.unlink()

    store = SessionStore(DB_PATH)
    dm = DialogManager(store=store)

    status_hits = 0
    slot_hits = 0
    n_turns = 0
    inherit_num = 0.0
    inherit_den = 0
    coref_cases: list[dict] = []
    failures: list[dict] = []

    for script in scripts:
        sid = f"eval-{script['id']}"
        prev_slots: dict = {}
        for idx, turn in enumerate(script["turns"], start=1):
            res = dm.handle(sid, turn["user"])
            n_turns += 1
            status_ok = res.status == turn["status"]
            slots_ok = res.slots == turn["slots"]
            status_hits += int(status_ok)
            slot_hits += int(slots_ok)
            if not status_ok or not slots_ok:
                failures.append(
                    {
                        "script": script["id"],
                        "turn": idx,
                        "user": turn["user"],
                        "status": {"expect": turn["status"], "actual": res.status},
                        "slots": {"expect": turn["slots"], "actual": res.slots},
                    }
                )

            if turn.get("coref"):
                expected = set(turn.get("inherited", []))
                actual = set(res.inherited)
                inherit_den += len(expected)
                # 分子必须是「继承成功的槽位数」，不能是每轮的继承比例，否则和分母（槽位总数）量纲不一致
                retain = {k for k in expected & actual if prev_slots.get(k) == res.slots.get(k)}
                inherit_num += len(retain)
                ok = bool(res.coreference) and expected <= actual and len(retain) == len(expected) and status_ok
                coref_cases.append({"script": script["id"], "turn": idx, "ok": ok, "inherited": res.inherited})

            prev_slots = dict(res.slots)

    # 重连：新建对象 + 同一个 SQLite 文件，模拟进程重启
    last = scripts[-1]
    sid = f"eval-{last['id']}"
    fresh_store = SessionStore(DB_PATH)
    fresh_dm = DialogManager(store=fresh_store)
    state = fresh_store.load_state(sid)
    expected_final = last["turns"][-1]["slots"]
    actual_final = (state or {}).get("values", {})
    restart_ok = actual_final == expected_final
    # 恢复后还能继续对话（不依赖内存里的历史）
    follow = fresh_dm.handle(sid, "那要是只欠了半个月呢")
    restart_continues = follow.status in ("answered", "input_required")

    report = {
        "scripts": len(scripts),
        "turns": n_turns,
        "status_accuracy": round(status_hits / n_turns, 4),
        "slot_accuracy": round(slot_hits / n_turns, 4),
        "inheritance_accuracy": round(inherit_num / max(inherit_den, 1), 4),
        "coref_cases": coref_cases,
        "coref_pass": f"{sum(1 for c in coref_cases if c['ok'])}/{len(coref_cases)}",
        "restart_ok": restart_ok,
        "restart_continues": restart_continues,
        "failures": failures,
    }
    out = Path("data/eval_out/dialog.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    log.info(
        "状态 %.3f | 槽位 %.3f | 继承 %.3f | 指代 %s | 重连 %s",
        report["status_accuracy"],
        report["slot_accuracy"],
        report["inheritance_accuracy"],
        report["coref_pass"],
        restart_ok,
    )
    for f in failures[:6]:
        log.info("失败样例 %s", json.dumps(f, ensure_ascii=False)[:220])


if __name__ == "__main__":
    main()
