"""Streamlit 演示前端：把 B3/B4/B5/B6 四个验收项各开一个页签。

刻意做成「一屏能看懂」：
* 问答页直接暴露三档编排（A/B/C）与三档检索（text/graph/hybrid），
  并逐项显示五道闸的通过情况 —— 这是 B6 的卖点，不能只给一个总分；
* 多轮页显示槽位、缺失项、本轮继承项与指代标记 —— 这是 B4 的可观测性；
* 复核页列出挂起任务并支持三种人工介入方式 —— 这是 B5 的闭环；
* 评测页直接读 ``data/eval_out/*.json``，页面上看到的就是报告里的数字。

启动：``streamlit run app/__009__ui/__000__app.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from config import settings
from common.llm import llm_status

EVAL_FILES = {
    "B2 金标准": "data/eval_out/gold.json",
    "B3 检索三档": "data/eval_out/retrieval.json",
    "B3 路由": "data/eval_out/router.json",
    "B4 多轮": "data/eval_out/dialog.json",
    "B5 人在环": "data/eval_out/hitl.json",
    "B6 消融": "data/eval_out/ablation.json",
}


@st.cache_resource(show_spinner="加载检索器与图（首次约 20s）…")
def load_runtime():
    from app.__004__retrieval.__005__retriever import get_retriever
    from app.__006__hitl.__003__pipeline import HitlPipeline
    from app.__007__agents.__001__gates import build_article_concepts
    from app.__007__agents.__002__flows import make_flow

    ret = get_retriever()
    art_concepts = build_article_concepts(ret.graph.store)
    flows = {t: make_flow(ret, ret.text.articles, t, art_concepts=art_concepts) for t in ("A", "B", "C")}
    return ret, flows, HitlPipeline(retriever=ret)


def gate_badge(check: dict) -> str:
    return ("✅ " if check["ok"] else "❌ ") + f"**{check['gate']}** — {check['detail']}"


def show_answer(state: dict) -> None:
    st.markdown(state.get("answer", ""))
    cols = st.columns(4)
    cols[0].metric("耗时(ms)", state.get("latency_ms", 0))
    cols[1].metric("LLM 调用", state.get("llm_calls", 0))
    cols[2].metric("覆盖度", state.get("retrieval", {}).get("coverage", 0))
    cols[3].metric("top1 相似", state.get("retrieval", {}).get("top1_similarity", 0))
    if state.get("refused"):
        st.warning("系统判定证据不足，给出的是「无法确定」类回答，应转人工复核。")
    if state.get("path_note"):
        with st.expander("推理链路（规划节点产出）"):
            st.write(state["path_note"])
    if state.get("checks"):
        st.subheader("五道闸")
        for c in state["checks"]:
            st.markdown(gate_badge(c))
        if state.get("repaired"):
            st.info(f"触发确定性修复，已重试 {state.get('attempts', 0)} 轮。")
    items = state.get("retrieval", {}).get("items", [])
    if items:
        st.subheader("检索到的条文")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "条文": i["citation"],
                        "分数": round(i.get("score", 0.0), 4),
                        "来源": i.get("source"),
                        "覆盖": i.get("coverage"),
                        "图路径": " -> ".join(i.get("path") or []) or "-",
                    }
                    for i in items
                ]
            ),
            use_container_width=True,
        )
        with st.expander("条文原文"):
            for i in items:
                st.markdown(f"**{i['citation']}**（{i.get('law_name')}）\n\n{i.get('content', '')}")
    linked = state.get("retrieval", {}).get("linked", {})
    if linked.get("concepts"):
        st.caption("命中概念：" + "、".join(f"{c['name']}({c['score']:.2f})" for c in linked["concepts"]))
    if linked.get("scenarios"):
        st.caption("命中场景：" + "、".join(f"{s['name']}({s['score']:.2f})" for s in linked["scenarios"]))


def render_qa(ret, flows) -> None:
    c1, c2, c3 = st.columns([3, 1, 1])
    query = c1.text_input("问题", value="公司拖了我两个月工资，我能要回来吗？", key="qa_query")
    tier = c2.selectbox("编排", ["A 单链路", "B 多 Agent", "C +五道闸"], index=2, key="qa_tier")
    mode = c3.selectbox("检索", ["hybrid", "text", "graph"], index=0, key="qa_mode")
    if st.button("提问", type="primary", key="qa_btn") and query.strip():
        with st.spinner("检索 + 生成 + 校验…"):
            state = flows[tier[0]](query, mode=mode)
        st.session_state["qa_state"] = state
    if "qa_state" in st.session_state:
        show_answer(st.session_state["qa_state"])


def render_dialog(ret) -> None:
    from app.__005__dialog.__002__dialog_manager import DialogManager

    if "dm" not in st.session_state:
        st.session_state.dm = DialogManager(retriever=ret)
        st.session_state.sid = st.session_state.dm.new_session()
        st.session_state.turns = []
    dm = st.session_state.dm
    st.caption(f"会话 ID：`{st.session_state.sid}`（刷新后仍可按此恢复，槽位状态存在 SQLite）")
    text = st.text_input("输入一轮，例如「2024年5月离职了」或「那如果没签合同呢」", key="dlg_input")
    if st.button("发送", key="dlg_btn") and text.strip():
        turn = dm.handle(st.session_state.sid, text)
        st.session_state.turns.append(turn.as_dict())
    for t in st.session_state.turns:
        st.markdown(f"**第 {t['turn_no']} 轮 · {t['status']}** · 意图 `{t['intent']}`")
        st.markdown(f"- 槽位：`{json.dumps(t['slots'], ensure_ascii=False)}`")
        st.markdown(f"- 缺失：`{t['missing']}`　本轮继承：`{t['inherited']}`　指代：`{t['coreference']}`")
        if t["question"]:
            st.info(f"追问：{t['question']}")
        if t["retrieval"]:
            items = t["retrieval"]["items"][:3]
            st.success("命中：" + "；".join(i["citation"] for i in items))


def render_hitl(pipe) -> None:
    stats = pipe.stats()
    st.metric("待复核", stats.get("pending", 0))
    st.caption(f"介入率 {stats.get('intervention_rate', 0):.2f}　修正类型分布 {stats.get('correction_types', {})}")
    q = st.text_input("发起一条需要复核的问题", value="公司股东之间转让股权需要缴什么税？", key="hitl_q")
    force = st.checkbox("强制复核（跳过自动判定）", value=True, key="hitl_force")
    if st.button("提交", key="hitl_start") and q.strip():
        with st.spinner("检索中…"):
            out = pipe.start(q, force_review=force)
        st.json(out.get("retrieval", {}))
        st.session_state["hitl_task"] = out.get("task_id")
    pending = pipe.pending()
    if pending:
        st.subheader("挂起任务")
        for p in pending:
            with st.expander(f"{p['task_id']} · {p['query'][:28]} · {p['reason'][:40]}"):
                st.write(f"top1={p['top1_similarity']:.3f}　coverage={p['coverage']:.3f}")
                action = st.radio("修正方式", ["accept", "edit_articles", "edit_answer", "provide_answer"], key=f"act_{p['task_id']}")
                payload: dict = {"type": action}
                if action == "edit_articles":
                    ids = st.text_input("条文 ID（如 lcl#85, ll#91）", key=f"ids_{p['task_id']}")
                    payload["article_ids"] = [x.strip() for x in ids.split(",") if x.strip()]
                if action in ("edit_answer", "provide_answer"):
                    payload["answer"] = st.text_area("人工答案", key=f"ans_{p['task_id']}")
                if st.button("提交修正", key=f"sub_{p['task_id']}"):
                    try:
                        res = pipe.resume(p["task_id"], payload)
                        st.success("已恢复")
                        st.markdown(res.get("answer", ""))
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))


def render_eval() -> None:
    names = st.multiselect("报告", list(EVAL_FILES), default=["B3 检索三档", "B6 消融"])
    for name in names:
        path = Path(EVAL_FILES[name])
        if not path.exists():
            st.warning(f"缺少 {path}，请先运行对应评测脚本。")
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        st.subheader(name)
        if name == "B3 检索三档":
            rows = [{"档位": k, **{m: v for m, v in val.items() if not isinstance(v, dict)}} for k, val in blob.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        elif name == "B6 消融":
            st.dataframe(pd.DataFrame(blob.get("tiers", {})).T, use_container_width=True)
            if blob.get("cost_table"):
                st.caption("成本表")
                st.dataframe(pd.DataFrame(blob["cost_table"]).T, use_container_width=True)
            boot = blob.get("bootstrap", {})
            if boot:
                st.caption("配对 bootstrap（100 次，零是否落在区间内）")
                st.dataframe(pd.DataFrame(boot).T, use_container_width=True)
        else:
            st.json(blob)


def main() -> None:
    st.set_page_config(page_title="LexWeave · 劳动法图谱问答", layout="wide")
    st.title("LexWeave · 劳动法知识图谱双路检索问答")
    st.caption(f"运行模式：{settings.run_mode}　LLM：{llm_status().get('model') or '未配置'}")

    ret, flows, pipe = load_runtime()
    backend = ret.graph.store.backend
    st.caption(f"图后端：{backend}　语料：{len(ret.text.articles)} 条")

    tabs = st.tabs(["图谱问答（B6）", "多轮槽位（B4）", "人工复核（B5）", "评测报告"])
    with tabs[0]:
        render_qa(ret, flows)
    with tabs[1]:
        render_dialog(ret)
    with tabs[2]:
        render_hitl(pipe)
    with tabs[3]:
        render_eval()


if __name__ == "__main__":
    main()
