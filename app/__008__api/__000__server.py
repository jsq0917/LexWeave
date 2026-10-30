"""FastAPI 服务：把检索 / 图问答消融 / 多轮对话 / 人在环暴露成 HTTP 接口。

存在的意义不是「再包一层」，而是让每个验收项都有一个可被第三方复现的入口：
* ``POST /retrieve``  —— B3 的 text / graph / hybrid 三档，返回 items + 图路径 + 路由决策；
* ``POST /ask``       —— B6 的 A / B / C 三档，返回答案、五道闸逐项结果、LLM 调用数与耗时；
* ``GET  /ask/stream``—— 同一次编排的 SSE 版本，按节点推 ``retrieve -> plan -> answer -> check``；
* ``POST /chat``      —— B4 的多轮槽位会话，``session_id`` 相同即共享状态；
* ``GET  /hitl`` ...  —— B5 的挂起列表与人工修正接口。

所有重对象（检索器、图、模型）都是懒加载且进程内单例，首请求慢、后续快；
``/health`` 明确回报当前真实后端（Neo4j 还是 NetworkX 离线镜像），不粉饰。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from common.llm import llm_status  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402

log = get_logger("api")

app = FastAPI(title="LexWeave", version="1.0.0", description="劳动法知识图谱双路检索 + 多 Agent 校验问答")

_MAX_ITEMS = 5


# --------------------------------------------------------------------------------
# 懒加载单例
# --------------------------------------------------------------------------------
def _retriever():
    from app.__004__retrieval.__005__retriever import get_retriever

    return get_retriever()


_FLOWS: dict[tuple[str, bool], Any] = {}


def _flow(tier: str, use_llm: bool = True):
    from app.__007__agents.__001__gates import build_article_concepts
    from app.__007__agents.__002__flows import make_flow

    key = (tier.upper(), use_llm)
    if key not in _FLOWS:
        ret = _retriever()
        art_concepts = build_article_concepts(ret.graph.store)
        _FLOWS[key] = make_flow(ret, ret.text.articles, key[0], use_llm=use_llm, art_concepts=art_concepts)
    return _FLOWS[key]


def _hitl():
    from app.__006__hitl.__003__pipeline import HitlPipeline

    return HitlPipeline(retriever=_retriever())


def _dialog():
    from app.__005__dialog.__002__dialog_manager import DialogManager

    return DialogManager(retriever=_retriever())


# --------------------------------------------------------------------------------
# 请求体
# --------------------------------------------------------------------------------
class RetrieveReq(BaseModel):
    query: str = Field(min_length=1)
    mode: str = Field(default="hybrid", pattern="^(text|graph|hybrid)$")
    level: str | None = Field(default=None, pattern="^(simple|medium|complex)$")
    top_k: int | None = Field(default=None, ge=1, le=20)


class AskReq(BaseModel):
    query: str = Field(min_length=1)
    mode: str = Field(default="hybrid", pattern="^(text|graph|hybrid)$")
    tier: str = Field(default="C", pattern="^[ABCabc]$")
    llm: bool = True


class ChatReq(BaseModel):
    session_id: str = ""
    text: str = Field(min_length=1)
    mode: str = Field(default="hybrid", pattern="^(text|graph|hybrid)$")


class HitlStartReq(BaseModel):
    query: str = Field(min_length=1)
    mode: str = Field(default="hybrid", pattern="^(text|graph|hybrid)$")
    force_review: bool = False


class HitlResolveReq(BaseModel):
    type: str = Field(pattern="^(accept|edit_articles|edit_answer|provide_answer)$")
    article_ids: list[str] | None = None
    answer: str | None = None
    citations: list[str] | None = None


# --------------------------------------------------------------------------------
# 基础
# --------------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    ret = _retriever()
    try:
        backend = ret.graph.store.backend
    except Exception:  # 图未构建时也要能返回 200，便于前端展示「先跑 build_graph」
        backend = "uninitialized"
    return {
        "ok": True,
        "graph_backend": backend,
        "corpus_articles": len(ret.text.articles),
        "llm": llm_status(),
    }


@app.get("/meta/laws")
def laws() -> dict[str, Any]:
    idx = _retriever().text.articles
    counts: dict[str, int] = {}
    for art in idx.values():
        counts[art["law_name"]] = counts.get(art["law_name"], 0) + 1
    return {"total": len(idx), "by_law": counts}


# --------------------------------------------------------------------------------
# B3：三档检索
# --------------------------------------------------------------------------------
@app.post("/retrieve")
def retrieve(req: RetrieveReq) -> dict[str, Any]:
    out = _retriever().retrieve(req.query, mode=req.mode, level=req.level, top_k=req.top_k).as_dict()
    out["items"] = out["items"][: max(req.top_k or _MAX_ITEMS, _MAX_ITEMS)]
    return out


# --------------------------------------------------------------------------------
# B6：三档编排
# --------------------------------------------------------------------------------
def _ask(req: AskReq) -> dict[str, Any]:
    flow = _flow(req.tier, req.llm)
    t0 = time.perf_counter()
    state = flow(req.query, mode=req.mode)
    state["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    state["tier"] = req.tier.upper()
    state["retrieval"]["items"] = state["retrieval"]["items"][:_MAX_ITEMS]
    return state


@app.post("/ask")
def ask(req: AskReq) -> dict[str, Any]:
    return _ask(req)


@app.get("/ask/stream")
def ask_stream(
    query: str = Query(min_length=1),
    mode: str = Query(default="hybrid"),
    tier: str = Query(default="C"),
) -> StreamingResponse:
    """SSE：按 LangGraph 节点逐个推送，前端可以逐段渲染而不是等整段答案。"""
    flow = _flow(tier)

    def gen() -> Iterator[str]:
        t0 = time.perf_counter()
        yield _sse("start", {"query": query, "mode": mode, "tier": tier.upper()})
        final: dict[str, Any] | None = None
        for chunk in flow.stream_steps({"query": query, "mode": mode, "attempts": 0, "repaired": False, "citations": [], "checks": []}):
            for node, upd in chunk.items():
                payload = {k: v for k, v in upd.items() if k in ("path_note", "answer", "checks", "attempts", "repaired", "llm_calls")}
                if "retrieval" in upd:
                    r = upd["retrieval"]
                    payload["retrieval"] = {
                        "level": r.get("level"),
                        "top1_similarity": r.get("top1_similarity"),
                        "coverage": r.get("coverage"),
                        "low_confidence": r.get("low_confidence"),
                        "items": r.get("items", [])[:_MAX_ITEMS],
                        "linked": r.get("linked"),
                    }
                final = {**(final or {}), **upd}
                yield _sse("node", {"node": node, **payload})
        if final:
            final.setdefault("answer", "")
            final["refused"] = "无法确定" in final["answer"] or "无法给出" in final["answer"]
            final["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            yield _sse("done", {"answer": final["answer"], "citations": final.get("citations", []), "refused": final["refused"], "latency_ms": final["latency_ms"]})

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# --------------------------------------------------------------------------------
# B4：多轮
# --------------------------------------------------------------------------------
@app.post("/chat")
def chat(req: ChatReq) -> dict[str, Any]:
    dm = _dialog()
    sid = req.session_id or dm.new_session()
    return dm.handle(sid, req.text, mode=req.mode).as_dict()


@app.get("/chat/{session_id}/history")
def chat_history(session_id: str) -> dict[str, Any]:
    return {"session_id": session_id, "turns": _dialog().history(session_id)}


# --------------------------------------------------------------------------------
# B5：人在环
# --------------------------------------------------------------------------------
@app.post("/hitl")
def hitl_start(req: HitlStartReq) -> dict[str, Any]:
    return _hitl().start(req.query, mode=req.mode, force_review=req.force_review)


@app.get("/hitl")
def hitl_list(status: str | None = None) -> dict[str, Any]:
    return {"pending": _hitl().pending(), "status_filter": status, "stats": _hitl().stats()}


@app.post("/hitl/{task_id}/resolve")
def hitl_resolve(task_id: str, req: HitlResolveReq) -> dict[str, Any]:
    try:
        action = {"type": req.type}
        if req.article_ids:
            action["article_ids"] = req.article_ids
        if req.answer:
            action["answer"] = req.answer
        if req.citations:
            action["citations"] = req.citations
        return _hitl().resume(task_id, action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.__008__api.__000__server:app", host="127.0.0.1", port=8000, reload=False)
