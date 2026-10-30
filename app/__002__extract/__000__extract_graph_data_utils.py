"""LLM 增量抽取（对照 TCM 的 ``__000__extract_graph_data_utils.py``）。

与原项目一致的三个工程点：
1. ``PromptTemplate | llm | JsonOutputParser(pydantic_object=...)`` 的链式调用；
2. **断点续传**：每抽完一条就 append 到 JSONL，重跑自动跳过已完成的 article_id；
3. 单条失败不中断整批，失败计入统计，最后统一重试。

和原项目不同的地方：LLM 只做「规则没覆盖到的补充」，且必须从**闭集概念表**里选，
避免自由发挥导致图 schema 漂移。
"""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.llm import get_llm  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import append_jsonl, read_jsonl  # noqa: E402

log = get_logger("extract_llm")

_SYSTEM = (
    "你是中国法律条文标注员。给定一条法条原文与一份封闭的概念表，"
    "判断该条文在规范哪些概念（可多选）、对应哪些典型纠纷场景。"
    "只允许从给定列表中选择，不得自造；不确定就少选。"
)

_USER = """【法条】{law_name} {article_no}
【章节】{chapter_path}
【原文】
{content}

【可选概念】
{concept_list}

【可选场景】
{scenario_list}

输出 JSON：
{{"concepts": ["概念名"], "scenarios": ["场景名"], "reason": "一句话理由"}}
"""


class ArticleGraph(BaseModel):
    concepts: list[str] = Field(default_factory=list)
    scenarios: list[str] = Field(default_factory=list)
    reason: str = ""


def _concept_list_for_law(concepts: list[dict], law_name: str) -> list[str]:
    names = [c["name"] for c in concepts if law_name in c["laws"]]
    # 其他法律的概念也可能被引用（如劳动合同法引用民法典），补充 30 个通用概念
    if len(names) < 40:
        names += [c["name"] for c in concepts if law_name not in c["laws"]][:30]
    return names


def extract_one(article: dict, concept_names: list[str], scenario_names: list[str]) -> dict[str, Any]:
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate.from_template(_USER)
    parser = JsonOutputParser(pydantic_object=ArticleGraph)
    rendered = prompt.format(
        law_name=article["law_name"],
        article_no=article["article_no"],
        chapter_path=article.get("chapter_path", ""),
        content=article["content"],
        concept_list="、".join(concept_names),
        scenario_list="、".join(scenario_names),
    )
    llm = get_llm()
    raw = llm.chat(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": rendered}],
        temperature=0.0,
        max_tokens=600,
    )
    parsed = parser.parse(raw)
    data = ArticleGraph(**parsed) if isinstance(parsed, dict) else ArticleGraph()
    return {
        "article_id": article["article_id"],
        "concepts": data.concepts,
        "scenarios": data.scenarios,
        "reason": data.reason,
    }


def run_llm_extraction(
    articles: list[dict],
    concepts: list[dict],
    scenarios: list[dict],
    out_path: Path,
    max_workers: int = 4,
    limit: int | None = None,
) -> dict[str, Any]:
    """断点续传 + 并发抽取。返回统计信息。"""
    done: set[str] = set()
    if out_path.exists():
        done = {r["article_id"] for r in read_jsonl(out_path) if "article_id" in r}
        log.info("断点续传：已完成 %d 条，跳过", len(done))

    todo = [a for a in articles if a["article_id"] not in done]
    if limit:
        todo = todo[:limit]
    if not todo:
        return {"requested": 0, "ok": 0, "failed": 0, "skipped": len(done)}

    scenario_names = [s["name"] for s in scenarios]
    by_law: dict[str, list[str]] = {
        law: _concept_list_for_law(concepts, law)
        for law in {a["law_name"] for a in articles}
    }
    name_to_id = {c["name"]: c["id"] for c in concepts}

    lock = threading.Lock()
    ok, failed = 0, 0

    def _task(art: dict) -> dict[str, Any]:
        row = extract_one(art, by_law[art["law_name"]], scenario_names)
        row["concepts"] = [name_to_id[n] for n in row["concepts"] if n in name_to_id]
        row["provenance"] = "llm"
        return row

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_task, a): a for a in todo}
        for fut in as_completed(futures):
            art = futures[fut]
            try:
                row = fut.result()
                with lock:
                    append_jsonl(out_path, row)
                    ok += 1
            except Exception as exc:
                failed += 1
                log.warning("抽取失败 %s: %s", art["article_id"], exc)
            if (ok + failed) % 50 == 0:
                log.info("LLM 抽取进度 %d/%d（失败 %d）", ok + failed, len(todo), failed)

    return {"requested": len(todo), "ok": ok, "failed": failed, "skipped": len(done)}
