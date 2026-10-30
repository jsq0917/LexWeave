"""构建 B2 金标集：概念链 -> 锁定条文 -> LLM 写题面 -> 两道校验 -> Pass A/B 标注与 κ。

流程刻意把「人」与「模型」的职责分开：
  Pass A（人）：场景 + 概念链（见 eval/gold_combos.py），条文由确定性规则锁定并由 check_gold 回原文核验
  Pass B（模型）：只看到题面 + 若干候选条文（含 2 条干扰项），独立判断哪些条文是必需的
  κ：Pass A 与 Pass B 的逐条二元标签一致性；另跑两遍 Pass B 测稳定性

真实局限（会写进 README）：Pass A 不是双人盲标，κ 衡量的是「人定骨架」与「模型判读」的一致性，
不能等同于人工标注者间一致性。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.__003__graph.__003__graph_store import get_graph_store  # noqa: E402
from common.llm import get_llm  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from eval.common import JsonCache, bigram_overlap, cohen_kappa  # noqa: E402
from eval.gold_combos import COMBOS  # noqa: E402

log = get_logger("build_gold")

LEAK_THRESHOLD = 0.45
DISTRACTORS = 2
CANDIDATE_LIMIT = 3
SEED = 20261029
RERANK_LOCK = threading.Lock()

QUESTION_SYSTEM = (
    "你是法律咨询用户，不是法条解说员。把给定的纠纷事实写成一句真实、具体、口语化的提问。"
    "禁止：出现条号（如「第X条」）、禁止摘抄法条原文、禁止使用「依据」「根据本法」等书面套话、"
    "禁止直接点出概念术语的全部关键词（最多出现一个自然说法）。只输出题面本身。"
)

# v3：v2 只说「必须覆盖所有争点」，模型仍会写窄。v3 把每个争点逐条列给模型，
# 并要求题面在同一句提问里分别点到，题面才真的需要全部被锁条文。
QUESTION_SYSTEM_STRICT = (
    QUESTION_SYSTEM
    + "另外：这是多争点问题，题面必须在同一个提问里**分别点到下面列出的每个争点**（每个争点至少用一个生活化说法点到，不得遗漏），"
    "但不得出现条号、不得摘抄原文句子。"
)
QUESTION_VERSION = "v3"

NECESSITY_SYSTEM = (
    "你是法律推理审校员。给定一个用户问题与概念链中的一个环节，判断：要正确回答该问题，"
    "是否必须先确定这个环节的法律概念。只输出 JSON：{\"necessary\": true/false, \"reason\": \"...\"}"
)

# 判「必要性」的对象必须是**条文**而不是概念名：只看概念名时审校员无法知道第 37 条讲的是
# 「提前三十日通知」，会把明明需要的条文判成多余。三次独立判读取多数，降低单次噪声。
ARTICLE_NECESSITY_SYSTEM = (
    "你是法律推理审校员。给定一个用户问题与一条法条全文，判断：要完整、正确地回答该问题，"
    "是否必须用到这条法条（若换成别的条文或省略它，答案就不完整）。"
    "只输出 JSON：{\"necessary\": true/false, \"reason\": \"...\"}"
)

PASSB_SYSTEM = (
    "你是独立标注员。只读用户问题与候选法条，判断哪些法条是回答该问题**必需**的（可多选）。"
    "不要揣测出题人意图，只按法条内容判断。输出 JSON：{\"necessary\": [编号], \"reason\": \"...\"}"
)


def article_candidates(store, concept: dict, limit: int = 3) -> list[str]:
    """在概念所属法律内，按「DEFINES 置信度 + 命中位置」排前若干条候选条文。

    为什么要有候选列表：概念到条文的映射不是唯一的（同一个概念常散落在多条的正文里），
    之前只取 top-1，结果大量金标锁定了与题面无关的条文（必要性 43% 判不通过）。
    """
    cands = store.concept_articles([concept["id"]], limit=60)
    scored = []
    for c in cands:
        detail = store.article_detail([c["article_id"]])[0]
        if detail["law_name"] not in concept["laws"]:
            continue
        text = detail["content"]
        needles = [concept["name"], *concept["aliases"]]
        if not any(n and n in text for n in needles):
            continue
        first_sentence = re.split(r"[。；]", text)[0]
        bonus = 0.0
        if any(n in first_sentence for n in needles):
            bonus += 0.06
        if any(n in detail.get("chapter_path", "") for n in needles):
            bonus += 0.04
        if any(k in text for k in ("应当", "不得", "可以", "有权要求")):
            bonus += 0.03
        scored.append((c["confidence"] + bonus, -detail["article_num"], c["article_id"]))
    scored.sort(reverse=True)
    return [x[2] for x in scored[:limit]]


def select_article(store, concept: dict) -> str | None:
    """兼容旧调用：取候选里的第一条（仅用于给题面生成提供「事实边界」）。"""
    cands = article_candidates(store, concept, limit=1)
    return cands[0] if cands else None


def generate_question(
    combo,
    concepts,
    articles,
    cache: JsonCache,
    use_llm: bool,
    strict: bool = False,
    issues: list[str] | None = None,
) -> tuple[str, list[str]]:
    cid, hop, scenario_name, chain, intent = combo
    expected = articles
    sources = "；".join(f"{articles[a]['citation']}：{articles[a]['content']}" for a in expected)
    user = (
        f"纠纷场景：{scenario_name}\n用户诉求：{intent}\n"
        f"回答该问题会用到这些法条（仅用于让你知道事实边界，**不要引用原文**）：\n{sources}\n\n"
        "请写一句用户会真实说出口的提问，不超过 60 个字。"
    )
    if strict and issues:
        user += "\n\n必须点到的争点（逐条都要在提问里点到，不得遗漏）：\n" + "\n".join(
            f"{i + 1}. {t}" for i, t in enumerate(issues)
        )
    if not use_llm:
        return f"{intent}？", ["template"]
    system = QUESTION_SYSTEM_STRICT if strict else QUESTION_SYSTEM
    key = cache.key("question", QUESTION_VERSION if strict else "v1", cid, user)
    cached = cache.get(key)
    if cached:
        return cached["question"], cached["notes"]
    notes = []
    question = ""
    for attempt in range(2):
        q = get_llm().try_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            default="",
            temperature=0.0 if attempt == 0 else 0.7,
            max_tokens=140,
        ).strip().strip("“”\"")
        if not q:
            notes.append("llm_failed")
            break
        worst = max(bigram_overlap(q, articles[a]["content"]) for a in expected)
        if worst <= LEAK_THRESHOLD:
            question = q
            notes.append(f"leak={worst:.3f}")
            break
        notes.append(f"regen(leak={worst:.3f})")
    if not question:
        question = f"{intent}？"
        notes.append("fallback_template")
    cache.set(key, {"question": question, "notes": notes})
    return question, notes


def judge_necessity(question: str, concept: dict, cache: JsonCache, use_llm: bool, salt: str) -> int:
    """返回 1/0；无 LLM 时返回 -1 表示未判。"""
    if not use_llm:
        return -1
    user = f"问题：{question}\n环节：{concept['name']}（{concept['id']}）"
    key = cache.key("necessity", salt, concept["id"], user)
    cached = cache.get(key)
    if cached is not None:
        return int(cached["necessary"])
    res = get_llm().try_json(
        [{"role": "system", "content": NECESSITY_SYSTEM}, {"role": "user", "content": user}],
        default=None,
        temperature=0.2 if salt == "a" else 0.5,
        max_tokens=120,
    )
    val = 1 if isinstance(res, dict) and res.get("necessary") else 0
    cache.set(key, {"necessary": bool(val), "raw": res})
    return val


def judge_article_necessity(question: str, article: dict, cache: JsonCache, use_llm: bool, salt: str) -> int:
    """按条文内容判必要性（1/0）；三次独立判读由调用方取多数。"""
    if not use_llm:
        return -1
    user = f"问题：{question}\n法条：{article['citation']}\n{article['content']}"
    key = cache.key("art_necessity", salt, article["article_id"], user)
    cached = cache.get(key)
    if cached is not None:
        return int(cached["necessary"])
    res = get_llm().try_json(
        [{"role": "system", "content": ARTICLE_NECESSITY_SYSTEM}, {"role": "user", "content": user}],
        default=None,
        temperature={"a": 0.0, "b": 0.4, "c": 0.7}[salt],
        max_tokens=140,
    )
    val = 1 if isinstance(res, dict) and res.get("necessary") else 0
    cache.set(key, {"necessary": bool(val), "raw": res})
    return val


def pass_b(question: str, candidates: list[str], articles: dict, cache: JsonCache, use_llm: bool, salt: str) -> dict[int, int]:
    if not use_llm:
        return {}
    listing = "\n".join(f"[{i + 1}] {articles[a]['citation']}\n{articles[a]['content']}" for i, a in enumerate(candidates))
    user = f"问题：{question}\n\n候选法条：\n{listing}"
    key = cache.key("passb", salt, question, listing)
    cached = cache.get(key)
    if cached is not None:
        return {i: int(v) for i, v in cached["labels"].items()}
    res = get_llm().try_json(
        [{"role": "system", "content": PASSB_SYSTEM}, {"role": "user", "content": user}],
        default={"necessary": []},
        temperature=0.0 if salt == "a" else 0.6,
        max_tokens=200,
    )
    nums = res.get("necessary", []) if isinstance(res, dict) else []
    labels = {i: 1 if (i + 1) in nums else 0 for i in range(len(candidates))}
    cache.set(key, {"labels": {str(k): v for k, v in labels.items()}, "raw": res})
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="只用模板题面，跳过 LLM 标注（会如实标注 κ 为空")
    ap.add_argument("--only", nargs="+", default=None, help="只重建这些 id（题面用 v2 严格版），并与既有 gold_60.jsonl 合并")
    args = ap.parse_args()
    strict_ids = {x.lower() for x in (args.only or [])}
    # 全量重建也一律用 v2 严格题面：v1 题面常常只问一半争点，会让「按条文判必要性」整片失败，
    # 那是题面写窄了，不是条文多余。
    strict_all = not strict_ids

    use_llm = not args.no_llm
    if use_llm:
        from config import settings

        if not settings.llm_enabled:
            log.warning("未配置 LLM_API_KEY，自动降级为 --no-llm")
            use_llm = False
        else:
            from common.llm import llm_probe

            if not llm_probe():
                log.warning("LLM 鉴权失败/不可用，降级为 deterministic（κ 将标记为不可用，而不是伪造）")
                use_llm = False

    articles = {a["article_id"]: a for a in read_jsonl("data/extracted/articles.jsonl")}
    concepts = {c["id"]: c for c in read_jsonl("data/lexicon/concepts.jsonl")}
    scenarios = {s["id"]: s for s in read_jsonl("data/lexicon/scenarios.jsonl")}
    requires = {(r["source"], r["target"]) for r in read_jsonl("data/lexicon/requires.jsonl")}
    store = get_graph_store()
    cache = JsonCache(Path("data/gold/llm_cache.json"))
    rng = random.Random(SEED)

    try:
        from common.embedding_model import get_reranker

        reranker = get_reranker()
        log.info("精排模型已加载，用于条文选择")
    except Exception as exc:  # 模型不可用时退回候选第一条，不伪造
        log.warning("精排模型不可用（%s），条文选择退回候选第一条", exc)
        reranker = None

    def build_one(combo) -> dict:
        cid, hop, sid, chain, intent = combo
        scenario = scenarios[sid]
        chain_concepts = [concepts[c] for c in chain]
        cands_map = {c["id"]: article_candidates(store, c, CANDIDATE_LIMIT) for c in chain_concepts}
        for c in chain_concepts:
            if not cands_map[c["id"]]:
                raise RuntimeError(f"{cid}: 概念 {c['id']} 找不到可用条文")
        seed: list[str] = []
        for c in chain_concepts:
            aid = cands_map[c["id"]][0]
            if aid not in seed:
                seed.append(aid)
        laws = sorted({articles[a]["law_name"] for a in seed})

        # 干扰项：同法律随机条文，排除本链任一概念的全部候选（候选可能是真正需要的条文）
        seen = {a for v in cands_map.values() for a in v}
        pool = [
            a for a, d in articles.items()
            if d["law_name"] in laws and a not in seen
        ]
        distractors = rng.sample(pool, min(DISTRACTORS, len(pool)))

        issues = []
        for c in chain_concepts:
            gist = re.split(r"[。；]", articles[cands_map[c["id"]][0]]["content"])[0][:26]
            issues.append(f"{c['name']}（{gist}）")

        question, notes = generate_question(
            (cid, hop, scenario["name"], chain, intent),
            chain_concepts,
            {a: articles[a] for a in seed},
            cache,
            use_llm,
            strict=strict_all or cid in strict_ids,
            issues=issues,
        )

        # 条文选择：先按「概念-条文」候选收窄，再用精排模型按**题面相关性**挑一条。
        # 选择过程只看相关性，不看必要性判定；必要性判定仍独立跑三遍做审计。
        chosen_map: dict[str, str] = {}
        flat = [(c["id"], aid) for c in chain_concepts for aid in cands_map[c["id"]]]
        if reranker is not None and flat:
            docs = [articles[aid]["content"] for _, aid in flat]
            with RERANK_LOCK:  # 同一模型实例并发推理不稳，串行化
                scores = [float(s) for s in reranker.score(question, docs)]
            best: dict[str, tuple[float, str]] = {}
            for (cid_, aid), s in zip(flat, scores):
                if cid_ not in best or s > best[cid_][0]:
                    best[cid_] = (s, aid)
            for c in chain_concepts:
                chosen_map[c["id"]] = best[c["id"]][1]
        else:
            for c in chain_concepts:
                chosen_map[c["id"]] = cands_map[c["id"]][0]
        expected: list[str] = []
        for a in chosen_map.values():
            if a not in expected:
                expected.append(a)
        leakage = max(bigram_overlap(question, articles[a]["content"]) for a in expected)

        structural = all((chain[i], chain[i + 1]) in requires or (chain[i + 1], chain[i]) in requires for i in range(len(chain) - 1))
        # 概念层必要性：每个「跳」（概念环节）是否是回答所必需 —— 清单里「逐跳必要性」的口径。
        con_votes = {
            c["id"]: [judge_necessity(question, c, cache, use_llm, s) for s in ("a", "b", "c")]
            for c in chain_concepts
        }
        concept_need = {
            cid_: (sum(1 for v in vs if v == 1) >= 2 if use_llm else structural)
            for cid_, vs in con_votes.items()
        }
        # 条文层必要性：更严的补充审计（概念对得上、条文未必对得上），只报数不强求通过。
        art_votes = {
            aid: [judge_article_necessity(question, articles[aid], cache, use_llm, s) for s in ("a", "b", "c")]
            for aid in expected
        }
        article_need = {
            aid: (sum(1 for v in vs if v == 1) >= 2 if use_llm else structural)
            for aid, vs in art_votes.items()
        }
        needed_ok = all(concept_need.values())

        candidates = expected + distractors
        order = list(candidates)
        rng.shuffle(order)
        labels_a = [1 if a in expected else 0 for a in order]
        b1 = pass_b(question, order, articles, cache, use_llm, "a")
        b2 = pass_b(question, order, articles, cache, use_llm, "b")
        labels_b1 = [b1.get(i, 0) for i in range(len(order))]
        labels_b2 = [b2.get(i, 0) for i in range(len(order))]

        return {
            "id": cid,
            "hop_type": hop,
            "scenario_id": sid,
            "scenario_name": scenario["name"],
            "question": question,
            "intent": intent,
            "concepts": [{"id": c["id"], "name": c["name"]} for c in chain_concepts],
            "concept_articles": chosen_map,
            "expected_articles": expected,
            "article_citations": [articles[a]["citation"] for a in expected],
            "expected_path": [f"sc:{sid}", *chain, *expected],
            "laws": laws,
            "hops": len(expected),
            "annotation": {
                "candidates": order,
                "pass_a": labels_a,
                "pass_b_run1": labels_b1,
                "pass_b_run2": labels_b2,
            },
            "validation": {
                "leakage_max": round(leakage, 4),
                "leakage_pass": leakage <= LEAK_THRESHOLD,
                "structural_ok": structural,
                "necessity_concept_votes": con_votes,
                "necessity_by_concept": concept_need,
                "necessity_article_votes": art_votes,
                "necessity_by_article": article_need,
                "necessity_pass": needed_ok,
                "necessity_article_pass": all(article_need.values()),
                "question_version": QUESTION_VERSION if (strict_all or cid in strict_ids) else "v1",
                "question_notes": notes,
            },
        }

    rows = []
    combos = [c for c in COMBOS if not strict_ids or c[0] in strict_ids]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(build_one, c): c for c in combos}
        for fut in as_completed(futures):
            combo = futures[fut]
            try:
                rows.append(fut.result())
                log.info("完成 %s", combo[0])
            except Exception as exc:
                log.error("构建失败 %s: %s", combo[0], exc)

    rows.sort(key=lambda r: r["id"])
    if strict_ids:
        merged = {r["id"]: r for r in read_jsonl("data/gold/gold_60.jsonl")}
        merged.update({r["id"]: r for r in rows})
        rows = [merged[k] for k in sorted(merged)]
        log.info("子集重建 %d 条，合并后共 %d 条", len(combos), len(rows))

    all_a, all_b1, all_b2 = [], [], []
    for r in rows:
        all_a += r["annotation"]["pass_a"]
        all_b1 += r["annotation"]["pass_b_run1"]
        all_b2 += r["annotation"]["pass_b_run2"]

    kappa_ab1 = cohen_kappa(all_a, all_b1) if use_llm else float("nan")
    kappa_ab2 = cohen_kappa(all_a, all_b2) if use_llm else float("nan")
    kappa_stability = cohen_kappa(all_b1, all_b2) if use_llm else float("nan")
    leak_fail = sum(1 for r in rows if not r["validation"]["leakage_pass"])
    need_fail = sum(1 for r in rows if not r["validation"]["necessity_pass"])
    art_need_fail = sum(1 for r in rows if not r["validation"]["necessity_article_pass"])
    report = {
        "total": len(rows),
        "by_hop": {t: sum(1 for r in rows if r["hop_type"] == t) for t in ("two_hop", "three_hop", "cross_law")},
        "leakage_fail": leak_fail,
        "leakage_fail_rate": round(leak_fail / max(len(rows), 1), 4),
        "necessity_fail": need_fail,
        "necessity_article_fail": art_need_fail,
        "kappa_pass_a_vs_b1": round(kappa_ab1, 4) if kappa_ab1 == kappa_ab1 else None,
        "kappa_pass_a_vs_b2": round(kappa_ab2, 4) if kappa_ab2 == kappa_ab2 else None,
        "kappa_pass_b_stability": round(kappa_stability, 4) if kappa_stability == kappa_stability else None,
        "llm_used": use_llm,
        "mean_articles": round(sum(len(r["expected_articles"]) for r in rows) / max(len(rows), 1), 2),
        "laws_covered": sorted({l for r in rows for l in r["laws"]}),
    }
    write_jsonl("data/gold/gold_60.jsonl", rows)
    write_json("data/gold/gold_report.json", report)
    log.info("金标 %s", report)


if __name__ == "__main__":
    main()
