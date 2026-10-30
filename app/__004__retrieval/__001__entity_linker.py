"""实体链接：查询文本 -> 图上的概念/场景节点。

两级策略，先便宜后昂贵：
1. 词典层：别名精确命中（含中文数字、口语说法），命中即 1.0，完全可解释；
2. 语义层：bge-large-zh-v1.5 编码，FAISS 近邻，阈值 0.85 —— 只处理词典漏掉的表述。
阈值不是拍脑袋：早期版本用 0.75，会把「经济补偿」误配到「经济赔偿」这类近义但法律后果不同的概念。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.__002__extract.__001__rule_extractor import AliasMatcher  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.path_utils import read_json, read_jsonl  # noqa: E402
from common.text_utils import normalize  # noqa: E402

log = get_logger("entity_link")


@dataclass
class LinkedEntity:
    id: str
    name: str
    score: float
    method: str  # exact | semantic | scenario


@dataclass
class LinkingResult:
    concepts: list[LinkedEntity] = field(default_factory=list)
    scenarios: list[LinkedEntity] = field(default_factory=list)

    @property
    def concept_ids(self) -> list[str]:
        return [c.id for c in self.concepts]


class ConceptIndex:
    """概念/场景别名向量索引（bge-large-zh-v1.5）。"""

    def __init__(self, index_dir: Path | None = None) -> None:
        from config import settings

        self.dir = index_dir or settings.index_dir
        self.faiss_index: Any = None
        self.entries: list[dict] = []

    def build(self, concepts: list[dict], scenarios: list[dict]) -> dict:
        from common.embedding_model import get_align_encoder

        entries: list[dict] = []
        texts: list[str] = []
        for c in concepts:
            for a in c["aliases"]:
                entries.append({"id": c["id"], "kind": "concept", "name": c["name"], "alias": a})
                texts.append(a)
        for s in scenarios:
            for a in s["aliases"]:
                entries.append({"id": s["id"], "kind": "scenario", "name": s["name"], "alias": a})
                texts.append(a)
        enc = get_align_encoder()
        vecs = enc.encode(texts)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        self.dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.dir / "concept.faiss"))
        (self.dir / "concept_meta.json").write_text(
            json.dumps({"entries": entries, "dim": int(vecs.shape[1])}, ensure_ascii=False), encoding="utf-8"
        )
        self.faiss_index = index
        self.entries = entries
        return {"aliases": len(entries), "dim": int(vecs.shape[1])}

    def load(self) -> "ConceptIndex":
        meta = self.dir / "concept_meta.json"
        if not meta.exists():
            raise FileNotFoundError("概念索引不存在，先运行 scripts/build_index.py")
        blob = json.loads(meta.read_text(encoding="utf-8"))
        self.entries = blob["entries"]
        self.faiss_index = faiss.read_index(str(self.dir / "concept.faiss"))
        return self

    def search(self, query: str, k: int = 8) -> list[tuple[dict, float]]:
        from common.embedding_model import get_align_encoder

        vec = get_align_encoder().encode([query])
        scores, idx = self.faiss_index.search(vec, k)
        out = []
        for s, i in zip(scores[0], idx[0]):
            if 0 <= i < len(self.entries):
                out.append((self.entries[i], float(s)))
        return out


class EntityLinker:
    def __init__(self) -> None:
        from config import settings

        self.threshold = settings.align_threshold
        self.concepts = {c["id"]: c for c in read_jsonl(settings.lexicon_dir / "concepts.jsonl")}
        self.scenarios = {s["id"]: s for s in read_jsonl(settings.lexicon_dir / "scenarios.jsonl")}
        self.alias_index = read_json(settings.lexicon_dir / "alias_index.json")
        self.alias_matcher = AliasMatcher(self.alias_index)
        self.concept_index = ConceptIndex().load()

    def link(self, query: str, max_concepts: int = 6, max_scenarios: int = 3) -> LinkingResult:
        res = LinkingResult()
        norm = normalize(query)
        seen_c: dict[str, LinkedEntity] = {}
        seen_s: dict[str, LinkedEntity] = {}

        for m in self.alias_matcher.find(query):
            for entry in self.alias_index.get(m.alias, []):
                base = 1.0 if m.alias == entry["name"] else 0.95
                target = seen_c if entry["kind"] == "concept" else seen_s
                prev = target.get(entry["id"])
                if prev is None or base > prev.score:
                    target[entry["id"]] = LinkedEntity(
                        id=entry["id"],
                        name=entry["name"],
                        score=base,
                        method="scenario" if entry["kind"] == "scenario" else "exact",
                    )

        # 语义层补齐：只取高于阈值的，且同一 id 只保留最高分
        for entry, sim in self.concept_index.search(query, k=12):
            if sim < self.threshold:
                continue
            target = seen_c if entry["kind"] == "concept" else seen_s
            target.setdefault(
                entry["id"],
                LinkedEntity(id=entry["id"], name=entry["name"], score=round(sim, 4), method="semantic"),
            )

        res.concepts = sorted(seen_c.values(), key=lambda e: -e.score)[:max_concepts]
        res.scenarios = sorted(seen_s.values(), key=lambda e: -e.score)[:max_scenarios]
        return res
