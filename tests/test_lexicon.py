"""术语库「种子 → jsonl」一致性守卫。

为什么单独测这个：术语库是单向编译产物，但历史上出现过直接改
``data/lexicon/*.jsonl``、不回写 ``__000__lexicon_seed.py`` 的情况。结果是一重建就少
10 个概念、23 条边，而 README 里的节点/边数还停在旧构建上——别人按 README 跑那三条命令，
数字必然对不上，等于自证不可复现。这条测试把「重建后与仓库里的 jsonl 不一致」变成红灯。
"""

from __future__ import annotations

from app.__001__ingest.__001__build_lexicon import check
from common.path_utils import read_json, read_jsonl


def test_lexicon_matches_seed() -> None:
    drift = check()
    assert not drift, (
        "data/lexicon 与 __000__lexicon_seed.py 不一致（重建会丢改动）：\n  - "
        + "\n  - ".join(drift)
        + "\n请把改动回写种子后重跑 app/__001__ingest/__001__build_lexicon.py"
    )


def test_lexicon_counts_match_readme() -> None:
    """锁住 README 与徽章里的 1,799 节点 / 5,697 边的上游口径。

    概念 281 + 场景 53 + 条文 1,465 = 1,799 节点；REQUIRES 196 / SPECIAL_OF 183 /
    APPLIES_TO 198 参与 5,697 条边。改词典必须同步改 README，否则三条复现命令会对不上。
    """
    stats = read_json("data/lexicon/stats.json")
    assert stats["concepts"] == 281
    assert stats["scenarios"] == 53
    assert stats["requires_edges"] == 196
    assert stats["special_of_edges"] == 183
    assert stats["scenario_concept_edges"] == 198

    concepts = read_jsonl("data/lexicon/concepts.jsonl")
    assert len(concepts) == stats["concepts"]
    assert len({c["id"] for c in concepts}) == len(concepts)
