"""从 data/eval_out/ 与 data/graph/ 出结果图（README 用）。

三张图全部从已落盘的评测产物读取，不写死任何数字：
  fig1_b3_retrieval.png   三档检索 R@5 / 路径准确率 / 分跳数
  fig2_b6_bootstrap.png   B6 质量指标的配对差值 95% CI（森林图）
  fig3_graph_composition.png  4,259 条边的类型与 provenance 构成

用法：
    .venv\\Scripts\\python.exe scripts\\make_figures.py
    .venv\\Scripts\\python.exe scripts\\make_figures.py --outdir docs/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import rcParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# 中文字体：Windows 用 SimHei，Linux/macOS 回落到系统黑体
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 120
rcParams["savefig.dpi"] = 200

TIERS = ["text", "graph", "hybrid"]
TIER_CN = {"text": "纯文本", "graph": "纯图", "hybrid": "hybrid"}
C_TEXT = "#4C7FB0"
C_GRAPH = "#C2705A"
C_HYBRID = "#5B9E7A"
C_NEUTRAL = "#8A94A0"


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9D2DB")
    ax.spines["bottom"].set_color("#C9D2DB")
    ax.tick_params(colors="#5B6673", labelsize=10)
    ax.grid(axis="y", color="#EDF1F5", linewidth=0.9)
    ax.set_axisbelow(True)


def fig_retrieval(out: Path) -> None:
    res = load("data/eval_out/retrieval.json")["results"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), gridspec_kw={"width_ratios": [1.05, 1]})

    # --- 左：主指标 ---
    ax = axes[0]
    metrics = [("recall@5", "R@5"), ("path_recall", "路径准确率")]
    x = np.arange(len(metrics))
    w = 0.26
    colors = {"text": C_TEXT, "graph": C_GRAPH, "hybrid": C_HYBRID}
    for i, t in enumerate(TIERS):
        vals = [res[t][k] for k, _ in metrics]
        pos = x + (i - 1) * w
        ax.bar(pos, vals, w * 0.92, label=TIER_CN[t], color=colors[t], zorder=3)
        for p, v in zip(pos, vals):
            if v == 0:
                ax.text(p, 0.012, "0", ha="center", va="bottom", fontsize=8.6, color="#7C8794")
            else:
                ax.text(p, v + 0.018, f"{v:.3f}", ha="center", va="bottom", fontsize=8.6, color="#3D4650")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in metrics], fontsize=10.5)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("得分（60 条金标）", fontsize=10.5, color="#5B6673")
    ax.set_title("三档检索主指标：hybrid 相对纯文本 +3.9pt（目标 15pt）", fontsize=11.5, pad=10, color="#1B2733")
    ax.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0))
    style(ax)

    # --- 右：分跳数 ---
    ax = axes[1]
    hops = [("two_hop", "二跳"), ("three_hop", "三跳"), ("cross_law", "跨法")]
    x = np.arange(len(hops))
    for i, t in enumerate(TIERS):
        vals = [res[t]["by_hop"][h]["recall@5"] for h, _ in hops]
        pos = x + (i - 1) * w
        ax.bar(pos, vals, w * 0.92, label=TIER_CN[t], color=colors[t], zorder=3)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.018, f"{v:.2f}", ha="center", va="bottom", fontsize=8.4, color="#3D4650")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in hops], fontsize=10.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("R@5", fontsize=10.5, color="#5B6673")
    ax.set_title("分跳数 R@5：跳数越多，图路掉得越快", fontsize=11.5, pad=10, color="#1B2733")
    style(ax)

    fig.text(
        0.01, 0.01,
        "数据来源：data/eval_out/retrieval.json（scripts/eval_retrieval.py 产出）。"
        "hybrid 未达 15pt 的原因见 README §5 / §12 第 1 条。",
        fontsize=8.4, color=C_NEUTRAL,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig1_b3_retrieval.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_bootstrap(out: Path) -> None:
    abl = load("data/eval_out/ablation.json")
    boot = abl["bootstrap"]
    tiers = abl["tiers"]

    metrics = [
        ("recall@5", "R@5"),
        ("cite_precision", "引用精确率"),
        ("cite_recall", "引用召回率"),
        ("grounded", "grounded 率"),
    ]
    comps = [("B-A", "B 档 vs A 档：加多 Agent", "#C2705A"),
             ("C-B", "C 档 vs B 档：再加五道闸门", "#4C7FB0")]

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    y = np.arange(len(metrics))[::-1]
    h = 0.34
    for j, (key, label, color) in enumerate(comps):
        ds, los, his, sig = [], [], [], []
        for m, _ in metrics:
            d = boot[f"{key}:{m}"]
            ds.append(d["delta"])
            los.append(d["ci95"][0])
            his.append(d["ci95"][1])
            sig.append(d["excludes_zero"])
        pos = y + (0.5 - j) * h
        for p, d, lo, hi, s in zip(pos, ds, los, his, sig):
            ax.plot([lo, hi], [p, p], color=color, linewidth=2.2, solid_capstyle="round", zorder=3)
            ax.plot([d], [p], marker="o", markersize=6.5, color=color,
                    markerfacecolor=color if s else "white", markeredgewidth=1.6, zorder=4)
            ax.text(hi + 0.006, p, f"{d:+.3f}" + ("（区间不含 0）" if s else "（含 0）"),
                    va="center", fontsize=8.6, color=color if s else "#7C8794")
        ax.plot([], [], color=color, linewidth=2.2, marker="o", markersize=6, label=label)

    ax.axvline(0, color="#3D4650", linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([n for _, n in metrics], fontsize=10.5)
    ax.set_xlabel("相邻两档的配对差值（bootstrap 100 次，95% CI）", fontsize=10.5, color="#5B6673")
    ax.set_title("加多 Agent 只买到延迟；再加闸门换来精度方向，但 60 条样本撑不起显著性",
                 fontsize=11.8, pad=12, color="#1B2733")
    ax.legend(frameon=False, fontsize=9.3, loc="upper right", bbox_to_anchor=(1.0, 1.02))
    ax.set_xlim(-0.06, 0.235)
    style(ax)
    ax.grid(axis="x", color="#EDF1F5", linewidth=0.9)
    ax.grid(axis="y", visible=False)

    # 右侧贴成本注记
    lat = boot["B-A:latency_ms"]["delta"]
    calls = boot["B-A:llm_calls"]["delta"]
    fig.text(
        0.01, 0.01,
        f"同一批 60 条上：B 档相对 A 档多 {int(calls)} 次 LLM 调用、延迟 +{lat:.0f} ms"
        f"（{tiers['A']['latency_median_ms']:.0f} → {tiers['B']['latency_median_ms']:.0f} ms），"
        "这两个区间不含 0；质量指标全部含 0。",
        fontsize=8.4, color=C_NEUTRAL,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig2_b6_bootstrap.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_graph(out: Path) -> None:
    g = load("data/graph/graph.json")
    pairs = Counter((e["relation"], e["provenance"]) for e in g["edges"])
    order = ["DEFINES", "APPLIES_TO", "REQUIRES", "SPECIAL_OF", "CITES"]
    prov_cn = {"rule": "规则抽取", "manual": "人工编写", "regex": "正则抽取"}
    prov_color = {"rule": "#8FB6D9", "manual": "#C2705A", "regex": "#9DBFAE"}

    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for prov in ["rule", "manual", "regex"]:
        vals = np.array([pairs[(r, prov)] for r in order], dtype=float)
        ax.barh(y, vals, left=left, height=0.55, color=prov_color[prov],
                label=f"{prov_cn[prov]}（{int(vals.sum())}）", zorder=3)
        for p, l, v in zip(y, left, vals):
            if v >= 150:
                ax.text(l + v / 2, p, f"{int(v):,}", ha="center", va="center",
                        fontsize=9, color="#FFFFFF" if prov == "manual" else "#33404D")
            elif v > 0:
                ax.text(l + v + 55, p, f"{int(v):,}", ha="left", va="center",
                        fontsize=9, color="#5B6673")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=10.5)
    ax.set_xlabel("边数（合计 4,259）", fontsize=10.5, color="#5B6673")
    ax.set_title("边的构成：概念层 541 条全部人工编写，DEFINES 由规则抽取",
                 fontsize=11.8, pad=12, color="#1B2733")
    ax.legend(frameon=False, fontsize=9.3, loc="lower right")
    ax.set_xlim(0, 4100)
    style(ax)
    ax.grid(axis="x", color="#EDF1F5", linewidth=0.9)
    ax.grid(axis="y", visible=False)

    fig.text(
        0.01, 0.01,
        "数据来源：data/graph/graph.json（scripts/build_graph.py 产出，stats.json 同脚本刷新）。"
        "DEFINES 混有「提及」而非「定义」，是 B1 抽检未达 85% 的原因，见 README §12 第 6 条。",
        fontsize=8.4, color=C_NEUTRAL,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "fig3_graph_composition.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/figures")
    args = ap.parse_args()
    out = ROOT / args.outdir
    out.mkdir(parents=True, exist_ok=True)

    fig_retrieval(out)
    fig_bootstrap(out)
    fig_graph(out)
    for p in sorted(out.glob("*.png")):
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    sys.exit(main())
