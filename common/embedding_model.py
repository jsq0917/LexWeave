"""向量模型与重排模型的统一入口（懒加载 + 本地目录优先）。

三个模型各司其职：
* ``bge-m3``      -> 文本路 dense 召回（多语言、长文本，支持 8192 token）
* ``bge-large-zh-v1.5`` -> 图路「文章实体 -> 概念节点」对齐（短文本语义匹配）
* ``bge-reranker-v2-m3`` -> 召回后的精排

设备选择：显式 ``cuda`` > ``EMBEDDING_DEVICE`` > 有卡用卡 > CPU。
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Sequence

import numpy as np

from config import settings
from common.logging_utils import get_logger

log = get_logger("embedding")

# 已知的本地模型目录（按优先级从环境变量到内置路径回落）
_HF_HUB = os.path.expanduser(r"~\.cache\huggingface\hub")

_LOCAL_CANDIDATES: dict[str, list[str]] = {
    "BAAI/bge-large-zh-v1.5": [
        os.getenv("ALIGN_MODEL_DIR", ""),
        r"E:\AI大模型开发\中医药知识图谱\04-模型\bge-large-zh-v1.5",
        r"E:\AI大模型开发\中医药知识图谱\04-模型\bge-large-zh",
    ],
    # 本机的 bge-m3 是「扁平模型目录」（直接放在 hub 缓存目录下，没有 snapshots/blobs 结构），
    # HuggingFace 无法按仓库名解析，必须显式传本地目录。
    "BAAI/bge-m3": [os.getenv("EMBEDDING_MODEL_DIR", ""), os.path.join(_HF_HUB, "models--BAAI--bge-m3")],
    "BAAI/bge-reranker-v2-m3": [
        os.getenv("RERANK_MODEL_DIR", ""),
        r"E:\project\模型\bge-reranker-v2-m3",
        r"E:\project\LawAnswer\models\bge-reranker-v2-m3",
    ],
}


def resolve_model_path(name: str) -> str:
    """本地目录优先，其次退回 HuggingFace 仓库名（走本地缓存）。"""
    for cand in _LOCAL_CANDIDATES.get(name, []):
        if cand and Path(cand).exists():
            return cand
    return name


def _pick_device() -> str:
    dev = (settings.embedding_device or "auto").lower()
    if dev in {"cpu", "cuda"}:
        if dev == "cuda":
            import torch

            if not torch.cuda.is_available():
                log.warning("EMBEDDING_DEVICE=cuda 但未检测到可用 GPU，回落 CPU")
                return "cpu"
        return dev
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


_encoder_lock = Lock()
_dense_cache: dict[str, "DenseEncoder"] = {}


class DenseEncoder:
    """sentence-transformers 封装，输出 L2 归一化后的 float32 向量。"""

    def __init__(self, model_name: str, device: str | None = None, max_seq_length: int = 512) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.path = resolve_model_path(model_name)
        self.device = device or _pick_device()
        self.max_seq_length = max_seq_length
        # 8GB 显卡要同时装 bge-m3 / bge-large / reranker，fp16 是必须的：
        # 否则三模型叠加会吃满显存，推理被换页拖到「看起来卡死」。
        kwargs: dict[str, object] = {}
        if self.device == "cuda":
            import torch

            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        self.model = SentenceTransformer(self.path, device=self.device, **kwargs)
        try:
            self.model.max_seq_length = max_seq_length
        except Exception:
            pass
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            list(texts),
            batch_size=batch_size or settings.embedding_batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)


def get_dense_encoder(model_name: str | None = None, max_seq_length: int = 512) -> DenseEncoder:
    name = model_name or settings.embedding_model_name
    if name not in _dense_cache:
        with _encoder_lock:
            if name not in _dense_cache:
                log.info("加载 dense 模型 %s -> %s", name, resolve_model_path(name))
                _dense_cache[name] = DenseEncoder(name, max_seq_length=max_seq_length)
    return _dense_cache[name]


def get_align_encoder() -> DenseEncoder:
    return get_dense_encoder(settings.align_model_name, max_seq_length=64)


_rerank_lock = Lock()
_reranker: "Reranker | None" = None


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or settings.rerank_model_name
        self.path = resolve_model_path(self.model_name)
        self.device = _pick_device()
        kwargs: dict[str, object] = {}
        if self.device == "cuda":
            import torch

            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        self.model = CrossEncoder(self.path, device=self.device, max_length=512, **kwargs)

    def score(self, query: str, docs: Sequence[str], batch_size: int = 32) -> list[float]:
        if not docs:
            return []
        pairs = [[query, d] for d in docs]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return [float(s) for s in np.asarray(scores).reshape(-1)]


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        with _rerank_lock:
            if _reranker is None:
                log.info("加载 rerank 模型 %s -> %s", settings.rerank_model_name, resolve_model_path(settings.rerank_model_name))
                _reranker = Reranker()
    return _reranker


def embedding_ready() -> bool:
    """探测 sentence-transformers / torch 是否可用（测试与文档用）。"""
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False
