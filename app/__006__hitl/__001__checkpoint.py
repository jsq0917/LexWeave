"""人在环的挂起点：每个待复核任务一个 JSON 文件，落盘即「可恢复」。

用文件而不是数据库，是因为挂起点本质是「一次任务快照」：体积大（含检索结果）、
不需要查询、需要人能直接打开看。目录结构保证进程重启后按 task_id 找得回来。
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.logging_utils import get_logger  # noqa: E402

log = get_logger("checkpoint")


@dataclass
class Checkpoint:
    task_id: str
    query: str
    mode: str
    status: str  # awaiting_review | completed
    reason: str
    top1_similarity: float
    coverage: float
    retrieval: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None
    resolution: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class CheckpointStore:
    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            from config import settings

            root = settings.checkpoint_path
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        return self.root / f"{task_id}.json"

    def save(self, ckpt: Checkpoint) -> Path:
        ckpt.updated_at = time.time()
        path = self._path(ckpt.task_id)
        path.write_text(json.dumps(ckpt.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    def load(self, task_id: str) -> Checkpoint | None:
        path = self._path(task_id)
        if not path.exists():
            return None
        blob = json.loads(path.read_text(encoding="utf-8"))
        return Checkpoint(**blob)

    def list(self, status: str | None = None) -> list[Checkpoint]:
        out = []
        for path in sorted(self.root.glob("*.json")):
            try:
                ckpt = Checkpoint(**json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
            if status is None or ckpt.status == status:
                out.append(ckpt)
        return out
