"""路径与 JSON 读写工具（断点续传的落盘格式统一在这里）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import orjson


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(orjson.loads(line))
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(orjson.dumps(row, option=orjson.OPT_NON_STR_KEYS).decode("utf-8"))
            f.write("\n")
            n += 1
    return n


def append_jsonl(path: Path | str, row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(orjson.dumps(row, option=orjson.OPT_NON_STR_KEYS).decode("utf-8"))
        f.write("\n")


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path | str, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_jsonl_indexed(path: Path | str, key: str = "id") -> dict[str, dict[str, Any]]:
    return {row[key]: row for row in read_jsonl(path)}
