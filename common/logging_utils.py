"""统一日志格式，所有阶段脚本共用。"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "lexweave") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        from config import settings

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("lexweave")
        root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    if name.startswith("lexweave"):
        return logging.getLogger(name)
    return logging.getLogger(f"lexweave.{name}")
