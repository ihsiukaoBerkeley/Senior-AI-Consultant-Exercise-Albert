"""Small on-disk JSON cache for model outputs, keyed by content hashes.

Keys always include the article hash, the prompt version and the model, so
an edited article, a changed prompt or another model never reuses a stale
answer. Recompiles of an unchanged KB, a second KB set that shares articles,
and the change tests therefore cost no repeat calls.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from ..config import POLICY_DIR

CACHE_DIR = POLICY_DIR / "_cache"


def key(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def _path(kind: str, k: str, cache_dir: Optional[Path]) -> Path:
    return Path(cache_dir or CACHE_DIR) / kind / f"{k}.json"


def get(kind: str, k: str, cache_dir: Optional[Path] = None) -> Optional[dict]:
    p = _path(kind, k, cache_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def put(kind: str, k: str, value: dict, cache_dir: Optional[Path] = None) -> None:
    p = _path(kind, k, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
