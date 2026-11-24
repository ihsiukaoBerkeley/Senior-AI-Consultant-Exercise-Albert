"""KB watcher (code only): which articles were added, changed or removed since
the last compiled policy version.

A KB set is a named list of folders. Later folders override earlier article
ids, and `remove` drops article ids (used by the KB-mutation change tests):

* "base"        -> data/knowledge-base/
* "with-kb013"  -> data/knowledge-base/ + data/kb-additions/

Change detection compares the sha256 of every article file with the
`kb_hashes` of the previous `PolicyTable` (normally the live version).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from ..config import KB_ADDITIONS_DIR, KB_DIR
from ..kb import Article, load_kb
from ..models import PolicyTable

KB_SETS: dict[str, tuple[Path, ...]] = {
    "base": (KB_DIR,),
    "with-kb013": (KB_DIR, KB_ADDITIONS_DIR),
}


@dataclass(frozen=True)
class KBSource:
    """A resolved KB set: a name (used for runtime/policy/<name>/), folders, removed ids."""

    name: str
    dirs: tuple[Path, ...]
    remove: tuple[str, ...] = ()
    allow_override: bool = False  # later folders may replace an id (KB change tests only)


KBSetLike = Union[str, KBSource, Sequence[Union[str, Path]]]


def resolve_source(kb_set: KBSetLike, *, name: Optional[str] = None, remove: Iterable[str] = ()) -> KBSource:
    """Turn a KB set name, a KBSource or a list of folders into a KBSource."""
    if isinstance(kb_set, KBSource):
        if name or remove:
            return KBSource(name or kb_set.name, kb_set.dirs, tuple(kb_set.remove) + tuple(remove),
                            kb_set.allow_override)
        return kb_set
    if isinstance(kb_set, str):
        if kb_set not in KB_SETS:
            raise ValueError(f"unknown KB set {kb_set!r}; known sets: {sorted(KB_SETS)} (or pass a list of folders)")
        return KBSource(name or kb_set, KB_SETS[kb_set], tuple(remove))
    dirs = tuple(Path(d) for d in kb_set)
    if not dirs:
        raise ValueError("empty folder list")
    return KBSource(name or "custom", dirs, tuple(remove))


def load_articles(source: KBSetLike) -> list[Article]:
    """Load every article of a KB set (Active and inactive), sorted by id."""
    src = resolve_source(source)
    for d in src.dirs:
        if not Path(d).is_dir():
            raise FileNotFoundError(f"KB folder not found: {d}")
    return [a for a in load_kb(src.dirs, allow_override=src.allow_override) if a.article_id not in set(src.remove)]


@dataclass
class KBChanges:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)  # current article_id -> sha256

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.changed or self.removed)

    @property
    def touched(self) -> list[str]:
        """Articles that need (re)extraction or whose rules disappear."""
        return sorted(set(self.added) | set(self.changed) | set(self.removed))

    def to_dict(self) -> dict:
        return {
            "added": self.added,
            "changed": self.changed,
            "removed": self.removed,
            "unchanged": self.unchanged,
        }

    def summary(self) -> str:
        if not self.has_changes:
            return f"no KB changes ({len(self.unchanged)} articles unchanged)"
        parts = []
        for label, ids in (("added", self.added), ("changed", self.changed), ("removed", self.removed)):
            if ids:
                parts.append(f"{label}: {', '.join(ids)}")
        parts.append(f"unchanged: {len(self.unchanged)}")
        return "; ".join(parts)


def detect_changes(
    articles: list[Article], previous: Union[PolicyTable, dict[str, str], None]
) -> KBChanges:
    """Compare current article hashes with a previous table (or a hash dict)."""
    old = previous.kb_hashes if isinstance(previous, PolicyTable) else dict(previous or {})
    cur = {a.article_id: a.sha256 for a in articles}
    ch = KBChanges(hashes=cur)
    for aid in sorted(cur):
        if aid not in old:
            ch.added.append(aid)
        elif old[aid] != cur[aid]:
            ch.changed.append(aid)
        else:
            ch.unchanged.append(aid)
    ch.removed = sorted(set(old) - set(cur))
    return ch


def watch(kb_set: KBSetLike, previous: Optional[PolicyTable] = None, *, root: Optional[Path] = None) -> KBChanges:
    """Changes in a KB set since its live policy version (or since `previous`)."""
    src = resolve_source(kb_set)
    if previous is None:
        from .compiler import load_live  # late import: compiler imports this module

        previous = load_live(src.name, root=root)
    return detect_changes(load_articles(src), previous)
