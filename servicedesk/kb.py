"""Load knowledge-base articles and search the Active ones."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

_META = re.compile(r"^-\s*(Owner|Last reviewed|Status|Superseded by|Applies to):\s*(.+?)\s*$", re.M)
_TITLE = re.compile(r"^#\s*(KB-\d+):\s*(.+?)\s*$", re.M)

# KB naming rules (config/working_assumptions.yaml): one Markdown file per article, named
# KB-<id>-<short-title>.md, whose heading "# KB-<id>: Title" carries the same id, with the
# four metadata lines below. An id may appear only once in a KB set.
_FILE = re.compile(r"^(KB-\d{3,})-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
REQUIRED_META = ("Owner", "Last reviewed", "Status", "Applies to")
STATUSES = ("Active", "Deprecated")
NAMING_HELP = ("Each article is one file named like KB-020-short-title.md (KB-, at least three digits, then "
               "lowercase words joined by hyphens), starting with the heading '# KB-020: Title' and the lines "
               "'- Owner:', '- Last reviewed:', '- Status: Active' (or Deprecated) and '- Applies to:'.")


class KBFormatError(ValueError):
    """A KB file that breaks the naming rules; the message names the file and the fix."""
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the and or of to in on for with is are be by as at it this that from not when if do does "
    "my our your can i we you me us has have was were will should must after before any all".split()
)


@dataclass(frozen=True)
class Article:
    article_id: str
    title: str
    owner: str
    last_reviewed: str
    status: str
    applies_to: str
    superseded_by: Optional[str]
    text: str
    path: Path
    sha256: str

    @property
    def active(self) -> bool:
        return self.status.lower() == "active"


def _where(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def parse_article(path: Path) -> Article:
    path = Path(path)
    named = _FILE.match(path.name)
    if not named:
        raise KBFormatError(f"{_where(path)}: not a valid KB file name. {NAMING_HELP}")
    text = path.read_text(encoding="utf-8")
    title = _TITLE.search(text)
    if not title:
        raise KBFormatError(f"{_where(path)}: missing the heading line '# {named.group(1)}: Title'. {NAMING_HELP}")
    if title.group(1) != named.group(1):
        raise KBFormatError(f"{_where(path)}: the file name says {named.group(1)} but the heading says "
                            f"{title.group(1)}; they must match")
    meta = {k: v for k, v in _META.findall(text)}
    missing = [k for k in REQUIRED_META if not meta.get(k)]
    if missing:
        raise KBFormatError(f"{_where(path)}: missing metadata line(s): "
                            + ", ".join(f"'- {k}: ...'" for k in missing))
    status = next((s for s in STATUSES if s.lower() == meta["Status"].lower()), None)
    if status is None:
        raise KBFormatError(f"{_where(path)}: Status is {meta['Status']!r}; use Active or Deprecated")
    meta["Status"] = status
    return Article(
        article_id=title.group(1),
        title=title.group(2),
        owner=meta.get("Owner", ""),
        last_reviewed=meta.get("Last reviewed", ""),
        status=meta.get("Status", "Unknown"),
        applies_to=meta.get("Applies to", ""),
        superseded_by=meta.get("Superseded by"),
        text=text,
        path=path,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def load_kb(dirs: Iterable[Path], *, allow_override: bool = False) -> list[Article]:
    """Load every Markdown article in the given folders, checking the naming rules.

    An article id may appear only once. `allow_override` lets a later folder replace an
    earlier folder's article with the same id; only the extractor's KB change tests use
    it (data/kb-mutations). Two files with one id in the same folder are always an error.
    """
    by_id: dict[str, Article] = {}
    for d in dirs:
        for path in sorted(Path(d).glob("*.md")):
            art = parse_article(path)
            prev = by_id.get(art.article_id)
            if prev is not None and (prev.path.parent == path.parent or not allow_override):
                raise KBFormatError(f"{art.article_id} appears twice: {_where(prev.path)} and {_where(path)}. "
                                    "Each article id may be used only once; give the new article an unused id.")
            by_id[art.article_id] = art
    return sorted(by_id.values(), key=lambda a: a.article_id)


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


class KBIndex:
    """BM25 over Active articles only, so deprecated guidance can never be retrieved."""

    def __init__(self, articles: list[Article], k1: float = 1.4, b: float = 0.75):
        self.articles = [a for a in articles if a.active]
        self.k1, self.b = k1, b
        self.docs = [_tokens(f"{a.title} {a.applies_to} {a.text}") for a in self.articles]
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for doc in self.docs for t in set(doc))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(doc) for doc in self.docs]

    def search(self, query: str, k: int = 3) -> list[tuple[Article, float]]:
        q = _tokens(query)
        scored = []
        for art, tf, doc in zip(self.articles, self.tf, self.docs):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            scored.append((art, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(a, s) for a, s in scored[:k] if s > 0]
