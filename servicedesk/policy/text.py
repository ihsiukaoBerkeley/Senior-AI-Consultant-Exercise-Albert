"""Text helpers shared by the extractor, checker and diff (code only)."""
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the and or of to in on for with is are be by as at it this that from when if do does "
    "not never no don t must should may can any all only their its his her user users s "
    "agent agents service desk".split()
)
# Verbs that policy texts use interchangeably; mapped to one form so a reworded prohibition still matches.
_SYNONYMS = {
    "request": "ask", "requesting": "ask", "asking": "ask",
    "keep": "store", "retain": "store", "save": "store", "storing": "store",
    "advise": "tell", "instruct": "tell", "recommend": "tell",
    "record": "log", "logging": "log",
    "place": "put", "include": "put",
}


def normalize(text: str) -> str:
    """Whitespace-normalised text for verbatim comparison.

    Collapses every run of whitespace (including CRLF line breaks), unifies
    curly quotes and dashes, and drops markdown code/emphasis markers, so a
    quote copied without the backticks around `Access` still counts as
    verbatim. Case and wording are kept: a paraphrase never matches.
    """
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", "-")
    t = t.replace("`", "").replace("*", "")
    return _WS.sub(" ", t).strip()


def quote_in_article(quote: str, article_text: str) -> bool:
    """True if the quote appears word for word in the article (whitespace-normalised).

    A quote stitched from pieces with an ellipsis counts only if every piece
    is itself verbatim and the pieces appear in order.
    """
    art = normalize(article_text)
    q = normalize(quote)
    if not q:
        return False
    pieces = [p for p in _ELLIPSIS.split(q) if p.strip()]
    if not pieces:
        return False
    pos = 0
    for piece in pieces:
        piece = piece.strip().strip(".,;:")  # trailing punctuation around a cut is fine
        if not piece:
            continue
        found = art.find(piece, pos)
        if found < 0:
            return False
        pos = found + len(piece)
    return True


def tokens(text: str) -> set[str]:
    """Content words, lower-cased, crude plural stripping; for fuzzy matching."""
    out = set()
    for w in _WORD.findall(text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(_SYNONYMS.get(w, w))
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def containment(a: set[str], b: set[str]) -> float:
    """Share of a's words that also appear in b."""
    if not a:
        return 1.0
    return len(a & b) / len(a)


def norm_entry(text: str) -> str:
    """Normalised list entry (required field, must-never item) for set comparison."""
    t = normalize(text).lower().strip(" .;:")
    t = re.sub(r"^(the|a|an)\s+", "", t)
    return t
