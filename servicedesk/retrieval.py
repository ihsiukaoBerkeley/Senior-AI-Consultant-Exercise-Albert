"""Finding the knowledge-base articles for an issue: keyword, vector or hybrid search.

Three modes, chosen per run (config.RETRIEVAL_MODE, the SD_RETRIEVAL variable, or
--retrieval on the command line):

- bm25:   keyword search (kb.KBIndex), the prototype's first version.
- vector: semantic search. Each Active article is split into chunks of whole paragraphs,
          each chunk is embedded locally (fastembed: a small ONNX model, by default
          BAAI/bge-small-en-v1.5, no API key) and held in a FAISS index. An article
          scores its best chunk.
- hybrid: both, the default. Keywords catch exact terms (product names, error codes);
          vectors catch the same problem said in other words. An article's hybrid score is
          the average of its keyword score and its vector score, each scaled to 0-1.

The embedding model sits behind one small interface (Embedder: embed passages, embed a
query). Anthropic offers no embedding model, so the prototype runs an open-source one
locally to need no second API key; in production the same interface would call a hosted
embedding API (for example Azure OpenAI embeddings).

Before every run, ensure_ready() makes search ready the way the policy compile makes the
policy ready: it downloads the embedding model into runtime/models if it is missing (and
again if the files there are damaged), then embeds any Active article whose text has no
stored vector. search_status() reports the same without changing anything. Both take the
article folders the run will search (`dirs`: the live policy version's copy of its articles,
compiler.article_dirs); without them, the KB set's folders.

In every mode only Active articles are indexed, so deprecated guidance is never found.
Embeddings are cached on disk by the hash of the model name and the chunk text: a run
embeds only articles that are new or changed since the last run, the same idea as the
policy watcher. The FAISS index is rebuilt in memory at start (milliseconds at this size;
a large KB would persist it and update it by article id).
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Protocol

from . import config
from .kb import Article, KBIndex

MODES = ("hybrid", "bm25", "vector")
SEARCH_DEPTH = 50     # how many hits each search returns before they are combined
CHUNK_WORDS = 180     # paragraphs are grouped into chunks of at most this many words
# Instructions some embedding models expect in front of a search query (not the passages).
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}
_HEADER_LINE = re.compile(r"^\s*(#\s*KB-\d+:.*|-\s*(Owner|Last reviewed|Status|Superseded by|Applies to):.*)$")

# No telemetry from the model download.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class RetrievalError(RuntimeError):
    """Vector search can't start (missing package, model download failed); the message says what to do."""


def check_mode(mode: Optional[str]) -> str:
    mode = mode or config.RETRIEVAL_MODE
    if mode not in MODES:
        raise RetrievalError(f"unknown retrieval mode {mode!r}; use one of {', '.join(MODES)}")
    return mode


# --------------------------------------------------------------------------
# Chunks
# --------------------------------------------------------------------------

def article_body(art: Article) -> str:
    """The article text without its heading and metadata lines."""
    return "\n".join(line for line in art.text.splitlines() if not _HEADER_LINE.match(line)).strip()


def chunk_texts(art: Article, max_words: int = CHUNK_WORDS) -> list[str]:
    """Whole paragraphs grouped up to `max_words`, each led by the title and "Applies to" line
    so a chunk carries its article's topic. Today's articles (about 120 words) are one chunk."""
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", article_body(art)) if p.strip()]
    chunks, cur, words = [], [], 0
    for p in paras:
        n = len(p.split())
        if cur and words + n > max_words:
            chunks.append(" ".join(cur))
            cur, words = [], 0
        cur.append(p)
        words += n
    if cur:
        chunks.append(" ".join(cur))
    head = f"{art.title}. Applies to: {art.applies_to}."
    return [f"{head}\n{c}" for c in chunks] or [head]


# --------------------------------------------------------------------------
# Embedding model
# --------------------------------------------------------------------------

class Embedder(Protocol):
    name: str

    def passages(self, texts: list[str]) -> "np.ndarray": ...  # noqa: F821
    def query(self, text: str) -> "np.ndarray": ...  # noqa: F821


class FastEmbedder:
    """A local ONNX embedding model (fastembed). Loaded on first use; one per process."""

    def __init__(self, name: str, cache_dir: Path):
        self.name = name
        self.cache_dir = Path(cache_dir)
        self.prefix = QUERY_PREFIX.get(name, "")
        self._model = None
        self._lock = threading.RLock()

    def downloaded(self) -> bool:
        short = self.name.split("/")[-1]
        return any(self.cache_dir.glob(f"models--*{short}*"))

    def load(self, log=None):
        with self._lock:
            return self._load(log)

    def _create(self):
        from fastembed import TextEmbedding

        try:  # Hugging Face's "set a HF_TOKEN" hint: no token is needed for this public model
            from huggingface_hub.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
        except Exception:
            pass
        return TextEmbedding(self.name, cache_dir=str(self.cache_dir))

    def _load(self, log=None):
        if self._model is not None:
            return self._model
        try:
            import fastembed  # noqa: F401
        except ImportError:
            raise RetrievalError("vector search needs the fastembed and faiss-cpu packages: run "
                                 "pip install -r requirements.txt, or set SD_RETRIEVAL=bm25 in .env (keyword search only)") from None
        try:
            self._model = self._create()
        except Exception as first:
            if not self.downloaded():
                raise RetrievalError(_download_failed(self.name, first)) from None
            # The files are there but unusable (an interrupted download): fetch them again, once.
            if log:
                log(f"The embedding model in {_rel(self.cache_dir)} is incomplete or damaged: downloading it again...")
            import shutil

            shutil.rmtree(self.cache_dir, ignore_errors=True)
            try:
                self._model = self._create()
            except Exception as second:
                raise RetrievalError(_download_failed(self.name, second)) from None
        return self._model

    def passages(self, texts: list[str]):
        import numpy as np

        with self._lock:
            return np.asarray(list(self._load().passage_embed(texts)), dtype="float32")

    def query(self, text: str):
        import numpy as np

        with self._lock:
            return np.asarray(next(iter(self._load().query_embed(self.prefix + text))), dtype="float32")


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(config.ROOT))
    except ValueError:
        return str(path)


def _download_failed(name: str, err: Exception) -> str:
    return (f"could not download or load the embedding model {name} ({type(err).__name__}: {str(err)[:200]}). "
            f"The first run downloads it from Hugging Face (about 70 MB): check that this machine can reach "
            f"huggingface.co and *.hf.co (a proxy can be set with HTTPS_PROXY), then run again. To run without "
            f"it, set SD_RETRIEVAL=bm25 in .env (keyword search only).")


@lru_cache(maxsize=4)
def get_embedder(name: str) -> Embedder:
    return FastEmbedder(name, config.MODELS_DIR)


# --------------------------------------------------------------------------
# Embedding cache (by content hash) and the FAISS index
# --------------------------------------------------------------------------

_cache_lock = threading.Lock()


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()


class EmbeddingCache:
    """Vectors on disk, keyed by sha256(model name + text). Safe to share between KB sets."""

    def __init__(self, model_name: str, root: Optional[Path] = None):
        self.path = Path(root or config.VECTOR_DIR) / _slug(model_name) / "embeddings.npz"
        self.vectors: dict[str, "np.ndarray"] = {}  # noqa: F821
        if self.path.exists():
            import numpy as np

            try:
                with np.load(self.path, allow_pickle=False) as data:
                    self.vectors = {k: data[k] for k in data.files}
            except Exception:
                self.vectors = {}  # a damaged cache is rebuilt, never trusted

    @staticmethod
    def key(model_name: str, text: str) -> str:
        return "h" + hashlib.sha256(f"{model_name}\n{text}".encode("utf-8")).hexdigest()

    def save(self) -> None:
        import numpy as np

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp.npz")
        np.savez(tmp, **self.vectors)
        os.replace(tmp, self.path)


class VectorIndex:
    """Active articles' chunks in a FAISS inner-product index over normalised vectors (cosine)."""

    def __init__(self, articles: list[Article], embedder: Embedder, cache_root: Optional[Path] = None):
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise RetrievalError("vector search needs the fastembed and faiss-cpu packages: run "
                                 "pip install -r requirements.txt, or set SD_RETRIEVAL=bm25 in .env (keyword search only)") from None
        self.embedder = embedder
        self.articles = [a for a in articles if a.active]
        texts, self.owners = [], []
        for i, art in enumerate(self.articles):
            for chunk in chunk_texts(art):
                texts.append(chunk)
                self.owners.append(i)
        keys = [EmbeddingCache.key(embedder.name, t) for t in texts]
        with _cache_lock:
            cache = EmbeddingCache(embedder.name, cache_root)
            todo = [i for i, k in enumerate(keys) if k not in cache.vectors]
            if todo:
                vecs = embedder.passages([texts[i] for i in todo])
                for i, v in zip(todo, vecs):
                    cache.vectors[keys[i]] = v
                cache.save()
        self.embedded_now = len(todo)          # chunks embedded by this run (new or changed text)
        self.chunks = len(texts)
        if not texts:
            self.index = None
            return
        mat = np.stack([cache.vectors[k] for k in keys]).astype("float32")
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        self.index = faiss.IndexFlatIP(mat.shape[1])
        self.index.add(mat)

    def search(self, query: str, k: int = 3) -> list[tuple[Article, float]]:
        import numpy as np

        if self.index is None or not query.strip():
            return []
        q = np.asarray(self.embedder.query(query), dtype="float32").reshape(1, -1)
        q /= np.linalg.norm(q) + 1e-12
        scores, ids = self.index.search(q, min(self.chunks, max(k * 8, SEARCH_DEPTH)))
        best: dict[int, float] = {}
        for s, i in zip(scores[0], ids[0]):
            if i >= 0:
                a = self.owners[i]
                best[a] = max(best.get(a, -1.0), float(s))
        ranked = sorted(best.items(), key=lambda x: -x[1])[:k]
        return [(self.articles[a], s) for a, s in ranked]


# --------------------------------------------------------------------------
# Retriever: the one entry point the pipeline uses
# --------------------------------------------------------------------------

def combine(keyword: list[tuple[Article, float]],
            vector: list[tuple[Article, float]]) -> list[tuple[Article, float]]:
    """Hybrid score = average of the keyword score and the vector score, each scaled to 0-1:
    BM25 divided by the query's best BM25 score, and cosine scaled between the lowest and
    highest similarity in the vector list. An article missing from a list scores 0 there.
    Ties keep the keyword order."""
    score: dict[str, float] = {}
    by_id: dict[str, Article] = {}
    best = max((s for _, s in keyword), default=0.0)
    for art, s in keyword:
        score[art.article_id] = (s / best if best > 0 else 0.0) / 2
        by_id[art.article_id] = art
    if vector:
        lo, hi = min(s for _, s in vector), max(s for _, s in vector)
        for art, s in vector:
            score[art.article_id] = score.get(art.article_id, 0.0) + ((s - lo) / (hi - lo) if hi > lo else 1.0) / 2
            by_id[art.article_id] = art
    order = {a.article_id: i for i, (a, _) in enumerate(keyword)}
    ranked = sorted(score, key=lambda a: (-score[a], order.get(a, len(order))))
    return [(by_id[a], score[a]) for a in ranked]


@dataclass
class SearchResult:
    hits: list[tuple[Article, float]]                               # final order (combined in hybrid)
    keyword: list[tuple[str, float]] = field(default_factory=list)  # top BM25 hits, for the audit log
    vector: list[tuple[str, float]] = field(default_factory=list)   # top cosine hits, for the audit log


class Retriever:
    def __init__(self, articles: list[Article], mode: Optional[str] = None,
                 embedder: Optional[Embedder] = None, cache_root: Optional[Path] = None):
        self.mode = check_mode(mode)
        self.keyword = KBIndex(articles)
        self.articles = self.keyword.articles  # Active only
        self.vector: Optional[VectorIndex] = None
        if self.mode in ("vector", "hybrid"):
            self.vector = VectorIndex(articles, embedder or get_embedder(config.EMBED_MODEL), cache_root)

    @property
    def embed_model(self) -> Optional[str]:
        return self.vector.embedder.name if self.vector else None

    def search(self, query: str, k: int = 5) -> SearchResult:
        depth = max(k, SEARCH_DEPTH)
        kw = self.keyword.search(query, k=depth) if self.mode != "vector" else []
        vec = self.vector.search(query, k=depth) if self.vector else []
        if self.mode == "bm25":
            hits = kw
        elif self.mode == "vector":
            hits = vec
        else:
            hits = combine(kw, vec)
        return SearchResult(
            hits=hits[:k],
            keyword=[(a.article_id, round(s, 3)) for a, s in kw[:5]],
            vector=[(a.article_id, round(s, 3)) for a, s in vec[:5]],
        )


# --------------------------------------------------------------------------
# The check every run makes first (like the policy compile): model, then vectors
# --------------------------------------------------------------------------

@dataclass
class SearchStatus:
    kb_set: str
    mode: str
    model: Optional[str] = None          # None for keyword search
    model_present: bool = True
    articles: list[str] = field(default_factory=list)   # Active articles that are searched
    pending: list[str] = field(default_factory=list)    # Active articles with no current vector
    model_downloaded_now: bool = False
    embedded_now: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.mode == "bm25" or (self.model_present and not self.pending)

    def summary(self) -> str:
        if self.mode == "bm25":
            return f"Article search for '{self.kb_set}': keyword only, no embedding model needed."
        model = "present" if self.model_present else "not downloaded yet (the next run downloads it)"
        vectors = (f"up to date for all {len(self.articles)} articles" if not self.pending else
                   f"{len(self.pending)} of {len(self.articles)} articles to embed on the next run")
        kind = "keyword + vector" if self.mode == "hybrid" else "vector only"
        return f"Article search for '{self.kb_set}': {kind}. Embedding model {self.model}: {model}; vectors {vectors}."


def _kb_articles(kb_set: str, dirs: Optional[list[Path]] = None) -> list[Article]:
    """The articles a run searches: `dirs` if given, else the KB set's folders."""
    from .kb import load_kb
    from .pipeline.inputs import kb_dirs_for

    return load_kb(dirs or kb_dirs_for(kb_set))


def _pending(articles: list[Article], model_name: str, cache_root: Optional[Path] = None) -> list[str]:
    """Active articles that have at least one chunk with no stored vector (new, edited or never embedded)."""
    cache = EmbeddingCache(model_name, cache_root)
    return [a.article_id for a in articles if a.active
            and any(EmbeddingCache.key(model_name, c) not in cache.vectors for c in chunk_texts(a))]


def search_status(kb_set: str, mode: Optional[str] = None, cache_root: Optional[Path] = None,
                  dirs: Optional[list[Path]] = None) -> SearchStatus:
    """Where the KB set's search stands, without downloading or embedding anything."""
    mode = check_mode(mode)
    articles = _kb_articles(kb_set, dirs)
    st = SearchStatus(kb_set=kb_set, mode=mode, articles=[a.article_id for a in articles if a.active])
    if mode == "bm25":
        return st
    emb = get_embedder(config.EMBED_MODEL)
    st.model = emb.name
    st.model_present = bool(getattr(emb, "downloaded", lambda: True)())
    st.pending = _pending(articles, emb.name, cache_root)
    return st


def ensure_ready(kb_set: str, mode: Optional[str] = None, *, log=print, quiet_when_ready: bool = False,
                 cache_root: Optional[Path] = None, dirs: Optional[list[Path]] = None) -> SearchStatus:
    """Make the KB set's article search ready, as every run does before it starts:

    1. the embedding model: downloaded into runtime/models if it isn't there (about 70 MB,
       once), and downloaded again if the files there are damaged;
    2. the article vectors: every Active article the run will search (`dirs`) needs a vector
       for its text in runtime/vector/; new or edited articles are embedded, the rest are reused.

    Keyword search (bm25) needs neither. A failure raises RetrievalError before the run
    changes anything; the message says what to check.
    """
    st = search_status(kb_set, mode, cache_root, dirs)
    if st.mode == "bm25":
        if not quiet_when_ready:
            log(st.summary())
        return st
    emb = get_embedder(config.EMBED_MODEL)
    if not st.model_present:
        log(f"Embedding model {emb.name} not found in {_rel(config.MODELS_DIR)}: downloading it from "
            f"Hugging Face (about 70 MB, once)...")
    load = getattr(emb, "load", None)
    if load is not None:
        load(log)
    if not st.model_present:
        st.model_present = st.model_downloaded_now = True
        log("Embedding model downloaded.")
    if st.pending:
        first = len(st.pending) == len(st.articles)
        names = ", ".join(st.pending[:6]) + (" ..." if len(st.pending) > 6 else "")
        log(f"No article vectors for '{kb_set}' yet: embedding its {len(st.articles)} Active articles..." if first
            else f"The KB changed since its vectors were made: embedding {len(st.pending)} new or changed "
                 f"article(s) ({names})...")
        VectorIndex(_kb_articles(kb_set, dirs), emb, cache_root)
        st.embedded_now, st.pending = st.pending, []
        log(f"Article vectors for '{kb_set}' ready: {len(st.articles)} articles.")
    elif not quiet_when_ready:
        log(st.summary())
    return st
