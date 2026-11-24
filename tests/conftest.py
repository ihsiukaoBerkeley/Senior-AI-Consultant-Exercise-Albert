"""Shared test setup: vector search runs on a small fake embedder, so no test downloads a model
or writes to runtime/vector."""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Deterministic hashed bag of words (64 dimensions): texts that share words point the same way."""

    name = "fake-embedder"
    dim = 64

    def __init__(self):
        self.passage_calls: list[list[str]] = []

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for w in _WORD.findall(text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def passages(self, texts: list[str]) -> np.ndarray:
        self.passage_calls.append(list(texts))
        return np.stack([self._vec(t) for t in texts])

    def query(self, text: str) -> np.ndarray:
        return self._vec(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture(autouse=True)
def _offline_vectors(monkeypatch, tmp_path_factory, fake_embedder):
    from servicedesk import config, retrieval

    monkeypatch.setattr(config, "VECTOR_DIR", tmp_path_factory.mktemp("vector"))
    monkeypatch.setattr(retrieval, "get_embedder", lambda name: fake_embedder)
