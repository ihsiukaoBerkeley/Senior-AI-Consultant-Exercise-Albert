"""Keyword, vector and hybrid article search (servicedesk/retrieval.py) and the retrieval eval.

No API calls and no model download: vectors come from the fake embedder in conftest.py.
One check uses the real embedding model, and only when an earlier run has downloaded it.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from servicedesk import config
from servicedesk.config import GOLD_DIR, KB_ADDITIONS_DIR, KB_DIR
from servicedesk.guards import redact_secrets
from servicedesk.kb import load_kb, parse_article
from servicedesk.retrieval import (
    EmbeddingCache,
    FastEmbedder,
    RetrievalError,
    Retriever,
    check_mode,
    chunk_texts,
    combine,
    ensure_ready,
    search_status,
)


def arts():
    return load_kb([KB_DIR, KB_ADDITIONS_DIR])


def art(aid):
    return next(a for a in arts() if a.article_id == aid)


# --------------------------------------------------------------------------
# Chunks
# --------------------------------------------------------------------------

def test_a_chunk_drops_the_header_lines_and_leads_with_the_title():
    chunks = chunk_texts(art("KB-004"))
    assert len(chunks) == 1
    assert chunks[0].startswith("Network drives and shared folders. Applies to: Managed network drives")
    assert "Owner:" not in chunks[0] and "# KB-004" not in chunks[0]
    assert "data-owner approval" in chunks[0]


def test_a_long_article_splits_at_paragraphs(tmp_path):
    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(5))  # 62 words each
    f = tmp_path / "KB-090-long-article.md"
    f.write_text("# KB-090: Long article\n\n- Owner: X\n- Last reviewed: 2026-01-01\n- Status: Active\n"
                 "- Applies to: Tests\n\n" + body)
    chunks = chunk_texts(parse_article(f), max_words=150)
    assert len(chunks) == 3  # paragraphs 0-1, 2-3, 4
    assert all(c.startswith("Long article. Applies to: Tests.") for c in chunks)
    assert "Paragraph 4" in chunks[-1] and "Paragraph 2" in chunks[1]


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_no_mode_ever_returns_the_deprecated_article(mode):
    hits = Retriever(arts(), mode).search("reinstall LegacyConnect VPN client gateway certificate", k=20).hits
    assert hits and all(a.article_id != "KB-011" for a, _ in hits)
    assert hits[0][0].article_id == "KB-003"


def test_a_hybrid_result_carries_both_lists_for_the_audit_log():
    res = Retriever(arts(), "hybrid").search("shared drive one folder access denied", k=3)
    assert res.hits[0][0].article_id == "KB-004" and len(res.hits) == 3
    assert res.keyword[0][0] == "KB-004" and res.vector


def test_unknown_mode_is_rejected():
    with pytest.raises(RetrievalError, match="unknown retrieval mode"):
        check_mode("semantic")


# --------------------------------------------------------------------------
# Embedding cache: only new or changed articles are embedded
# --------------------------------------------------------------------------

def test_only_new_or_changed_articles_are_embedded(fake_embedder):
    a = arts()
    active = sum(x.active for x in a)
    Retriever(a, "hybrid")
    assert sum(len(call) for call in fake_embedder.passage_calls) == active  # one chunk per article today
    fake_embedder.passage_calls.clear()

    again = Retriever(a, "hybrid")
    assert again.vector.embedded_now == 0 and fake_embedder.passage_calls == []

    kb005 = next(x for x in a if x.article_id == "KB-005")
    edited = replace(kb005, text=kb005.text + "\nToner cartridges are ordered through the facilities portal.\n")
    changed = Retriever([edited if x.article_id == "KB-005" else x for x in a], "hybrid")
    assert changed.vector.embedded_now == 1
    assert len(fake_embedder.passage_calls) == 1 and "Toner cartridges" in fake_embedder.passage_calls[0][0]


def test_a_damaged_cache_is_rebuilt_not_trusted(tmp_path, fake_embedder):
    cache = EmbeddingCache(fake_embedder.name, tmp_path)
    cache.path.parent.mkdir(parents=True)
    cache.path.write_bytes(b"not a zip file")
    r = Retriever(arts(), "vector", cache_root=tmp_path)
    assert r.vector.embedded_now == sum(a.active for a in arts())


# --------------------------------------------------------------------------
# Hybrid score: the average of the two scores, each scaled to 0-1
# --------------------------------------------------------------------------

def test_hybrid_score_is_the_average_of_the_scaled_scores():
    kw = [(art("KB-004"), 10.0), (art("KB-012"), 5.0)]
    vec = [(art("KB-012"), 0.70), (art("KB-004"), 0.60), (art("KB-001"), 0.50)]
    out = dict((a.article_id, s) for a, s in combine(kw, vec))
    assert out["KB-004"] == pytest.approx((1.0 + 0.5) / 2)   # best keyword match, middle vector match
    assert out["KB-012"] == pytest.approx((0.5 + 1.0) / 2)   # half the best keyword score, best vector match
    assert out["KB-001"] == pytest.approx((0.0 + 0.0) / 2)   # no keyword match, lowest vector match
    assert [a.article_id for a, _ in combine(kw, vec)][:2] == ["KB-004", "KB-012"]  # a tie keeps the keyword order


def test_a_strong_keyword_match_beats_a_slightly_closer_vector_match():
    kw = [(art("KB-004"), 13.5), (art("KB-012"), 2.5)]
    vec = [(art("KB-012"), 0.70), (art("KB-001"), 0.69), (art("KB-004"), 0.68), (art("KB-010"), 0.40),
           (art("KB-009"), 0.35)]
    assert combine(kw, vec)[0][0].article_id == "KB-004"


def test_with_no_keyword_hits_the_vector_order_decides():
    vec = [(art("KB-008"), 0.65), (art("KB-002"), 0.62), (art("KB-003"), 0.45)]
    assert [a.article_id for a, _ in combine([], vec)] == ["KB-008", "KB-002", "KB-003"]


# --------------------------------------------------------------------------
# Runs: the mode is recorded, kept on resume, and a failed model load changes nothing
# --------------------------------------------------------------------------

class _Broken:
    name = "broken-model"

    def downloaded(self):
        return False

    def load(self, log=None):
        raise RetrievalError("could not load the embedding model broken-model (offline)")


def test_a_failed_model_load_stops_the_run_before_it_resets_the_folder(tmp_path, monkeypatch, capsys):
    from servicedesk import cli, retrieval
    from servicedesk.evals import request_eval

    monkeypatch.setattr(retrieval, "get_embedder", lambda name: _Broken())
    monkeypatch.setattr(request_eval, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(request_eval, "make_llm", lambda: object())
    previous = tmp_path / "runs" / "mine" / "keep.txt"
    previous.parent.mkdir(parents=True)
    previous.write_text("earlier run")

    code = cli.main(["run", "--kb", "with-kb013", "--name", "mine", "--policy", "gold", "--retrieval", "hybrid"])
    assert code == 2 and "Error: could not load the embedding model" in capsys.readouterr().err
    assert previous.read_text() == "earlier run"
    assert ensure_ready("with-kb013", "bm25", log=lambda _m: None).ready  # keyword search needs no model


def test_runs_record_their_search_and_resume_with_it(tmp_path, monkeypatch):
    from servicedesk import cli
    from servicedesk.evals import request_eval
    from servicedesk.policy import compiler

    monkeypatch.setattr(request_eval, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(request_eval, "make_llm", lambda: object())
    pol = compiler.load_gold("with-kb013")
    hybrid = request_eval.write_run_meta(tmp_path / "runs" / "r1", "with-kb013", pol, "gold", "hybrid")
    keyword = request_eval.write_run_meta(tmp_path / "runs" / "r2", "with-kb013", pol, "gold", "bm25")
    assert (hybrid["retrieval"], hybrid["embed_model"]) == ("hybrid", config.EMBED_MODEL)
    assert (keyword["retrieval"], keyword["embed_model"]) == ("bm25", None)

    pipeline, _, _ = cli.build_pipeline_for_run("r1", log=lambda _m: None)
    assert pipeline.retriever.mode == "hybrid"
    # A run made before vector search (no "retrieval" in run_meta.json) resumes with keywords.
    meta_file = tmp_path / "runs" / "r1" / "run_meta.json"
    data = json.loads(meta_file.read_text())
    data.pop("retrieval")
    meta_file.write_text(json.dumps(data))
    pipeline, _, _ = cli.build_pipeline_for_run("r1", log=lambda _m: None)
    assert pipeline.retriever.mode == "bm25"


# --------------------------------------------------------------------------
# The check before every run: model first, then vectors
# --------------------------------------------------------------------------

def test_first_run_embeds_every_article_then_only_new_or_edited_ones(monkeypatch, fake_embedder):
    from servicedesk import retrieval

    logs: list[str] = []
    first = ensure_ready("with-kb013", "hybrid", log=logs.append)
    active = sum(a.active for a in arts())
    assert first.ready and len(first.embedded_now) == active
    assert any("No article vectors for 'with-kb013' yet" in m for m in logs)

    logs.clear()
    again = ensure_ready("with-kb013", "hybrid", log=logs.append)
    assert again.ready and again.embedded_now == [] and "vectors up to date" in logs[-1]
    assert ensure_ready("with-kb013", "hybrid", log=logs.append, quiet_when_ready=True).ready
    assert len(logs) == 1  # quiet when there was nothing to do

    edited = [replace(a, text=a.text + "\nToner is ordered through the facilities portal.\n")
              if a.article_id == "KB-005" else a for a in arts()]
    monkeypatch.setattr(retrieval, "_kb_articles", lambda kb_set, dirs=None: edited)
    assert search_status("with-kb013", "hybrid").pending == ["KB-005"]
    logs.clear()
    changed = ensure_ready("with-kb013", "hybrid", log=logs.append)
    assert changed.embedded_now == ["KB-005"] and "1 new or changed article(s) (KB-005)" in logs[0]


class _Downloadable:
    """Stands in for the real model: absent until load() is called."""

    name = "fake-model"

    def __init__(self, inner):
        self.inner, self.present, self.loads = inner, False, 0

    def downloaded(self):
        return self.present

    def load(self, log=None):
        self.loads += 1
        self.present = True

    def passages(self, texts):
        return self.inner.passages(texts)

    def query(self, text):
        return self.inner.query(text)


def test_a_missing_model_is_downloaded_before_the_vectors_are_made(monkeypatch, fake_embedder):
    from servicedesk import retrieval

    model = _Downloadable(fake_embedder)
    monkeypatch.setattr(retrieval, "get_embedder", lambda name: model)
    before = search_status("base", "hybrid")
    assert not before.model_present and not before.ready and "not downloaded yet" in before.summary()
    logs: list[str] = []
    st = ensure_ready("base", "hybrid", log=logs.append)
    assert model.loads == 1 and st.model_downloaded_now and st.ready
    assert "not found" in logs[0] and "downloading it from Hugging Face" in logs[0]
    assert logs[1] == "Embedding model downloaded." and "No article vectors for 'base' yet" in logs[2]


def test_a_damaged_model_is_downloaded_again_once(tmp_path):
    model = FastEmbedder("BAAI/bge-small-en-v1.5", tmp_path / "models")
    (tmp_path / "models" / "models--Qdrant--bge-small-en-v1.5-onnx-Q").mkdir(parents=True)  # files present
    attempts = []

    def create():
        attempts.append(model.downloaded())
        if len(attempts) == 1:
            raise ValueError("model file is truncated")
        return "loaded"

    model._create = create
    logs: list[str] = []
    assert model.load(logs.append) == "loaded"
    assert attempts == [True, False]  # the damaged files were removed before the second try
    assert "incomplete or damaged: downloading it again" in logs[0]


def test_a_failed_download_names_the_sites_to_allow_and_the_keyword_fallback(tmp_path):
    model = FastEmbedder("BAAI/bge-small-en-v1.5", tmp_path / "models")

    def create():
        raise ConnectionError("Connection refused")

    model._create = create
    with pytest.raises(RetrievalError) as e:
        model.load()
    msg = str(e.value)
    assert "huggingface.co" in msg and "*.hf.co" in msg and "SD_RETRIEVAL=bm25" in msg and "Connection refused" in msg


def test_keyword_search_never_touches_the_model(monkeypatch):
    from servicedesk import retrieval

    monkeypatch.setattr(retrieval, "get_embedder", lambda name: _Broken())
    st = ensure_ready("base", "bm25", log=lambda _m: None)
    assert st.ready and st.model is None and "keyword only" in st.summary()


def test_cli_policy_show_reports_the_search_status(capsys):
    from servicedesk import cli

    cli.main(["policy", "show", "--kb", "base"])
    assert "Article search for 'base'" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Retrieval eval
# --------------------------------------------------------------------------

def test_the_query_set_matches_the_answer_key():
    from servicedesk.evals.retrieval_eval import QUERY_FILE
    from servicedesk.policy import compiler

    rows = [json.loads(line) for line in QUERY_FILE.read_text().splitlines() if line.strip()]
    key = {r["request_id"]: r for r in json.loads((GOLD_DIR / "answer_key.json").read_text())}
    situations = {r.article_id for r in compiler.load_gold("with-kb013").rules if r.scope == "situation"}
    expected = sum(1 for r in key.values() for i in r["issues"]
                   if any(a in situations for a in (i.get("article_id") or [])))
    assert len(rows) == expected == 27
    for row in rows:
        issue = key[row["request_id"]]["issues"][row["issue"] - 1]
        assert set(row["articles"]) == {a for a in issue["article_id"] if a in situations}
        assert row["text_from"] in ("request", "reply") and row["agent_query"]
        assert "message" not in row  # read from the request files at eval time, then redacted


def test_eval_messages_are_redacted_like_the_pipeline_input():
    from servicedesk.evals.retrieval_eval import load_queries

    rows = load_queries()
    assert any("[REDACTED-password]" in row["message"] for row in rows)  # SYN-013 offers a password
    for row in rows:
        assert redact_secrets(row["message"])[0] == row["message"]  # nothing left to redact


def test_retrieval_eval_scores_every_method_and_writes_the_report(tmp_path):
    from servicedesk.evals.retrieval_eval import METHODS, run_retrieval_eval

    data = run_retrieval_eval(log=lambda _m: None, out_dir=tmp_path)
    assert set(data["results"]) == set(METHODS)
    assert data["candidates"] == config.KB_CANDIDATES == 5
    for res in data["results"].values():
        for qs in ("agent", "message"):
            assert res[qs]["n"] == 27 and 0 <= res[qs]["top1"] <= res[qs]["top3"] <= res[qs]["top5"] <= 27
    assert data["results"]["keyword (BM25)"]["agent"]["top5"] == 27  # keywords don't use the embedder
    assert (tmp_path / "retrieval_eval.md").read_text().startswith("# Retrieval eval")
    assert json.loads((tmp_path / "retrieval_eval.json").read_text())["queries"] == 27


# --------------------------------------------------------------------------
# The real model (only when an earlier run has downloaded it)
# --------------------------------------------------------------------------

def _real_model():
    emb = FastEmbedder(config.EMBED_MODEL, config.MODELS_DIR)
    return emb if emb.downloaded() else None


@pytest.mark.skipif(_real_model() is None, reason="embedding model not downloaded yet; the first hybrid run does that")
def test_the_real_model_finds_the_security_article_in_other_words(tmp_path):
    r = Retriever(arts(), "vector", embedder=_real_model(), cache_root=tmp_path)
    hits = r.search("someone keeps asking me to approve sign-ins on my phone that I didn't start", k=2).hits
    assert {a.article_id for a, _ in hits} & {"KB-008", "KB-002"}
