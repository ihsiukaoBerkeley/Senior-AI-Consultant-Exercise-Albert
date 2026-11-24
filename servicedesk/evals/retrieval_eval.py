"""Retrieval eval: keyword vs vector vs hybrid article search on the gold issues. No API calls.

The query set (data/gold/retrieval_queries.jsonl) has one row per answer-key issue whose
correct article has situation rules (27 of the 29 issues; NEW-011 is not an IT request).
Each issue is searched two ways:

- agent:   the search keywords and summary the reading agent wrote for the issue in a
           recorded run, which is what the pipeline searches on.
- message: the employee's own words, redacted as the pipeline redacts them (for SYN-009's
           second issue, the reply that first raised it). Harder: nothing has rewritten the
           text into the KB's vocabulary, so it shows how each method copes with other words.

Articles are ranked as the pipeline ranks them (Active articles that have situation rules)
and scored on whether a correct article is in the top 1, top 3 and top 5. The top
KB_CANDIDATES (5) decide the outcome: the rule picker reads those articles and can't
choose one it never sees.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

from .. import config
from ..config import GOLD_DIR, KB_CANDIDATES, REPLIES_FILE
from ..guards import redact_secrets
from ..retrieval import Retriever, ensure_ready
from ..stores import now_iso

QUERY_FILE = GOLD_DIR / "retrieval_queries.jsonl"
METHODS = {"keyword (BM25)": "bm25", "vector": "vector", "hybrid": "hybrid"}
QUERY_SETS = {"agent": "agent's search keywords + summary", "message": "employee's own words"}
CUTOFFS = (1, 3, 5)
Log = Callable[[str], None]


def load_queries(path: Path = QUERY_FILE) -> list[dict]:
    from ..pipeline import load_requests

    messages = {r.request_id: r.message for r in load_requests("all")}
    replies = json.loads(Path(REPLIES_FILE).read_text()) if Path(REPLIES_FILE).exists() else {}
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        raw = replies[row["request_id"]] if row.get("text_from") == "reply" else messages[row["request_id"]]
        row["message"] = redact_secrets(raw)[0]
        rows.append(row)
    return rows


def rank_of(ranked: list[str], correct: list[str]) -> Optional[int]:
    ranks = [ranked.index(a) + 1 for a in correct if a in ranked]
    return min(ranks) if ranks else None


def score(retriever: Retriever, rows: list[dict], query_key: str, eligible: set[str]) -> dict:
    per, t = [], 0.0
    for r in rows:
        start = time.perf_counter()
        hits = retriever.search(r[query_key], k=len(retriever.articles)).hits
        t += time.perf_counter() - start
        ranked = [a.article_id for a, _ in hits if a.article_id in eligible]
        per.append({"id": f"{r['request_id']}#{r['issue']}", "about": r["about"], "correct": r["articles"],
                    "rank": rank_of(ranked, r["articles"]), "candidates": ranked[:KB_CANDIDATES]})
    n = len(per)
    out = {"n": n, "query_ms": round(1000 * t / n, 2) if n else 0.0, "issues": per}
    for c in CUTOFFS:
        out[f"top{c}"] = sum(p["rank"] is not None and p["rank"] <= c for p in per)
    return out


def run_retrieval_eval(kb_set: str = "with-kb013", log: Log = print, out_dir: Optional[Path] = None) -> dict:
    from ..kb import load_kb
    from ..pipeline import kb_dirs_for
    from ..policy import compiler

    articles = load_kb(kb_dirs_for(kb_set))
    ensure_ready(kb_set, "hybrid", log=log, quiet_when_ready=True)  # downloads the model / embeds if needed
    eligible = {r.article_id for r in compiler.load_gold(kb_set).rules if r.scope == "situation"}
    rows = load_queries()
    results = {}
    for label, mode in METHODS.items():
        ret = Retriever(articles, mode)
        results[label] = {qs: score(ret, rows, "message" if qs == "message" else "agent_query", eligible)
                          for qs in QUERY_SETS}
    data = {
        "generated_at": now_iso(),
        "kb_set": kb_set,
        "articles_indexed": sum(a.active for a in articles),
        "eligible_articles": sorted(eligible),
        "embed_model": config.EMBED_MODEL,
        "candidates": KB_CANDIDATES,
        "queries": len(rows),
        "results": results,
    }
    out = Path(out_dir or config.RUNTIME_DIR / "reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_eval.json").write_text(json.dumps(data, indent=2))
    (out / "retrieval_eval.md").write_text(render_markdown(data))
    data["report_md"] = out / "retrieval_eval.md"
    for line in summary_lines(data):
        log(line)
    return data


def _cells(res: dict, n: int) -> list[str]:
    return [f"{res[f'top{c}']}/{n}" for c in CUTOFFS]


def summary_lines(data: dict) -> list[str]:
    n = data["queries"]
    cuts = "/".join(str(c) for c in CUTOFFS)
    L = [f"{n} issues, {data['articles_indexed']} Active articles ({data['kb_set']}), "
         f"vectors: {data['embed_model']} (local); the picker reads the top {data['candidates']}",
         f"{'Method':16s} {'Agent query, top ' + cuts:>24s}   {'Employee words, top ' + cuts:>26s}"]
    for label, res in data["results"].items():
        L.append(f"{label:16s} {'  '.join(_cells(res['agent'], n)):>24s}   {'  '.join(_cells(res['message'], n)):>26s}")
    return L


def render_markdown(data: dict) -> str:
    n, k = data["queries"], data["candidates"]
    cols = " | ".join(f"top {c}" for c in CUTOFFS)
    L = ["# Retrieval eval: keyword vs vector vs hybrid", "",
         f"Generated {data['generated_at']}. KB set `{data['kb_set']}`: {data['articles_indexed']} Active articles, "
         f"of which {len(data['eligible_articles'])} have situation rules and can be candidates. "
         f"Vectors: `{data['embed_model']}`, run locally. {n} gold issues. No API calls.", "",
         f"The rule picker reads the **top {k}** articles, so top {k} decides the outcome: it can't choose an "
         f"article it never sees.", "",
         f"| Method | Agent query: {cols} | Employee's words: {cols} | ms/query |",
         "|---|" + "---|" * (2 * len(CUTOFFS) + 1)]
    for label, res in data["results"].items():
        a, m = res["agent"], res["message"]
        L.append(f"| {label} | " + " | ".join(_cells(a, n) + _cells(m, n)) + f" | {m['query_ms']} |")
    labels = list(data["results"])
    L += ["", "## Issues not ranked first", "",
          "Rank of the first correct article among the candidate articles (- = not found).", "",
          "| Issue | Correct | " + " | ".join(f"{lab} ({qs})" for lab in labels for qs in QUERY_SETS) + " |",
          "|---|---|" + "---|" * (len(labels) * len(QUERY_SETS))]
    first = data["results"][labels[0]]["agent"]["issues"]
    for i, p in enumerate(first):
        ranks = [data["results"][lab][qs]["issues"][i]["rank"] for lab in labels for qs in QUERY_SETS]
        if any(r != 1 for r in ranks):
            L.append(f"| {p['id']} {p['about']} | {', '.join(p['correct'])} | "
                     + " | ".join("-" if r is None else str(r) for r in ranks) + " |")
    L.append("")
    return "\n".join(L)
