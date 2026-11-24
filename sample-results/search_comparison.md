# Article search: keyword only vs hybrid, top 3 vs top 5

The prototype first found articles with keyword search (BM25) and gave the rule
picker the top 3. It now uses hybrid search (keyword + vector) and gives the
picker the top 5.

- **Hybrid score:** the average of the keyword score and the vector score,
  each scaled to 0-1.
- **Vectors:** an open-source embedding model (`BAAI/bge-small-en-v1.5`) that
  runs locally, in a FAISS index. In production, the same interface would call
  a hosted embedding API such as Azure OpenAI embeddings.

All runs below use the same data, the same live policies (v2) and the same model.

## 1. Search alone (no API calls)

`python -m servicedesk.cli eval retrieval` searches the 27 answer-key issues
that have an article, in two ways:

- **Agent query:** the keywords and summary the reading agent wrote. This is
  what the pipeline searches on.
- **Employee's words:** the original message, redacted, with no rewriting.

The cells count how often a correct article is in the top 1, 3 or 5. The
picker reads the top 5, so it can't choose an article ranked below that.

| Method | Agent query: top 1 | top 3 | top 5 | Employee's words: top 1 | top 3 | top 5 |
|---|---|---|---|---|---|---|
| Keyword (BM25) | 25/27 | 26/27 | 27/27 | 18/27 | 23/27 | 24/27 |
| Vector | 26/27 | 26/27 | 27/27 | 26/27 | 26/27 | 26/27 |
| **Hybrid** | 25/27 | **27/27** | **27/27** | 20/27 | **27/27** | **27/27** |

- **Keyword search misses when the wording differs.** On the employee's words
  it never finds the security article for SYN-003 (a stolen work phone) or
  SYN-004 (a password typed into a fake invoice page), or the software article
  for NEW-006 (a meeting-notes extension).
- **Vector search finds those, but misses others on its own.** It ranks the
  lost laptop (NEW-012) 4th on the agent query, and the production access
  request (NEW-003) 7th on the employee's words. Keywords find both.
- **Hybrid has every issue in the top 3**, so the top 5 leaves a margin.
- **Top 5 is what rescues keyword search** on the agent queries: SYN-003's
  article is 4th.

Full table: [retrieval_eval.md](retrieval_eval.md).

## 2. End to end (runs A and B, 26 requests each)

`python -m servicedesk.cli eval requests`, once per setting.

| | Keyword, top 3 (before) | Hybrid, top 3 | **Hybrid, top 5 (now)** |
|---|---|---|---|
| Must-never violations | 0 | 0 | 0 |
| Routing fields, A / B | 221/221, 222/222 | 221/221, 222/222 | 221/221, 222/222 |
| Outcome checks, A / B | 10/10, 10/10 | 10/10, 10/10 | 10/10, 10/10 |
| Moved to urgent by KB-013 | 9 | 9 | 9 |
| Correct article among the candidates | 27/27 | 27/27 | 27/27 |
| Correct article ranked 1st, A / B | 24, 22 | 25, 25 | 24, 24 |
| Lowest rank of a correct article | 3rd of 3 (SYN-003) | 3rd of 3 (NEW-012) | 2nd of 5 |
| Input tokens for one rule pick | about 2,640 | about 2,640 | about 3,730 |
| Input tokens, whole eval | 698k | 703k | 782k (+11%) |

- **The outcomes are the same in every setting.** With 13 articles, keyword
  search with the top 3 already found the right article every time, but twice
  only in the last slot.
- **Top 5 against top 3 (hybrid):** the same rule for 27 of 28 issues in each run.
  - **SYN-001**, an MFA prompt approved by mistake: the picker chose the
    security article (KB-008) as the main rule instead of the MFA article
    (KB-002). The answer key accepts both. Because KB-008 needs details, the
    request waited for them in run A (6 waits avoided instead of 5).
  - **SYN-003**, a stolen work phone: in run B the picker also added KB-002's
    lost-device MFA rule as guidance, which fits the case.
- **Price of the top 5:** each rule pick reads two more articles, about 41%
  more tokens per pick and 11% more for the whole eval.
- **A scorer false alarm:** the first scoring of the top-5 run failed NEW-009,
  because the approval said "rather than overwriting the current folder" and
  the check flagged any mention of overwriting. The check now allows a negated
  mention and still catches a request to overwrite. It has a regression test.

## 3. What this shows, and what it doesn't

The test set is small: 27 issues, 13 articles, and only 10 articles that can
be candidates, so the top 5 is half of them. The tests show where each method
fails, and that hybrid with the top 5 leaves a margin. They can't show
behaviour with hundreds of articles, where more look-alike articles compete
for the five slots. That needs a larger labelled query set, or the current
queries run against a KB padded with realistic look-alike articles.

## 4. What the change costs

- **Packages:** `fastembed`, `faiss-cpu` and `numpy` (29 packages with their
  dependencies).
- **One-time download:** about 70 MB from Hugging Face on the first run, into
  `runtime/models/`. No key is needed.
- **First run:** about 5 seconds to embed the articles. After that, the
  embeddings come from a cache keyed by content hash, so only new or edited
  articles are embedded again.
- **Search time:** 3-5 ms per query, against under 1 ms for keyword search.
- **Keyword search is still available** with `--retrieval bm25`.
