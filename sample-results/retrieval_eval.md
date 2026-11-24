# Retrieval eval: keyword vs vector vs hybrid

Generated 2026-11-24T01:28:01+00:00. KB set `with-kb013`: 12 Active articles, of which 10 have situation rules and can be candidates. Vectors: `BAAI/bge-small-en-v1.5`, run locally. 27 gold issues. No API calls.

The rule picker reads the **top 5** articles, so top 5 decides the outcome: it can't choose an article it never sees.

| Method | Agent query: top 1 | top 3 | top 5 | Employee's words: top 1 | top 3 | top 5 | ms/query |
|---|---|---|---|---|---|---|---|
| keyword (BM25) | 25/27 | 26/27 | 27/27 | 18/27 | 23/27 | 24/27 | 0.01 |
| vector | 26/27 | 26/27 | 27/27 | 26/27 | 26/27 | 26/27 | 4.36 |
| hybrid | 25/27 | 27/27 | 27/27 | 20/27 | 27/27 | 27/27 | 4.5 |

## Issues not ranked first

Rank of the first correct article among the candidate articles (- = not found).

| Issue | Correct | keyword (BM25) (agent) | keyword (BM25) (message) | vector (agent) | vector (message) | hybrid (agent) | hybrid (message) |
|---|---|---|---|---|---|---|---|
| NEW-003#2 production access needed for a customer call | KB-001 | 1 | 2 | 1 | 7 | 1 | 3 |
| NEW-006#1 install an AI meeting-notes browser extension | KB-006 | 1 | - | 1 | 1 | 1 | 2 |
| NEW-008#1 entered company password on a phishing page | KB-008 | 1 | 5 | 1 | 1 | 1 | 2 |
| NEW-012#1 company laptop lost at the airport | KB-008 | 1 | 1 | 4 | 1 | 2 | 1 |
| SYN-001#1 approved an MFA prompt they didn't start; now locked out | KB-008, KB-002 | 1 | 2 | 1 | 1 | 1 | 1 |
| SYN-002#1 endpoint protection quarantined a file | KB-008 | 1 | 2 | 1 | 1 | 1 | 1 |
| SYN-003#1 work phone stolen | KB-008 | 4 | - | 1 | 1 | 2 | 2 |
| SYN-004#1 typed password into a fake invoice page | KB-008 | 2 | - | 1 | 1 | 1 | 2 |
| SYN-009#2 battery swelling and hot, first raised in the reply | KB-005 | 1 | 3 | 1 | 1 | 1 | 2 |
| SYN-012#1 Singapore branch printer disconnects for everyone (not the HQ incident MAJ-303) | KB-005 | 1 | 2 | 1 | 1 | 1 | 2 |
