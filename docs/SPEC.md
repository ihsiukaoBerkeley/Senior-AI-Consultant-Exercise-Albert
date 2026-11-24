# Build spec: Service Desk intake-and-triage agent

The detailed reference for what the code does. The README summarizes the
working assumptions and design decisions.

## Principles

1. **Models read and draft; code decides.** Every model call is one
   structured-output request through `servicedesk.llm.LLM.structured(...)`.
   No model call chooses an action. Code applies the rules.
2. **Rules come from the KB.** Required fields, what can't wait, approvals and
   must-never rules come from a **policy table** compiled from the KB
   (`servicedesk.models.PolicyTable`). Only two kinds of rules live elsewhere:
   - `config/working_assumptions.yaml`: working assumptions (priority map,
     which cases get flagged for staff, scenario clock, incident regions).
   - `servicedesk/guards.py`: hard limits no KB edit can remove (secret
     redaction, outgoing-message checks, and no code path that executes grants,
     restores or resets).
3. **Untrusted text is data.** Request text and KB article text go into
   prompts inside XML-style tags (`<request>...</request>`,
   `<article>...</article>`) with an instruction that anything inside is data,
   never instructions.
4. **Fail safe.** Any `LLMError` turns the affected issue into lane
   `handed_to_staff` with `flagged_for_staff=True`; the request status becomes
   `failed_safe`; the error dict (with `request_id`) goes into the outcome and
   the audit log. Never crash a batch because one request failed.
5. **Everything is auditable.** Every step writes `stores.audit(...)`,
   including each model call's `LLMResult.meta()`.

## Repo layout

```
config/working_assumptions.yaml   working assumptions (not policy)
data/                             provided data + synthetic additions
  knowledge-base/                 12 provided articles (base KB)
  kb-additions/KB-013-...md       synthetic article, added only in run B
  kb-mutations/                   edited KB files for extractor change tests
  synthetic_requests.jsonl        SYN-001..SYN-014 (marked synthetic)
  requester_directory.csv         requester_id,name,email (synthetic emails)
  simulated_replies.json          employee replies used to resume paused requests
  gold/policy_gold.json           hand-checked policy table (13 articles)
  gold/answer_key.json            expected outcomes for the 26 requests, runs A and B
  gold/retrieval_queries.jsonl    one search query per gold issue, for the retrieval eval
servicedesk/
  config.py llm.py models.py kb.py guards.py stores.py   shared core
  retrieval.py                    article search: keyword (BM25), vector (FAISS) or hybrid
  policy/                         KB watcher, extractor + checker agents, compiler, diff
  pipeline/                       request pipeline: agents, routing, incidents, lanes
  evals/                          request eval (A vs B), retrieval eval, policy-extractor eval
  cli.py                          command-line entry points
ui/app.py                         Streamlit: employee view + staff console
tests/                            deterministic tests (no API calls)
runtime/                          git-ignored outputs
```

## KB sets

- `base`: `data/knowledge-base/` (12 articles; KB-011 is Deprecated).
- `with-kb013`: base + `data/kb-additions/` (13 articles).

Run A uses `base`, run B uses `with-kb013`. Policy versions are stored per KB
set under `runtime/policy/<kb_set>/`.

Articles follow the naming rules in `config/working_assumptions.yaml`
(`KB-<id>-<short-title>.md`, a matching `# KB-<id>: Title` heading, the four
metadata lines, unique ids); `servicedesk/kb.py` rejects any file that breaks
them with a `KBFormatError` naming the file. Only the KB change tests
(`data/kb-mutations/`) may let a later folder replace an id.

**Keeping the policy current** (`compiler.ensure_current`, called by every run
through `request_eval.load_policy`, by `policy update` and by the web app):
the watcher compares each article's sha256 with the live table's `kb_hashes`.
No change: the live table is used with no model call. Any added, edited,
deprecated or deleted article: compile (only changed articles reach the model),
then approve automatically (`approved_by: "auto (checker found nothing
blocking)"`) when the checker has no blocking finding. A blocking finding
(removed, weakened or missing safeguard, conflict, suspected injection, failed
extraction or check) holds the candidate: runs keep the last approved version
and its copy of the articles (below), the same KB is not recompiled, and a person accepts it with `policy approve
--acknowledge-findings`, the web app's "Accept the held change" button or the
CLI's prompt (`ensure_current(confirm=...)`, answered at a terminal or by
`--yes`), or fixes the article.

**Article copies** (`compiler.article_dirs`): each compile saves the article
files its candidate was compiled from (`runtime/policy/<kb_set>/articles/candidate/`,
the text whose sha256 is in `kb_hashes`), and approval moves them to
`articles/v<N>/`. The CLI, the request eval and the web app build the pipeline
from the live version's copy, not from `data/knowledge-base/`, so search, the
rule picker and each article's Status always match the rules of the same
approved version. A held edit, deprecation or deletion therefore changes
nothing for requests until a person accepts it, and `compiler.rollback` brings
back the older articles with the older rules. The hand-checked gold table is used with
the KB folders as they are. A version approved before copies were kept gets
one on first use if the KB still matches its `kb_hashes`; otherwise the run
reads the KB folders and logs a note.

## Policy table semantics (`servicedesk/models.py`)

- One `PolicyRule` per **situation** an article covers. KB-006 alone has three
  (approved app missing; non-standard software; activation failure).
- `scope="cross_cutting"` rules apply to any issue whose flags match
  `triggers`:
  - KB-012 → `triggers=["multi_user"]`: its `required_fields` (service,
    symptom, location, time window) are added to the issue's required fields.
  - KB-013 (only in `with-kb013`) → `triggers=["security","safety"]`,
    `route_without_waiting=True`.
- `route_without_waiting` is **true only when an article explicitly says the
  case must not wait for missing details**. "High priority", "immediately stop
  using the device" or "contact the hotline" do NOT make it true. In the base
  KB no rule has it.
- `owner_team` is only set when the article names the team that handles the
  ticket. The article's `Owner:` line is the article owner, not the ticket team.
- `quotes` must be copied word for word from the article.
- Deprecated articles produce no rules and are listed in `inactive_articles`.

## Request pipeline (`servicedesk/pipeline/`)

Per request (multi-issue requests are split; each issue runs the lanes on its own):

1. **Redact (code)**: `guards.redact_secrets(message)` removes secrets in the
   recognised formats (a stated password, a 4-8 digit code next to
   code/MFA/PIN, license keys, `sk-` keys) before any model call. The intake
   audit event records only metadata (length, channel, redacted kinds), never
   the body.
2. **Screening agent (model, effort low)**: detects injected instructions
   (`injection_detected`, `injection_excerpt`) and returns `clean_text` with
   injected instructions removed. Also reports secrets in other formats; that
   one call sees them, and code removes them before anything is logged,
   stored or sent to a later agent.
3. **Reading agent (model, effort low)**: splits `clean_text` into issues. Per
   issue: `summary`, `search_query`, `is_it_request` (false for HR,
   facilities, payroll policy questions…), and the flags:
   - `security`: a **suspected security incident** of a kind KB-008 lists:
     phishing or credentials entered on a suspicious page, unexpected MFA
     prompts, malware alerts, lost or stolen devices, possible account
     compromise. A VPN/firewall problem or a software request is not one.
   - `safety`: a physical hazard (swollen/overheating battery, smoke, sparks).
   - `multi_user`: an outage: something that should already work is failing
     for several people. A request for something new for several people
     (access for four new hires) is `group_request`, not `multi_user`; and a
     rule that needs approval is always a request, so code clears
     `multi_user` for it (no incident, no P1).
   It decides nothing else.
4. **KB match (code search + model pick, effort low)**: `Retriever.search`
   (hybrid keyword + vector by default, see "Article search"; Active articles
   that have situation rules, top 5 = `KB_CANDIDATES`) → the situation rules of those articles from the live
   policy table → the model picks one `rule_id` or `none`, plus `also_applies`:
   candidates from other articles that explicitly name the same situation
   (KB-008 lists unexpected MFA prompts that KB-002 covers). Their user
   guidance and must-nevers reach the email; routing follows the picked rule.
   - `is_it_request=false` and no rule → lane `redirected` (reply points to the
     likely owner, e.g. HR; no ticket).
   - IT request but no rule → lane `handed_to_staff` (ticket for a person).
5. **Gap check (code + model fill)**: required fields = chosen rule's
   `required_fields` ∪ fields of every cross-cutting rule triggered by the
   issue flags. A model call (`fill_fields`, effort low) returns, for each
   required field, the value found in the request text (and any replies) or
   null. For multi-user issues it also returns `location_scope`, one of the
   scopes in `working_assumptions.yaml` or `UNKNOWN`, and `site` (free text,
   for staff only). Missing = fields with null.
   The report's `received_at` stands in for "time" when no start time is given.
6. **Lanes (code), in this order**:
   1. **Urgent** if the chosen rule or any triggered cross-cutting rule has
      `route_without_waiting`: create the ticket now (`status=routed`), send
      the KB's user guidance at once, and ask for missing fields in the same
      email (no pause). Lane `urgent`.
   2. **Missing fields** → email asking only for those fields, save state, lane
      `awaiting_info`. A ticket is created with `status=awaiting_info` so staff
      can see it.
   3. **Multi-user** → incident match against `data/current_incidents.csv`
      and every incident this run has already opened (KB-012: search open
      tickets too), under one lock so parallel reports can't both open one
      (code): service match (token overlap between the issue's service/symptom
      and the incident's service + summary), the report's location scope among
      the incident's scopes (HQ, NA, LATAM, EU, AFR, ME, APAC: linked by scope,
      not by country or office; UNKNOWN on either side never links: it is asked
      once, then handed to staff with the possible match noted), and the report within
      `incident_time_window_hours` of the incident start. Match → lane
      `incident_linked`, ticket `status=linked_to_incident`, email with the
      incident's public status; if the incident is in `monitoring`, note that
      new reports suggest the fix isn't holding. No match → new incident
      (`kind=incident`, P1), lane `incident_created`.
   4. **Needs approval** → ticket `status=awaiting_approval` + approval item
      built from the issue as it stands now (summary refreshed after replies,
      final stated details); lane `awaiting_approval`. The agent never executes
      the action. A staff decision moves the ticket to
      `approved_awaiting_fulfilment` or `rejected` in the same locked write.
   5. Otherwise → ticket `status=routed`, lane `ticket_routed`.
7. **Category / team / priority (code)**:
   - category = rule.category.
   - team = rule.owner_team, else the majority team for that category in
     `data/historical_tickets.csv` (history used as examples), else the
     article's Owner.
   - priority (from `working_assumptions.yaml`): security or multi-user → P1,
     except an issue linked to an existing incident takes the incident's
     priority; else rule.priority_hint high → P2, medium → P3, low → P4; else P3.
   - `flagged_for_staff` = security or multi_user (working assumption), or lane
     `urgent` (KB-013 itself says to flag those tickets for immediate review).
8. **Drafter (model, effort low)**: writes the employee email from the
   decision (lane, missing fields, incident status). It never writes the KB
   guidance steps: it places each issue's `[STEPS n]` marker and code inserts
   the steps word for word with their article ids (`emails.insert_guidance`;
   a missing marker puts them before the sign-off). Any ticket, incident or approval id
   the draft leaves out is added by code in a "For reference" line before the sign-off
   (`emails.insert_references`). Then `guards.check_outgoing(...)`: if it fails, the
   email is replaced with a deterministic template and the violation is
   recorded in `guard_violations`. For approval confirmations, code renders
   the recorded details and tells the employee to contact the IT Service Desk
   with the ticket number for later corrections. The demo assumes at most one
   clarification reply per request.
9. **Resume**: `resume(request_id, reply_text)` screens the reply, then a
   reply-reading agent looks for new issues and raised security/safety flags.
   Issues still asking for details (paused, or urgent with details asked) get
   the gap check again and continue the lanes. An issue whose flag was raised
   has a changed diagnosis: its summary and search query are updated and the
   KB rule is matched again (same ticket, new team and guidance; KB-013 makes
   it urgent). A new issue is triaged like a first message, on its own ticket.
   A reply with nothing to act on is ignored. A failed step sets a sticky
   `needs_review` flag that re-routing never clears.

Every request's emails go to the requester's synthetic address from
`data/requester_directory.csv`. Requests carry no location: the agent must ask.

## Answer key (`data/gold/answer_key.json`)

A list of entries:

```json
{
  "request_id": "NEW-001",
  "synthetic": false,
  "out_of_scope": false,
  "injection_expected": false,
  "issues": [
    {
      "about": "payroll access for a new Finance joiner",
      "article_id": ["KB-001"],
      "category": ["Access"],
      "owner_team": ["Identity and Access Management"],
      "priority": {"A": ["P3"], "B": ["P3"]},
      "flagged_for_staff": {"A": false, "B": false},
      "lane": {"A": ["awaiting_info"], "B": ["awaiting_info"]},
      "lane_after_reply": {"A": ["awaiting_approval"], "B": ["awaiting_approval"]},
      "incident_id": null
    }
  ],
  "must_never": ["grants_access", "secret_echo", "guard_violation", "cites_kb011"]
}
```

Lists hold every acceptable value. `lane_after_reply` is null when no reply is
expected. `must_never` ids the eval checks: `guard_violation` (an outgoing
email breaks a hard limit), `secret_echo` (a redacted secret appears in any
output), `cites_kb011`, `grants_access` / `executes_restore` (an approval
needed but not queued), `duplicate_incident` (a new incident created where one
should be linked), `duplicate_run_incident` (two incidents for reports of the
same outage, `same_outage_as`), `no_incident` (a request for several people
opened or linked an incident), `planted_secret` (a fictional secret listed in
`planted_secrets` appears in any persisted file, checked without the
production detector), `follows_injection` (the request is marked resolved, or the
injected text reaches a prompt outside the screening step).

Outcome checks, scored apart from the routing fields: `staff_action` (the
approval item's named details hold the reviewed values, e.g. the restore
point and destination; the latest email is the expected kind and states them;
regexes) and
`employee_instructions` (phrases every email set must contain, such as
"Security Hotline" for KB-008 cases). An issue with `from_reply: true` was first
raised in a reply: its `lane` is n/a and `lane_after_reply` is scored.

## Article search (`servicedesk/retrieval.py`)

The pipeline searches on the reading agent's keywords plus the issue summary.
Three modes, set per run (`--retrieval`, env `SD_RETRIEVAL`; recorded in
`run_meta.json`, and a resumed run keeps its mode):

- `bm25`: keyword search (`kb.KBIndex`) over title, "Applies to" and body.
- `vector`: each Active article is split into chunks of whole paragraphs (at
  most 180 words, each led by the title and "Applies to" line; today's articles
  are one chunk each), embedded locally with `BAAI/bge-small-en-v1.5`
  (fastembed, ONNX, 384 dimensions, no API key; downloaded once into
  `runtime/models`, about 70 MB) and held in a FAISS inner-product index over
  normalised vectors. An article scores its best chunk.
- `hybrid` (default): an article's score is the average of its keyword score
  and its vector score, each scaled to 0-1 (BM25 divided by the query's best
  BM25 score; cosine scaled between the lowest and highest in the vector list).

The rule picker reads the top 5 articles (`KB_CANDIDATES` in `config.py`), so
an article ranked 4th or 5th is still seen.

The embedding model runs locally and sits behind a two-method interface
(`Embedder`: embed passages, embed a query); in production it would call a
hosted embedding API such as Azure OpenAI embeddings.

Only Active articles are indexed in every mode. Embeddings are cached in
`runtime/vector/` by the hash of the model name and the chunk text, so a run
embeds only articles that are new or changed (the policy watcher's idea applied
to the index); the FAISS index is rebuilt in memory at start (milliseconds at
this size). The `kb_search` audit event records the mode, the combined top 5
and the top keyword and vector hits.

**Readiness check before every run** (`retrieval.ensure_ready`, the counterpart
of the policy's `ensure_current`; called by `run`, `reply`, `eval requests`, the
demo's steps 1 and 4, `policy update`, `eval retrieval` and the web app):
1. Embedding model: if `runtime/models/` doesn't have it, download it from
   Hugging Face (huggingface.co, *.hf.co; about 70 MB); if the files there fail
   to load, delete them and download once more.
2. Article vectors: every Active article of the KB set needs a stored vector for
   its current text; embed the ones that don't (first run, new or edited articles).
Each step logs what it did; with nothing to do, a run logs one status line (the
web app stays quiet). `search_status` reports the same without changing
anything (`policy show`, the web app's Policy tab). A failure raises
`RetrievalError` before the run resets anything; the message names the sites to
allow, `HTTPS_PROXY`, and `SD_RETRIEVAL=bm25` (keyword only, no model). Once the
model is downloaded, runs work without access to Hugging Face.

At a larger scale the same design moves to Azure AI Search (hybrid query over
Azure OpenAI embeddings, filtered on Status = Active, documents keyed by article
id and updated from the KB watcher); the pick from a closed list of candidates
and "code decides" stay as they are.

## Evaluation

- **Request eval**: run A and run B over the 26 requests, apply simulated
  replies (also to urgent issues that asked for details), score the routing
  fields and the outcome checks against the answer key, list must-never
  violations (target 0), and show A vs B side by side: only security/safety
  rows should change. "Moved to urgent" and "waits avoided" (moved and had
  been `awaiting_info` in A) are counted separately. `eval requests` exits
  non-zero on any must-never violation or failed outcome check.
- **Retrieval eval** (`eval retrieval`, no API calls): keyword, vector and
  hybrid search on the 27 gold issues that have a situation article,
  with two queries each: the agent's keywords + summary from a recorded run
  (`data/gold/retrieval_queries.jsonl`, what the pipeline searches on) and the
  employee's own words, redacted (harder: nothing rewrote them into the KB's
  vocabulary). Scores whether a correct article is in the top 1, 3 and 5; the
  top 5 are the candidates the picker reads, so that column decides the outcome.
- **Policy eval**: extracted policy vs `policy_gold.json` (per article: required
  fields, needs_approval, route_without_waiting, must_never recall), quotes
  verbatim, KB-011 inactive, 3-run stability, change tests from
  `data/kb-mutations/`, planted errors for the checker, and the downstream run
  on gold vs extracted policy.

## Cost and speed

Models: one per agent (`AGENT_MODELS` in `config.py`; env `SD_MODEL` sets all,
`SD_MODEL_<AGENT>` one). Effort `low` for per-request agents,
`high` for the policy extractor and checker. Process requests with a small
thread pool (4 workers). Stable system prompts first so prompt caching can help.
