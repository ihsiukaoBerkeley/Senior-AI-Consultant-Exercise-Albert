"""Streamlit app: employee view, staff console and policy view over one run folder.

Run with:  streamlit run ui/app.py      (or: python -m servicedesk.cli ui)

Every view reads the run's local store files under runtime/runs/<run>/, so it
also works on runs made by the CLI or the request eval. Submitting a request
or a reply makes live API calls through the same pipeline the CLI uses.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from servicedesk.config import (  # noqa: E402
    CUSTOM_FILE,
    DIRECTORY_FILE,
    INCOMING_FILE,
    REPLIES_FILE,
    SYNTHETIC_FILE,
    assumptions,
    model_for,
)
from servicedesk.evals.request_eval import (  # noqa: E402
    META_FILE,
    REPORTS_DIR,
    RUNS_DIR,
    PIPELINE_AGENTS,
    SNAPSHOT_FILE,
    cost_usd,
    model_calls,
    read_json,
)
from servicedesk.llm import LLMError  # noqa: E402
from servicedesk.pipeline.inputs import RequestFileError, load_request_file  # noqa: E402
from servicedesk.stores import Stores  # noqa: E402

KB_SETS = ["base", "with-kb013"]
DEFAULT_KB = "with-kb013"  # the agreed policy: security and safety cases never wait
KB_LABELS = {"with-kb013": "Current policy (with KB-013)", "base": "Before KB-013 (run A comparison)"}


def kb_label(kb_set: Optional[str]) -> str:
    return KB_LABELS.get(kb_set or "", kb_set or "unknown")
UI_RUN = "ui"
NEW_UI_RUN = "ui (new run)"
PRIORITY_ORDER = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

st.set_page_config(page_title="Service desk agent", layout="wide")


# --------------------------------------------------------------------------
# data helpers (all tolerate missing files)
# --------------------------------------------------------------------------


def md_escape(text: str) -> str:
    """Show plain text through st.markdown without markdown/LaTeX surprises; keep line breaks."""
    out = []
    for ch in text or "":
        out.append("\\" + ch if ch in "\\`*_{}[]<>()#+-.!|$~" else ch)
    return "".join(out).replace("\n", "  \n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@st.cache_data(show_spinner=False)
def builtin_requests() -> dict[str, dict]:
    return {r["request_id"]: r for r in load_jsonl(INCOMING_FILE) + load_jsonl(SYNTHETIC_FILE)}


def custom_requests() -> tuple[dict[str, dict], dict[str, str], Optional[str]]:
    """Your own cases from data/custom_requests.jsonl (read fresh, so edits show on the next rerun)."""
    if not CUSTOM_FILE.exists():
        return {}, {}, None
    try:
        reqs, replies = load_request_file(CUSTOM_FILE)
    except RequestFileError as e:
        return {}, {}, str(e)
    return {r.request_id: {**r.model_dump(), "custom": True} for r in reqs}, replies, None


def sample_requests() -> dict[str, dict]:
    return {**builtin_requests(), **custom_requests()[0]}


@st.cache_data(show_spinner=False)
def directory() -> dict[str, dict]:
    if not DIRECTORY_FILE.exists():
        return {}
    with open(DIRECTORY_FILE, newline="") as f:
        return {row["requester_id"]: row for row in csv.DictReader(f)}


@st.cache_data(show_spinner=False)
def builtin_replies() -> dict[str, str]:
    return read_json(REPLIES_FILE, {}) or {}


def simulated_replies() -> dict[str, str]:
    return {**builtin_replies(), **custom_requests()[1]}


def list_runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
    return [d.name for d in sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)]


SEARCH_LABELS = {"hybrid": "keyword + vector", "bm25": "keyword only", "vector": "vector only"}


def run_meta(run_dir: Path) -> dict:
    meta = read_json(run_dir / META_FILE) or read_json(run_dir / SNAPSHOT_FILE) or {}
    out = {k: meta.get(k) for k in ("kb_set", "policy_source", "policy_version", "retrieval")}
    if not out["kb_set"]:  # a run made without the eval runner: infer from the saved request state
        states = read_json(run_dir / "request_state.json", {}) or {}
        for s in states.values():
            dirs = s.get("kb_dirs") or []
            if dirs:
                out["kb_set"] = "with-kb013" if any("kb-additions" in d or "with-kb013" in d for d in dirs) else "base"
            pol = s.get("policy") or {}
            out["policy_source"] = out["policy_source"] or ("gold" if pol.get("source") == "gold" else "live")
            out["policy_version"] = out["policy_version"] or pol.get("version")
            break
    return out


def show_llm_error(e: LLMError) -> None:
    hint = " Check ANTHROPIC_API_KEY in .env." if e.kind == "auth" else ""
    st.error(f"API error ({e.kind}): {e}  \nrequest_id: {e.request_id or '-'}.{hint}")


def build_pipeline(run_dir: Path, meta: dict):
    """A fresh pipeline bound to the run folder (cheap: KB load + index, no API call)."""
    from servicedesk.evals.request_eval import load_policy, make_llm
    from servicedesk.pipeline import Pipeline
    from servicedesk.policy.compiler import article_dirs

    from servicedesk.retrieval import ensure_ready

    notes: list[str] = []
    policy, source = load_policy(meta["kb_set"], meta.get("policy_source") or "live", log=notes.append)
    # The run keeps the search it started with; runs from before vector search used keywords.
    # Like the policy: download the embedding model and embed new or edited articles if needed.
    retrieval = meta.get("retrieval") or "bm25"
    # The articles as they were when this policy version was approved (a held KB edit changes nothing).
    dirs = article_dirs(meta["kb_set"], policy, log=notes.append)
    ensure_ready(meta["kb_set"], retrieval, log=notes.append, quiet_when_ready=True, dirs=dirs)
    show_policy_notes(notes)
    return Pipeline(policy, dirs, Stores(run_dir), llm=make_llm(), retrieval=retrieval)


def show_policy_notes(notes: list[str], where=st) -> None:
    """Messages from the automatic KB check: a held change or a fallback is a warning, the rest is info."""
    if not notes:
        return
    text = "  \n".join(n.strip() for n in notes)
    if any("held for review" in n or n.startswith("WARNING") for n in notes):
        where.warning(text)
    else:
        where.info(text)


def create_ui_run(kb_set: str, source: str, reset: bool) -> None:
    from servicedesk.evals.request_eval import load_policy, write_run_meta

    run_dir = RUNS_DIR / UI_RUN
    Stores(run_dir, reset=reset)
    notes: list[str] = []
    policy, used = load_policy(kb_set, source, log=notes.append)
    write_run_meta(run_dir, kb_set, policy, used)
    if notes:
        st.session_state["flash"] = notes


def outcome_of(stores: Stores, rid: str) -> Optional[dict]:
    state = stores.load_state(rid) or {}
    return state.get("outcome")


def issue_table(outcome: dict) -> list[dict]:
    rows = []
    for i in outcome.get("issues", []):
        rows.append({
            "#": i.get("issue_index"),
            "summary": i.get("summary"),
            "lane": i.get("lane"),
            "article": i.get("article_id"),
            "rule": i.get("rule_id"),
            "priority": i.get("priority"),
            "team": i.get("owner_team"),
            "flagged": i.get("flagged_for_staff"),
            "flags": ", ".join(f for f in ("security", "safety", "multi_user") if i.get(f)),
            "missing fields": ", ".join(i.get("missing_fields") or []),
            "incident": i.get("incident_id"),
            "ticket": i.get("ticket_id"),
        })
    return rows


def show_email(e) -> None:
    with st.container(border=True):
        st.markdown(f"**{md_escape(e.subject)}**")
        st.caption(f"{e.email_id} · to {e.to} · {e.kind} · cites {', '.join(e.cites) or '-'} · {e.created_at}")
        st.markdown(md_escape(e.body))


# --------------------------------------------------------------------------
# sidebar: run folder
# --------------------------------------------------------------------------

st.sidebar.title("Service desk agent")
st.sidebar.caption("Models: " + ", ".join(sorted({model_for(a) for a in PIPELINE_AGENTS})))
runs = list_runs()
options = runs + ([NEW_UI_RUN] if UI_RUN not in runs else [])
default_idx = options.index(UI_RUN) if UI_RUN in options else 0
choice = st.sidebar.selectbox("Run folder (runtime/runs/)", options, index=default_idx)

run_dir: Optional[Path] = None
meta: dict = {}
if choice == NEW_UI_RUN:
    st.sidebar.info("Create a run named 'ui' to try requests here.")
    kb_new = st.sidebar.radio("Policy", KB_SETS, index=KB_SETS.index(DEFAULT_KB), format_func=kb_label, key="new_kb",
                              help="KB-013 (security and safety cases never wait) is the agreed policy. "
                                   "'Before KB-013' reproduces run A for comparison.")
    src_new = st.sidebar.radio("Policy table", ["live", "gold"], key="new_src",
                               help="live = the approved compiled policy; gold = the hand-checked table")
    if st.sidebar.button("Create run 'ui'", type="primary"):
        try:
            create_ui_run(kb_new, src_new, reset=False)
            st.rerun()
        except Exception as e:  # keep the app up
            st.sidebar.error(f"Could not create the run: {e}")
else:
    run_dir = RUNS_DIR / choice
    meta = run_meta(run_dir)
    st.sidebar.markdown(
        f"**Policy:** {kb_label(meta.get('kb_set'))}  \n"
        f"**Policy table:** {meta.get('policy_source') or '-'} v{meta.get('policy_version') or '-'}  \n"
        f"**Article search:** {SEARCH_LABELS.get(meta.get('retrieval') or 'bm25')}"
    )
    if choice == UI_RUN:
        with st.sidebar.expander("Start the 'ui' run over"):
            kb_r = st.radio("Policy", KB_SETS, format_func=kb_label, key="reset_kb",
                            index=KB_SETS.index(meta["kb_set"] if meta.get("kb_set") in KB_SETS else DEFAULT_KB))
            src_r = st.radio("Policy table", ["live", "gold"], key="reset_src")
            if st.button("Reset run 'ui'"):
                try:
                    create_ui_run(kb_r, src_r, reset=True)
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not reset: {e}")
if st.session_state.get("flash"):
    show_policy_notes(list(st.session_state.pop("flash")), where=st.sidebar)
if not runs:
    st.sidebar.caption("No runs yet. CLI: `python -m servicedesk.cli run --kb with-kb013 --name demo` "
                       "or `eval requests`.")

stores: Optional[Stores] = Stores(run_dir) if run_dir is not None and run_dir.exists() else None

tab_emp, tab_staff, tab_policy = st.tabs(["Employee", "Staff console", "Policy"])


# --------------------------------------------------------------------------
# Employee
# --------------------------------------------------------------------------


def employee_tab() -> None:
    st.subheader("Ask the service desk")
    if stores is None:
        st.info("Pick or create a run folder in the sidebar first.")
        return
    if not meta.get("kb_set"):
        st.warning("This run has no run_meta.json, so its KB set is unknown. Use the 'ui' run or a run made by the CLI.")
        return
    samples = sample_requests()
    people = directory()
    st.caption(f"Policy for this run: **{kb_label(meta.get('kb_set'))}**")
    mode = st.radio("Request", ["From the data files", "New message"], horizontal=True)
    scenario_now = assumptions().get("scenario_now", "2026-09-05T09:00:00Z")

    custom_error = custom_requests()[2]
    if custom_error:
        st.warning(f"data/custom_requests.jsonl was skipped: {custom_error}")

    if mode == "From the data files":
        if not samples:
            st.warning("No request files found under data/.")
            return
        rid = st.selectbox("Request", list(samples), format_func=lambda r: f"{r}: {samples[r]['message'][:80]}")
        row = samples[rid]
        who = people.get(row["requester_id"], {})
        st.markdown(f"**From:** {who.get('name', row['requester_id'])} ({row['requester_id']}) · "
                    f"{row['channel']} · {row['received_at']}"
                    + (" · your case (data/custom_requests.jsonl)" if row.get("custom")
                       else " · synthetic" if row.get("synthetic") else ""))
        st.markdown("> " + md_escape(row["message"]))
        request_row = row
    else:
        ids = list(people)
        requester = st.selectbox("Requester", ids, format_func=lambda r: f"{r}: {people[r]['name']}") if ids else "REQ-000"
        message = st.text_area("Message", placeholder="Describe the problem...", height=120)
        existing = [k for k in (stores.all_states() or {}) if k.startswith("UI-")]
        n = max([int(k.split("-")[1]) for k in existing if k.split("-")[1].isdigit()] + [0]) + 1
        rid = f"UI-{n:03d}"
        st.caption(f"New id {rid}; received_at is the scenario clock {scenario_now} so incident matching works.")
        request_row = {"request_id": rid, "received_at": scenario_now, "channel": "web",
                       "requester_id": requester, "message": message, "synthetic": True}

    if stores.load_state(rid) is not None and mode == "From the data files":
        st.caption("Already processed in this run. Submitting again re-processes it and adds new tickets and emails.")

    if st.button("Submit to the agent", type="primary", disabled=not request_row["message"].strip()):
        from servicedesk.models import IncomingRequest

        with st.spinner("Screening, reading, matching the KB and drafting the reply (live API calls)..."):
            try:
                build_pipeline(run_dir, meta).process(IncomingRequest(**request_row))
                st.session_state["employee_rid"] = rid
            except LLMError as e:
                show_llm_error(e)
            except Exception as e:
                st.error(f"The request could not be processed: {type(e).__name__}: {e}")

    show_rid = rid if stores.load_state(rid) is not None else st.session_state.get("employee_rid")
    if not show_rid or stores.load_state(show_rid) is None:
        return
    outcome = outcome_of(stores, show_rid) or {}
    st.divider()
    st.markdown(f"### {show_rid}: status `{outcome.get('status', '?')}`")
    notes = []
    if outcome.get("injection_detected"):
        notes.append("injected instructions detected and removed")
    if outcome.get("secrets_redacted"):
        notes.append("redacted: " + ", ".join(outcome["secrets_redacted"]))
    if outcome.get("guard_violations"):
        notes.append("guard replaced a draft: " + ", ".join(outcome["guard_violations"]))
    if outcome.get("llm_errors"):
        notes.append(f"{len(outcome['llm_errors'])} model error(s): handed to staff")
    if notes:
        st.caption(" · ".join(notes))
    if outcome.get("issues"):
        st.dataframe(issue_table(outcome), hide_index=True, width="stretch")
    emails = sorted((e for e in stores.outbox() if e.request_id == show_rid), key=lambda e: e.email_id)
    st.markdown("**Emails from the agent**")
    if not emails:
        st.caption("No emails for this request yet.")
    for e in emails:
        show_email(e)

    # Urgent issues are routed at once but can still ask for details (KB-013): the employee
    # must be able to answer those too, without the ticket waiting for them.
    if outcome.get("reply_requested") or outcome.get("status") == "awaiting_info":
        st.markdown("**Reply with the details we asked for**")
        sim = simulated_replies().get(show_rid, "")
        with st.form(f"reply-{show_rid}"):
            text = st.text_area("Your reply", value=sim, height=110)
            if sim:
                st.caption("Pre-filled with the scripted reply for this case (simulated_replies.json or your case's \"reply\" field).")
            sent = st.form_submit_button("Send reply")
        if sent and text.strip():
            with st.spinner("Checking the reply and continuing (live API calls)..."):
                try:
                    build_pipeline(run_dir, meta).resume(show_rid, text)
                    st.session_state["employee_rid"] = show_rid
                    st.rerun()
                except LLMError as e:
                    show_llm_error(e)
                except Exception as e:
                    st.error(f"The reply could not be processed: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# Staff console
# --------------------------------------------------------------------------


def staff_tab() -> None:
    st.subheader("Staff console")
    if stores is None:
        st.info("Pick a run folder in the sidebar. Runs are created by the CLI (`run`, `eval requests`, `demo`) "
                "or by the Employee tab.")
        return
    tickets = stores.tickets()
    approvals = stores.approvals()
    outbox = stores.outbox()
    states = stores.all_states() or {}
    events = stores.audit_events()
    calls = [c for ev in events for c in model_calls(ev)]
    tin = sum(int(c.get("input_tokens") or 0) for c in calls)
    tout = sum(int(c.get("output_tokens") or 0) for c in calls)
    cw = sum(int(c.get("cache_creation_input_tokens") or 0) for c in calls)
    cr = sum(int(c.get("cache_read_input_tokens") or 0) for c in calls)
    cost = sum(cost_usd(int(c.get("input_tokens") or 0), int(c.get("output_tokens") or 0),
                        int(c.get("cache_creation_input_tokens") or 0), int(c.get("cache_read_input_tokens") or 0),
                        model=c.get("model")) for c in calls)

    m = st.columns(5)
    m[0].metric("Tickets", len(tickets))
    m[1].metric("Flagged for staff", sum(t.flagged_for_staff for t in tickets))
    m[2].metric("Pending approvals", sum(a.status == "pending" for a in approvals))
    m[3].metric("Awaiting info", sum(t.status == "awaiting_info" for t in tickets))
    m[4].metric("Model calls", len(calls), help=f"{tin + cw + cr:,} input (incl. cache) / {tout:,} output tokens, approx. ${cost:.2f}")

    if not (tickets or approvals or outbox or states):
        st.info("This run folder is empty. Submit a request in the Employee tab or run the CLI.")
        return

    flags: dict[tuple, str] = {}
    for rid, s in states.items():
        for i in (s.get("outcome") or {}).get("issues", []):
            flags[(rid, i.get("issue_index"))] = ", ".join(f for f in ("security", "safety", "multi_user") if i.get(f))

    st.markdown("#### Queue")
    queue = sorted(tickets, key=lambda t: (not t.flagged_for_staff, PRIORITY_ORDER.get(t.priority or "", 9), t.ticket_id))
    st.dataframe(
        [{
            "ticket": t.ticket_id,
            "kind": t.kind,
            "request": t.request_id,
            "status": t.status,
            "priority": t.priority,
            "team": t.owner_team,
            "category": t.category,
            "flagged": t.flagged_for_staff,
            "flags": flags.get((t.request_id, t.issue_index), ""),
            "missing fields": ", ".join(t.missing_fields),
            "incident": t.linked_incident,
            "article": t.article_id,
            "summary": t.summary,
        } for t in queue],
        hide_index=True,
        width="stretch",
    )

    st.markdown("#### Approvals")
    st.info("Approving or rejecting only records the decision. The prototype never executes the action "
            "(no access is granted, nothing is restored or reset): a person does that in the real system.")
    staff_name = st.text_input("Decided by", value="service-desk staff")
    if not approvals:
        st.caption("No approval items in this run.")
    for a in sorted(approvals, key=lambda a: (a.status != "pending", a.approval_id)):
        c = st.columns([6, 1, 1])
        decided = f" by {a.decided_by} at {a.decided_at}" if a.decided_by else ""
        c[0].markdown(f"**{a.approval_id}** · {a.request_id} · {a.ticket_id} · `{a.status}`{md_escape(decided)}  \n"
                      f"{md_escape(a.action)}  \nApprover: {md_escape(a.approver or 'not named in the KB')}")
        if a.details:
            with c[0].expander("Details for the decision", expanded=a.status == "pending"):
                st.dataframe([{"field": k, "value": v} for k, v in a.details.items()], hide_index=True,
                             width="stretch")
        if a.status == "pending":
            if c[1].button("Approve", key=f"approve-{a.approval_id}"):
                stores.decide_approval(a.approval_id, True, staff_name or "staff")
                st.rerun()
            if c[2].button("Reject", key=f"reject-{a.approval_id}"):
                stores.decide_approval(a.approval_id, False, staff_name or "staff")
                st.rerun()

    st.markdown("#### Outbox")
    rids = sorted({e.request_id for e in outbox})
    pick = st.selectbox("Request", ["all"] + rids, key="outbox_filter")
    for e in sorted(outbox, key=lambda e: e.email_id, reverse=True):
        if pick != "all" and e.request_id != pick:
            continue
        with st.expander(f"{e.email_id} · {e.request_id} · {e.kind} · {e.subject}"):
            st.caption(f"to {e.to} · cites {', '.join(e.cites) or '-'} · {e.created_at}")
            st.markdown(md_escape(e.body))

    st.markdown("#### Audit trail")
    by_rid: dict[str, list[dict]] = {}
    for ev in events:
        by_rid.setdefault(ev.get("request_id", "?"), []).append(ev)
    if not by_rid:
        st.caption("No audit events.")
    for rid in sorted(by_rid):
        evs = by_rid[rid]
        rc = [(ev.get("step"), c) for ev in evs for c in model_calls(ev)]
        status = (states.get(rid, {}).get("outcome") or {}).get("status", "-")
        with st.expander(f"{rid} · {status} · {len(evs)} events · {len(rc)} model calls"):
            if rc:
                st.markdown("Model calls")
                st.dataframe([{
                    "step": step,
                    "call": c.get("call"),
                    "request_id": c.get("request_id"),
                    "stop_reason": c.get("stop_reason"),
                    "input_tokens": c.get("input_tokens"),
                    "output_tokens": c.get("output_tokens"),
                    "latency_s": c.get("latency_s"),
                    "fallback": c.get("fallback_used"),
                } for step, c in rc], hide_index=True, width="stretch")
            rows = []
            for ev in evs:
                detail = {k: v for k, v in ev.items() if k not in ("ts", "request_id", "step", "llm")}
                text = json.dumps(detail, ensure_ascii=False, default=str)
                rows.append({"ts": ev.get("ts"), "step": ev.get("step"),
                             "detail": text if len(text) <= 300 else text[:297] + "..."})
            st.dataframe(rows, hide_index=True, width="stretch")


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


def latest_policy_files(kb_set: str) -> tuple[Optional[Path], Optional[Path], str]:
    """(diff.md, findings.json, label): the pending candidate's, else the live version's."""
    from servicedesk.policy import compiler

    d = compiler.policy_dir(kb_set)
    if (d / "diff.md").exists():
        return d / "diff.md", d / "findings.json", "pending candidate"
    ptr = compiler.live_pointer(kb_set)
    if ptr:
        v = ptr["version"]
        return d / f"v{v}.diff.md", d / f"v{v}.findings.json", f"live v{v}"
    return None, None, ""


def held_policy_review(compiler, kb_set: str, stat: dict) -> None:
    """A recompiled policy the checker held: show why, and let the person viewing accept it."""
    from servicedesk.policy.watcher import detect_changes, load_articles

    cand, findings = compiler.load_candidate(kb_set) or (None, [])
    blocking = [f for f in findings if f.blocking]
    st.warning(f"A recompiled policy (candidate v{stat['pending_candidate']}) is held for review: "
               f"{len(blocking)} blocking checker finding(s). Until a person accepts it, runs keep the approved policy and the articles as they were when it was approved: the change has no effect on requests. "
               "Accept it here, or fix the article and click **Check the KB now**.")
    for f in blocking:
        st.markdown(f"- **{f.kind}** ({f.rule_id or f.article_id or 'table'}): {md_escape(f.explanation)}")
    current = detect_changes(load_articles(kb_set), None).hashes
    if cand is None or cand.kb_hashes != current:
        st.caption("The KB changed again since this candidate was compiled: click **Check the KB now** first.")
        return
    if st.button("Accept the held change", key="policy_accept_held",
                 help="Approve the recompiled policy with these findings; it becomes the live policy."):
        table = compiler.approve(kb_set, approved_by="web app reviewer (accepted the checker's findings)",
                                 acknowledge_findings=True)
        st.success(f"'{kb_set}' policy v{table.version} is live.")
        st.rerun()


def policy_tab() -> None:
    st.subheader("Policy compiled from the KB")
    try:
        from servicedesk.evals.request_eval import load_policy
        from servicedesk.policy import compiler
    except ImportError as e:
        st.warning(f"The policy compiler is not available: {e}")
        return
    default = KB_SETS.index(meta["kb_set"] if meta.get("kb_set") in KB_SETS else DEFAULT_KB)
    kb_set = st.radio("Policy", KB_SETS, index=default, format_func=kb_label, horizontal=True, key="policy_kb")
    stat = compiler.status(kb_set)
    changes = stat.get("kb_changes_since_live") or {}
    if changes.get("error"):
        st.error(f"The KB has a file that breaks the naming rules: {changes['error']}")
    else:
        touched = {k: v for k, v in changes.items() if k in ("added", "changed", "removed") and v}
        if stat.get("pending_blocking_findings") and touched:
            held_policy_review(compiler, kb_set, stat)
        elif touched:
            st.info("The KB changed since the approved policy (" + "; ".join(f"{k}: {', '.join(v)}" for k, v in touched.items())
                    + "). It is recompiled automatically before the next request, or now:")
    try:
        from servicedesk.retrieval import search_status

        st.caption(search_status(kb_set, dirs=compiler.article_dirs(kb_set, compiler.load_live(kb_set))).summary())
    except Exception as e:
        st.caption(f"Article search status unavailable: {e}")
    if st.button("Check the KB now", key="policy_update",
                 help="Recompile if any article was added, edited or removed; approve automatically when the checker "
                      "finds nothing blocking. Then make article search ready: download the embedding model if it is "
                      "missing and embed new or edited articles. Runs do all this on their own."):
        notes: list[str] = []
        with st.spinner("Comparing article hashes, recompiling and embedding what changed "
                        "(live API calls if anything did)..."):
            try:
                from servicedesk.retrieval import ensure_ready

                up = compiler.ensure_current(kb_set, log=notes.append)
                if up.action == "unchanged":
                    notes.append(f"No article changed since policy v{up.table.version}.")
                ensure_ready(kb_set, log=notes.append, dirs=compiler.article_dirs(kb_set, up.table, log=notes.append))
            except LLMError as e:
                show_llm_error(e)
            except Exception as e:
                st.error(f"Could not update the policy or the article search: {type(e).__name__}: {e}")
        show_policy_notes(notes)
    live = compiler.load_live(kb_set)
    table = live
    if live is None:
        st.info(f"No approved policy for '{kb_set}' yet. It is compiled from the KB automatically the first time a "
                "request is processed, or with the button above.")
        if st.toggle("Show the hand-checked gold policy instead", key="show_gold"):
            table, _ = load_policy(kb_set, "gold", log=lambda _m: None)
    else:
        ptr = compiler.live_pointer(kb_set) or {}
        st.caption(f"Live v{live.version} · {len(live.rules)} rules · approved by {ptr.get('approved_by', '-')} "
                   f"at {ptr.get('approved_at', '-')} · inactive: {', '.join(live.inactive_articles) or 'none'}")

    if table is not None:
        articles = sorted({r.article_id for r in table.rules})
        pick = st.multiselect("Filter by article", articles, key="policy_articles")
        rules = [r for r in table.rules if not pick or r.article_id in pick]
        st.dataframe([{
            "rule": r.rule_id,
            "scope": r.scope,
            "triggers": ", ".join(r.triggers),
            "category": r.category,
            "owner team": r.owner_team,
            "required fields": ", ".join(r.required_fields),
            "needs approval": r.needs_approval,
            "approver": r.approver,
            "route without waiting": r.route_without_waiting,
            "priority hint": r.priority_hint,
            "must never": "; ".join(r.must_never),
        } for r in rules], hide_index=True, width="stretch")
        if rules:
            rid = st.selectbox("Rule details", [r.rule_id for r in rules], key="policy_rule")
            r = table.get(rid)
            if r:
                st.markdown(f"**Situation:** {md_escape(r.situation)}")
                if r.user_guidance:
                    st.markdown("**User guidance:**  \n" + "  \n".join("- " + md_escape(g) for g in r.user_guidance))
                if r.quotes:
                    st.markdown("**Quotes (verbatim from the article):**  \n"
                                + "  \n".join("> " + md_escape(q) for q in r.quotes))

    st.markdown("#### Latest diff and checker findings")
    cols = st.columns(len(KB_SETS))
    for col, kb in zip(cols, KB_SETS):
        with col:
            st.markdown(f"**{kb}**")
            diff_path, findings_path, label = latest_policy_files(kb)
            if diff_path is None or not diff_path.exists():
                st.caption("No compile yet.")
                continue
            st.caption(f"{label}: {diff_path.relative_to(ROOT)}")
            findings = (read_json(findings_path, {}) or {}).get("findings", []) if findings_path else []
            if findings:
                st.dataframe([{
                    "kind": f.get("kind"),
                    "where": f.get("rule_id") or f.get("article_id") or "table",
                    "needs decision": f.get("blocking"),
                    "action": f.get("action"),
                    "explanation": f.get("explanation"),
                } for f in findings], hide_index=True, width="stretch")
            else:
                st.caption("No checker findings.")
            with st.expander("diff.md"):
                st.markdown(diff_path.read_text())

    st.markdown("#### Eval reports")
    reports = sorted(REPORTS_DIR.glob("*.md")) if REPORTS_DIR.exists() else []
    if not reports:
        st.caption("No reports yet: `python -m servicedesk.cli eval requests` and `eval policy`.")
    for p in reports:
        with st.expander(p.name, expanded=p.name == "request_eval.md"):
            st.markdown(p.read_text())


for tab, fn in ((tab_emp, employee_tab), (tab_staff, staff_tab), (tab_policy, policy_tab)):
    with tab:
        try:
            fn()
        except Exception as e:  # never let one view take the app down
            st.error(f"This view hit an error: {type(e).__name__}: {e}")
