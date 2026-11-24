"""Command line: python -m servicedesk.cli <command>

  policy update  --kb base|with-kb013      recompile if the KB changed; auto-approve when the checker is clean;
                                           make article search ready (model download, new or edited articles)
                                           (every run does this first, so it is rarely needed)
  policy compile --kb base|with-kb013      extract + check a candidate policy, write diff.md
  policy approve --kb ... [--acknowledge-findings] [--by NAME]
  policy show    --kb ...                  live version summary + latest diff.md path
  run    --kb base|with-kb013 --name NAME [--policy live|gold] [--no-replies] [--request ID ...]
         [--requests FILE]                   your own cases, e.g. data/custom_requests.jsonl
         [--retrieval hybrid|bm25|vector]    article search (default hybrid: keyword + vector)
  reply  --name NAME --request NEW-007 --text "..."   resume one paused request
  eval   requests [--policy live|gold] [--no-replies] [--retrieval ...]   run A and B, score, write report
  eval   retrieval [--kb with-kb013|base]   keyword vs vector vs hybrid search on the gold issues (no API calls)
  eval   policy                             extracted policy vs gold
  demo   [--policy live|gold] [--retrieval ...]   the KB-013 story end to end
  ui     [--port 8501]                      streamlit run ui/app.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any, Optional

from .config import POLICY_DIR, ROOT, model_for
from .kb import KBFormatError
from .llm import LLMError
from .pipeline.inputs import RequestFileError
from .retrieval import MODES as RETRIEVAL_MODES, RetrievalError

KB_SETS = ["base", "with-kb013"]
RETRIEVAL_HELP = ("article search: hybrid (keyword + vector, the default), bm25 (keyword only) or vector; "
                  "the default can also be set with SD_RETRIEVAL")

HINTS = {
    "auth": "Check ANTHROPIC_API_KEY in .env (see .env.example).",
    "permission": "The API key has no access to this model or feature; check the key's workspace.",
    "not_found": "Check the model names (config.AGENT_MODELS, or SD_MODEL / SD_MODEL_<AGENT> in the environment).",
    "billing": "The account has no credit; check billing in the Claude Console.",
    "rate_limited": "Still rate limited after retries; wait a minute or use fewer workers.",
    "overloaded": "The API is overloaded; try again shortly.",
    "connection": "Could not reach the API; check the network connection.",
    "timeout": "The request timed out after retries; try again.",
}


# --------------------------------------------------------------------------
# small output helpers
# --------------------------------------------------------------------------


def _rel(path: Any) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        print(f"{question} [y/N] y (--yes)")
        return True
    if not sys.stdin.isatty():
        print(f"{question} -> no (not an interactive terminal; pass --yes to accept)")
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def _prepare_policy(kb_set: str, args) -> Any:
    """Bring the KB set's policy up to date before running requests (what every run does).

    Recompiles when an article changed and approves automatically when the checker finds
    nothing blocking. A change the checker holds is put to the person at the terminal
    (or accepted by --yes); unattended, it stays held and runs keep the approved policy and
    the articles as they were when it was approved.
    """
    if getattr(args, "policy", "live") != "live":
        return None
    from .policy import compiler

    return compiler.ensure_current(kb_set, confirm=lambda q: _confirm(q, bool(getattr(args, "yes", False))))


def _prepare_search(kb_set: str, args) -> Any:
    """Make article search ready before running requests (what every run does): download the
    embedding model if it is missing and embed new or edited articles. Keyword search needs neither.
    Checks the articles the run will search: the live policy version's copy of them."""
    from .policy import compiler
    from .retrieval import ensure_ready

    live = compiler.load_live(kb_set) if getattr(args, "policy", "live") == "live" else None
    return ensure_ready(kb_set, getattr(args, "retrieval", None), dirs=compiler.article_dirs(kb_set, live))


def _step(n: int, total: int, title: str) -> None:
    print()
    print(f"== Step {n}/{total}: {title}")


def _issue_rows(snap: dict, only: Optional[list[str]] = None) -> list[tuple]:
    rows = []
    for rid, s in snap.get("requests", {}).items():
        if only and rid not in only:
            continue
        before, after = s.get("before"), s.get("after")
        if before is None:
            rows.append((rid, "-", "-", "-", "", f"not processed: {s.get('error') or 'unknown error'}", ""))
            continue
        after_issues = {i["issue_index"]: i for i in (after or {}).get("issues", [])}
        for i in before.get("issues", []):
            a = after_issues.get(i["issue_index"])
            lane = i["lane"] + (f" -> {a['lane']}" if a and a["lane"] != i["lane"] else "")
            final = a or i
            flags = ",".join(f for f in ("security", "safety", "multi_user") if final.get(f))
            flag = "FLAG" if final.get("flagged_for_staff") else ""
            rows.append((rid, str(i["issue_index"]), final.get("article_id") or "-", final.get("priority") or "-",
                         flag, lane, final.get("owner_team") or "-", flags))
    return rows


def _print_issue_table(snap: dict, only: Optional[list[str]] = None) -> None:
    rows = _issue_rows(snap, only)
    print(f"{'Request':<9}{'#':<3}{'Article':<9}{'Pri':<5}{'Flag':<6}{'Lane (before -> after reply)':<40}Team")
    for r in rows:
        rid, idx, art, pri, flag, lane, team = r[:7]
        flags = f" [{r[7]}]" if len(r) > 7 and r[7] else ""
        print(f"{rid:<9}{idx:<3}{art:<9}{pri:<5}{flag:<6}{lane:<40}{team}{flags}")


def _lane_counts(snap: dict) -> str:
    counts: dict[str, int] = {}
    for s in snap.get("requests", {}).values():
        final = s.get("after") or s.get("before")
        for i in (final or {}).get("issues", []):
            counts[i["lane"]] = counts.get(i["lane"], 0) + 1
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))


def _print_email(e: dict) -> None:
    print(f"  {e['email_id']}  to {e['to']}  [{e['kind']}]  cites {', '.join(e.get('cites') or []) or '-'}")
    print(f"  Subject: {e['subject']}")
    for para in e["body"].splitlines():
        print(textwrap.fill(para, width=100, initial_indent="    ", subsequent_indent="    ") if para.strip() else "")


def _print_headline(h: dict) -> None:
    mn = h["must_never_violations"]
    print(f"Must-never violations: {mn['total']} (target 0; A {mn['A']}, B {mn['B']})")
    print("Routing fields (A / B):")
    for f, v in h["field_accuracy"].items():
        print(f"  {f:<20}{v['A']:>8} / {v['B']}")
    oc = h.get("outcome_checks") or {}
    if oc:
        print(f"Outcome checks, staff action and employee instructions (A / B): {oc['A']} / {oc['B']}")
        for f in oc.get("failures", []):
            print(f"  FAIL {f['run']} {f['request_id']}: {f['id']}")
    print("Incident handling:")
    for c in h["incident_handling"]:
        print(f"  {c['request_id']} -> {c['expected']}: A {'pass' if c['A'] else 'FAIL'}, B {'pass' if c['B'] else 'FAIL'}")
    ab = h["ab_changes"]
    print(f"A vs B: changed {', '.join(ab['requests_changed']) or 'nothing'}")
    waits = ab.get("waits_avoided", [])
    print(f"  moved to urgent in B: {len(ab['moved_to_urgent'])} ({', '.join(ab['moved_to_urgent']) or 'none'})")
    print(f"  of those, waited for details in A (waits avoided): {len(waits)} ({', '.join(waits) or 'none'})")
    print(f"  unexpected changes: {ab['unexpected_changes']}; expected changes missing: {ab['missed_changes']}")
    print(f"Controls SYN-006..SYN-008 identical: {'yes' if h['controls_identical'] else 'NO'}")
    llm = h["llm"]
    print(f"Guard-replaced drafts: A {h['guard_replaced']['A']}, B {h['guard_replaced']['B']}")
    print(f"LLM: {llm['calls']} calls, {llm['input_tokens']:,} in (incl. cache) / {llm['output_tokens']:,} out tokens, "
          f"approx. ${llm['approx_cost_usd']:.2f} (prompt-cache tokens included)")


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------


def _print_findings(findings: list, limit: int = 20) -> None:
    for f in findings[:limit]:
        where = f.rule_id or f.article_id or "table"
        tag = "NEEDS DECISION" if f.blocking else "info"
        expl = f.explanation if len(f.explanation) <= 160 else f.explanation[:157] + "..."
        print(f"  - [{f.kind}, {tag}] {where}: {expl}")
    if len(findings) > limit:
        print(f"  ... {len(findings) - limit} more in findings.json")


def _compile(kb_set: str, use_cache: bool = True):
    from .evals.request_eval import make_llm
    from .policy import compiler

    print(f"Compiling policy for KB set '{kb_set}' (extractor {model_for('extractor')}, checker "
          f"{model_for('checker')}, effort high; cached outputs reused)...")
    result = compiler.compile(kb_set, llm=make_llm(), use_cache=use_cache)
    print(result.summary())
    if result.findings:
        print(f"Checker findings ({len(result.findings)}, {len(result.blocking)} need a reviewer decision):")
        _print_findings(result.findings)
    print(f"Diff: {_rel(result.paths.get('diff.md', POLICY_DIR / kb_set / 'diff.md'))}")
    return result


def cmd_policy_compile(args) -> int:
    result = _compile(args.kb, use_cache=not args.no_cache)
    print(f"Approve with: python -m servicedesk.cli policy approve --kb {args.kb}"
          + (" --acknowledge-findings" if result.blocking else ""))
    return 0


def _approve(kb_set: str, by: str, acknowledge: bool) -> Optional[Any]:
    from .policy import compiler

    try:
        table = compiler.approve(kb_set, approved_by=by, acknowledge_findings=acknowledge)
    except compiler.PolicyApprovalError as e:
        print(f"Approval refused: {e}")
        return None
    print(f"Approved: '{kb_set}' live policy is now v{table.version} ({len(table.rules)} rules), approved by {by}.")
    return table


def cmd_policy_approve(args) -> int:
    return 0 if _approve(args.kb, args.by, args.acknowledge_findings) else 2


def cmd_policy_update(args) -> int:
    """What every run does first, on its own: recompile if the KB changed, auto-approve if clean."""
    from .policy import compiler

    up = compiler.ensure_current(args.kb, confirm=lambda q: _confirm(q, args.yes))
    if up.action == "unchanged":
        print(f"'{args.kb}' policy v{up.table.version} is live, and no article changed since it was compiled.")
    _prepare_search(args.kb, args)
    return 2 if up.action == "held" else 0


def latest_diff_path(kb_set: str) -> Optional[Path]:
    """Pending candidate's diff.md, else the live version's, else the newest vN.diff.md."""
    from .policy import compiler

    d = compiler.policy_dir(kb_set)
    if (d / "diff.md").exists():
        return d / "diff.md"
    ptr = compiler.live_pointer(kb_set)
    if ptr and (d / f"v{ptr['version']}.diff.md").exists():
        return d / f"v{ptr['version']}.diff.md"
    diffs = sorted(d.glob("v*.diff.md"), key=lambda p: p.stat().st_mtime)
    return diffs[-1] if diffs else None


def cmd_policy_show(args) -> int:
    from .policy import compiler

    kb = args.kb
    st = compiler.status(kb)
    live = compiler.load_live(kb)
    ptr = compiler.live_pointer(kb) or {}
    if live is None:
        print(f"KB set '{kb}': no approved policy yet. The next run compiles it from the KB "
              f"(or now: python -m servicedesk.cli policy update --kb {kb}).")
    else:
        print(f"KB set '{kb}': live v{live.version} ({len(live.rules)} rules, source {live.source}), "
              f"approved by {ptr.get('approved_by', '-')} at {ptr.get('approved_at', '-')}")
        print(f"  versions: {', '.join(f'v{v}' for v in st['versions'])}")
        print(f"  inactive articles: {', '.join(live.inactive_articles) or 'none'}")
        per_article: dict[str, int] = {}
        for r in live.rules:
            per_article[r.article_id] = per_article.get(r.article_id, 0) + 1
        print("  rules per article: " + ", ".join(f"{a} {n}" for a, n in sorted(per_article.items())))
        cross = [r for r in live.rules if r.scope == "cross_cutting"]
        print("  cross-cutting: " + (", ".join(f"{r.rule_id} (triggers {'/'.join(r.triggers)}"
                                              f"{', route without waiting' if r.route_without_waiting else ''})"
                                              for r in cross) or "none"))
        appr = [r.rule_id for r in live.rules if r.needs_approval]
        print(f"  needs approval ({len(appr)}): {', '.join(appr) or 'none'}")
        rww = [r.rule_id for r in live.rules if r.route_without_waiting]
        print(f"  route without waiting: {', '.join(rww) or 'none'}")
        if ptr.get("acknowledged_findings"):
            print(f"  findings acknowledged at approval: {len(ptr['acknowledged_findings'])}")
    if st.get("pending_candidate") is not None:
        print(f"  pending candidate: v{st['pending_candidate']} "
              f"({st['pending_blocking_findings']} finding(s) need a reviewer decision)")
    changes = st.get("kb_changes_since_live") or {}
    if isinstance(changes, dict) and not changes.get("error"):
        touched = {k: v for k, v in changes.items() if isinstance(v, list) and v and k != "unchanged"}
        print("  KB changes since live: " + ("; ".join(f"{k}: {', '.join(map(str, v))}" for k, v in touched.items())
                                             or "none"))
    diff = latest_diff_path(kb)
    print(f"  latest diff: {_rel(diff) if diff else 'none yet'}")
    from .retrieval import search_status

    print(search_status(kb, dirs=compiler.article_dirs(kb, compiler.load_live(kb))).summary())
    return 0


# --------------------------------------------------------------------------
# run / reply
# --------------------------------------------------------------------------


def cmd_run(args) -> int:
    from .evals.request_eval import run_dir_for, run_requests

    up = _prepare_policy(args.kb, args)
    _prepare_search(args.kb, args)
    snap = run_requests(args.kb, args.name, policy_source=args.policy, reset=True,
                        apply_replies=not args.no_replies, workers=args.workers, only=args.request or None,
                        request_file=Path(args.requests) if args.requests else None, retrieval=getattr(args, "retrieval", None),
                        policy_update=up)
    print()
    _print_issue_table(snap)
    print()
    print(f"Final lanes: {_lane_counts(snap)}")
    print(f"Run folder: {_rel(run_dir_for(args.name))}")
    return 0


def load_run_meta(run_dir: Path) -> dict:
    from .evals.request_eval import META_FILE, SNAPSHOT_FILE, read_json

    meta = read_json(run_dir / META_FILE) or read_json(run_dir / SNAPSHOT_FILE) or {}
    return {k: meta.get(k) for k in ("kb_set", "policy_source", "policy_version", "retrieval")}


def build_pipeline_for_run(run_name: str, log=print, policy_update=None):
    """Pipeline bound to an existing run folder (its KB set and policy source). `policy_update`:
    a policy check the caller already made (see request_eval.load_policy)."""
    from .evals.request_eval import load_policy, make_llm, run_dir_for
    from .pipeline import Pipeline
    from .policy.compiler import article_dirs
    from .stores import Stores

    run_dir = run_dir_for(run_name)
    if not run_dir.exists():
        raise FileNotFoundError(f"no run named '{run_name}' under {_rel(run_dir.parent)}")
    meta = load_run_meta(run_dir)
    if not meta.get("kb_set"):
        raise FileNotFoundError(f"{_rel(run_dir)} has no run_meta.json: start it with `run` or the UI")
    policy, source = load_policy(meta["kb_set"], meta.get("policy_source") or "live", log=log, update=policy_update)
    if meta.get("policy_version") and policy.version != meta["policy_version"]:
        log(f"Note: run used policy v{meta['policy_version']}; resuming with {source} v{policy.version}.")
    stores = Stores(run_dir)
    # A run keeps the search it started with; runs from before vector search used keywords.
    retrieval = meta.get("retrieval") or "bm25"
    from .retrieval import ensure_ready

    # The articles as they were when this policy version was approved (a held KB edit changes nothing).
    dirs = article_dirs(meta["kb_set"], policy, log=log)
    ensure_ready(meta["kb_set"], retrieval, log=log, dirs=dirs)
    return Pipeline(policy, dirs, stores, llm=make_llm(), retrieval=retrieval), stores, meta


def cmd_reply(args) -> int:
    from .evals.request_eval import run_dir_for

    meta0 = load_run_meta(run_dir_for(args.name))
    up = None
    if meta0.get("kb_set") and (meta0.get("policy_source") or "live") == "live":
        up = _prepare_policy(meta0["kb_set"], args)
    pipeline, stores, meta = build_pipeline_for_run(args.name, policy_update=up)
    state = stores.load_state(args.request)
    if state is None:
        print(f"Request {args.request} was not processed in run '{args.name}'.")
        return 2
    seen = {e.email_id for e in stores.outbox() if e.request_id == args.request}
    print(f"Resuming {args.request} in run '{args.name}' (KB set {meta['kb_set']})...")
    outcome = pipeline.resume(args.request, args.text)
    snap = {"requests": {args.request: {"before": (state.get("outcome") or outcome.model_dump()),
                                        "after": outcome.model_dump()}}}
    after = stores.load_state(args.request) or {}
    if (after.get("history") or [{}])[-1].get("stage", "").endswith("_ignored"):
        print(f"Reply ignored: nothing in {args.request} is waiting for details and the reply raises nothing new. "
              f"(Runs apply the simulated replies unless started with --no-replies.)")
        return 0
    print(f"Status: {outcome.status}")
    _print_issue_table(snap)
    new = [e for e in stores.outbox() if e.request_id == args.request and e.email_id not in seen]
    for e in new:
        print()
        _print_email(e.model_dump())
    if outcome.guard_violations:
        print(f"Guard replaced a draft: {', '.join(outcome.guard_violations)}")
    return 0


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------


def cmd_eval_requests(args) -> int:
    from .evals.request_eval import RUN_KB, run_request_eval

    ups = {}
    for kb_set in RUN_KB.values():
        ups[kb_set] = _prepare_policy(kb_set, args)
        _prepare_search(kb_set, args)
    res = run_request_eval(policy_source=args.policy, apply_replies=not args.no_replies, retrieval=getattr(args, "retrieval", None),
                           policy_updates=ups)
    print()
    _print_headline(res["headline"])
    print(f"Report: {_rel(res['report_md'])} (+ .json)")
    return eval_exit_status(res["headline"])


def eval_exit_status(headline: dict) -> int:
    """0 only when nothing failed: no must-never violation and no failed outcome check."""
    failures = (headline.get("outcome_checks") or {}).get("failures") or []
    if headline["must_never_violations"]["total"] or failures:
        print(f"FAILED: {headline['must_never_violations']['total']} must-never violation(s), "
              f"{len(failures)} failed outcome check(s)")
        return 1
    return 0


def cmd_eval_retrieval(args) -> int:
    from .evals.retrieval_eval import run_retrieval_eval

    res = run_retrieval_eval(args.kb)
    print(f"Report: {_rel(res['report_md'])} (+ .json)")
    return 0


def cmd_eval_policy(args) -> int:
    try:
        from .evals.policy_eval import run_policy_eval
    except ImportError as e:
        print(f"The policy eval is not available ({e}).")
        return 2

    res = run_policy_eval()
    if isinstance(res, dict):
        for key in ("headline", "summary"):
            if isinstance(res.get(key), dict):
                for k, v in res[key].items():
                    print(f"  {k}: {v if not isinstance(v, (dict, list)) else json.dumps(v, default=str)[:160]}")
                break
        from .evals.request_eval import REPORTS_DIR

        if (REPORTS_DIR / "policy_eval.md").exists():
            print(f"Report: {_rel(REPORTS_DIR / 'policy_eval.md')} (+ .json)")
    elif res is not None:
        print(res if not isinstance(res, (tuple, list)) else "\n".join(map(str, res)))
    return 0


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------


def _security_cases(snap: dict) -> tuple[list[str], list[str]]:
    """Security/safety requests in run A: those that waited for details, and those routed
    straight away anyway (KB-013 can't avoid a wait that never happened)."""
    waited, routed = [], []
    for rid, s in snap.get("requests", {}).items():
        issues = [i for i in (s.get("before") or {}).get("issues", []) if i.get("security") or i.get("safety")]
        if any(i["lane"] == "awaiting_info" for i in issues):
            waited.append(rid)
        elif any(i["lane"] != "urgent" for i in issues):
            routed.append(rid)
    return waited, routed


def cmd_demo(args) -> int:
    from .evals.request_eval import RUN_NAMES, run_requests, score_and_report
    from .policy import compiler
    from .policy.diff import diff_tables

    total = 6
    gold = args.policy == "gold"
    print("Demo: the KB adds KB-013 ('security and safety cases must not wait'); the policy changes; "
          "only security/safety requests change lane.")

    _step(1, total, "Base policy and article search, from the 12 provided KB articles")
    if gold:
        print("Using the hand-checked gold policy (no compile).")
        base_table = compiler.load_gold("base")
    else:
        up_a = _prepare_policy("base", args)
        if up_a.table is None:
            print("Stopped: the base policy is held for review (see above). Accept it with "
                  "`python -m servicedesk.cli policy approve --kb base --acknowledge-findings`, or rerun the demo "
                  "and answer y.")
            return 2
        if up_a.action == "unchanged":
            print(f"'base' policy v{up_a.table.version} is live, and no article changed since it was compiled.")
        base_table = up_a.table
    _prepare_search("base", args)

    _step(2, total, "Run A: base KB, all 26 requests, simulated replies")
    snap_a = run_requests("base", RUN_NAMES["A"], policy_source=args.policy, workers=args.workers,
                          retrieval=getattr(args, "retrieval", None), policy_update=None if gold else up_a)

    _step(3, total, "Security and safety requests in run A")
    waited, routed = _security_cases(snap_a)
    if waited:
        print(f"Without KB-013 these {len(waited)} waited for details before the ticket was complete:")
        _print_issue_table({"requests": {rid: dict(s, after=None) for rid, s in snap_a["requests"].items()}}, only=waited)
    else:
        print("No security/safety request waited for details in run A.")
    if routed:
        print(f"Already routed without waiting in run A (not flagged urgent): {', '.join(routed)}")

    _step(4, total, "KB-013 added: the watcher sees a new article, the policy is recompiled (diff below) "
                    "and the article is embedded for search")
    up_b = None
    if gold:
        table_b = compiler.load_gold("with-kb013")
    else:
        up_b = _prepare_policy("with-kb013", args)
        if up_b.action == "unchanged":
            print(f"'with-kb013' policy v{up_b.table.version} is live, and no article changed since it was compiled.")
        table_b = up_b.result.candidate if up_b.action == "held" and up_b.result else up_b.table
    if table_b is not None:
        d = diff_tables(base_table, table_b)
        c = d.counts()
        print(f"Base v{base_table.version} -> with-kb013 v{table_b.version}: +{c['added']} added, -{c['removed']} removed, "
              f"~{c['changed']} changed, {c['unchanged']} unchanged.")
        for r in d.added:
            print(f"  + {r.rule_id} ({r.scope}, triggers {'/'.join(r.triggers) or '-'}, "
                  f"route_without_waiting={r.route_without_waiting})")
            for q in r.quotes[:2]:
                print(f"      \"{q}\"")
        for ch in d.changed:
            print(f"  ~ {ch.new_rule_id}: {', '.join(ch.changes) or ', '.join(ch.added) or ', '.join(ch.removed)}")
    _prepare_search("with-kb013", args)

    _step(5, total, "Approval gate")
    if gold:
        print("Gold policy: hand-checked, nothing to approve.")
    elif up_b.action == "held":
        print("Stopped before run B: the change is held for review (see above). Accept it with "
              "`python -m servicedesk.cli policy approve --kb with-kb013 --acknowledge-findings`, or rerun the demo "
              "and answer y.")
        return 2
    elif up_b.action == "approved" and up_b.approved_by == compiler.PERSON_APPROVER:
        print(f"Approved by you after reviewing {len(up_b.blocking)} checker finding(s).")
    elif up_b.action == "approved":
        print("Approved automatically: the checker found nothing blocking. A removed or weakened safeguard, a "
              "conflict or injected text would have held the change for you to decide.")
    else:
        ptr = compiler.live_pointer("with-kb013") or {}
        print(f"Approved earlier (v{up_b.table.version}, by {ptr.get('approved_by', '-')}).")

    _step(6, total, "Run B: base + KB-013")
    run_requests("with-kb013", RUN_NAMES["B"], policy_source=args.policy, workers=args.workers,
                 retrieval=getattr(args, "retrieval", None), policy_update=up_b)

    print()
    print("== A vs B")
    res = score_and_report()
    comp = res["comparison"]
    for m in comp["moved_to_urgent"]:
        why = "/".join(k for k in ("security", "safety") if m[k]) or "no flag"
        print(f"  {m['request_id']:<9}{str(m['A_lane']):<16}-> urgent   ({why}) {m['about']}")
    waits = comp.get("waits_avoided", [])
    print(f"  {len(comp['moved_to_urgent'])} moved to urgent; {len(waits)} of them had been waiting for details in A.")
    if comp["security_safety_not_urgent_in_B"]:
        print("  still waiting in B: " + ", ".join(m["request_id"] for m in comp["security_safety_not_urgent_in_B"]))
    if comp["unexpected_changes"]:
        print("  unexpected changes: " + ", ".join(sorted({c['request_id'] for c in comp['unexpected_changes']})))
    print()
    _print_headline(res["headline"])
    print(f"Report: {_rel(res['report_md'])}")
    return 0


# --------------------------------------------------------------------------
# ui
# --------------------------------------------------------------------------


def cmd_ui(args) -> int:
    url = f"http://localhost:{args.port}"
    # Headless: on a machine that has never run Streamlit, it otherwise stops at a first-run
    # prompt asking for an email address. The browser is opened here instead, once the app answers.
    # Bound to localhost: not reachable from the network, and no public-IP lookup.
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "ui" / "app.py"), "--server.port", str(args.port),
           "--server.headless", "true", "--server.address", "localhost", "--browser.gatherUsageStats", "false"]
    print(f"Starting the web app at {url} (Ctrl+C to stop)...")
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    return subprocess.call(cmd, cwd=ROOT)


def _open_when_ready(url: str, timeout_s: float = 60.0) -> None:
    import time
    import urllib.request
    import webbrowser

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{url}/_stcore/health", timeout=2)
        except OSError:
            time.sleep(0.5)
            continue
        webbrowser.open(url)
        return


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m servicedesk.cli", description="Service desk intake-and-triage prototype")
    sub = p.add_subparsers(dest="command", required=True)

    pol = sub.add_parser("policy", help="compile, approve and inspect the KB policy table")
    psub = pol.add_subparsers(dest="policy_command", required=True)
    c = psub.add_parser("compile", help="compile a candidate policy and write diff.md")
    c.add_argument("--kb", choices=KB_SETS, required=True)
    c.add_argument("--no-cache", action="store_true", help="ignore cached model outputs")
    c.set_defaults(func=cmd_policy_compile)
    a = psub.add_parser("approve", help="promote the candidate to the live version")
    a.add_argument("--kb", choices=KB_SETS, required=True)
    a.add_argument("--by", default="reviewer", help="name recorded as the approver")
    a.add_argument("--acknowledge-findings", action="store_true", help="approve despite checker findings")
    a.set_defaults(func=cmd_policy_approve)
    up = psub.add_parser("update", help="recompile if the KB changed and auto-approve when the checker is clean; "
                                        "download the embedding model and embed new or edited articles if needed "
                                        "(runs do all this on their own)")
    up.add_argument("--kb", choices=KB_SETS, required=True)
    up.add_argument("--yes", action="store_true", help="accept a change the checker holds, without asking")
    up.set_defaults(func=cmd_policy_update)
    s = psub.add_parser("show", help="live version summary, the latest diff.md and the article search status")
    s.add_argument("--kb", choices=KB_SETS, required=True)
    s.set_defaults(func=cmd_policy_show)

    r = sub.add_parser("run", help="process the 26 built-in requests (or --requests FILE) into runtime/runs/NAME")
    r.add_argument("--kb", choices=KB_SETS, required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--policy", choices=["live", "gold"], default="live")
    r.add_argument("--no-replies", action="store_true", help="do not apply the simulated replies")
    r.add_argument("--request", action="append", metavar="ID", help="process only this request (repeatable)")
    r.add_argument("--yes", action="store_true",
                   help="if the KB changed and the checker holds the recompiled policy, accept it without asking")
    r.add_argument("--requests", metavar="FILE",
                   help="process the requests in this JSONL file instead of the built-in ones "
                        "(see data/custom_requests.jsonl)")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--retrieval", choices=RETRIEVAL_MODES, help=RETRIEVAL_HELP)
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("reply", help="resume one paused request in an existing run")
    rp.add_argument("--name", required=True)
    rp.add_argument("--request", required=True)
    rp.add_argument("--text", required=True)
    rp.set_defaults(func=cmd_reply)

    ev = sub.add_parser("eval", help="request eval (A vs B) or policy eval")
    esub = ev.add_subparsers(dest="eval_command", required=True)
    er = esub.add_parser("requests", help="run A (base) and B (with-kb013), score and write the report")
    er.add_argument("--policy", choices=["live", "gold"], default="live")
    er.add_argument("--no-replies", action="store_true")
    er.add_argument("--yes", action="store_true", help="accept a policy change the checker holds, without asking")
    er.add_argument("--retrieval", choices=RETRIEVAL_MODES, help=RETRIEVAL_HELP)
    er.set_defaults(func=cmd_eval_requests)
    et = esub.add_parser("retrieval", help="keyword vs vector vs hybrid article search on the gold issues "
                                           "(no API calls)")
    et.add_argument("--kb", choices=KB_SETS, default="with-kb013")
    et.set_defaults(func=cmd_eval_retrieval)
    ep = esub.add_parser("policy", help="extracted policy vs policy_gold.json")
    ep.set_defaults(func=cmd_eval_policy)

    d = sub.add_parser("demo", help="KB-013 story: run A, KB-013 recompiled + diff, approval gate, run B, compare")
    d.add_argument("--yes", action="store_true",
                   help="accept a policy change the checker holds, without asking (clean changes need no answer)")
    d.add_argument("--policy", choices=["live", "gold"], default="live",
                   help="gold skips compiling and uses the hand-checked policy")
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--retrieval", choices=RETRIEVAL_MODES, help=RETRIEVAL_HELP)
    d.set_defaults(func=cmd_demo)

    u = sub.add_parser("ui", help="launch the Streamlit app")
    u.add_argument("--port", type=int, default=8501)
    u.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    u.set_defaults(func=cmd_ui)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except LLMError as e:
        print(f"\nAPI error ({e.kind}): {e}", file=sys.stderr)
        print(f"  request_id: {e.request_id or '-'}" + (f", HTTP {e.status}" if e.status else ""), file=sys.stderr)
        if e.kind in HINTS:
            print(f"  {HINTS[e.kind]}", file=sys.stderr)
        return 3
    except FileNotFoundError as e:
        print(f"Not found: {e}", file=sys.stderr)
        return 2
    except (RequestFileError, KBFormatError, RetrievalError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
