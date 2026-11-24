"""Request pipeline: redact -> screen -> read -> KB match -> gap check -> lanes -> draft -> guard.

Models read and draft; code decides. Every step writes an audit event; every
model call's LLMResult.meta() goes into the audit log. Any LLMError turns the
affected issue into lane handed_to_staff (flagged) and the request status into
failed_safe; a batch never crashes because one request failed.

Per-request state (redacted text only) is saved with stores.save_state so a
paused request can be resumed, and a UI or eval runner can read every outcome
from the run folder alone:

    state["outcome"]          latest RequestOutcome (model_dump)
    state["initial_outcome"]  outcome before any reply
    state["history"]          [{"stage", "at", "outcome"}, ...] one entry per pass
    state["issues"]           per-issue working state (rule, fields, lane, ids)
    state["replies"]          screened, redacted replies
"""
from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from ..config import KB_CANDIDATES, assumptions
from ..guards import redact_secrets
from ..kb import load_kb
from ..llm import LLM, LLMError
from ..models import IncomingRequest, IssueOutcome, PolicyRule, PolicyTable, RequestOutcome
from ..retrieval import Retriever
from ..stores import DECIDED_TICKET_STATUS, Stores, now_iso
from . import agents
from .emails import (email_kind, finalize_email, insert_guidance, insert_references, load_directory, missing_steps,
                     render_template, requester_for, subject_for)
from .incidents import Incident, incident_from_ticket, load_incidents, match_incident, status_note
from .lanes import FLAG_NAMES, LaneDecision, decide_lane, is_report_time_field, required_fields_for, triggered_cross_rules
from .routing import STAFF_TEAM, category_for, flagged_for, priority_for, team_for

_NULLISH = {"", "null", "none", "unknown", "n/a", "na", "not provided", "not stated", "not specified"}


def accepts_reply(issue: dict) -> bool:
    """An issue still asks the employee for details: it is paused for them, or it was routed
    at once (KB-013) and the email asked for them alongside the guidance."""
    return issue["lane"] == "awaiting_info" or (issue["lane"] == "urgent" and bool(issue["missing_fields"]))


class Pipeline:
    def __init__(self, policy: PolicyTable, kb_dirs: list[Path], stores: Stores, llm: Optional[LLM] = None,
                 retrieval: Optional[str] = None):
        self.policy = policy
        self.kb_dirs = [Path(d) for d in kb_dirs]
        self.stores = stores
        self.llm = llm if llm is not None else LLM()
        self.articles = {a.article_id: a for a in load_kb(self.kb_dirs)}
        self.active_ids = {aid for aid, a in self.articles.items() if a.active}
        # Keyword, vector or hybrid search over the Active articles (servicedesk/retrieval.py).
        self.retriever = Retriever(list(self.articles.values()), retrieval)
        # A rule only applies when its article is loaded and Active (so KB-013's rule
        # can't act in a base run even if a policy table still carries it).
        self.live_rules = [r for r in policy.rules if r.article_id in self.active_ids]
        self.incidents = load_incidents()
        self.directory = load_directory()
        self.conf = assumptions()
        self.window_hours = float(self.conf.get("incident_time_window_hours", 12))
        # Matching an incident and opening one must be one step: run_batch processes requests
        # in parallel, and two reports of the same outage must not both open an incident.
        self._decide_lock = threading.RLock()

    def known_incidents(self) -> list[Incident]:
        """The service-status snapshot plus every incident this run has opened (KB-012: search
        open tickets too before creating one)."""
        return self.incidents + [incident_from_ticket(t) for t in self.stores.tickets() if t.kind == "incident"]

    # ======================================================================
    # Public API
    # ======================================================================

    def process(self, req: IncomingRequest) -> RequestOutcome:
        try:
            return self._process(req)
        except Exception as e:  # a bug, not a model failure: still never crash the batch
            return self._crash(req.request_id, e, req=req)

    def resume(self, request_id: str, reply_text: str) -> RequestOutcome:
        state = self.stores.load_state(request_id)
        if state is None:
            raise KeyError(f"no saved state for request {request_id}")
        try:
            return self._resume(state, reply_text)
        except Exception as e:
            return self._crash(request_id, e, state=state)

    def run_batch(self, reqs: list[IncomingRequest], workers: int = 4) -> list[RequestOutcome]:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(self.process, reqs))

    # ======================================================================
    # First pass
    # ======================================================================

    def _process(self, req: IncomingRequest) -> RequestOutcome:
        rid = req.request_id
        redacted, kinds = redact_secrets(req.message)
        who = requester_for(req.requester_id, self.directory)
        rs: dict[str, Any] = {
            "request_id": rid,
            "received_at": req.received_at,
            "channel": req.channel,
            "requester_id": req.requester_id,
            "synthetic": req.synthetic,
            "to": who.email,
            "first_name": who.first_name,
            # Saved only after screening succeeds (with any secret the screening model found
            # removed). Until then the text exists in memory only; if screening fails it is
            # never stored: staff read the message in its original channel.
            "redacted_message": None,
            "message_withheld": None,
            "clean_text": None,
            "replies": [],
            "secrets_redacted": kinds,
            "injection_detected": False,
            "injection_excerpt": None,
            "issues": [],
            "llm_errors": [],
            "guard_violations": [],
            "email_ids": [],
            "kb_dirs": [str(d) for d in self.kb_dirs],
            "retrieval": self.retriever.mode,
            "policy": {"version": self.policy.version, "source": self.policy.source},
            "history": [],
        }
        # No message body before screening: a secret in a format only the screening model spots
        # must not reach the log. The screened text is logged by _apply_screening.
        self.stores.audit(rid, "received", channel=req.channel, requester_id=req.requester_id,
                          synthetic=req.synthetic, chars=len(req.message), secrets_redacted=kinds)

        # -- 2. screening ---------------------------------------------------
        try:
            scr = agents.screen(self.llm, redacted, kind="request")
        except LLMError as e:
            return self._fail_unread(rs, e, stage="screening")
        rs["clean_text"] = self._apply_screening(rs, scr, redacted, source="request")

        # -- 3. reading -----------------------------------------------------
        try:
            reading = agents.read(self.llm, rs["clean_text"])
        except LLMError as e:
            return self._fail_unread(rs, e, stage="reading")
        reads = reading.parsed.issues
        self.stores.audit(rid, "reading", llm=reading.meta(), issues=[r.model_dump() for r in reads])
        if not reads:
            rs["issues"] = [self._new_issue(0, None)]
        else:
            rs["issues"] = [self._new_issue(i, r) for i, r in enumerate(reads)]

        # -- 4-7. per issue: KB match, gap check, lanes ---------------------
        for issue in rs["issues"]:
            try:
                self._triage(rs, issue)
            except LLMError as e:
                self._fail_issue(rs, issue, e)

        # -- 8. email -------------------------------------------------------
        self._send_email(rs, rs["issues"], stage="first_response")
        return self._finish(rs, "initial")

    def _new_issue(self, index: int, read: Optional[agents.IssueRead]) -> dict[str, Any]:
        r = read or agents.IssueRead(
            summary="Request with no readable issue; a person needs to review it",
            search_query="", is_it_request=True, non_it_owner=None, security=False, safety=False, multi_user=False,
            group_request=False,
        )
        return {
            "issue_index": index,
            "summary": redact_secrets(r.summary)[0],
            "search_query": redact_secrets(r.search_query)[0],
            "is_it_request": r.is_it_request,
            "non_it_owner": r.non_it_owner,
            "security": r.security,
            "safety": r.safety,
            "multi_user": r.multi_user,
            "group_request": r.group_request,
            "search_hits": [],
            "rule_id": None,
            "also_rule_ids": [],  # rules from other articles that name the same situation: their guidance applies
            "cross_rule_ids": [],
            "required_fields": [],
            "provided_fields": {},
            "missing_fields": [],
            "location_scope": None,  # HQ, NA, LATAM, EU, AFR, ME, APAC or UNKNOWN (working assumption)
            "site": None,  # the office as stated, for staff; never used to link
            "lane": None,
            "lane_reason": None,
            "approver": None,
            "ticket_id": None,
            "incident_id": None,
            "approval_id": None,
            "category": None,
            "owner_team": None,
            "priority": None,
            "flagged": False,
            "needs_review": False,  # sticky: a failed step asked for a person; re-routing never clears it
            "status_note": None,
            "notes": [],
            "failed": False,
        }

    def _apply_screening(self, rs: dict, res, text: str, *, source: str) -> str:
        """Use the screened text; redact secrets the model found that the regex missed."""
        scr = res.parsed
        extra_kinds = []
        if scr.injection_detected:
            clean = (scr.clean_text or "").strip()
            excerpt = (scr.injection_excerpt or "").strip()
            if excerpt and excerpt in clean:  # belt and braces: cut it if the model left it in
                clean = clean.replace(excerpt, "[removed: injected instructions]")
            if not clean:
                clean = text.replace(excerpt, "[removed: injected instructions]") if excerpt else ""
        else:
            clean = text  # nothing injected: keep the employee's words exactly
        for s in scr.extra_secrets:
            v = (s.value or "").strip()
            if len(v) < 4 or "[REDACTED" in v:
                continue
            if v in clean or v in text:
                clean = clean.replace(v, f"[REDACTED-{s.kind}]")
                text = text.replace(v, f"[REDACTED-{s.kind}]")
                if rs.get("redacted_message"):
                    rs["redacted_message"] = rs["redacted_message"].replace(v, f"[REDACTED-{s.kind}]")
                extra_kinds.append(s.kind)
        if source == "request":
            rs["redacted_message"] = redact_secrets(text)[0]  # the screened original, now safe to store
        clean, more = redact_secrets(clean)
        excerpt = redact_secrets(scr.injection_excerpt or "")[0] or None
        for v in (s.value for s in scr.extra_secrets if s.value and len(s.value.strip()) >= 4):
            if excerpt:
                excerpt = excerpt.replace(v.strip(), "[REDACTED]")
        rs["secrets_redacted"] = sorted(set(rs["secrets_redacted"]) | set(extra_kinds) | set(more))
        if scr.injection_detected:
            rs["injection_detected"] = True
            rs["injection_excerpt"] = rs.get("injection_excerpt") or excerpt
        self.stores.audit(
            rs["request_id"], f"screening:{source}", llm=res.meta(), injection_detected=scr.injection_detected,
            injection_excerpt=excerpt, extra_secret_kinds=sorted(set(extra_kinds)), clean_text=clean,
        )
        return clean

    # ======================================================================
    # Per issue
    # ======================================================================

    def _search(self, issue: dict) -> tuple[list, dict]:
        """The top KB_CANDIDATES (5) Active articles that have situation rules (cross-cutting-only
        articles such as KB-012/KB-013 can't be picked, so they must not take a candidate slot),
        plus the keyword and vector lists behind them for the audit log."""
        query = f"{issue['search_query']} {issue['summary']}".strip()
        if not query:
            return [], {}
        with_situations = {r.article_id for r in self.live_rules if r.scope == "situation"}
        res = self.retriever.search(query, k=len(self.retriever.articles))
        hits = [(a, s) for a, s in res.hits if a.article_id in with_situations][:KB_CANDIDATES]
        return hits, {"mode": self.retriever.mode, "keyword": res.keyword, "vector": res.vector}

    def _triage(self, rs: dict, issue: dict, *, context: Optional[str] = None) -> None:
        rid = rs["request_id"]
        issue["also_rule_ids"] = []
        hits, lists = self._search(issue)
        issue["search_hits"] = [[a.article_id, round(s, 4)] for a, s in hits]
        art_ids = [a.article_id for a, _ in hits]
        candidates = [r for r in self.live_rules if r.article_id in art_ids and r.scope == "situation"]
        self.stores.audit(rid, "kb_search", issue_index=issue["issue_index"], query=issue["search_query"],
                          hits=issue["search_hits"], candidates=[r.rule_id for r in candidates], **lists)
        rule: Optional[PolicyRule] = None
        if candidates:
            res = agents.pick_rule(self.llm, context or rs["clean_text"], issue["summary"], candidates, self.articles)
            pick = res.parsed.rule_id
            self.stores.audit(rid, "kb_pick", issue_index=issue["issue_index"], llm=res.meta(),
                              rule_id=pick, reason=res.parsed.reason, also_applies=res.parsed.also_applies)
            if pick != "none":
                rule = self.policy.get(pick)
            if rule is not None:
                also = [self.policy.get(x) for x in dict.fromkeys(res.parsed.also_applies)]
                issue["also_rule_ids"] = [r.rule_id for r in also if r is not None and r.article_id != rule.article_id]
        issue["rule_id"] = rule.rule_id if rule else None
        if rule is not None:
            self._rule_fields(issue)
            self._fill(rs, issue)
        self._decide_and_apply(rs, issue)

    def _rule_fields(self, issue: dict) -> None:
        """Cross-cutting rules and required fields follow from the chosen rule and the issue's
        flags, so they are re-derived whenever a flag changes (for example after a reply)."""
        rule = self.policy.get(issue["rule_id"]) if issue["rule_id"] else None
        if rule is None:
            return
        if rule.needs_approval and issue["multi_user"]:
            # A rule that needs approval is a request for a change (access, software, a restore),
            # never an outage: several people asking for it opens no incident and gets no P1.
            issue["multi_user"] = False
            issue["group_request"] = True
            note = "Covers several people, but it is a request that needs approval, not an outage: no incident."
            if note not in issue["notes"]:
                issue["notes"].append(note)
        flags = {k: issue[k] for k in FLAG_NAMES}
        cross = triggered_cross_rules(self.live_rules, flags)
        issue["cross_rule_ids"] = [r.rule_id for r in cross]
        issue["required_fields"] = required_fields_for(rule, cross)

    def _fill(self, rs: dict, issue: dict) -> None:
        """Gap check: what the employee stated for each required field (request + replies)."""
        fields = issue["required_fields"]
        with_location = bool(issue["multi_user"])
        if not fields and not with_location:
            issue["provided_fields"], issue["missing_fields"] = {}, []
            return
        res = agents.fill_fields(self.llm, rs["clean_text"], rs["replies"], issue["summary"], fields,
                                 with_location=with_location)
        values: dict[str, str] = {}
        for fv in res.parsed.values:
            v = (fv.value or "").strip()
            if v.lower().strip(" .") not in _NULLISH:
                values[fv.name] = redact_secrets(v)[0]
        scope = getattr(res.parsed, "location_scope", None) if with_location else None
        provided, missing = {}, []
        for f in fields:
            if f not in values and f.strip().lower() in ("region", "location scope") and scope and scope != "UNKNOWN":
                values[f] = scope  # the normalised scope answers a KB field called "region"
            if f in values:
                provided[f] = values[f]
            elif is_report_time_field(f):  # the report's received_at stands in for the time
                provided[f] = f"{rs['received_at']} (time of report)"
            else:
                missing.append(f)
        issue["provided_fields"], issue["missing_fields"] = provided, missing
        summary_now = (getattr(res.parsed, "summary_now", None) or "").strip()
        if rs["replies"] and summary_now:
            # A reply can change what the employee wants (for example the restore destination):
            # the ticket, the approval item and the email all describe the issue as it stands now.
            issue["summary"] = redact_secrets(summary_now)[0]
        if with_location:
            issue["location_scope"] = scope or "UNKNOWN"
            site = getattr(res.parsed, "site", None)
            issue["site"] = redact_secrets(site)[0] if site else None
        self.stores.audit(rs["request_id"], "fill_fields", issue_index=issue["issue_index"], llm=res.meta(),
                          required=fields, provided=provided, missing=missing,
                          location_scope=issue["location_scope"], site=issue["site"], replies=len(rs["replies"]))

    def _service_symptom(self, issue: dict) -> tuple[Optional[str], Optional[str]]:
        pf = issue["provided_fields"]
        service = next((v for k, v in pf.items() if "service" in k.lower()), None)
        symptom = next((v for k, v in pf.items() if "symptom" in k.lower()), None)
        if not service and not symptom:
            symptom = issue["summary"]
        return service, symptom

    def _decide_and_apply(self, rs: dict, issue: dict) -> LaneDecision:
        with self._decide_lock:  # no model calls inside: holding it is cheap
            return self._decide_and_apply_locked(rs, issue)

    def _decide_and_apply_locked(self, rs: dict, issue: dict) -> LaneDecision:
        rid = rs["request_id"]
        rule = self.policy.get(issue["rule_id"]) if issue["rule_id"] else None
        cross = [r for r in (self.policy.get(x) for x in issue["cross_rule_ids"]) if r is not None]
        flags = {k: issue[k] for k in FLAG_NAMES}

        match = None
        defer_to_staff = None  # an incident that may match, but a location scope is UNKNOWN
        waits_or_urgent = bool(issue["missing_fields"]) or any(r.route_without_waiting for r in ([rule] if rule else []) + cross)
        if rule is not None and issue["multi_user"] and not waits_or_urgent:
            service, symptom = self._service_symptom(issue)
            match = match_incident(service=service, symptom=symptom, location_scope=issue["location_scope"],
                                   reported_at=rs["received_at"], incidents=self.known_incidents(),
                                   window_hours=self.window_hours)
            self.stores.audit(rid, "incident_match", issue_index=issue["issue_index"], service=service,
                              symptom=symptom, location_scope=issue["location_scope"], site=issue["site"],
                              matched=match.incident.incident_id if match.incident else None,
                              needs_location=match.needs_location.incident_id if match.needs_location else None,
                              checks=match.checks)
            if match.incident is None and match.needs_location is not None:
                maybe = match.needs_location
                if issue["location_scope"] == "UNKNOWN" and not issue.get("location_asked"):
                    # UNKNOWN defers automatic linking: ask once where it happens rather than
                    # link the report to the wrong place or open a duplicate incident.
                    issue["location_asked"] = True
                    issue["missing_fields"] = [*issue["missing_fields"], "location (office, city or country)"]
                else:
                    defer_to_staff = maybe

        decision = decide_lane(rule=rule, cross_rules=cross, is_it_request=issue["is_it_request"],
                               multi_user=issue["multi_user"], missing_fields=issue["missing_fields"],
                               incident=match.incident if match else None)
        if defer_to_staff is not None and decision.lane == "incident_created":
            # Still unresolved: no automatic link and no possible duplicate. A person decides.
            scope = "; ".join(defer_to_staff.scope) or "location unknown"
            note = (f"May be the same problem as {defer_to_staff.incident_id} ({scope}), but the location scope "
                    f"is unresolved, so it was not linked automatically: staff decide whether to link.")
            if note not in issue["notes"]:
                issue["notes"].append(note)
            decision = LaneDecision("handed_to_staff", "routed", f"possible match {defer_to_staff.incident_id}; "
                                    "location scope unresolved", needs_approval=decision.needs_approval,
                                    approver=decision.approver)
        lane = decision.lane
        art_owner = self.articles[rule.article_id].owner if rule and rule.article_id in self.articles else None
        inc = decision.incident
        issue.update(
            lane=lane,
            lane_reason=decision.reason,
            approver=decision.approver,
            category=category_for(rule, self.policy.rules) if rule else None,
            owner_team=(team_for(rule, art_owner, category=category_for(rule, self.policy.rules)) if rule
                        else (STAFF_TEAM if lane == "handed_to_staff" else None)),
            priority=None if lane == "redirected" else priority_for(
                rule, security=issue["security"], multi_user=issue["multi_user"],
                incident_priority=inc.priority if inc else None, conf=self.conf),
            flagged=flagged_for(flags, lane=lane, conf=self.conf) or bool(issue.get("needs_review")),
        )
        if inc is not None:
            issue["incident_id"] = inc.incident_id
            issue["status_note"] = status_note(inc)
            if issue["status_note"] and issue["status_note"] not in issue["notes"]:
                issue["notes"].append(issue["status_note"])
        self.stores.audit(rid, "lane", issue_index=issue["issue_index"], lane=lane, reason=decision.reason,
                          rule_id=issue["rule_id"], cross_rules=issue["cross_rule_ids"],
                          missing=issue["missing_fields"], category=issue["category"], team=issue["owner_team"],
                          priority=issue["priority"], flagged=issue["flagged"], flags=flags)
        self._apply(rs, issue, decision, rule)
        return decision

    def _ticket_fields(self, issue: dict, rule: Optional[PolicyRule]) -> dict[str, Any]:
        fields = dict(issue["provided_fields"])
        if issue["multi_user"]:
            if issue["location_scope"]:
                fields["location scope"] = issue["location_scope"]
            if issue["site"]:
                fields["site"] = issue["site"]
        return dict(
            category=issue["category"],
            owner_team=issue["owner_team"],
            priority=issue["priority"],
            flagged_for_staff=issue["flagged"],
            summary=issue["summary"],
            fields=fields,
            missing_fields=list(issue["missing_fields"]),
            article_id=rule.article_id if rule else None,
            rule_id=rule.rule_id if rule else None,
        )

    def _upsert_ticket(self, rs: dict, issue: dict, **fields: Any) -> None:
        rid = rs["request_id"]
        if issue["ticket_id"]:
            t = self.stores.update_ticket(issue["ticket_id"], **fields)
            self.stores.audit(rid, "ticket_updated", issue_index=issue["issue_index"], ticket_id=t.ticket_id,
                              status=t.status)
        else:
            t = self.stores.add_ticket(request_id=rid, issue_index=issue["issue_index"], kind="ticket", **fields)
            issue["ticket_id"] = t.ticket_id
            self.stores.audit(rid, "ticket_created", issue_index=issue["issue_index"], ticket_id=t.ticket_id,
                              status=t.status, priority=t.priority, team=t.owner_team, flagged=t.flagged_for_staff)

    def _apply(self, rs: dict, issue: dict, decision: LaneDecision, rule: Optional[PolicyRule]) -> None:
        """Write the decision to the stores. Grants, restores and resets are never executed:
        an action that needs approval only ever becomes an approval-queue item."""
        rid, lane = rs["request_id"], decision.lane
        if lane == "redirected":
            return
        base = self._ticket_fields(issue, rule)
        if lane == "incident_created":
            # started_at lets later reports in this run match it inside the time window
            inc_fields = {**base, "fields": {**base["fields"], "started_at": rs["received_at"]}}
            inc = self.stores.add_ticket(request_id=rid, issue_index=issue["issue_index"], kind="incident",
                                         status="routed", **inc_fields)
            issue["incident_id"] = inc.ticket_id
            self.stores.audit(rid, "incident_created", issue_index=issue["issue_index"], incident_id=inc.ticket_id,
                              priority=inc.priority, team=inc.owner_team)
            if issue["ticket_id"]:  # the report was paused first: link its ticket to the new incident
                self._upsert_ticket(rs, issue, status="linked_to_incident", linked_incident=inc.ticket_id, **base)
            else:
                issue["ticket_id"] = inc.ticket_id
            return
        extra = {"linked_incident": decision.incident.incident_id} if decision.incident else {}
        status = decision.ticket_status
        current = (next((a for a in self.stores.approvals() if a.approval_id == issue["approval_id"]), None)
                   if issue["approval_id"] else None)
        if lane == "awaiting_approval" and current is not None and current.status != "pending":
            status = DECIDED_TICKET_STATUS[current.status]  # a later reply never undoes a recorded decision
        self._upsert_ticket(rs, issue, status=status, **base, **extra)
        if lane == "awaiting_approval":
            # Built from the issue as it stands now (summary refreshed after replies, final
            # stated details), so the approver decides on what the employee actually asked for.
            item_fields = dict(action=f"{issue['summary']} [{rule.rule_id if rule else 'no rule'}]",
                               approver=decision.approver, rule_id=rule.rule_id if rule else None,
                               details=dict(base["fields"]))
            if not issue["approval_id"]:
                item = self.stores.add_approval(ticket_id=issue["ticket_id"], request_id=rid, **item_fields)
                issue["approval_id"] = item.approval_id
                self.stores.audit(rid, "approval_queued", issue_index=issue["issue_index"], approval_id=item.approval_id,
                                  ticket_id=issue["ticket_id"], approver=decision.approver)
            else:
                if current is not None and current.status == "pending":
                    self.stores.update_approval(issue["approval_id"], **item_fields)
                    self.stores.audit(rid, "approval_updated", issue_index=issue["issue_index"],
                                      approval_id=issue["approval_id"])

    # ======================================================================
    # Fail safe
    # ======================================================================

    def _record_error(self, rs: dict, err: LLMError, *, stage: str, issue_index: Optional[int] = None) -> None:
        d = err.to_dict()
        d.update(stage=stage, issue_index=issue_index, service_request=rs["request_id"])
        rs["llm_errors"].append(d)
        self.stores.audit(rs["request_id"], "llm_error", error=d)

    def _fail_issue(self, rs: dict, issue: dict, err: LLMError, *, stage: Optional[str] = None) -> None:
        """Any LLMError: the issue goes to a person, flagged."""
        self._record_error(rs, err, stage=stage or err.call or "issue", issue_index=issue["issue_index"])
        rule = self.policy.get(issue["rule_id"]) if issue["rule_id"] else None
        issue.update(
            lane="handed_to_staff",
            lane_reason=f"failed safe: {err.kind} in {err.call or stage}",
            flagged=True,
            needs_review=True,
            failed=True,
            owner_team=STAFF_TEAM,
            category=category_for(rule, self.policy.rules) if rule else issue["category"],
            priority=priority_for(None, security=issue["security"], multi_user=issue["multi_user"], conf=self.conf),
        )
        issue["notes"].append(f"Pipeline failed safe ({err.kind} in {err.call or stage}); a person must review.")
        self._upsert_ticket(rs, issue, status="routed", **self._ticket_fields(issue, rule))
        self.stores.audit(rs["request_id"], "lane", issue_index=issue["issue_index"], lane="handed_to_staff",
                          reason=issue["lane_reason"], flagged=True)

    def _fail_unread(self, rs: dict, err: LLMError, *, stage: str) -> RequestOutcome:
        """Screening or reading failed: nothing was read, so one staff ticket for the whole request.
        If screening failed, the unscreened message is never persisted (metadata-only hand-off)."""
        self._record_error(rs, err, stage=stage)
        if rs.get("redacted_message") is None:
            rs["message_withheld"] = (f"screening failed ({err.kind}): the message was not stored; "
                                      "staff read it in its original channel")
        issue = self._new_issue(0, None)
        issue["summary"] = "Request could not be read automatically; a person needs to review it"
        rs["issues"] = [issue]
        issue.update(lane="handed_to_staff", lane_reason=f"failed safe: {err.kind} in {stage}", flagged=True,
                     failed=True, owner_team=STAFF_TEAM,
                     priority=priority_for(None, security=False, multi_user=False, conf=self.conf))
        issue["notes"].append(f"Pipeline failed safe ({err.kind} in {stage}); a person must review.")
        self._upsert_ticket(rs, issue, status="routed", **self._ticket_fields(issue, None))
        self._send_email(rs, rs["issues"], stage="first_response", use_model=False)
        return self._finish(rs, "initial")

    def _crash(self, request_id: str, exc: Exception, *, req: Optional[IncomingRequest] = None,
               state: Optional[dict] = None) -> RequestOutcome:
        msg = redact_secrets(f"{type(exc).__name__}: {exc}")[0]
        err = {"kind": "internal", "call": "pipeline", "message": msg, "service_request": request_id,
               "trace": redact_secrets(traceback.format_exc(limit=6))[0]}
        self.stores.audit(request_id, "internal_error", error=err)
        outcome = RequestOutcome(
            request_id=request_id, status="failed_safe", llm_errors=[err],
            issues=[IssueOutcome(issue_index=0, summary="Pipeline error; a person needs to review this request",
                                 lane="handed_to_staff", flagged_for_staff=True,
                                 notes=["internal error; see audit log"])],
        )
        rs = state or {"request_id": request_id, "history": []}
        rs["outcome"] = outcome.model_dump()
        rs.setdefault("history", []).append({"stage": "internal_error", "at": now_iso(), "outcome": rs["outcome"]})
        try:
            self.stores.save_state(request_id, rs)
        except Exception:
            pass
        return outcome

    # ======================================================================
    # Email
    # ======================================================================

    def _email_issue_ctx(self, issue: dict) -> dict:
        rule = self.policy.get(issue["rule_id"]) if issue["rule_id"] else None
        cross = [r for r in (self.policy.get(x) for x in issue["cross_rule_ids"]) if r is not None]
        lane = issue["lane"]
        also = [r for r in (self.policy.get(x) for x in issue.get("also_rule_ids", [])) if r is not None]
        guidance = []
        if rule is not None and not issue["failed"]:
            # Situation rules carry the user's steps, including rules from other articles that
            # name the same situation (KB-008's Security Hotline step for an unexpected MFA prompt).
            # Cross-cutting rules' guidance is addressed to the agent, not the user.
            seen: set[str] = set()
            for r in [rule, *also]:
                for g in r.user_guidance:
                    if g.strip().lower() not in seen:
                        seen.add(g.strip().lower())
                        guidance.append({"step": g, "article": r.article_id})
        articles = [] if lane == "redirected" or rule is None else list(
            dict.fromkeys([rule.article_id, *(r.article_id for r in also if guidance), *(r.article_id for r in cross)])
        )
        inc_public = None
        if lane == "incident_linked" and issue["incident_id"]:
            inc = next((i for i in self.known_incidents() if i.incident_id == issue["incident_id"]), None)
            inc_public = inc.public() if inc else None
        return {
            "summary": issue["summary"],
            # what the employee stated for this issue; after a reply these are the latest values
            "details": {} if lane == "redirected" else dict(issue["provided_fields"]),
            "lane": lane,
            "ticket_id": issue["ticket_id"],
            "incident_id": issue["incident_id"] if lane in ("incident_linked", "incident_created") else None,
            "approval_id": issue["approval_id"],
            "approver": issue["approver"] if lane == "awaiting_approval" else None,
            "team": issue["owner_team"],
            "missing_details": issue["missing_fields"] if lane in ("urgent", "awaiting_info") else [],
            "guidance": guidance,
            "must_never": list(dict.fromkeys([*(rule.must_never if rule else []), *(m for r in also for m in r.must_never)])),
            "incident": inc_public,
            "status_note": issue["status_note"] if lane == "incident_linked" else None,
            "redirect_to": issue["non_it_owner"] if lane == "redirected" else None,
            "articles": articles,
        }

    def _send_email(self, rs: dict, issues: list[dict], *, stage: str, use_model: bool = True) -> None:
        rid = rs["request_id"]
        ctx = {
            "employee_first_name": rs.get("first_name"),
            "stage": "reply to the employee's first message" if stage == "first_response"
            else "reply after the employee sent the details we asked for",
            "issues": [self._email_issue_ctx(i) for i in issues],
        }
        for n, it in enumerate(ctx["issues"], 1):  # where the drafter places each issue's KB steps
            it["steps_marker"] = f"[STEPS {n}]" if it["guidance"] else None
        lanes = [i["lane"] for i in issues]
        kind = email_kind(lanes)
        # Confirm the exact approval details and give employees a stable contact for corrections.
        if kind == "confirmation" and "awaiting_approval" in lanes:
            use_model = False
        cites = sorted({a for it in ctx["issues"] for a in it["articles"]})
        subject = body = None
        if use_model:
            try:
                res = agents.draft(self.llm, ctx)
                # KB steps and any ticket, incident or approval id the draft left out go in by code
                subject, body = subject_for(ctx), insert_references(insert_guidance(res.parsed.body, ctx), ctx)
                self.stores.audit(rid, "draft", llm=res.meta(), stage=stage)
            except LLMError as e:
                # The decision stands; the email falls back to the template and a person is told.
                self._record_error(rs, e, stage="drafter")
                for i in issues:
                    i["flagged"] = True
                    i["needs_review"] = True
                    note = "Drafter failed; the deterministic template email was sent."
                    if note not in i["notes"]:
                        i["notes"].append(note)
                    if i["ticket_id"]:
                        self.stores.update_ticket(i["ticket_id"], flagged_for_staff=True)
        if subject is None or body is None:
            subject, body = render_template(ctx)
        drafted_subject, drafted_body = subject, body
        subject, body, violations, used_template = finalize_email(subject, body, ctx)
        lost = missing_steps(body, ctx)
        if lost and not used_template:  # never expected after insert_guidance; fail safe anyway
            violations = [*violations, "missing_kb_steps"]
            subject, body = render_template(ctx)
            used_template = True
        if violations:
            rs["guard_violations"].extend(violations)
            self.stores.audit(rid, "guard", violations=violations, replaced_with_template=used_template, stage=stage,
                              rejected_subject=redact_secrets(drafted_subject)[0],
                              rejected_body=redact_secrets(drafted_body)[0])
        email = self.stores.add_email(request_id=rid, to=rs["to"], subject=subject, body=body,
                                      kind=kind, cites=cites)
        rs["email_ids"].append(email.email_id)
        self.stores.audit(rid, "email_sent", email_id=email.email_id, to=email.to, kind=email.kind, cites=cites,
                          template=used_template or not use_model, stage=stage)

    # ======================================================================
    # Resume
    # ======================================================================

    def _resume(self, rs: dict, reply_text: str) -> RequestOutcome:
        rid = rs["request_id"]
        redacted, kinds = redact_secrets(reply_text)
        rs["secrets_redacted"] = sorted(set(rs.get("secrets_redacted", [])) | set(kinds))
        # No message body is logged before screening: a secret only the screening model spots
        # must not already be in the audit log. The screened text is logged by _apply_screening.
        self.stores.audit(rid, "reply_received", chars=len(reply_text), secrets_redacted=kinds)
        targets = [i for i in rs["issues"] if accepts_reply(i)]
        n = len(rs["replies"]) + 1
        try:
            scr = agents.screen(self.llm, redacted, kind="reply")
        except LLMError as e:
            if not targets:  # nothing was waiting, but an unread reply still needs a person
                issue = self._new_issue(len(rs["issues"]), None)
                issue["summary"] = "Reply could not be read automatically; a person needs to review it"
                rs["issues"].append(issue)
                targets = [issue]
            for issue in targets:
                self._fail_issue(rs, issue, e, stage="screening:reply")
            self._send_email(rs, targets, stage="after_reply", use_model=False)
            return self._finish(rs, f"after_reply_{n}")
        clean = self._apply_screening(rs, scr, redacted, source="reply")
        earlier = list(rs["replies"])
        rs["replies"].append(clean)

        # A reply can raise a new problem, or show that a tracked issue is a security
        # incident or a hazard. Flags are only ever raised here, never lowered.
        new_issues: list[dict] = []
        raised: list[dict] = []
        try:
            rr = agents.read_reply(self.llm, rs["clean_text"], earlier, clean, rs["issues"])
            for fr in rr.parsed.raised_flags:
                idx = fr.issue_number - 1
                if not 0 <= idx < len(rs["issues"]):
                    continue
                issue = rs["issues"][idx]
                up = [k for k in ("security", "safety") if getattr(fr, k) and not issue[k]]
                if up and not issue["failed"] and issue["lane"] != "redirected":
                    for k in up:
                        issue[k] = True
                    # the diagnosis may have changed: the KB rule is matched again from this
                    if fr.summary.strip():
                        issue["summary"] = redact_secrets(fr.summary.strip())[0]
                    if fr.search_query.strip():
                        issue["search_query"] = redact_secrets(fr.search_query.strip())[0]
                    issue["notes"].append(f"Reply {n} shows this is a {' and '.join(up)} issue.")
                    raised.append(issue)
            for r in rr.parsed.new_issues:
                issue = self._new_issue(len(rs["issues"]), r)
                issue["notes"].append(f"Raised in the employee's reply {n}.")
                rs["issues"].append(issue)
                new_issues.append(issue)
            self.stores.audit(rid, "reply_reading", llm=rr.meta(), new_issues=[i["summary"] for i in new_issues],
                              raised=[{"issue_index": i["issue_index"], "security": i["security"], "safety": i["safety"]}
                                      for i in raised])
        except LLMError as e:
            # The reply still answers the waiting issues; a person reads it for anything new.
            self._record_error(rs, e, stage="reply_reading")
            for issue in targets:
                issue["flagged"] = True
                issue["needs_review"] = True  # survives the re-routing below
                issue["notes"].append("The reply could not be checked for new problems; a person should read it.")

        plain = [i for i in targets if i not in raised]
        if not plain and not raised and not new_issues:
            self.stores.audit(rid, "reply_ignored", reason="no issue is waiting for details and the reply raises nothing new")
            return self._finish(rs, f"reply_{n}_ignored")
        for issue in plain:
            try:
                self._rule_fields(issue)
                self._fill(rs, issue)
                self._decide_and_apply(rs, issue)
            except LLMError as e:
                self._fail_issue(rs, issue, e)
        context = f"{rs['clean_text']}\n\nLater reply from the employee:\n{clean}"
        # A raised flag can mean a changed diagnosis (a failed VPN login is a suspected compromise):
        # match the KB rule again, so the team, guidance and lanes follow the issue as it now is.
        # The ticket keeps its id: _upsert_ticket updates it in place.
        for issue in raised + new_issues:
            try:
                before = issue["rule_id"]
                self._triage(rs, issue, context=context)
                if issue in raised and issue["rule_id"] != before:
                    issue["notes"].append(f"Rule changed after reply {n}: {before or 'none'} -> {issue['rule_id'] or 'none'}.")
            except LLMError as e:
                self._fail_issue(rs, issue, e)
        self._send_email(rs, plain + raised + new_issues, stage="after_reply")
        return self._finish(rs, f"after_reply_{n}")

    # ======================================================================
    # Outcome + state
    # ======================================================================

    def _outcome(self, rs: dict) -> RequestOutcome:
        issues = [
            IssueOutcome(
                issue_index=i["issue_index"],
                summary=i["summary"],
                rule_id=i["rule_id"],
                article_id=i["rule_id"].split(":")[0] if i["rule_id"] else None,
                category=i["category"],
                owner_team=i["owner_team"],
                priority=i["priority"],
                lane=i["lane"],
                flagged_for_staff=i["flagged"],
                security=i["security"],
                safety=i["safety"],
                multi_user=i["multi_user"],
                group_request=i.get("group_request", False),
                missing_fields=i["missing_fields"],
                provided_fields=i["provided_fields"],
                incident_id=i["incident_id"],
                ticket_id=i["ticket_id"],
                approval_id=i["approval_id"],
                awaiting_reply=accepts_reply(i),
                also_applies=i.get("also_rule_ids", []),
                applied_rules=([i["rule_id"]] if i["rule_id"] else []) + i["cross_rule_ids"],
                notes=i["notes"],
            )
            for i in rs["issues"]
        ]
        if rs["llm_errors"]:
            status = "failed_safe"
        elif any(i.lane == "awaiting_info" for i in issues):
            status = "awaiting_info"
        else:
            status = "done"
        return RequestOutcome(
            request_id=rs["request_id"],
            status=status,
            injection_detected=rs["injection_detected"],
            injection_excerpt=rs["injection_excerpt"],
            secrets_redacted=rs["secrets_redacted"],
            out_of_scope=bool(issues) and all(i.lane == "redirected" for i in issues),
            reply_requested=any(i.awaiting_reply for i in issues),
            issues=issues,
            email_ids=rs["email_ids"],
            guard_violations=rs["guard_violations"],
            llm_errors=rs["llm_errors"],
        )

    def _finish(self, rs: dict, stage: str) -> RequestOutcome:
        outcome = self._outcome(rs)
        dump = outcome.model_dump()
        rs["outcome"] = dump
        if stage == "initial":
            rs["initial_outcome"] = dump
        rs["history"].append({"stage": stage, "at": now_iso(), "outcome": dump})
        self.stores.save_state(rs["request_id"], rs)
        self.stores.audit(rs["request_id"], "outcome", stage=stage, status=outcome.status,
                          lanes=[i.lane for i in outcome.issues])
        return outcome
