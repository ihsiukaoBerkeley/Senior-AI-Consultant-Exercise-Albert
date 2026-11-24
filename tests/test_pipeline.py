"""Deterministic tests for the request pipeline (no API calls).

Pure parts are tested directly (lanes, routing, incident matcher, emails); the
orchestration is tested end to end with a scripted fake LLM.
"""
from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pytest

from servicedesk.config import INCIDENTS_FILE, KB_ADDITIONS_DIR, KB_DIR, assumptions
from servicedesk.guards import check_outgoing
from servicedesk.llm import LLMError, LLMResult, Usage
from servicedesk.models import IncomingRequest, PolicyRule, PolicyTable
from servicedesk.pipeline import Pipeline, kb_dirs_for, load_requests
from servicedesk.pipeline.emails import (
    email_kind,
    finalize_email,
    load_directory,
    render_template,
    requester_for,
)
from servicedesk.pipeline.incidents import Incident, load_incidents, match_incident, status_note
from servicedesk.pipeline.lanes import decide_lane, is_report_time_field, required_fields_for, triggered_cross_rules
from servicedesk.pipeline.routing import flagged_for, majority_team, priority_for, team_for
from servicedesk.stores import Stores

CONF = assumptions()


# --------------------------------------------------------------------------
# In-memory policy fixtures
# --------------------------------------------------------------------------


def rule(rule_id: str, **kw) -> PolicyRule:
    base = dict(
        rule_id=rule_id,
        article_id=rule_id.split(":")[0],
        situation=kw.pop("situation", "test situation"),
        scope="situation",
        triggers=[],
        category=None,
        owner_team=None,
        required_fields=[],
        needs_approval=False,
        approver=None,
        route_without_waiting=False,
        priority_hint=None,
        must_never=[],
        user_guidance=[],
        quotes=[],
    )
    base.update(kw)
    return PolicyRule(**base)


ACCESS = rule("KB-001:new-access", category="Access", required_fields=["application", "requested role", "manager approval"],
              needs_approval=True, approver="manager", priority_hint="medium",
              user_guidance=["Sign out and back in after approved access is granted"])
SECURITY = rule("KB-008:suspicious-activity", category="Security", owner_team="Security Operations",
                required_fields=["affected device or account identifier", "time observed"], priority_hint="high",
                user_guidance=["Stop interacting with the suspicious content", "Contact the Security Hotline"])
BATTERY = rule("KB-005:battery-hazard", category="Hardware", user_guidance=["Stop using and charging the device immediately"])
PRINTERS = rule("KB-005:printer-multi-user", category="Network")
EMAIL_OUTAGE = rule("KB-007:multi-user-outage", category="Communication", priority_hint="high",
                    required_fields=["service", "start time"])
KB012 = rule("KB-012:shared-service-incident", scope="cross_cutting", triggers=["multi_user"],
             required_fields=["service", "symptom", "location", "start time"], priority_hint="high")
KB013 = rule("KB-013:route-without-waiting", scope="cross_cutting", triggers=["security", "safety"],
             route_without_waiting=True)

T = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))  # noqa: E731


def inc(incident_id, service, status, priority, scope, summary, started):
    return Incident(incident_id, service, status, priority, tuple(scope), summary, T(started))


MAJ_301 = inc("MAJ-301", "Email", "investigating", "P1", ["EU", "APAC"], "Outbound email delivery failures", "2026-09-05T08:20:00Z")
MAJ_302 = inc("MAJ-302", "VDI", "monitoring", "P2", ["NA"], "Intermittent freezing in engineering VDI pool", "2026-09-05T07:40:00Z")
MAJ_303 = inc("MAJ-303", "Third-floor printer", "investigating", "P2", ["HQ"], "Shared printer repeatedly disconnects", "2026-09-05T08:00:00Z")
INCIDENTS = [MAJ_301, MAJ_302, MAJ_303]


# --------------------------------------------------------------------------
# Lanes: decision table
# --------------------------------------------------------------------------

LANE_CASES = [
    # (id, rule, cross, is_it, multi_user, missing, incident, expected lane, expected ticket status)
    ("not IT, no rule", None, [], False, False, [], None, "redirected", None),
    ("IT, no rule", None, [], True, False, [], None, "handed_to_staff", "routed"),
    ("missing fields wait", ACCESS, [], True, False, ["requested role"], None, "awaiting_info", "awaiting_info"),
    ("complete, approval", ACCESS, [], True, False, [], None, "awaiting_approval", "awaiting_approval"),
    ("complete, no approval", SECURITY, [], True, False, [], None, "ticket_routed", "routed"),
    ("security waits in base KB", SECURITY, [], True, False, ["time observed"], None, "awaiting_info", "awaiting_info"),
    ("security urgent with KB-013", SECURITY, [KB013], True, False, ["time observed"], None, "urgent", "routed"),
    ("safety urgent with KB-013", BATTERY, [KB013], True, False, [], None, "urgent", "routed"),
    ("urgent beats approval", ACCESS, [KB013], True, False, [], None, "urgent", "routed"),
    ("multi-user missing waits", PRINTERS, [KB012], True, True, ["location"], None, "awaiting_info", "awaiting_info"),
    ("multi-user linked", PRINTERS, [KB012], True, True, [], MAJ_303, "incident_linked", "linked_to_incident"),
    ("multi-user new incident", PRINTERS, [KB012], True, True, [], None, "incident_created", "routed"),
    ("incident beats approval", ACCESS, [KB012], True, True, [], None, "incident_created", "routed"),
    ("not-IT flag but rule matched", SECURITY, [], False, False, [], None, "ticket_routed", "routed"),
]


@pytest.mark.parametrize("case", LANE_CASES, ids=[c[0] for c in LANE_CASES])
def test_lane_decision_table(case):
    _, r, cross, is_it, multi, missing, incident, lane, status = case
    d = decide_lane(rule=r, cross_rules=cross, is_it_request=is_it, multi_user=multi,
                    missing_fields=missing, incident=incident)
    assert d.lane == lane
    assert d.ticket_status == status
    if lane == "incident_linked":
        assert d.incident is incident
    if lane == "incident_created":
        assert d.create_incident


def test_approval_lane_carries_approver():
    d = decide_lane(rule=ACCESS, cross_rules=[], is_it_request=True, multi_user=False, missing_fields=[], incident=None)
    assert d.needs_approval and d.approver == "manager"


def test_cross_rules_trigger_on_flags_only():
    rules = [ACCESS, KB012, KB013]
    assert triggered_cross_rules(rules, {"security": False, "safety": False, "multi_user": False}) == []
    assert triggered_cross_rules(rules, {"multi_user": True}) == [KB012]
    assert triggered_cross_rules(rules, {"safety": True}) == [KB013]
    assert triggered_cross_rules(rules, {"security": True, "multi_user": True}) == [KB012, KB013]


def test_required_fields_are_union_without_duplicates():
    fields = required_fields_for(EMAIL_OUTAGE, [KB012])
    assert fields == ["service", "start time", "symptom", "location"]
    assert required_fields_for(ACCESS, []) == ACCESS.required_fields


def test_report_time_fields():
    assert is_report_time_field("start time") and is_report_time_field("Time observed")
    assert not is_report_time_field("approximate last-known-good time")
    assert not is_report_time_field("timestamp")


# --------------------------------------------------------------------------
# Priority, team, flag
# --------------------------------------------------------------------------


def test_priority_rules():
    assert priority_for(SECURITY, security=True, multi_user=False, conf=CONF) == "P1"
    assert priority_for(PRINTERS, security=False, multi_user=True, conf=CONF) == "P1"
    # a linked issue takes the incident's priority, even when multi-user
    assert priority_for(PRINTERS, security=False, multi_user=True, incident_priority="P2", conf=CONF) == "P2"
    assert priority_for(EMAIL_OUTAGE, security=False, multi_user=False, conf=CONF) == "P2"  # high
    assert priority_for(ACCESS, security=False, multi_user=False, conf=CONF) == "P3"  # medium
    assert priority_for(rule("KB-X:low", priority_hint="low"), security=False, multi_user=False, conf=CONF) == "P4"
    assert priority_for(BATTERY, security=False, multi_user=False, conf=CONF) == "P3"  # no hint -> default
    assert priority_for(None, security=False, multi_user=False, conf=CONF) == "P3"


def test_team_from_rule_then_history_majority_then_article_owner():
    counts = {"access": Counter({"Identity and Access Management": 5}),
              "hardware": Counter({"Workplace Technology": 4, "Workplace Infrastructure": 1}),
              "identity": Counter({"Identity and Access Management": 3, "Security Operations": 2})}
    assert team_for(SECURITY, "Security Operations", counts) == "Security Operations"  # rule.owner_team
    assert team_for(ACCESS, "Identity and Access Management", counts) == "Identity and Access Management"
    assert team_for(BATTERY, "Workplace Technology", counts) == "Workplace Technology"
    assert team_for(rule("KB-005:printer-single-user"), "Workplace Technology", counts) == "Workplace Technology"
    assert team_for(rule("KB-X:y", category="Unknown"), "Article Owner", counts) == "Article Owner"


def test_majority_team_from_real_history_and_ties():
    assert majority_team("Network") == "Workplace Infrastructure"  # 3 vs 2
    assert majority_team("Access") == "Identity and Access Management"
    assert majority_team("Performance") == "End User Computing"
    assert majority_team("access") == "Identity and Access Management"  # case-insensitive
    assert majority_team(None) is None
    tie = {"x": Counter({"Zeta": 2, "Alpha": 2})}
    assert majority_team("x", tie) == "Alpha"  # stable tie-break


def test_flagged_for_staff():
    assert flagged_for({"security": True}, conf=CONF)
    assert flagged_for({"multi_user": True}, conf=CONF)
    assert not flagged_for({"safety": True}, conf=CONF)
    assert flagged_for({"safety": True}, lane="urgent", conf=CONF)  # KB-013 says flag urgent tickets
    assert not flagged_for({}, lane="ticket_routed", conf=CONF)


# --------------------------------------------------------------------------
# Incident matcher
# --------------------------------------------------------------------------


def match(service, symptom, scope, at="2026-09-05T08:31:00Z", incidents=INCIDENTS, hours=12):
    return match_incident(service=service, symptom=symptom, location_scope=scope, reported_at=at,
                          incidents=incidents, window_hours=hours).incident


def test_loads_incident_snapshot():
    loaded = {i.incident_id: i for i in load_incidents(INCIDENTS_FILE)}
    assert loaded["MAJ-301"].scopes == {"EU", "APAC"}
    assert loaded["MAJ-303"].scopes == {"HQ"}
    assert loaded["MAJ-302"].status == "monitoring"


def test_multi_region_incident_links_each_of_its_scopes_only():
    assert match("email", "can't send email", "EU") is MAJ_301
    assert match("Email", "outbound mail not delivered", "APAC") is MAJ_301
    for other in ("NA", "LATAM", "AFR", "ME", "HQ"):
        assert match("email", "can't send email", other) is None, other


def test_hq_only_incident_links_hq_reports_only():
    at = "2026-09-05T08:21:00Z"
    assert match("third-floor printer", "keeps going offline", "HQ", at=at) is MAJ_303
    for other in ("NA", "LATAM", "EU", "AFR", "ME", "APAC"):
        assert match("printer", "repeatedly disconnecting", other, at=at) is None, other


@pytest.mark.parametrize("scope", ["HQ", "NA", "LATAM", "EU", "AFR", "ME", "APAC"])
def test_each_location_scope_links_only_to_itself(scope):
    outage = inc("MAJ-500", "Email", "investigating", "P1", [scope], "Outbound email delivery failures", "2026-09-05T08:20:00Z")
    assert match("email", "can't send email", scope, incidents=[outage]) is outage
    for other in {"HQ", "NA", "LATAM", "EU", "AFR", "ME", "APAC"} - {scope}:
        assert match("email", "can't send email", other, incidents=[outage]) is None


def test_scope_is_regional_not_by_country():
    # Working assumption: Tokyo and Singapore are both APAC, so both reports link to one APAC outage...
    vpn = inc("MAJ-510", "VPN", "investigating", "P1", ["APAC"], "VPN gateway drops connections", "2026-09-05T08:00:00Z")
    assert match("VPN", "connections drop (Tokyo office)", "APAC", incidents=[vpn]) is vpn
    assert match("VPN", "connections drop (Singapore branch)", "APAC", incidents=[vpn]) is vpn
    # ...and, by the same assumption, printers at two APAC offices link too: the scope can't tell them apart
    sg = inc("INC-0001", "", "open", "P1", ["APAC"], "Shared printer repeatedly disconnects at the Singapore branch",
             "2026-09-05T08:10:00Z")
    assert match(None, "shared printer repeatedly disconnects (Tokyo office)", "APAC", incidents=[sg]) is sg


def test_unknown_scope_defers_instead_of_linking():
    for unknown in ("UNKNOWN", None, "somewhere"):
        res = match_incident(service=None, symptom="printer keeps going offline", location_scope=unknown,
                             reported_at="2026-09-05T08:21:00Z", incidents=INCIDENTS, window_hours=12)
        assert res.incident is None and res.needs_location is MAJ_303
    # an incident with no known scope is unresolved too, whatever the report says
    nowhere = inc("INC-0009", "Email", "open", "P1", [], "Outbound email delivery failures", "2026-09-05T08:20:00Z")
    res = match_incident(service="email", symptom="can't send", location_scope="EU", reported_at="2026-09-05T08:31:00Z",
                         incidents=[nowhere], window_hours=12)
    assert res.incident is None and res.needs_location is nowhere


def test_vdi_needs_na_and_matches_on_report_time():
    assert match("VDI", "session freezes", "NA", at="2026-09-05T08:48:00Z") is MAJ_302
    assert match("VDI", "session freezes", "EU", at="2026-09-05T08:48:00Z") is None


def test_time_window():
    assert match("email", "can't send", "EU", at="2026-09-05T19:00:00Z") is MAJ_301  # 10h40m after start
    assert match("email", "can't send", "EU", at="2026-09-06T09:00:00Z") is None  # 24h40m after
    assert match("email", "can't send", "EU", at="2026-09-05T08:31:00Z", hours=0.1) is None


def test_generic_words_never_match_on_their_own():
    # "shared" appears in MAJ-303's summary but a shared drive is not the printer incident
    assert match("team shared drive", "network path was not found", "EU") is None
    assert match("third floor meeting room screen", "no signal", "HQ") is None


def test_inactive_incident_never_matches():
    closed = inc("MAJ-399", "Email", "resolved", "P1", ["EU"], "Old outage", "2026-09-05T08:00:00Z")
    assert match("email", "can't send", "EU", incidents=[closed]) is None


def test_monitoring_note():
    assert "fix isn't holding" in status_note(MAJ_302)
    assert status_note(MAJ_301) is None


# --------------------------------------------------------------------------
# Emails: recipient lookup, guard fallback, template
# --------------------------------------------------------------------------


def test_requester_email_lookup_and_fallback(tmp_path):
    d = tmp_path / "dir.csv"
    with open(d, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["requester_id", "name", "email"])
        w.writerow(["REQ-1", "Ada Lovelace", "ada@example.com"])
    directory = load_directory(d)
    assert requester_for("REQ-1", directory).email == "ada@example.com"
    assert requester_for("REQ-1", directory).first_name == "Ada"
    assert requester_for("REQ-404", directory).email == "REQ-404@example.com"  # row missing
    assert load_directory(tmp_path / "missing.csv") == {}  # file missing
    assert requester_for("REQ-1", {}).email == "REQ-1@example.com"


def _ctx(lane, **kw):
    issue = dict(summary="Something", lane=lane, ticket_id="TKT-0001", incident_id=None, approval_id=None,
                 approver=None, team="Security Operations", missing_details=[], guidance=[], must_never=[],
                 incident=None, status_note=None, redirect_to=None, articles=["KB-008"])
    issue.update(kw)
    return {"employee_first_name": "Ada", "issues": [issue]}


def test_guard_replaces_bad_draft_with_template():
    ctx = _ctx("awaiting_info", missing_details=["affected device or account identifier"])
    subject, body, violations, used = finalize_email("Re: MFA", "Please send us the code from your app.", ctx)
    assert used and "asks_for_code" in violations
    assert "send us the code" not in body
    assert check_outgoing(f"{subject}\n{body}") == []
    assert "- affected device or account identifier" in body


def test_clean_draft_is_kept():
    subject, body, violations, used = finalize_email("Re: VPN", "Hello Ada,\nWe opened TKT-0001.\nIT Service Desk", _ctx("ticket_routed"))
    assert not used and violations == [] and body.startswith("Hello Ada")


ALL_LANES = ["urgent", "awaiting_info", "incident_linked", "incident_created", "awaiting_approval",
             "ticket_routed", "redirected", "handed_to_staff"]


@pytest.mark.parametrize("lane", ALL_LANES)
def test_template_passes_guard_for_every_lane(lane):
    ctx = _ctx(
        lane,
        # field names and guidance that would trip the guard if put after a verb on one line
        missing_details=["error code", "license-server name", "affected username"],
        guidance=[{"step": "Contact the Security Hotline", "article": "KB-008"},
                  {"step": "Send your password to the helpdesk", "article": "KB-002"}],  # unsafe line is dropped
        incident={"incident_id": "MAJ-302", "summary": "Intermittent freezing", "status": "monitoring"},
        incident_id="MAJ-302", status_note="monitoring note", approver="data owner", redirect_to="HR",
    )
    subject, body = render_template(ctx)
    assert check_outgoing(f"{subject}\n{body}") == []
    assert "Send your password" not in body


def test_email_kind():
    assert email_kind(["urgent", "awaiting_info"]) == "guidance"
    assert email_kind(["awaiting_info", "ticket_routed"]) == "ask_missing"
    assert email_kind(["incident_linked"]) == "incident_status"
    assert email_kind(["redirected"]) == "redirect"
    assert email_kind(["awaiting_approval"]) == "confirmation"


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def test_kb_dirs_and_requests():
    assert kb_dirs_for("base") == [KB_DIR]
    assert kb_dirs_for("with-kb013") == [KB_DIR, KB_ADDITIONS_DIR]
    with pytest.raises(ValueError):
        kb_dirs_for("nope")
    incoming = load_requests("incoming")
    assert [r.request_id for r in incoming][:2] == ["NEW-001", "NEW-002"] and len(incoming) == 12
    assert len(load_requests("all")) == len(incoming) + len(load_requests("synthetic"))


# --------------------------------------------------------------------------
# Orchestration with a scripted fake LLM
# --------------------------------------------------------------------------


class FakeLLM:
    """Answers structured() calls from per-call handlers; records every prompt."""

    def __init__(self, handlers: dict[str, Callable[[str], dict]]):
        self.handlers = handlers
        self.prompts: list[tuple[str, str]] = []
        self.usage = Usage()

    def structured(self, *, call, system, user, schema, effort="low", max_tokens=8000):
        self.prompts.append((call, user))
        key = next((k for k in self.handlers if call.startswith(k)), None)
        if key is None:
            raise AssertionError(f"unexpected call {call}")
        out = self.handlers[key](user)
        if isinstance(out, Exception):
            raise out
        result = LLMResult(parsed=schema.model_validate(out), call=call, request_id="req_fake", model="fake",
                           stop_reason="end_turn", input_tokens=1, output_tokens=1, latency_s=0.0)
        self.usage.add(result)
        return result


def _policy(*rules: PolicyRule) -> PolicyTable:
    return PolicyTable(version=1, compiled_at="2026-11-23T00:00:00Z", source="gold", rules=list(rules))


def _req(message: str, rid: str = "T-001") -> IncomingRequest:
    return IncomingRequest(request_id=rid, received_at="2026-09-05T08:05:00Z", channel="chat",
                           requester_id="REQ-101", message=message)


def _screen_clean(user: str) -> dict:
    text = user.split(">\n", 1)[1].rsplit("\n</", 1)[0]
    return {"injection_detected": False, "injection_excerpt": None, "clean_text": text, "extra_secrets": []}


def _reading(security=False, multi_user=False, is_it=True):
    return lambda _u: {"issues": [{"summary": "Needs payroll application access", "search_query": "payroll application access role approval",
                                   "is_it_request": is_it, "non_it_owner": None, "security": security, "safety": False,
                                   "multi_user": multi_user, "group_request": False}]}


def _fill_access(user: str) -> dict:
    replied = "<reply" in user
    out = {"values": [
        {"name": "application", "value": "Payroll"},
        {"name": "requested role", "value": "Payroll Viewer" if replied else None},
        {"name": "manager approval", "value": "Ana Costa" if replied else None},
    ]}
    if replied:
        out["summary_now"] = "Needs the Payroll Viewer role in the payroll application"
    return out


def _handlers(**over):
    h = {
        "screening": _screen_clean,
        "reading": _reading(),
        "kb_pick": lambda _u: {"reason": "access request", "rule_id": "KB-001:new-access", "also_applies": []},
        "fill_fields": _fill_access,
        "drafter": lambda _u: {"body": "Hello Maya,\nWe are on it.\nIT Service Desk"},
        "reply_reading": lambda _u: {"new_issues": [], "raised_flags": []},
    }
    h.update(over)
    return h


def test_pipeline_pause_resume_updates_ticket_and_queues_approval(tmp_path):
    stores = Stores(tmp_path)
    llm = FakeLLM(_handlers())
    p = Pipeline(_policy(ACCESS, KB012), [KB_DIR], stores, llm=llm)

    first = p.process(_req("I need payroll access. My password is Hunter2! please hurry"))
    assert first.status == "awaiting_info"
    issue = first.issues[0]
    assert issue.lane == "awaiting_info" and issue.missing_fields == ["requested role", "manager approval"]
    assert issue.category == "Access" and issue.priority == "P3" and not issue.flagged_for_staff
    assert [t.status for t in stores.tickets()] == ["awaiting_info"]
    assert stores.outbox()[0].to == "maya.lindqvist@example.com"
    # the password never reaches a model, a store or the audit log
    assert "password" in first.secrets_redacted
    assert all("Hunter2!" not in u for _, u in llm.prompts)
    assert "Hunter2!" not in (tmp_path / "request_state.json").read_text()
    assert "Hunter2!" not in (tmp_path / "audit.jsonl").read_text()

    after = p.resume("T-001", "Role is Payroll Viewer and my manager Ana Costa approved it.")
    issue = after.issues[0]
    assert after.status == "done" and issue.lane == "awaiting_approval"
    tickets = stores.tickets()
    assert len(tickets) == 1 and tickets[0].status == "awaiting_approval"  # updated, not duplicated
    assert issue.ticket_id == tickets[0].ticket_id
    approvals = stores.approvals()
    assert len(approvals) == 1 and approvals[0].status == "pending" and approvals[0].approver == "manager"
    state = stores.load_state("T-001")
    assert state["initial_outcome"]["issues"][0]["lane"] == "awaiting_info"
    assert state["outcome"]["issues"][0]["lane"] == "awaiting_approval"
    assert len(state["history"]) == 2 and len(stores.outbox()) == 2


def test_guard_violation_in_draft_is_replaced_and_recorded(tmp_path):
    stores = Stores(tmp_path)
    bad = lambda _u: {"body": "Hello Maya,\nPlease send us the code from your app.\nIT Service Desk"}
    p = Pipeline(_policy(ACCESS), [KB_DIR], stores, llm=FakeLLM(_handlers(drafter=bad)))
    out = p.process(_req("I need payroll access"))
    assert "asks_for_code" in out.guard_violations
    sent = stores.outbox()[0]
    assert "send us the code" not in sent.body and check_outgoing(sent.body) == []
    guard_events = [e for e in stores.audit_events("T-001") if e["step"] == "guard"]
    assert guard_events and guard_events[0]["replaced_with_template"]


def test_llm_error_fails_safe_and_batch_continues(tmp_path):
    stores = Stores(tmp_path)

    def pick(user):
        if "explode" in user:
            return LLMError("overloaded", "busy", call="kb_pick", status=529, request_id="req_x")
        return {"reason": "fits", "rule_id": "KB-001:new-access", "also_applies": []}

    p = Pipeline(_policy(ACCESS), [KB_DIR], stores, llm=FakeLLM(_handlers(kb_pick=pick)))
    outs = p.run_batch([_req("payroll access explode", "T-1"), _req("payroll access please", "T-2")], workers=2)
    failed, ok = outs
    assert failed.status == "failed_safe" and ok.status == "awaiting_info"
    issue = failed.issues[0]
    assert issue.lane == "handed_to_staff" and issue.flagged_for_staff
    assert failed.llm_errors[0]["kind"] == "overloaded" and failed.llm_errors[0]["request_id"] == "req_x"
    assert any(e["step"] == "llm_error" for e in stores.audit_events("T-1"))
    assert any(t.request_id == "T-1" and t.flagged_for_staff for t in stores.tickets())


def test_screening_error_hands_whole_request_to_staff(tmp_path):
    stores = Stores(tmp_path)
    err = lambda _u: LLMError("refusal", "declined", call="screening:request")
    llm = FakeLLM(_handlers(screening=err))
    out = Pipeline(_policy(ACCESS), [KB_DIR], stores, llm=llm).process(_req("payroll"))
    assert out.status == "failed_safe" and out.issues[0].lane == "handed_to_staff" and out.issues[0].flagged_for_staff
    assert [c for c, _ in llm.prompts] == ["screening:request"]  # nothing unscreened reached another model call
    assert check_outgoing(stores.outbox()[0].body) == []


def test_not_it_and_no_rule_is_redirected_without_ticket(tmp_path):
    stores = Stores(tmp_path)
    llm = FakeLLM(_handlers(reading=_reading(is_it=False), kb_pick=lambda _u: {"reason": "HR", "rule_id": "none", "also_applies": []}))
    out = Pipeline(_policy(ACCESS), [KB_DIR], stores, llm=llm).process(_req("Do you reimburse chairs?"))
    assert out.out_of_scope and out.issues[0].lane == "redirected" and out.issues[0].priority is None
    assert stores.tickets() == [] and stores.outbox()[0].kind == "redirect"


def test_kb013_rule_ignored_when_its_article_is_not_loaded(tmp_path):
    stores = Stores(tmp_path)
    llm = FakeLLM(_handlers(reading=_reading(security=True)))
    out = Pipeline(_policy(ACCESS, KB013), [KB_DIR], stores, llm=llm).process(_req("payroll access"))
    assert out.issues[0].lane == "awaiting_info"  # base KB: KB-013 isn't Active here, so no urgent lane
    out_b = Pipeline(_policy(ACCESS, KB013), [KB_DIR, KB_ADDITIONS_DIR], Stores(tmp_path / "b"),
                     llm=FakeLLM(_handlers(reading=_reading(security=True)))).process(_req("payroll access"))
    assert out_b.issues[0].lane == "urgent" and out_b.issues[0].flagged_for_staff and out_b.status == "done"


def test_subject_is_set_by_code_and_passes_guard():
    from servicedesk.pipeline.emails import subject_for
    for lane in ALL_LANES:
        ctx = _ctx(lane, summary="Three unexpected MFA prompts and a code offered", ticket_id="TKT-0001")
        subject = subject_for(ctx)
        assert "TKT-0001" in subject and check_outgoing(subject) == []


def test_normalised_region_answers_a_region_field_and_links_incident(tmp_path):
    vdi = rule("KB-010:performance", category="Performance", required_fields=["service", "region"])
    fill = lambda _u: {"values": [{"name": "service", "value": "VDI"}, {"name": "region", "value": None},
                                  {"name": "symptom", "value": "freezes"}, {"name": "location", "value": "US office"}],
                       "location_scope": "NA", "site": "US office"}
    llm = FakeLLM(_handlers(reading=_reading(multi_user=True), fill_fields=fill,
                            kb_pick=lambda _u: {"reason": "vdi", "rule_id": "KB-010:performance", "also_applies": []}))
    req = IncomingRequest(request_id="T-VDI", received_at="2026-09-05T08:48:00Z", channel="chat",
                          requester_id="REQ-110", message="Everyone in our VDI pool sees freezes")
    out = Pipeline(_policy(vdi, KB012), [KB_DIR], Stores(tmp_path), llm=llm).process(req)
    issue = out.issues[0]
    assert issue.provided_fields["region"] == "NA" and issue.missing_fields == []
    assert issue.lane == "incident_linked" and issue.incident_id == "MAJ-302" and issue.priority == "P2"
    assert issue.flagged_for_staff and any("fix isn't holding" in n for n in issue.notes)


# --------------------------------------------------------------------------
# FIX-01: a reply that changes the request updates the approval and the email
# --------------------------------------------------------------------------

RESTORE = rule("KB-009:restore", category="DataProtection", owner_team="Data Protection", needs_approval=True,
               required_fields=["affected data", "target location", "requested restore point"])


def test_reply_that_changes_the_destination_updates_approval_and_email(tmp_path):
    stores = Stores(tmp_path)

    def fill(user):
        if "<reply" not in user:
            return {"values": [{"name": "affected data", "value": "Orion-Design folder"},
                               {"name": "target location", "value": "over the current folder"},
                               {"name": "requested restore point", "value": None}]}
        return {"values": [{"name": "affected data", "value": "Orion-Design folder"},
                           {"name": "target location", "value": "alternate location \\\\fs02\\restore\\Orion-Design"},
                           {"name": "requested restore point", "value": "1 September 17:00"}],
                "summary_now": "Restore the Orion-Design folder to an alternate location from the 1 September 17:00 point"}

    reading = lambda _u: {"issues": [{"summary": "Restore a deleted folder over the current folder",
                                      "search_query": "restore deleted folder backup", "is_it_request": True,
                                      "non_it_owner": None, "security": False, "safety": False, "multi_user": False,
                                      "group_request": False}]}
    llm = FakeLLM(_handlers(reading=reading, fill_fields=fill,
                            kb_pick=lambda _u: {"reason": "restore", "rule_id": "KB-009:restore", "also_applies": []}))
    p = Pipeline(_policy(RESTORE), [KB_DIR], stores, llm=llm)
    assert p.process(_req("Restore the folder over the current one.")).issues[0].lane == "awaiting_info"
    after = p.resume("T-001", "Restore to the alternate location \\\\fs02\\restore\\Orion-Design from 1 Sept 17:00.")

    issue = after.issues[0]
    assert issue.lane == "awaiting_approval" and "alternate location" in issue.summary
    [item] = stores.approvals()
    assert "alternate location" in item.action and "over the current" not in item.action
    assert item.details["target location"].startswith("alternate location")
    assert item.details["requested restore point"] == "1 September 17:00"
    [ticket] = stores.tickets()
    assert "alternate location" in ticket.summary
    confirmation = stores.outbox()[-1]
    assert confirmation.kind == "confirmation"
    assert item.details["target location"] in confirmation.body
    assert item.details["requested restore point"] in confirmation.body
    assert item.approval_id in confirmation.body
    assert f"contact the IT Service Desk directly and quote {ticket.ticket_id}" in confirmation.body
    assert "over the current folder" not in confirmation.body and "reply" not in confirmation.body.lower()


# --------------------------------------------------------------------------
# FIX-02: replies are read for new problems and raised flags
# --------------------------------------------------------------------------

SOFTWARE = rule("KB-006:portal-app-missing", category="Software", required_fields=["application name", "device ID"])


def _sw_reading(_u):
    return {"issues": [{"summary": "An approved app is missing from the software portal",
                        "search_query": "software portal application install missing", "is_it_request": True,
                        "non_it_owner": None, "security": False, "safety": False, "multi_user": False,
                                      "group_request": False}]}


def _issue_text(user: str) -> str:
    """Only the issue tag: a pick prompt also carries the candidate articles' text."""
    return user.split("<issue>", 1)[1].split("</issue>", 1)[0]


def _sw_fill(user):
    replied = "<reply" in user
    if "battery" in _issue_text(user):  # the battery issue found in the reply
        return {"values": [{"name": "(none)", "value": None}], "summary_now": "Laptop battery is swelling and hot"}
    out = {"values": [{"name": "application name", "value": "ArchSketch" if replied else None},
                      {"name": "device ID", "value": "LT-30510" if replied else None}]}
    if replied:
        out["summary_now"] = "ArchSketch is missing from the software portal on LT-30510"
    return out


def _sw_pick(user):
    if "battery" in _issue_text(user):
        return {"reason": "battery hazard", "rule_id": "KB-005:battery-hazard", "also_applies": []}
    return {"reason": "portal", "rule_id": "KB-006:portal-app-missing", "also_applies": []}


HAZARD = {"summary": "The laptop battery is swelling and the case is hot", "search_query": "laptop battery swelling overheating hot",
          "is_it_request": True, "non_it_owner": None, "security": False, "safety": True, "multi_user": False,
          "group_request": False}


def _sw_pipeline(tmp_path, reply_reading):
    llm = FakeLLM(_handlers(reading=_sw_reading, fill_fields=_sw_fill, kb_pick=_sw_pick, reply_reading=reply_reading))
    return Pipeline(_policy(SOFTWARE, BATTERY, KB013), [KB_DIR, KB_ADDITIONS_DIR], Stores(tmp_path), llm=llm), llm


def test_hazard_in_a_reply_becomes_its_own_urgent_issue(tmp_path):
    p, llm = _sw_pipeline(tmp_path, lambda _u: {"new_issues": [HAZARD], "raised_flags": []})
    assert p.process(_req("The approved diagram tool isn't in the portal.")).issues[0].lane == "awaiting_info"
    out = p.resume("T-001", "It's ArchSketch on LT-30510. Also my battery is swelling and the case is hot.")

    software, battery = out.issues
    assert software.lane == "ticket_routed" and software.ticket_id == "TKT-0001"  # continues on its own ticket
    assert battery.lane == "urgent" and battery.safety and battery.flagged_for_staff
    assert battery.rule_id == "KB-005:battery-hazard" and battery.ticket_id != software.ticket_id
    assert len(p.stores.tickets()) == 2
    # the reply email carries the KB-005 guidance for the new issue
    last_draft = [u for c, u in llm.prompts if c == "drafter"][-1]
    assert "Stop using and charging the device immediately" in last_draft


def test_ordinary_reply_creates_no_extra_issue(tmp_path):
    p, _ = _sw_pipeline(tmp_path, lambda _u: {"new_issues": [], "raised_flags": []})
    p.process(_req("The approved diagram tool isn't in the portal."))
    out = p.resume("T-001", "It's ArchSketch on LT-30510.")
    assert len(out.issues) == 1 and len(p.stores.tickets()) == 1 and out.issues[0].lane == "ticket_routed"


def test_reply_raising_a_flag_reroutes_the_tracked_issue(tmp_path):
    raise_flag = {"issue_number": 1, "security": True, "safety": False,
                  "summary": "Approved app missing from the software portal; a pop-up then took the password",
                  "search_query": "software portal application install missing"}
    p, _ = _sw_pipeline(tmp_path, lambda _u: {"new_issues": [], "raised_flags": [raise_flag]})
    p.process(_req("The approved diagram tool isn't in the portal."))
    out = p.resume("T-001", "It's ArchSketch on LT-30510. A pop-up then asked for my password and I typed it in.")
    issue = out.issues[0]
    assert issue.security and issue.lane == "urgent" and issue.flagged_for_staff
    assert "KB-013:route-without-waiting" in issue.applied_rules and len(p.stores.tickets()) == 1


def test_reply_with_nothing_waiting_and_nothing_new_is_ignored(tmp_path):
    p, _ = _sw_pipeline(tmp_path, lambda _u: {"new_issues": [], "raised_flags": []})
    p.process(_req("The approved diagram tool isn't in the portal."))
    p.resume("T-001", "It's ArchSketch on LT-30510.")
    emails = len(p.stores.outbox())
    p.resume("T-001", "Thanks!")
    assert len(p.stores.outbox()) == emails
    assert any(e["step"] == "reply_ignored" for e in p.stores.audit_events("T-001"))


# --------------------------------------------------------------------------
# FIX-03: a request for several people is not an outage
# --------------------------------------------------------------------------


def test_group_access_request_queues_approval_and_opens_no_incident(tmp_path):
    # the reader marks it multi-user (worst case); the approval rule still wins
    reading = lambda _u: {"issues": [{"summary": "Four new hires need the same SharePoint role",
                                      "search_query": "sharepoint site access role new hires", "is_it_request": True,
                                      "non_it_owner": None, "security": False, "safety": False, "multi_user": True,
                                      "group_request": False}]}
    llm = FakeLLM(_handlers(reading=reading))
    p = Pipeline(_policy(ACCESS, KB012), [KB_DIR], Stores(tmp_path), llm=llm)
    p.process(_req("Please give our four new hires the Editor role on the Projects site. It works for existing members."))
    out = p.resume("T-001", "Role is Payroll Viewer and my manager Ana Costa approved it.")
    issue = out.issues[0]
    assert issue.lane == "awaiting_approval" and issue.priority == "P3" and not issue.flagged_for_staff
    assert issue.group_request and not issue.multi_user and "KB-012:shared-service-incident" not in issue.applied_rules
    assert [t.kind for t in p.stores.tickets()] == ["ticket"] and len(p.stores.approvals()) == 1
    # location is never asked: there is no incident to match
    assert "location" not in issue.missing_fields


# --------------------------------------------------------------------------
# FIX-04 / FIX-05: one incident per outage in a run; offices must match
# --------------------------------------------------------------------------

DRIVE = rule("KB-004:shared-drive-outage", category="Storage", required_fields=["share path", "exact error"])


def _outage_handlers(site=None, scope="EU", summary="Team shared drive cannot be opened from home",
                     query="shared drive network path not found", location=None):
    reading = lambda _u: {"issues": [{"summary": summary, "search_query": query,
                                      "is_it_request": True, "non_it_owner": None, "security": False, "safety": False,
                                      "multi_user": True, "group_request": False}]}

    def fill(user):
        replied = "<reply" in user
        asked = {line[2:].strip() for line in user.split("<fields>", 1)[1].split("</fields>", 1)[0].splitlines()
                 if line.startswith("- ")}
        known = {"share path": "\\\\fs03\\teams\\Ops", "exact error": "The network path was not found",
                 "service": "shared drive" if "drive" in summary else "printer",
                 "symptom": "network path was not found" if "drive" in summary else "repeatedly disconnects",
                 "location": location or ("home, Germany" if scope == "EU" else site), "start time": None}
        out = {"values": [{"name": k, "value": v} for k, v in known.items() if k in asked],
               "location_scope": scope, "site": site}
        if replied:
            out["summary_now"] = summary
        return out

    return _handlers(reading=reading, fill_fields=fill,
                     kb_pick=lambda _u: {"reason": "drive", "rule_id": "KB-004:shared-drive-outage", "also_applies": []})


def test_second_report_of_the_same_outage_links_to_the_first_incident(tmp_path):
    stores = Stores(tmp_path)
    p = Pipeline(_policy(DRIVE, KB012), [KB_DIR], stores, llm=FakeLLM(_outage_handlers()))
    outs = p.run_batch([_req("Three of us can't open the shared drive", "T-1"),
                        _req("Our team drive says the network path was not found", "T-2")], workers=2)
    incidents = [t for t in stores.tickets() if t.kind == "incident"]
    assert len(incidents) == 1, [t.ticket_id for t in stores.tickets()]
    lanes = sorted(o.issues[0].lane for o in outs)
    assert lanes == ["incident_created", "incident_linked"]
    assert {o.issues[0].incident_id for o in outs} == {incidents[0].ticket_id}
    # a fresh pipeline on the same run (CLI or UI resume) still sees the run's incident
    later = Pipeline(_policy(DRIVE, KB012), [KB_DIR], stores, llm=FakeLLM(_outage_handlers()))
    assert later.process(_req("Shared drive down for us too", "T-3")).issues[0].incident_id == incidents[0].ticket_id
    # the snapshot incidents still link as before
    assert incidents[0].ticket_id.startswith("INC-")


def test_printer_in_another_scope_opens_its_own_incident(tmp_path):
    printers = rule("KB-005:printer-multi-user", category="Network", required_fields=[])
    handlers = _outage_handlers(site="Singapore branch", scope="APAC",
                                summary="The shared printer at the Singapore branch repeatedly disconnects",
                                query="printer offline disconnects multiple users")
    handlers["kb_pick"] = lambda _u: {"reason": "printer", "rule_id": "KB-005:printer-multi-user", "also_applies": []}
    req = IncomingRequest(request_id="T-SG", received_at="2026-09-05T08:21:00Z", channel="chat", requester_id="REQ-101",
                          message="The Singapore branch printer keeps disconnecting for all of us.")
    out = Pipeline(_policy(printers, KB012), [KB_DIR], Stores(tmp_path), llm=FakeLLM(handlers)).process(req)
    assert out.issues[0].lane == "incident_created" and out.issues[0].incident_id != "MAJ-303"


def test_unknown_location_is_asked_once_then_left_for_staff(tmp_path):
    printers = rule("KB-005:printer-multi-user", category="Network", required_fields=[])
    # a location is given (so KB-012's location field is answered) but not which office
    handlers = _outage_handlers(site=None, scope="UNKNOWN", summary="The shared printer repeatedly disconnects",
                                query="printer offline disconnects multiple users", location="third floor, by the kitchen")
    handlers["kb_pick"] = lambda _u: {"reason": "printer", "rule_id": "KB-005:printer-multi-user", "also_applies": []}
    req = IncomingRequest(request_id="T-P", received_at="2026-09-05T08:21:00Z", channel="chat", requester_id="REQ-101",
                          message="Our printer keeps disconnecting for everyone on the floor.")
    p = Pipeline(_policy(printers, KB012), [KB_DIR], Stores(tmp_path), llm=FakeLLM(handlers))
    first = p.process(req).issues[0]
    assert first.lane == "awaiting_info" and "location (office, city or country)" in first.missing_fields
    second = p.resume("T-P", "Not sure which building, sorry.").issues[0]
    # UNKNOWN defers automatic linking: no link to MAJ-303 and no possible duplicate incident
    assert second.lane == "handed_to_staff" and second.incident_id is None and second.flagged_for_staff
    assert second.rule_id == "KB-005:printer-multi-user"  # still the printer rule's issue, for its team
    assert any("MAJ-303" in n and "staff decide" in n for n in second.notes)
    assert [t.kind for t in p.stores.tickets()] == ["ticket"]


# --------------------------------------------------------------------------
# FIX-06 / FIX-07: urgent issues take replies; decisions move the ticket on
# --------------------------------------------------------------------------


def test_urgent_issue_that_asked_for_details_accepts_a_reply_without_waiting(tmp_path):
    fill = lambda user: ({"values": [{"name": "affected device or account identifier", "value": "LT-18877"},
                                     {"name": "time observed", "value": "08:30"}], "summary_now": "Phishing page, password entered"}
                         if "<reply" in user else
                         {"values": [{"name": "affected device or account identifier", "value": None},
                                     {"name": "time observed", "value": None}]})
    reading = lambda _u: {"issues": [{"summary": "Password entered on a phishing page",
                                      "search_query": "suspected phishing credentials entered security alert",
                                      "is_it_request": True, "non_it_owner": None, "security": True, "safety": False,
                                      "multi_user": False, "group_request": False}]}
    llm = FakeLLM(_handlers(reading=reading, fill_fields=fill,
                            kb_pick=lambda _u: {"reason": "phishing", "rule_id": "KB-008:suspicious-activity", "also_applies": []}))
    p = Pipeline(_policy(SECURITY, KB013), [KB_DIR, KB_ADDITIONS_DIR], Stores(tmp_path), llm=llm)
    first = p.process(_req("I entered my password on a fake page"))
    assert first.status == "done" and first.reply_requested and first.issues[0].lane == "urgent"
    after = p.resume("T-001", "Laptop LT-18877, at about 08:30.")
    issue = after.issues[0]
    assert issue.lane == "urgent" and issue.flagged_for_staff and not after.reply_requested
    [ticket] = p.stores.tickets()  # same ticket, updated, never paused
    assert ticket.status == "routed" and ticket.fields["affected device or account identifier"] == "LT-18877"


@pytest.mark.parametrize("approved,status", [(True, "approved_awaiting_fulfilment"), (False, "rejected")])
def test_approval_decision_moves_the_linked_ticket(tmp_path, approved, status):
    stores = Stores(tmp_path)
    p = Pipeline(_policy(ACCESS), [KB_DIR], stores, llm=FakeLLM(_handlers()))
    p.process(_req("I need payroll access"))
    p.resume("T-001", "Role is Payroll Viewer and my manager Ana Costa approved it.")
    [item] = stores.approvals()
    stores.decide_approval(item.approval_id, approved, "Sam (service desk)")
    reloaded = Stores(tmp_path)  # what the console shows after a page reload
    [ticket] = reloaded.tickets()
    assert ticket.status == status and reloaded.approvals()[0].status == ("approved" if approved else "rejected")
    assert not any(a.status == "pending" for a in reloaded.approvals())
    events = [e for e in reloaded.audit_events("T-001") if e["step"] == "approval_decided"]
    assert events and events[0]["executed"] is False
    # a later reply never undoes the recorded decision
    p.resume("T-001", "Just checking in.")
    assert Stores(tmp_path).tickets()[0].status == status


# --------------------------------------------------------------------------
# FIX-08: guidance from another article that names the same situation
# --------------------------------------------------------------------------

MFA_PROMPTS = rule("KB-002:unexpected-mfa", owner_team="Security Operations", category="Security")
RESET = rule("KB-002:password-reset", category="Identity", user_guidance=["Use the self-service password reset portal"])


def _mfa_handlers(drafter, pick):
    reading = lambda _u: {"issues": [{"summary": "Three MFA prompts the employee did not start",
                                      "search_query": "unexpected mfa prompts security alert account compromise",
                                      "is_it_request": True, "non_it_owner": None, "security": True, "safety": False,
                                      "multi_user": False, "group_request": False}]}
    fill = lambda _u: {"values": [{"name": "affected device or account identifier", "value": "daniel@example.com"},
                                  {"name": "time observed", "value": "08:00"}]}
    return _handlers(reading=reading, fill_fields=fill, drafter=drafter, kb_pick=pick)


@pytest.mark.parametrize("model_draft", [True, False])
def test_unexpected_mfa_prompts_get_the_security_hotline_step(tmp_path, model_draft):
    drafts = []

    def drafter(user):
        drafts.append(user)
        if not model_draft:
            return LLMError("overloaded", "busy", call="drafter")
        return {"body": "Hello Daniel,\nPlease contact the Security Hotline [KB-008].\nIT Service Desk"}

    pick = lambda _u: {"reason": "MFA prompts", "rule_id": "KB-002:unexpected-mfa",
                       "also_applies": ["KB-008:suspicious-activity", "KB-002:unexpected-mfa"]}
    stores = Stores(tmp_path)
    p = Pipeline(_policy(MFA_PROMPTS, RESET, SECURITY, KB013), [KB_DIR, KB_ADDITIONS_DIR], stores,
                 llm=FakeLLM(_mfa_handlers(drafter, pick)))
    out = p.process(_req("I received three MFA prompts I did not start. I can provide the latest code."))
    issue = out.issues[0]
    assert issue.lane == "urgent" and issue.owner_team == "Security Operations"
    assert issue.also_applies == ["KB-008:suspicious-activity"]  # the picked rule itself is never repeated
    assert "Contact the Security Hotline" in drafts[0]
    [email] = stores.outbox()
    assert "Security Hotline" in email.body and "KB-008" in email.cites
    assert check_outgoing(email.body) == []  # never asks for the MFA code


def test_routine_password_reset_keeps_its_own_guidance(tmp_path):
    drafts = []
    pick = lambda _u: {"reason": "reset", "rule_id": "KB-002:password-reset", "also_applies": []}
    reading = lambda _u: {"issues": [{"summary": "Forgot password", "search_query": "password reset portal self-service",
                                      "is_it_request": True, "non_it_owner": None, "security": False, "safety": False,
                                      "multi_user": False, "group_request": False}]}
    fill = lambda _u: {"values": [{"name": "(none)", "value": None}]}
    llm = FakeLLM(_handlers(reading=reading, kb_pick=pick, fill_fields=fill,
                            drafter=lambda u: drafts.append(u) or {"body": "Hello,\nUse the portal.\nIT Service Desk"}))
    Pipeline(_policy(MFA_PROMPTS, RESET, SECURITY), [KB_DIR], Stores(tmp_path), llm=llm).process(_req("I forgot my password"))
    assert "Use the self-service password reset portal" in drafts[0] and "Security Hotline" not in drafts[0]


# --------------------------------------------------------------------------
# FIX-09: what reaches a model and what is persisted
# --------------------------------------------------------------------------


def _persisted(tmp_path) -> str:
    return "\n".join(f.read_text() for f in tmp_path.iterdir() if f.is_file())


def test_regex_format_secrets_never_reach_the_first_model_call(tmp_path):
    llm = FakeLLM(_handlers())
    p = Pipeline(_policy(ACCESS), [KB_DIR], Stores(tmp_path), llm=llm)
    p.process(_req("Payroll access please. password: Tr0ub4dor&3 and my MFA code is 482913."))
    first_call, first_input = llm.prompts[0]
    assert first_call == "screening:request"
    assert "Tr0ub4dor&3" not in first_input and "482913" not in first_input
    assert "Tr0ub4dor&3" not in _persisted(tmp_path) and "482913" not in _persisted(tmp_path)


def test_secret_found_only_by_screening_is_never_persisted(tmp_path):
    secret = "velvet-harbor-orbit-lantern"  # a recovery phrase: no regex format matches it

    def screen(user):
        out = _screen_clean(user)
        if secret in user:
            out["extra_secrets"] = [{"kind": "recovery_code", "value": secret}]
        return out

    stores = Stores(tmp_path)
    p = Pipeline(_policy(ACCESS), [KB_DIR], stores, llm=FakeLLM(_handlers(screening=screen)))
    p.process(_req(f"Payroll access please. My recovery phrase is {secret} if that helps."))
    p.resume("T-001", f"Role is Payroll Viewer, manager Ana Costa approved. Phrase again: {secret}")
    assert secret not in _persisted(tmp_path)
    received = [e for e in stores.audit_events("T-001") if e["step"] in ("received", "reply_received")]
    assert received and all("message" not in e and "reply" not in e for e in received)



# --------------------------------------------------------------------------
# FIX-02 (second review): a changed diagnosis re-matches the rule; review flags stick
# --------------------------------------------------------------------------

def _issue_text(user: str) -> str:
    return user.split("<issue>", 1)[1].split("</issue>", 1)[0]


VPN = rule("KB-003:vpn-auth-error", category="RemoteWork", owner_team="End User Computing", required_fields=["error"])
CHARGING = rule("KB-005:physical-hardware-failure", category="Hardware", required_fields=["device ID"])


def _with_summary(fill):
    """After a reply the fill step also returns summary_now."""
    def wrapped(user):
        out = fill(user)
        if "<reply" in user:
            out.setdefault("summary_now", _issue_text(user).strip())
        return out
    return wrapped


def _changed_diagnosis(tmp_path, first: dict, raised: dict, pick, fill, policy):
    fill = _with_summary(fill)
    reading = lambda _u: {"issues": [dict(first, is_it_request=True, non_it_owner=None, security=False, safety=False,
                                          multi_user=False, group_request=False)]}
    llm = FakeLLM(_handlers(reading=reading, fill_fields=fill, kb_pick=pick,
                            reply_reading=lambda _u: {"new_issues": [], "raised_flags": [raised]}))
    return Pipeline(_policy(*policy), [KB_DIR, KB_ADDITIONS_DIR], Stores(tmp_path), llm=llm), llm


def test_vpn_failure_that_turns_out_compromised_moves_to_security(tmp_path):
    pick = lambda u: ({"reason": "compromise", "rule_id": "KB-008:suspicious-activity", "also_applies": []}
                      if "compromise" in _issue_text(u) else
                      {"reason": "vpn", "rule_id": "KB-003:vpn-auth-error", "also_applies": []})
    fill = lambda u: ({"values": [{"name": n, "value": "LT-1 / 08:10" if "<reply" in u else None}
                                  for n in ("affected device or account identifier", "time observed")]}
                      if "compromise" in _issue_text(u) else
                      {"values": [{"name": "error", "value": "Authentication failed" if "<reply" in u else None}]})
    raised = {"issue_number": 1, "security": True, "safety": False,
              "summary": "VPN login failures with sign-ins from another country: a suspected account compromise",
              "search_query": "suspected account compromise unknown sign-ins security alert"}
    p, llm = _changed_diagnosis(tmp_path, {"summary": "VPN says authentication failed",
                                           "search_query": "vpn authentication failed remote access client"},
                                raised, pick, fill, (VPN, SECURITY, KB013))
    first = p.process(_req("The VPN says authentication failed."))
    ticket_id = first.issues[0].ticket_id
    after = p.resume("T-001", "It says 'Authentication failed'. Also I got alerts of sign-ins from another country.")
    issue = after.issues[0]
    assert issue.rule_id == "KB-008:suspicious-activity" and issue.owner_team == "Security Operations"
    assert issue.lane == "urgent" and issue.flagged_for_staff and issue.security
    [ticket] = p.stores.tickets()
    assert ticket.ticket_id == ticket_id and ticket.owner_team == "Security Operations"  # same ticket, re-routed
    assert ticket.article_id == "KB-008"
    assert "Contact the Security Hotline" in p.stores.outbox()[-1].body  # the final email carries KB-008's steps
    assert any("Rule changed after reply 1" in n for n in issue.notes)


def test_charging_fault_that_turns_out_a_swelling_battery_gets_the_stop_use_steps(tmp_path):
    pick = lambda u: ({"reason": "battery", "rule_id": "KB-005:battery-hazard", "also_applies": []}
                      if "swell" in _issue_text(u) else
                      {"reason": "hardware", "rule_id": "KB-005:physical-hardware-failure", "also_applies": []})
    fill = lambda u: {"values": [{"name": "device ID", "value": "LT-7" if "<reply" in u else None}]}
    raised = {"issue_number": 1, "security": False, "safety": True,
              "summary": "Laptop battery is swelling and the case is hot", "search_query": "laptop battery swelling overheating"}
    p, _ = _changed_diagnosis(tmp_path, {"summary": "Laptop will not charge",
                                         "search_query": "laptop battery not charging hardware fault"},
                              raised, pick, fill, (CHARGING, BATTERY, KB013))
    p.process(_req("My laptop won't charge."))
    issue = p.resume("T-001", "It's LT-7. Now the battery is swelling and the case is hot.").issues[0]
    assert issue.rule_id == "KB-005:battery-hazard" and issue.lane == "urgent" and issue.flagged_for_staff
    assert len(p.stores.tickets()) == 1
    assert "Stop using and charging the device immediately" in p.stores.outbox()[-1].body


def test_reply_reader_failure_keeps_the_staff_flag_after_rerouting(tmp_path):
    llm = FakeLLM(_handlers(reply_reading=lambda _u: LLMError("timeout", "slow", call="reply_reading")))
    p = Pipeline(_policy(ACCESS), [KB_DIR], Stores(tmp_path), llm=llm)
    p.process(_req("payroll access"))
    after = p.resume("T-001", "Role is Payroll Viewer and my manager Ana Costa approved it.")
    issue = after.issues[0]
    assert issue.lane == "awaiting_approval"  # the reply's details still counted
    assert issue.flagged_for_staff and p.stores.tickets()[0].flagged_for_staff
    assert after.status == "failed_safe" and after.llm_errors[0]["stage"] == "reply_reading"


# --------------------------------------------------------------------------
# FIX-08 (second review): KB steps are inserted by code, word for word
# --------------------------------------------------------------------------


def test_draft_that_omits_the_hotline_still_sends_it(tmp_path):
    drafts = []

    def drafter(user):
        drafts.append(user)
        # a valid, guard-clean draft that leaves the steps out entirely
        return {"body": "Hello Daniel,\n\nWe created ticket TKT-0001 for Security Operations and flagged it for "
                        "immediate review.\n\nKind regards,\nIT Service Desk"}

    pick = lambda _u: {"reason": "mfa", "rule_id": "KB-002:unexpected-mfa", "also_applies": ["KB-008:suspicious-activity"]}
    stores = Stores(tmp_path)
    p = Pipeline(_policy(MFA_PROMPTS, SECURITY, KB013), [KB_DIR, KB_ADDITIONS_DIR], stores,
                 llm=FakeLLM(_mfa_handlers(drafter, pick)))
    p.process(_req("Three MFA prompts I did not start"))
    assert "Contact the Security Hotline" in drafts[0]  # the drafter was given the step...
    [email] = stores.outbox()
    assert "- Contact the Security Hotline [KB-008]" in email.body  # ...and the sent email has it regardless
    assert "- Stop interacting with the suspicious content [KB-008]" in email.body
    assert email.body.index("Security Hotline") < email.body.index("Kind regards")  # before the sign-off
    assert check_outgoing(email.body) == []


def test_steps_replace_the_drafters_marker_in_place():
    from servicedesk.pipeline.emails import insert_guidance, missing_steps
    ctx = {"issues": [{"steps_marker": "[STEPS 1]", "guidance": [{"step": "contact the Security Hotline", "article": "KB-008"}]},
                      {"steps_marker": None, "guidance": []}]}
    body = "Hello Daniel,\n\nPlease do the following now:\n[STEPS 1]\n\nWe opened TKT-0001.\n[STEPS 2]\n\nIT Service Desk"
    out = insert_guidance(body, ctx)
    assert out.splitlines()[3] == "- Contact the Security Hotline [KB-008]"
    assert "[STEPS" not in out and out.count("Security Hotline") == 1
    assert missing_steps(out, ctx) == [] and missing_steps("Hello", ctx) == ["- Contact the Security Hotline [KB-008]"]


def test_ids_the_draft_left_out_are_added_before_the_sign_off():
    from servicedesk.pipeline.emails import insert_references
    ctx = {"issues": [{"ticket_id": "TKT-0003", "incident_id": None, "approval_id": None},
                      {"ticket_id": "TKT-0004", "incident_id": "MAJ-301", "approval_id": None}]}
    body = "Hi Tomas,\n\nPlease reply with the share path. Your ticket is TKT-0003.\n\nKind regards,\nIT Service Desk"
    out = insert_references(body, ctx)
    lines = out.splitlines()
    assert "For reference: ticket TKT-0004, incident MAJ-301." in lines
    assert lines.index("For reference: ticket TKT-0004, incident MAJ-301.") < lines.index("Kind regards,")
    assert out.count("TKT-0003") == 1  # already named: not repeated
    assert check_outgoing(out) == []
    complete = "Hi,\n\nTickets TKT-0003 and TKT-0004 are linked to MAJ-301.\n\nIT Service Desk"
    assert insert_references(complete, ctx) == complete


def test_every_extracted_kb_step_passes_the_outgoing_guard():
    import json
    from servicedesk.config import RUNTIME_DIR
    steps = set()
    for kb in ("base", "with-kb013"):
        for f in (RUNTIME_DIR / "policy" / kb).glob("v*.json"):
            if f.name.count(".") == 1:
                steps |= {g for r in json.loads(f.read_text())["rules"] for g in r["user_guidance"]}
    gold = json.loads((Path(__file__).parent.parent / "data" / "gold" / "policy_gold.json").read_text())
    steps |= {g for r in gold["rules"] for g in r["user_guidance"]}
    assert steps
    assert {s: check_outgoing(f"- {s} [KB-000]") for s in steps if check_outgoing(f"- {s} [KB-000]")} == {}


# --------------------------------------------------------------------------
# FIX-09 (second review): nothing unscreened is persisted when screening fails
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["timeout", "refusal"])
def test_failed_screening_persists_no_message_body(tmp_path, kind):
    canary = "canary-plum-42"  # harmless text that only the unreviewed body contains
    llm = FakeLLM(_handlers(screening=lambda _u: LLMError(kind, "no screening", call="screening:request")))
    stores = Stores(tmp_path)
    out = Pipeline(_policy(ACCESS), [KB_DIR], stores, llm=llm).process(_req(f"payroll access please {canary}"))
    assert out.status == "failed_safe" and out.issues[0].lane == "handed_to_staff" and out.issues[0].flagged_for_staff
    assert canary not in _persisted(tmp_path)
    state = stores.load_state("T-001")
    assert state["redacted_message"] is None and "screening failed" in state["message_withheld"]
    assert len(stores.tickets()) == 1 and len(stores.outbox()) == 1  # staff hand-off and a template email


def test_failed_reply_screening_persists_no_reply_body(tmp_path):
    canary = "canary-fig-77"
    calls = {"n": 0}

    def screen(user):
        calls["n"] += 1
        return _screen_clean(user) if calls["n"] == 1 else LLMError("timeout", "slow", call="screening:reply")

    stores = Stores(tmp_path)
    p = Pipeline(_policy(ACCESS), [KB_DIR], stores, llm=FakeLLM(_handlers(screening=screen)))
    p.process(_req("payroll access please"))
    p.resume("T-001", f"Role is Payroll Viewer {canary}")
    assert canary not in _persisted(tmp_path)
    assert stores.load_state("T-001")["redacted_message"] == "payroll access please"  # screened request is kept


def test_successful_screening_stores_the_screened_message(tmp_path):
    secret = "velvet-harbor-orbit-lantern"

    def screen(user):
        out = _screen_clean(user)
        out["extra_secrets"] = [{"kind": "recovery_code", "value": secret}] if secret in user else []
        return out

    stores = Stores(tmp_path)
    Pipeline(_policy(ACCESS), [KB_DIR], stores, llm=FakeLLM(_handlers(screening=screen))).process(
        _req(f"payroll access, phrase {secret}"))
    assert stores.load_state("T-001")["redacted_message"] == "payroll access, phrase [REDACTED-recovery_code]"
