"""Lane decision (pure code, no I/O): easy to read and to unit test.

Order (docs/SPEC.md, step 6):
  0. no matching rule -> redirected (not IT) or handed_to_staff (IT)
  1. urgent           -> the rule or a triggered cross-cutting rule has route_without_waiting
  2. awaiting_info    -> required fields are missing
  3. incident_linked / incident_created -> multi-user issue, matched or not
  4. awaiting_approval -> the rule (or a triggered cross-cutting rule) needs approval
  5. ticket_routed
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from ..models import PolicyRule
from .incidents import Incident

FLAG_NAMES = ("security", "safety", "multi_user")

# Fields the report's received_at stands in for when no start time is given.
_REPORT_TIME_FIELDS = {"time", "start time", "time observed", "time window", "when it started", "time of report"}


def is_report_time_field(name: str) -> bool:
    return re.sub(r"\s+", " ", name.strip().lower()) in _REPORT_TIME_FIELDS


def triggered_cross_rules(rules: Iterable[PolicyRule], flags: dict[str, bool]) -> list[PolicyRule]:
    """Cross-cutting rules whose triggers match any of the issue's flags."""
    return [r for r in rules if r.scope == "cross_cutting" and any(flags.get(t) for t in r.triggers)]


def required_fields_for(rule: Optional[PolicyRule], cross: list[PolicyRule]) -> list[str]:
    """Chosen rule's fields + every triggered cross-cutting rule's fields, deduplicated in order."""
    seen: dict[str, str] = {}
    for r in ([rule] if rule else []) + list(cross):
        for f in r.required_fields:
            key = re.sub(r"\s+", " ", f.strip().lower())
            if r.needs_approval and is_approval_gate_field(key):
                continue  # obtained through the approval queue, never asked of the employee
            if key and key not in seen:
                seen[key] = f.strip()
    return list(seen.values())


def is_approval_gate_field(key: str) -> bool:
    """'approval', 'approval when required', 'owner approval': the approval itself.
    'manager approval' stays a field: the employee names their manager (KB-001)."""
    return bool(re.fullmatch(r"(?:(?:data|application|channel|group|resource)[ -]owner )?approvals?(?: when required| if required)?", key))


@dataclass(frozen=True)
class LaneDecision:
    lane: str
    ticket_status: Optional[str]  # None -> no ticket
    reason: str
    route_without_waiting: bool = False
    needs_approval: bool = False
    approver: Optional[str] = None
    incident: Optional[Incident] = None  # matched incident (incident_linked)
    create_incident: bool = False  # incident_created


def decide_lane(
    *,
    rule: Optional[PolicyRule],
    cross_rules: list[PolicyRule],
    is_it_request: bool,
    multi_user: bool,
    missing_fields: list[str],
    incident: Optional[Incident],
) -> LaneDecision:
    if rule is None:
        if not is_it_request:
            return LaneDecision("redirected", None, "not an IT request and no KB rule matches")
        return LaneDecision("handed_to_staff", "routed", "IT request but no KB rule matches: a person takes it")

    applied = [rule, *cross_rules]
    rww = [r.rule_id for r in applied if r.route_without_waiting]
    approvals = [r for r in applied if r.needs_approval]
    needs_approval = bool(approvals)
    approver = next((r.approver for r in approvals if r.approver), None)
    common = dict(route_without_waiting=bool(rww), needs_approval=needs_approval, approver=approver)

    if rww:
        return LaneDecision("urgent", "routed", f"route_without_waiting from {', '.join(rww)}", **common)
    if missing_fields:
        return LaneDecision("awaiting_info", "awaiting_info", f"missing: {', '.join(missing_fields)}", **common)
    if multi_user:
        if incident is not None:
            return LaneDecision(
                "incident_linked", "linked_to_incident", f"matches active incident {incident.incident_id}",
                incident=incident, **common,
            )
        return LaneDecision("incident_created", "routed", "multi-user issue, no matching active incident",
                            create_incident=True, **common)
    if needs_approval:
        return LaneDecision("awaiting_approval", "awaiting_approval", f"needs approval from {approver or 'approver'}",
                            **common)
    return LaneDecision("ticket_routed", "routed", "complete ticket, no approval needed", **common)
