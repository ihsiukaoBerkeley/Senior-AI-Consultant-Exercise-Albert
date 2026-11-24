"""Deterministic incident matching against data/current_incidents.csv (KB-012).

A multi-user issue matches an active incident when all three hold:
1. service: the issue's service/symptom tokens overlap the incident's service +
   summary tokens, and the overlap includes a word of the incident's service
   name (or at least two specific words). Generic words ("shared", "users",
   floor numbers...) never count on their own.
2. scope: linking is by location scope, not by country or office (working
   assumption; scopes in config/working_assumptions.yaml: HQ, NA, LATAM, EU,
   AFR, ME, APAC). The report's scope must be one of the incident's scopes.
   UNKNOWN on either side is unresolved: it never links, and the incident is
   reported back as `needs_location` so the caller can ask or defer to staff.
3. time: the report (its received_at) is within `incident_time_window_hours`
   of the incident start.

Incidents opened during a run are matched the same way (`incident_from_ticket`),
so a second report of the same outage links to the first one (KB-012).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import INCIDENTS_FILE, assumptions

SCOPES = frozenset((assumptions().get("location_scopes") or {}).keys())  # HQ, NA, LATAM, EU, AFR, ME, APAC
UNKNOWN = "UNKNOWN"
INACTIVE_STATUSES = {"resolved", "closed", "cancelled"}

_WORD = re.compile(r"[a-z0-9]+")
_SYNONYMS = {
    "mail": "email", "e": "", "outlook": "email", "emails": "email",
    "print": "printer", "printing": "printer",
    "freezing": "freeze", "frozen": "freeze", "freezes": "freeze", "froze": "freeze",
    "disconnect": "disconnects", "disconnecting": "disconnects", "disconnected": "disconnects", "drop": "disconnects",
    "dropping": "disconnects", "offline": "disconnects",
}
GENERIC = set(
    "a an the and or of to in on for with is are be by as at it this that from not no can cannot t "
    "our my we us they all everyone nobody anyone people person users user team teams department "
    "shared share common service services system systems issue issues problem problems error errors "
    "failure failures fail failing failed working work works broken down slow intermittent repeatedly "
    "keeps keep going today now since morning afternoon floor first second third fourth fifth level "
    "office site building hq home remote same other several many multiple".split()
)


def tokens(text: str) -> set[str]:
    out = set()
    for t in _WORD.findall((text or "").lower()):
        t = _SYNONYMS.get(t, t)
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss") and t not in ("disconnects",):
            t = t[:-1]
        if t and t not in GENERIC:
            out.add(t)
    return out


def parse_time(value: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class Incident:
    incident_id: str
    service: str
    status: str
    priority: str
    scope: tuple[str, ...]
    summary: str
    started_at: Optional[datetime]

    @property
    def scopes(self) -> set[str]:
        """The incident's location scopes; anything outside the agreed list is ignored."""
        return {s.strip().upper() for s in self.scope if s.strip().upper() in SCOPES}

    @property
    def active(self) -> bool:
        return self.status.lower() not in INACTIVE_STATUSES

    def public(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "service": self.service,
            "status": self.status,
            "priority": self.priority,
            "scope": "; ".join(self.scope),
            "summary": self.summary,
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }


def load_incidents(path: Path = INCIDENTS_FILE) -> list[Incident]:
    if not Path(path).exists():
        return []
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scope = tuple(s.strip() for s in re.split(r"[;,/]", row.get("scope") or "") if s.strip())
            out.append(
                Incident(
                    incident_id=row["incident_id"].strip(),
                    service=(row.get("service") or "").strip(),
                    status=(row.get("status") or "").strip(),
                    priority=(row.get("priority") or "").strip(),
                    scope=scope,
                    summary=(row.get("summary") or "").strip(),
                    started_at=parse_time(row.get("started_at") or ""),
                )
            )
    return out


def normalise_scope(value: Optional[str]) -> str:
    v = (value or "").strip().upper()
    return v if v in SCOPES else UNKNOWN


def incident_from_ticket(ticket) -> Incident:
    """An incident opened earlier in this run (an INC- ticket), in the matcher's terms."""
    f = dict(ticket.fields)
    scope = normalise_scope(f.get("location scope"))
    skip = {"location scope", "site", "started_at"}
    details = " ".join(v for k, v in f.items() if k not in skip and not k.lower().startswith(("location", "start", "time", "region")))
    return Incident(
        incident_id=ticket.ticket_id,
        service=f.get("service") or "",
        status="open",
        priority=ticket.priority or "",
        scope=(scope,) if scope != UNKNOWN else (),
        summary=f"{ticket.summary} {details}".strip(),
        started_at=parse_time(f.get("started_at") or ""),
    )


@dataclass
class MatchResult:
    incident: Optional[Incident]
    checks: list[dict] = field(default_factory=list)  # one entry per incident, for the audit log
    needs_location: Optional[Incident] = None  # would match on service and time, but a scope is UNKNOWN

    @property
    def matched(self) -> bool:
        return self.incident is not None


def match_incident(
    *,
    service: Optional[str],
    symptom: Optional[str],
    location_scope: Optional[str],
    reported_at: str,
    incidents: list[Incident],
    window_hours: float,
) -> MatchResult:
    issue_tokens = tokens(f"{service or ''} {symptom or ''}")
    reported = parse_time(reported_at)
    scope = normalise_scope(location_scope)
    best: Optional[tuple[int, Incident]] = None
    needs_location: Optional[Incident] = None
    checks = []
    for inc in incidents:
        svc = tokens(inc.service)
        overlap = issue_tokens & (svc | tokens(inc.summary))
        service_ok = bool(overlap & svc) or len(overlap) >= 2
        inc_scopes = inc.scopes
        # None = unresolved: an UNKNOWN scope on either side defers automatic linking
        scope_ok: Optional[bool] = None if scope == UNKNOWN or not inc_scopes else scope in inc_scopes
        if reported and inc.started_at:
            hours = abs((reported - inc.started_at).total_seconds()) / 3600
            time_ok = hours <= window_hours
        else:
            hours, time_ok = None, False
        ok = bool(inc.active and service_ok and scope_ok and time_ok)
        if scope_ok is None and inc.active and service_ok and time_ok and needs_location is None:
            needs_location = inc
        checks.append({
            "incident_id": inc.incident_id,
            "active": inc.active,
            "overlap": sorted(overlap),
            "service_ok": service_ok,
            "scope": list(inc.scope),
            "location_scope": scope,
            "scope_ok": scope_ok,
            "hours_apart": round(hours, 2) if hours is not None else None,
            "time_ok": time_ok,
            "match": ok,
        })
        if ok:
            key = len(overlap)
            if best is None or key > best[0] or (
                key == best[0] and (inc.started_at or datetime.min.replace(tzinfo=timezone.utc))
                > (best[1].started_at or datetime.min.replace(tzinfo=timezone.utc))
            ):
                best = (key, inc)
    found = best[1] if best else None
    return MatchResult(incident=found, checks=checks, needs_location=None if found else needs_location)


def status_note(incident: Incident) -> Optional[str]:
    """KB-012: a new report on a monitored incident suggests the fix isn't holding."""
    if incident.status.lower() == "monitoring":
        return (
            f"{incident.incident_id} is in monitoring after a fix; this new report suggests the fix "
            f"isn't holding, so the owning team has been told."
        )
    return None
