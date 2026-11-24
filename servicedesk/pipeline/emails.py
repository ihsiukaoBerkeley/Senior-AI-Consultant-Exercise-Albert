"""Employee emails: recipient lookup, email kind, guard check and the
deterministic template that replaces a draft that breaks a hard limit."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import DIRECTORY_FILE
from ..guards import check_outgoing, redact_secrets


# --------------------------------------------------------------------------
# Requester directory
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Requester:
    requester_id: str
    name: Optional[str]
    email: str

    @property
    def first_name(self) -> Optional[str]:
        return self.name.split()[0] if self.name else None


def load_directory(path: Path = DIRECTORY_FILE) -> dict[str, Requester]:
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = (row.get("requester_id") or "").strip()
            email = (row.get("email") or "").strip()
            if rid and email:
                out[rid] = Requester(rid, (row.get("name") or "").strip() or None, email)
    return out


def requester_for(requester_id: str, directory: dict[str, Requester]) -> Requester:
    """Directory entry, or a fallback "<requester_id>@example.com" when the row or file is missing."""
    return directory.get(requester_id) or Requester(requester_id, None, f"{requester_id}@example.com")


# --------------------------------------------------------------------------
# Email kind and citations
# --------------------------------------------------------------------------


def approver_phrase(approver: str) -> str:
    """"data owner" -> "the data owner"; names and phrases that already read well are kept."""
    who = (approver or "").strip()
    if not who:
        return "the approver"
    if who[0].islower() and not re.match(r"(the|your|a|an)\s", who, re.I):
        return f"the {who}"
    return who


def email_kind(lanes: list[str]) -> str:
    if "urgent" in lanes:
        return "guidance"
    if "awaiting_info" in lanes:
        return "ask_missing"
    if any(l in ("incident_linked", "incident_created") for l in lanes):
        return "incident_status"
    if lanes and all(l == "redirected" for l in lanes):
        return "redirect"
    return "confirmation"


_SUBJECTS = {
    "guidance": "Important steps for your IT report",
    "ask_missing": "A few details needed for your IT request",
    "incident_status": "Update on the service problem you reported",
    "redirect": "About your question to the IT Service Desk",
    "confirmation": "Your IT request is with the right team",
}


def subject_for(ctx: dict) -> str:
    """Set by code: model-written subjects tend to put a ticket id next to "MFA", which the
    one-time-code check reads as a leaked code."""
    issues = ctx.get("issues", [])
    refs = [i.get("ticket_id") or i.get("incident_id") for i in issues if i.get("ticket_id") or i.get("incident_id")]
    base = _SUBJECTS[email_kind([i.get("lane") for i in issues])]
    return base + (f" ({', '.join(dict.fromkeys(refs))})" if refs else "")


# --------------------------------------------------------------------------
# Deterministic template (used when the draft fails the guard or the drafter fails)
# --------------------------------------------------------------------------


def _safe(line: str) -> Optional[str]:
    """Keep a dynamic line only if it passes the outgoing checks on its own."""
    line = redact_secrets(line)[0].strip()
    return line if line and not check_outgoing(line) else None


def _bullets(items: list[str]) -> list[str]:
    return [f"- {s}" for s in (_safe(i) for i in items) if s]


def render_template(ctx: dict) -> tuple[str, str]:
    name = _safe(ctx.get("employee_first_name") or "") or "there"
    issues = ctx.get("issues", [])
    subject = subject_for(ctx)
    lines = [f"Hello {name},", "", "Thank you for contacting the IT Service Desk."]
    for n, it in enumerate(issues, 1):
        lines.append("")
        about = _safe(it.get("summary") or "")
        lines.append(f"{n}. {about}" if about else f"{n}. Your request")
        lane, tid, team = it.get("lane"), it.get("ticket_id"), it.get("team") or "the service desk"
        guidance = [f"{g['step']} [{g['article']}]" for g in it.get("guidance", [])]
        missing = it.get("missing_details", [])
        if lane == "urgent":
            lines.append(f"We have opened ticket {tid} for {team} and flagged it for immediate review by our staff.")
            if guidance:
                lines += ["Please follow these steps now:", *_bullets(guidance)]
            if missing:
                lines += ["When you can, please reply with these details:", *_bullets(missing)]
        elif lane == "awaiting_info":
            lines.append(f"To route this to the right team we need a few more details (reference {tid}).")
            lines += ["Please reply with:", *_bullets(missing)]
            if guidance:
                lines += ["In the meantime:", *_bullets(guidance)]
        elif lane == "incident_linked":
            inc = it.get("incident") or {}
            lines.append(
                f"This matches an active incident, {inc.get('incident_id')} ({_safe(inc.get('summary') or '') or 'service incident'}). "
                f"Its current status is: {inc.get('status')}."
            )
            lines.append(f"We have linked your report ({tid}) to it, so there is no need to report it again.")
            if it.get("status_note"):
                lines.append("The incident is being monitored after a fix, and your report tells the team the problem may be continuing.")
        elif lane == "incident_created":
            lines.append(
                f"Because several people are affected, we have opened incident {it.get('incident_id')} for {team}."
            )
        elif lane == "awaiting_approval":
            who = approver_phrase(_safe(it.get("approver") or ""))
            approval = it.get("approval_id")
            ref = f" ({approval})" if approval else ""
            lines.append(f"Your request is with {who} for a decision{ref}.")
            details = [f"{k}: {v}" for k, v in (it.get("details") or {}).items()]
            if details:
                lines += ["What we sent for the decision:", *_bullets(details)]
            lines.append(
                "Automated intake for this ticket is complete. "
                f"If any details change, contact the IT Service Desk directly and quote {tid}."
            )
        elif lane == "ticket_routed":
            lines.append(f"We have created ticket {tid} and routed it to {team}.")
        elif lane == "redirected":
            owner = _safe(it.get("redirect_to") or "") or "the team that owns this topic (for example HR or Facilities)"
            lines.append(f"This does not look like an IT request, so we have not opened a ticket. Please contact {owner}.")
        else:  # handed_to_staff
            ref = f" (ticket {tid})" if tid else ""
            lines.append(f"A member of the service desk will review this personally and contact you{ref}.")
        if lane not in ("urgent", "awaiting_info") and guidance:
            lines += ["Steps you can take:", *_bullets(guidance)]
    lines += [
        "",
        "Never share your password or one-time codes with anyone, including the service desk.",
        "",
        "Kind regards,",
        "IT Service Desk",
    ]
    cites = sorted({a for it in issues for a in it.get("articles", [])})
    if cites:
        lines.append(f"Articles: {', '.join(cites)}")
    return subject, "\n".join(lines)


# --------------------------------------------------------------------------
# KB guidance steps: inserted by code, word for word (the model never writes them)
# --------------------------------------------------------------------------

_CLOSING = re.compile(r"^\s*(kind regards|best regards|warm regards|regards|best wishes|many thanks|thanks|"
                      r"thank you|it service desk)\b", re.I)
_ANY_MARKER = re.compile(r"\[\s*STEPS\s*\d+\s*\]", re.I)


def steps_lines(issue: dict) -> list[str]:
    """The issue's KB guidance as email lines: '- Step text [KB-xxx]'."""
    out = []
    for g in issue.get("guidance") or []:
        step = (g.get("step") or "").strip()
        if step:
            out.append(f"- {step[0].upper()}{step[1:]} [{g.get('article')}]")
    return out


def insert_guidance(body: str, ctx: dict) -> str:
    """Put every issue's KB steps into a drafted body, word for word.

    Each step block replaces the issue's steps marker ([STEPS n]); a block whose
    marker the draft left out goes in just before the sign-off. Stray markers are
    removed. So a mandatory instruction (KB-008: contact the Security Hotline;
    KB-005: stop using and charging the device) can never be dropped or reworded
    by the drafter."""
    lines = body.splitlines()
    pending: list[str] = []
    for it in ctx.get("issues", []):
        marker, block = it.get("steps_marker"), steps_lines(it)
        if not marker or not block:
            continue
        n = re.sub(r"\D", "", marker)
        pat = re.compile(r"\[\s*STEPS\s*" + n + r"\s*\]", re.I)
        idx = next((k for k, line in enumerate(lines) if pat.search(line)), None)
        if idx is None:
            pending += block
            continue
        rest = pat.sub("", lines[idx]).strip()
        lines[idx:idx + 1] = ([rest] if rest else []) + block
    lines = [_ANY_MARKER.sub("", line).rstrip() if _ANY_MARKER.search(line) else line for line in lines]
    lines = [line for k, line in enumerate(lines) if line.strip() or not (k and not lines[k - 1].strip())]
    if pending:
        _before_signoff(lines, ["Please follow these steps from our knowledge base:", *pending])
    return "\n".join(lines).strip("\n")


def _before_signoff(lines: list[str], block: list[str]) -> None:
    """Insert a block of lines just before the email's sign-off (or at the end)."""
    at = len(lines)
    for k in range(len(lines) - 1, max(-1, len(lines) - 6), -1):
        if _CLOSING.match(lines[k]):
            at = k
    block = [*block, ""]
    if at and lines[at - 1].strip():
        block.insert(0, "")
    lines[at:at] = block


_REF_LABELS = (("ticket_id", "ticket"), ("incident_id", "incident"), ("approval_id", "approval request"))


def insert_references(body: str, ctx: dict) -> str:
    """Make sure a drafted body names every ticket, incident and approval of the decision.

    The drafter is asked to; when it leaves one out, code adds a reference line before
    the sign-off. The line stands on its own: the one-time-code check reads digits next
    to words like "MFA" or "code" as a leaked code."""
    refs = {}
    for it in ctx.get("issues", []):
        for key, label in _REF_LABELS:
            if it.get(key) and it[key] not in body:
                refs.setdefault(it[key], label)
    if not refs:
        return body
    lines = body.splitlines()
    _before_signoff(lines, ["For reference: " + ", ".join(f"{label} {ref}" for ref, label in refs.items()) + "."])
    return "\n".join(lines).strip("\n")


def missing_steps(body: str, ctx: dict) -> list[str]:
    """KB steps of the decision that the email does not contain word for word."""
    text = body.lower()
    return [line for it in ctx.get("issues", []) for line in steps_lines(it)
            if line[2:].rsplit(" [", 1)[0].lower() not in text]


_FALLBACK_BODY = (
    "Hello,\n\nThank you for contacting the IT Service Desk. We have received your request and a member "
    "of staff will review it.\n\nKind regards,\nIT Service Desk"
)


def finalize_email(subject: str, body: str, ctx: dict) -> tuple[str, str, list[str], bool]:
    """Check a drafted email against the hard limits.

    Returns (subject, body, violations, used_template). If the draft breaks a
    limit, it is replaced by the deterministic template; if even that fails
    (it shouldn't), a minimal fixed message is sent instead.
    """
    violations = check_outgoing(f"{subject}\n{body}")
    if not violations:
        return subject, body, [], False
    t_subject, t_body = render_template(ctx)
    if check_outgoing(f"{t_subject}\n{t_body}"):
        return "Your IT request", _FALLBACK_BODY, violations, True
    return t_subject, t_body, violations, True
