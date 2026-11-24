"""Shared data shapes: requests, the compiled policy table, outcomes, store records."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class IncomingRequest(BaseModel):
    request_id: str
    received_at: str
    channel: str
    requester_id: str
    message: str
    synthetic: bool = False


# --------------------------------------------------------------------------
# Policy table: compiled from the KB, read by the request pipeline.
# --------------------------------------------------------------------------

Trigger = Literal["security", "safety", "multi_user"]


class PolicyRule(BaseModel):
    """One situation an article covers, or one cross-cutting rule.

    situation rules: matched to an issue by the KB-match step.
    cross_cutting rules: apply to any issue whose flags match `triggers`
    (KB-012 for multi-user issues; KB-013 for security and safety cases).
    """

    rule_id: str = Field(description='"<article_id>:<short-slug>", e.g. "KB-004:single-folder-denied"')
    article_id: str
    situation: str = Field(description="When this rule applies, in plain words")
    scope: Literal["situation", "cross_cutting"]
    triggers: list[Trigger] = Field(description="Cross-cutting rules only: issue flags that make it apply; [] otherwise")
    category: Optional[str] = Field(description="Ticket category the article names, e.g. Access; null if none")
    owner_team: Optional[str] = Field(description="Team that owns the ticket; null if the article doesn't say")
    required_fields: list[str] = Field(description="Details the article says to capture before the ticket is complete")
    needs_approval: bool = Field(description="True if the article requires someone's approval before the action")
    approver: Optional[str] = Field(description="Who approves, e.g. application owner, data owner; null if none")
    route_without_waiting: bool = Field(
        description="True only if the article explicitly says the case must be routed without waiting for missing details"
    )
    priority_hint: Optional[Literal["high", "medium", "low"]] = Field(
        description="Priority word the article uses for this situation; null if none"
    )
    must_never: list[str] = Field(description="Things the agent must never do or say in this situation")
    user_guidance: list[str] = Field(description="Steps the article says to give the user")
    quotes: list[str] = Field(description="Sentences copied word for word from the article that support this rule")


class PolicyTable(BaseModel):
    version: int
    compiled_at: str
    source: Literal["extracted", "gold"]
    kb_hashes: dict[str, str] = Field(default_factory=dict)  # article_id -> sha256 of the file
    inactive_articles: list[str] = Field(default_factory=list)  # deprecated: no rules
    rules: list[PolicyRule]

    def for_articles(self, article_ids: list[str]) -> list[PolicyRule]:
        return [r for r in self.rules if r.article_id in article_ids and r.scope == "situation"]

    def cross_cutting(self, trigger: Optional[str] = None) -> list[PolicyRule]:
        return [
            r for r in self.rules
            if r.scope == "cross_cutting" and (trigger is None or trigger in r.triggers)
        ]

    def get(self, rule_id: str) -> Optional[PolicyRule]:
        return next((r for r in self.rules if r.rule_id == rule_id), None)


# --------------------------------------------------------------------------
# Outcomes: what the pipeline decided, per request and per issue.
# --------------------------------------------------------------------------

Lane = Literal[
    "urgent",              # routed now without waiting (a route_without_waiting rule applied)
    "awaiting_info",       # paused: email sent asking for missing fields
    "incident_linked",     # multi-user issue linked to an active incident
    "incident_created",    # multi-user issue, no active incident: new P1 incident
    "awaiting_approval",   # complete, but needs a person's approval
    "ticket_routed",       # complete ticket routed to the owning team
    "redirected",          # not an IT request: pointed to the right owner
    "handed_to_staff",     # IT but no matching article, or the pipeline failed safe
]


class IssueOutcome(BaseModel):
    issue_index: int
    summary: str
    rule_id: Optional[str] = None
    article_id: Optional[str] = None
    category: Optional[str] = None
    owner_team: Optional[str] = None
    priority: Optional[str] = None
    lane: Lane
    flagged_for_staff: bool = False
    security: bool = False
    safety: bool = False
    multi_user: bool = False  # an outage: something shared is failing for several people
    group_request: bool = False  # a request for something new for several people (not an outage)
    awaiting_reply: bool = False  # the employee was asked for details (paused, or urgent with details asked)
    missing_fields: list[str] = Field(default_factory=list)
    provided_fields: dict[str, str] = Field(default_factory=dict)
    incident_id: Optional[str] = None
    ticket_id: Optional[str] = None
    approval_id: Optional[str] = None
    applied_rules: list[str] = Field(default_factory=list)  # every policy rule id that shaped the decision
    also_applies: list[str] = Field(default_factory=list)  # other articles' rules for the same situation (guidance only)
    notes: list[str] = Field(default_factory=list)


class RequestOutcome(BaseModel):
    request_id: str
    status: Literal["done", "awaiting_info", "failed_safe"]
    injection_detected: bool = False
    injection_excerpt: Optional[str] = None
    secrets_redacted: list[str] = Field(default_factory=list)
    out_of_scope: bool = False
    reply_requested: bool = False  # any issue is awaiting a reply: the UI shows the reply form
    issues: list[IssueOutcome] = Field(default_factory=list)
    email_ids: list[str] = Field(default_factory=list)
    guard_violations: list[str] = Field(default_factory=list)  # caught and replaced before sending
    llm_errors: list[dict] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Store records (local JSON files under runtime/<run>/).
# --------------------------------------------------------------------------


class Ticket(BaseModel):
    ticket_id: str
    request_id: str
    issue_index: int
    kind: Literal["ticket", "incident"] = "ticket"
    # approved_awaiting_fulfilment / rejected: recorded by staff in the console; the prototype
    # never carries out the action itself
    status: Literal["routed", "awaiting_info", "awaiting_approval", "linked_to_incident",
                    "approved_awaiting_fulfilment", "rejected"]
    category: Optional[str] = None
    owner_team: Optional[str] = None
    priority: Optional[str] = None
    flagged_for_staff: bool = False
    summary: str
    fields: dict[str, str] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    article_id: Optional[str] = None
    rule_id: Optional[str] = None
    linked_incident: Optional[str] = None
    created_at: str


class ApprovalItem(BaseModel):
    approval_id: str
    ticket_id: str
    request_id: str
    action: str
    approver: Optional[str] = None
    rule_id: Optional[str] = None
    details: dict[str, str] = Field(default_factory=dict)  # the ticket's final stated details, shown beside the decision
    status: Literal["pending", "approved", "rejected"] = "pending"
    decided_by: Optional[str] = None
    decided_at: Optional[str] = None


class OutboxEmail(BaseModel):
    email_id: str
    request_id: str
    to: str
    subject: str
    body: str
    kind: Literal["guidance", "ask_missing", "incident_status", "redirect", "confirmation"]
    cites: list[str] = Field(default_factory=list)  # KB article ids the message relies on
    created_at: str
