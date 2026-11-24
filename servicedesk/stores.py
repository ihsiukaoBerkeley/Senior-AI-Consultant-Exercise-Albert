"""Local JSON stores for one run: tickets, approvals, outbox, request state, audit log.

Everything lands in runtime/<run>/ so it can be opened during the demo.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import ApprovalItem, OutboxEmail, Ticket


# The linked ticket's status once staff record a decision. Nothing is executed: a person
# carries out an approved action in the real system.
DECIDED_TICKET_STATUS = {"approved": "approved_awaiting_fulfilment", "rejected": "rejected"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Stores:
    def __init__(self, run_dir: Path, reset: bool = False):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._files = {
            "tickets": self.dir / "tickets.json",
            "approvals": self.dir / "approvals.json",
            "outbox": self.dir / "outbox.json",
            "state": self.dir / "request_state.json",
        }
        self.audit_file = self.dir / "audit.jsonl"
        if reset:
            for f in [*self._files.values(), self.audit_file]:
                f.unlink(missing_ok=True)
        self._counters = {"T": 0, "A": 0, "E": 0}
        for kind, prefix, key in (("tickets", "T", "ticket_id"), ("approvals", "A", "approval_id"), ("outbox", "E", "email_id")):
            items = self._read(kind)
            if items:
                self._counters[prefix] = max(int(i[key].split("-")[-1]) for i in items.values())

    # ---- low level ----
    def _read(self, kind: str) -> dict[str, Any]:
        path = self._files[kind]
        return json.loads(path.read_text()) if path.exists() else {}

    def _write(self, kind: str, data: dict[str, Any]) -> None:
        tmp = self._files[kind].with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.replace(self._files[kind])

    def _next_id(self, prefix: str, label: str) -> str:
        self._counters[prefix] += 1
        return f"{label}-{self._counters[prefix]:04d}"

    # ---- tickets ----
    def add_ticket(self, **fields: Any) -> Ticket:
        with self._lock:
            data = self._read("tickets")
            label = "INC" if fields.get("kind") == "incident" else "TKT"
            ticket = Ticket(ticket_id=self._next_id("T", label), created_at=now_iso(), **fields)
            data[ticket.ticket_id] = ticket.model_dump()
            self._write("tickets", data)
            return ticket

    def update_ticket(self, ticket_id: str, **changes: Any) -> Ticket:
        with self._lock:
            data = self._read("tickets")
            data[ticket_id].update(changes)
            self._write("tickets", data)
            return Ticket(**data[ticket_id])

    def tickets(self) -> list[Ticket]:
        return [Ticket(**t) for t in self._read("tickets").values()]

    # ---- approvals ----
    def add_approval(self, **fields: Any) -> ApprovalItem:
        with self._lock:
            data = self._read("approvals")
            item = ApprovalItem(approval_id=self._next_id("A", "APR"), **fields)
            data[item.approval_id] = item.model_dump()
            self._write("approvals", data)
            return item

    def update_approval(self, approval_id: str, **changes: Any) -> ApprovalItem:
        with self._lock:
            data = self._read("approvals")
            data[approval_id].update(changes)
            self._write("approvals", data)
            return ApprovalItem(**data[approval_id])

    def decide_approval(self, approval_id: str, approved: bool, decided_by: str) -> ApprovalItem:
        """Record the decision and move the linked ticket on, in one locked step, so the approval
        record, the ticket and the staff queue always agree."""
        with self._lock:
            data = self._read("approvals")
            decision = "approved" if approved else "rejected"
            data[approval_id].update(status=decision, decided_by=decided_by, decided_at=now_iso())
            self._write("approvals", data)
            item = ApprovalItem(**data[approval_id])
            tickets = self._read("tickets")
            if item.ticket_id in tickets:
                tickets[item.ticket_id]["status"] = DECIDED_TICKET_STATUS[decision]
                self._write("tickets", tickets)
            self.audit(item.request_id, "approval_decided", approval_id=approval_id, ticket_id=item.ticket_id,
                       decision=decision, decided_by=decided_by, ticket_status=DECIDED_TICKET_STATUS[decision],
                       executed=False)
            return item

    def approvals(self) -> list[ApprovalItem]:
        return [ApprovalItem(**a) for a in self._read("approvals").values()]

    # ---- outbox ----
    def add_email(self, **fields: Any) -> OutboxEmail:
        with self._lock:
            data = self._read("outbox")
            email = OutboxEmail(email_id=self._next_id("E", "MAIL"), created_at=now_iso(), **fields)
            data[email.email_id] = email.model_dump()
            self._write("outbox", data)
            return email

    def outbox(self) -> list[OutboxEmail]:
        return [OutboxEmail(**e) for e in self._read("outbox").values()]

    # ---- per-request state (for pause and resume) ----
    def save_state(self, request_id: str, state: dict[str, Any]) -> None:
        with self._lock:
            data = self._read("state")
            data[request_id] = state
            self._write("state", data)

    def load_state(self, request_id: str) -> Optional[dict[str, Any]]:
        return self._read("state").get(request_id)

    def all_states(self) -> dict[str, Any]:
        return self._read("state")

    # ---- audit log ----
    def audit(self, request_id: str, step: str, **detail: Any) -> None:
        event = {"ts": now_iso(), "request_id": request_id, "step": step, **detail}
        with self._lock, open(self.audit_file, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def audit_events(self, request_id: Optional[str] = None) -> list[dict]:
        if not self.audit_file.exists():
            return []
        events = [json.loads(line) for line in self.audit_file.read_text().splitlines() if line.strip()]
        return [e for e in events if request_id is None or e["request_id"] == request_id]
