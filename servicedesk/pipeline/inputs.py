"""Loading requests and resolving KB sets."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Union

from ..config import INCOMING_FILE, KB_ADDITIONS_DIR, KB_DIR, SYNTHETIC_FILE, assumptions
from ..models import IncomingRequest

KB_SETS = {
    "base": [KB_DIR],
    "with-kb013": [KB_DIR, KB_ADDITIONS_DIR],
}

CHANNELS = ("chat", "web", "email", "phone_transcript")
CUSTOM_REQUESTER = "REQ-900"          # the test employee in requester_directory.csv
CUSTOM_FIELDS = ("request_id", "message", "received_at", "channel", "requester_id", "reply", "synthetic")
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class RequestFileError(ValueError):
    """A request file (or a requested id) the run can't use; the message says what to fix."""


def kb_dirs_for(kb_set: str) -> list[Path]:
    """"base" -> [KB_DIR]; "with-kb013" -> [KB_DIR, KB_ADDITIONS_DIR]."""
    try:
        return list(KB_SETS[kb_set])
    except KeyError:
        raise ValueError(f"unknown KB set {kb_set!r}; expected one of {sorted(KB_SETS)}") from None


def _read_jsonl(path: Path) -> list[IncomingRequest]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(IncomingRequest(**json.loads(line)))
    return out


def load_requests(which: str = "all") -> list[IncomingRequest]:
    """"incoming" (NEW-001..012), "synthetic" (SYN-001..014) or "all" (both, in that order)."""
    if which == "incoming":
        return _read_jsonl(INCOMING_FILE)
    if which == "synthetic":
        return _read_jsonl(SYNTHETIC_FILE)
    if which == "all":
        return _read_jsonl(INCOMING_FILE) + _read_jsonl(SYNTHETIC_FILE)
    raise ValueError(f"unknown request set {which!r}; expected incoming, synthetic or all")


def _valid_time(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def load_request_file(path: Union[str, Path]) -> tuple[list[IncomingRequest], dict[str, str]]:
    """Requests a user adds (data/custom_requests.jsonl), plus their scripted replies.

    One JSON object per line. Only "request_id" and "message" are required:
    received_at defaults to the scenario clock, channel to "web" and requester_id
    to REQ-900 (a test employee). An optional "reply" is sent as the employee's
    answer if the agent asks for details. Ids must be unique and must not reuse a
    built-in id (NEW-..., SYN-...).
    """
    path = Path(path)
    if not path.exists():
        raise RequestFileError(f"{path} does not exist")
    name = path.name
    builtin = {r.request_id for r in load_requests("all")}
    reqs: list[IncomingRequest] = []
    replies: dict[str, str] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{name} line {n}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise RequestFileError(f"{where}: not valid JSON ({e.msg}). Each line must be one JSON object.") from None
        if not isinstance(row, dict):
            raise RequestFileError(f"{where}: expected a JSON object")
        unknown = sorted(set(row) - set(CUSTOM_FIELDS))
        if unknown:
            raise RequestFileError(f"{where}: unknown field(s) {', '.join(unknown)}; "
                                   f"allowed: {', '.join(CUSTOM_FIELDS)}")
        rid = str(row.get("request_id") or "").strip()
        message = str(row.get("message") or "").strip()
        if not rid or not message:
            raise RequestFileError(f"{where}: \"request_id\" and \"message\" are required")
        if not _ID_RE.match(rid):
            raise RequestFileError(f"{where}: request_id {rid!r} may use only letters, digits, '-' and '_'")
        if rid in builtin:
            raise RequestFileError(f"{where}: {rid} is a built-in request id; use your own, for example CUS-001")
        if any(r.request_id == rid for r in reqs):
            raise RequestFileError(f"{where}: duplicate request_id {rid}")
        channel = str(row.get("channel") or "web").strip()
        if channel not in CHANNELS:
            raise RequestFileError(f"{where}: channel must be one of {', '.join(CHANNELS)}")
        received_at = str(row.get("received_at") or assumptions().get("scenario_now")).strip()
        if not _valid_time(received_at):
            raise RequestFileError(f"{where}: received_at {received_at!r} is not an ISO time "
                                   f"such as 2026-09-05T08:30:00Z")
        reqs.append(IncomingRequest(
            request_id=rid, message=message, channel=channel, received_at=received_at,
            requester_id=str(row.get("requester_id") or CUSTOM_REQUESTER).strip(),
            synthetic=bool(row.get("synthetic", True)),
        ))
        reply = str(row.get("reply") or "").strip()
        if reply:
            replies[rid] = reply
    return reqs, replies
