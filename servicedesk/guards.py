"""Hard limits that stay in code. No KB edit can switch these off.

1. Secrets in recognised formats (a stated password, a 4-8 digit code next to
   "code"/"MFA"/"PIN", license keys, sk- API keys) are redacted before any model
   call or log entry. Other formats are found by the screening model, so that one
   call sees them; they are removed before anything is stored or sent on.
2. Outgoing messages are checked: never ask for a password or code, never
   advise disabling security controls, never promise safety, approval,
   access or a successful restore.
3. The pipeline has no code path that grants access, restores data or resets
   credentials: those only ever become approval-queue items.
"""
from __future__ import annotations

import re

# (name, pattern, group to redact or 0 for the whole match)
_SECRET_PATTERNS = [
    ("password", re.compile(r"(?i)\b(?:password|passcode|passwd|pwd)\b\s*(?:is|was|:|=)\s*(\S+)"), 1),
    ("one_time_code", re.compile(r"(?i)\b(?:code|otp|mfa|2fa|token|pin)\b[^\d\n]{0,25}\b(\d{4,8})\b"), 1),
    ("one_time_code", re.compile(r"\b(\d{3}[ -]\d{3})\b(?=[^\n]{0,25}\b(?:code|otp|mfa)\b)", re.I), 1),
    ("license_key", re.compile(r"\b([A-Z0-9]{4,6}(?:-[A-Z0-9]{4,6}){3,})\b"), 1),
    ("api_key", re.compile(r"\b(sk-[A-Za-z0-9_\-]{16,})\b"), 1),
]


def redact_secrets(text: str) -> tuple[str, list[str]]:
    """Replace secret values with [REDACTED-<kind>]; return the kinds found."""
    found: list[str] = []
    for name, pattern, group in _SECRET_PATTERNS:
        def _sub(m: re.Match, name=name, group=group) -> str:
            found.append(name)
            whole, secret = m.group(0), m.group(group)
            return whole.replace(secret, f"[REDACTED-{name}]")
        text = pattern.sub(_sub, text)
    return text, sorted(set(found))


_NEGATION = re.compile(r"(?i)\b(never|not|don't|do not|doesn't|no one|nobody|cannot|can't|unable|won't|avoid|without)\b")

# Checked sentence by sentence; a sentence with a negation ("never share your code") is allowed.
_OUTGOING_RULES = [
    ("asks_for_password", re.compile(r"(?i)\b(send|share|provide|tell|give|reply with|type|enter|confirm)\b[^.?!\n]{0,40}\bpassword\b")),
    ("asks_for_code", re.compile(r"(?i)\b(send|share|provide|tell|give|reply with|read out|forward)\b[^.?!\n]{0,40}\b(code|otp|recovery key|license key)\b")),
    ("advises_disabling_security", re.compile(r"(?i)\b(disable|turn off|switch off|uninstall|pause)\b[^.?!\n]{0,30}\b(firewall|antivirus|endpoint protection|security software|compliance|certificate)\b")),
    ("promises_safety", re.compile(r"(?i)\b(account|device|laptop|computer|phone|machine)\b[^.?!\n]{0,20}\b(is|are) (now )?(safe|secure|clean|not compromised)\b")),
    ("claims_approval", re.compile(r"(?i)\b(is|has been|have been|was) (now )?(approved|licensed|authorised|authorized)\b")),
    ("claims_access_granted", re.compile(r"(?i)\b(access|permission|role)s? (has|have) been (granted|added|given)\b|\b(i|we) (have )?(granted|added you)\b")),
    ("claims_restore_done", re.compile(r"(?i)\b(restore|restoration|recovery)\b[^.?!\n]{0,20}\b(is complete|completed|succeeded|was successful|is done)\b")),
    ("claims_resolved", re.compile(r"(?i)\b(this|your) (request|ticket|issue|incident) (is|has been) (resolved|closed)\b")),
]


def check_outgoing(text: str) -> list[str]:
    """Return the hard-limit rules an outgoing message breaks (empty = OK)."""
    violations: list[str] = []
    for sentence in re.split(r"(?<=[.?!])\s+|\n+", text):
        if not sentence.strip() or _NEGATION.search(sentence):
            continue
        for name, pattern in _OUTGOING_RULES:
            if pattern.search(sentence):
                violations.append(name)
    _, leaked = redact_secrets(text)
    violations.extend(f"leaks_{k}" for k in leaked)
    return sorted(set(violations))
