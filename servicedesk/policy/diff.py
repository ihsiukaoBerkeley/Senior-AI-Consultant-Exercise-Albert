"""Diff two policy tables (code only): rules added / removed / changed, field by field.

Rules are matched across versions by rule_id first, then (for rules of the
same article whose slug changed) by the words of their slug and situation.
Quote and situation wording changes are listed separately as `reworded`:
they are not policy changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models import PolicyRule, PolicyTable
from .text import jaccard, norm_entry, tokens

SCALAR_FIELDS = (
    "scope",
    "triggers",
    "category",
    "owner_team",
    "needs_approval",
    "approver",
    "route_without_waiting",
    "priority_hint",
)
LIST_FIELDS = ("required_fields", "must_never", "user_guidance")
WORDING_FIELDS = ("situation", "quotes")
# The fields that decide what the pipeline does with a case: used by stability and no-change tests.
SAFETY_FIELDS = ("scope", "triggers", "needs_approval", "route_without_waiting", "category")

MATCH_THRESHOLD = 0.2


def _rule_words(r: PolicyRule) -> set[str]:
    slug = r.rule_id.split(":", 1)[-1].replace("-", " ")
    return tokens(f"{slug} {r.situation}")


def similarity(a: PolicyRule, b: PolicyRule) -> float:
    if a.article_id != b.article_id or a.scope != b.scope:
        return 0.0
    s = jaccard(_rule_words(a), _rule_words(b))
    if a.category and a.category == b.category:
        s += 0.05
    return s


def match_rules(
    old: list[PolicyRule], new: list[PolicyRule], threshold: float = MATCH_THRESHOLD
) -> list[tuple[Optional[PolicyRule], Optional[PolicyRule]]]:
    """Pair old and new rules: (old, new), (old, None) = removed, (None, new) = added."""
    pairs: list[tuple[Optional[PolicyRule], Optional[PolicyRule]]] = []
    new_by_id = {r.rule_id: r for r in new}
    used_new: set[str] = set()
    rest_old: list[PolicyRule] = []
    for o in old:
        n = new_by_id.get(o.rule_id)
        if n is not None and n.rule_id not in used_new:
            pairs.append((o, n))
            used_new.add(n.rule_id)
        else:
            rest_old.append(o)
    rest_new = [n for n in new if n.rule_id not in used_new]
    cands = sorted(
        (
            (similarity(o, n), i, j)
            for i, o in enumerate(rest_old)
            for j, n in enumerate(rest_new)
            if o.article_id == n.article_id
        ),
        key=lambda c: (-c[0], c[1], c[2]),
    )
    mo: set[int] = set()
    mn: set[int] = set()
    for s, i, j in cands:
        if s < threshold or i in mo or j in mn:
            continue
        pairs.append((rest_old[i], rest_new[j]))
        mo.add(i)
        mn.add(j)
    pairs += [(o, None) for i, o in enumerate(rest_old) if i not in mo]
    pairs += [(None, n) for j, n in enumerate(rest_new) if j not in mn]
    return pairs


def _scalar(v):
    if isinstance(v, list):
        return sorted(v)
    if isinstance(v, str):
        return v.strip().lower()
    return v


@dataclass
class RuleChange:
    article_id: str
    old_rule_id: str
    new_rule_id: str
    changes: dict[str, tuple] = field(default_factory=dict)  # scalar field -> (old, new)
    added: dict[str, list[str]] = field(default_factory=dict)  # list field -> entries added
    removed: dict[str, list[str]] = field(default_factory=dict)  # list field -> entries removed
    reworded: list[str] = field(default_factory=list)  # situation / quotes changed

    @property
    def substantive(self) -> bool:
        return bool(self.changes or self.added or self.removed)

    @property
    def safety_relevant(self) -> bool:
        return any(f in self.changes for f in ("needs_approval", "route_without_waiting", "scope", "triggers")) or bool(
            self.removed.get("must_never") or self.added.get("must_never")
        )

    def to_dict(self) -> dict:
        return {
            "article_id": self.article_id,
            "old_rule_id": self.old_rule_id,
            "new_rule_id": self.new_rule_id,
            "changes": {k: list(v) for k, v in self.changes.items()},
            "added": self.added,
            "removed": self.removed,
            "reworded": self.reworded,
        }


def compare_rules(old: PolicyRule, new: PolicyRule) -> RuleChange:
    ch = RuleChange(article_id=new.article_id, old_rule_id=old.rule_id, new_rule_id=new.rule_id)
    for f in SCALAR_FIELDS:
        a, b = getattr(old, f), getattr(new, f)
        if _scalar(a) != _scalar(b):
            ch.changes[f] = (a, b)
    for f in LIST_FIELDS:
        a, b = getattr(old, f), getattr(new, f)
        na, nb = {norm_entry(x) for x in a}, {norm_entry(x) for x in b}
        added = [x for x in b if norm_entry(x) not in na]
        removed = [x for x in a if norm_entry(x) not in nb]
        if added:
            ch.added[f] = added
        if removed:
            ch.removed[f] = removed
    if " ".join(old.situation.split()) != " ".join(new.situation.split()):
        ch.reworded.append("situation")
    if {norm_entry(q) for q in old.quotes} != {norm_entry(q) for q in new.quotes}:
        ch.reworded.append("quotes")
    if old.rule_id != new.rule_id:
        ch.reworded.append("rule_id")
    return ch


@dataclass
class PolicyDiff:
    old_version: Optional[int]
    new_version: Optional[int]
    added: list[PolicyRule] = field(default_factory=list)
    removed: list[PolicyRule] = field(default_factory=list)
    changed: list[RuleChange] = field(default_factory=list)  # substantive changes
    reworded: list[RuleChange] = field(default_factory=list)  # wording / quotes only
    unchanged: int = 0
    articles_added: list[str] = field(default_factory=list)
    articles_removed: list[str] = field(default_factory=list)
    newly_inactive: list[str] = field(default_factory=list)
    reactivated: list[str] = field(default_factory=list)
    kept_previous: list[str] = field(default_factory=list)  # extraction failed: previous rules kept

    @property
    def is_empty(self) -> bool:
        return not (
            self.added
            or self.removed
            or self.changed
            or self.articles_removed
            or self.newly_inactive
            or self.reactivated
            or self.kept_previous
        )

    def for_article(self, article_id: str) -> "PolicyDiff":
        return PolicyDiff(
            old_version=self.old_version,
            new_version=self.new_version,
            added=[r for r in self.added if r.article_id == article_id],
            removed=[r for r in self.removed if r.article_id == article_id],
            changed=[c for c in self.changed if c.article_id == article_id],
            reworded=[c for c in self.reworded if c.article_id == article_id],
            articles_added=[a for a in self.articles_added if a == article_id],
            articles_removed=[a for a in self.articles_removed if a == article_id],
            newly_inactive=[a for a in self.newly_inactive if a == article_id],
            reactivated=[a for a in self.reactivated if a == article_id],
            kept_previous=[a for a in self.kept_previous if a == article_id],
        )

    def counts(self) -> dict:
        return {
            "added": len(self.added),
            "removed": len(self.removed),
            "changed": len(self.changed),
            "reworded": len(self.reworded),
            "unchanged": self.unchanged,
        }

    def to_dict(self) -> dict:
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "counts": self.counts(),
            "added": [r.rule_id for r in self.added],
            "removed": [r.rule_id for r in self.removed],
            "changed": [c.to_dict() for c in self.changed],
            "reworded": [c.to_dict() for c in self.reworded],
            "articles_added": self.articles_added,
            "articles_removed": self.articles_removed,
            "newly_inactive": self.newly_inactive,
            "reactivated": self.reactivated,
            "kept_previous": self.kept_previous,
        }

    def to_markdown(
        self,
        title: str = "Policy diff",
        findings: Optional[list] = None,
        notes: Optional[list[str]] = None,
        old_label: Optional[str] = None,
        new_label: Optional[str] = None,
    ) -> str:
        return render_markdown(self, title, findings or [], notes or [], old_label, new_label)


def diff_tables(old: Optional[PolicyTable], new: PolicyTable, kept_previous: Optional[list[str]] = None) -> PolicyDiff:
    old_rules = old.rules if old else []
    d = PolicyDiff(old_version=old.version if old else None, new_version=new.version, kept_previous=sorted(kept_previous or []))
    for o, n in match_rules(old_rules, new.rules):
        if o is None:
            d.added.append(n)
        elif n is None:
            d.removed.append(o)
        else:
            ch = compare_rules(o, n)
            if ch.substantive:
                d.changed.append(ch)
            elif ch.reworded:
                d.reworded.append(ch)
            else:
                d.unchanged += 1
    old_hashes = set(old.kb_hashes) if old else set()
    old_inactive = set(old.inactive_articles) if old else set()
    d.articles_added = sorted(set(new.kb_hashes) - old_hashes) if old else []
    d.articles_removed = sorted(old_hashes - set(new.kb_hashes))
    d.newly_inactive = sorted(set(new.inactive_articles) - old_inactive) if old else []
    d.reactivated = sorted(old_inactive - set(new.inactive_articles) - set(d.articles_removed))
    return d


# ---------------------------------------------------------------- markdown


def _fmt(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "[" + ", ".join(map(str, v)) + "]" if v else "[]"
    return str(v)


def rule_lines(r: PolicyRule) -> list[str]:
    head = f"- **{r.rule_id}** ({r.scope}"
    if r.scope == "cross_cutting":
        head += f", triggers: {', '.join(r.triggers) or 'none'}"
    head += f"): {r.situation}"
    lines = [head]
    flags = []
    if r.route_without_waiting:
        flags.append("**route without waiting**")
    if r.needs_approval:
        flags.append(f"**needs approval** ({r.approver or 'approver not named'})")
    if r.category:
        flags.append(f"category {r.category}")
    if r.owner_team:
        flags.append(f"team {r.owner_team}")
    if r.priority_hint:
        flags.append(f"priority {r.priority_hint}")
    if flags:
        lines.append("  - " + "; ".join(flags))
    if r.required_fields:
        lines.append("  - required fields: " + "; ".join(r.required_fields))
    if r.must_never:
        lines.append("  - must never: " + "; ".join(r.must_never))
    if r.user_guidance:
        lines.append("  - user guidance: " + "; ".join(r.user_guidance))
    return lines


def change_lines(c: RuleChange) -> list[str]:
    name = c.new_rule_id if c.old_rule_id == c.new_rule_id else f"{c.old_rule_id} -> {c.new_rule_id}"
    lines = [f"- **{name}**"]
    for f, (a, b) in c.changes.items():
        lines.append(f"  - {f}: {_fmt(a)} -> {_fmt(b)}")
    for f, items in c.added.items():
        lines.append(f"  - {f} added: " + "; ".join(items))
    for f, items in c.removed.items():
        lines.append(f"  - {f} removed: " + "; ".join(items))
    if c.reworded:
        lines.append(f"  - reworded: {', '.join(c.reworded)}")
    return lines


def render_markdown(
    d: PolicyDiff, title: str, findings: list, notes: list[str], old_label: Optional[str] = None, new_label: Optional[str] = None
) -> str:
    old = old_label or ("live v" + str(d.old_version) if d.old_version is not None else "(no live version)")
    new = new_label or f"candidate v{d.new_version}"
    out = [f"# {title}", "", f"{old} -> {new}", ""]
    out += [f"- {n}" for n in notes]
    if notes:
        out.append("")
    c = d.counts()
    out.append(
        f"**Summary:** {c['added']} rule(s) added, {c['removed']} removed, {c['changed']} changed, "
        f"{c['reworded']} reworded only, {c['unchanged']} unchanged."
    )
    blocking = [f for f in findings if getattr(f, "blocking", False)]
    if findings:
        out.append(f"**Checker findings:** {len(findings)} ({len(blocking)} need a reviewer decision).")
    out.append("")
    if d.is_empty and not d.reworded:
        out += ["No policy changes.", ""]
    for label, ids in (
        ("Articles added", d.articles_added),
        ("Articles removed", d.articles_removed),
        ("Articles newly inactive (deprecated: rules removed)", d.newly_inactive),
        ("Articles reactivated", d.reactivated),
        ("Extraction failed: previous rules kept", d.kept_previous),
    ):
        if ids:
            out.append(f"**{label}:** {', '.join(ids)}")
    out.append("")
    if d.added:
        out += ["## Rules added", ""]
        for r in d.added:
            out += rule_lines(r)
        out.append("")
    if d.removed:
        out += ["## Rules removed", ""]
        for r in d.removed:
            out += rule_lines(r)
        out.append("")
    if d.changed:
        out += ["## Rules changed", ""]
        for ch in d.changed:
            out += change_lines(ch)
        out.append("")
    if d.reworded:
        out += ["## Reworded only (no policy change)", ""]
        for ch in d.reworded:
            out.append(f"- {ch.new_rule_id}: {', '.join(ch.reworded)}")
        out.append("")
    if findings:
        out += ["## Checker findings", ""]
        for f in findings:
            where = f.rule_id or f.article_id or "table"
            tag = "needs decision" if f.blocking else "info"
            out.append(f"- [{f.kind}, {tag}, {f.action}] {where}: {f.explanation}")
        out.append("")
    return "\n".join(out)
