"""Category, team, priority and staff flag (code only).

- team: the rule's owner_team, else the majority team for the category in the
  historical tickets (history used as examples, not truth), else the article's Owner.
- priority: from config/working_assumptions.yaml.
- flagged_for_staff: security or multi_user (working assumption), or lane urgent (KB-013).
"""
from __future__ import annotations

import csv
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..config import HISTORY_FILE, assumptions
from ..models import PolicyRule

STAFF_TEAM = "Service Desk"  # a person at the service desk picks up handed_to_staff tickets


@lru_cache(maxsize=4)
def history_team_counts(path: Path = HISTORY_FILE) -> dict[str, Counter]:
    """category (lower-case) -> Counter of assigned teams in the historical tickets."""
    counts: dict[str, Counter] = {}
    if not Path(path).exists():
        return counts
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat, team = (row.get("category") or "").strip(), (row.get("assigned_team") or "").strip()
            if cat and team:
                counts.setdefault(cat.lower(), Counter())[team] += 1
    return counts


def majority_team(category: Optional[str], counts: Optional[dict[str, Counter]] = None) -> Optional[str]:
    """Most frequent team for the category; ties broken by team name so the result is stable."""
    if not category:
        return None
    counts = history_team_counts() if counts is None else counts
    c = counts.get(category.strip().lower())
    if not c:
        return None
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def known_teams(counts: Optional[dict[str, Counter]] = None) -> set[str]:
    counts = history_team_counts() if counts is None else counts
    return {team for c in counts.values() for team in c}


def canonical_team(name: Optional[str], counts: Optional[dict[str, Counter]] = None) -> Optional[str]:
    """Map a short team name from an article ("Security") to the one team in the
    historical tickets that starts with it ("Security Operations"); otherwise keep it."""
    if not name:
        return None
    teams = known_teams(counts)
    if name in teams:
        return name
    starts = [t for t in teams if t.lower().startswith(name.strip().lower() + " ")]
    return starts[0] if len(starts) == 1 else name


def category_for(
    rule: Optional[PolicyRule], rules: list[PolicyRule], counts: Optional[dict[str, Counter]] = None
) -> Optional[str]:
    """The rule's category; if the article names none for this situation, the majority
    category of the named team in history, else the article's other situations' category."""
    if rule is None:
        return None
    if rule.category:
        return rule.category
    counts = history_team_counts() if counts is None else counts
    team = canonical_team(rule.owner_team, counts)
    if team:
        per_cat = {cat: c[team] for cat, c in counts.items() if c.get(team)}
        if per_cat:
            best = sorted(per_cat.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            return _display_category(best, counts)
    siblings = Counter(
        r.category for r in rules
        if r.article_id == rule.article_id and r.scope == "situation" and r.category
    )
    if siblings:
        return sorted(siblings.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return None


@lru_cache(maxsize=4)
def _category_names(path: Path = HISTORY_FILE) -> dict[str, str]:
    names: dict[str, str] = {}
    if Path(path).exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cat = (row.get("category") or "").strip()
                if cat:
                    names.setdefault(cat.lower(), cat)
    return names


def _display_category(key: str, counts: Optional[dict[str, Counter]] = None) -> str:
    return _category_names().get(key, key)


def team_for(
    rule: Optional[PolicyRule], article_owner: Optional[str], counts: Optional[dict[str, Counter]] = None,
    category: Optional[str] = None,
) -> Optional[str]:
    if rule is None:
        return None
    return (canonical_team(rule.owner_team, counts)
            or majority_team(category or rule.category, counts)
            or (article_owner or None))


def priority_for(
    rule: Optional[PolicyRule],
    *,
    security: bool,
    multi_user: bool,
    incident_priority: Optional[str] = None,
    conf: Optional[dict] = None,
) -> str:
    p = (conf or assumptions())["priority"]
    if incident_priority:  # linked to an existing incident: take its priority
        return incident_priority
    if security or multi_user:
        return p["security_or_multi_user"]
    if rule is not None and rule.priority_hint:
        return p[rule.priority_hint]
    return p["default"]


def flagged_for(flags: dict[str, bool], *, lane: Optional[str] = None, conf: Optional[dict] = None) -> bool:
    review = (conf or assumptions()).get("flag_for_immediate_review", ["security", "multi_user"])
    return any(flags.get(name) for name in review) or lane == "urgent"
