"""Fallbacks for gaps in the extracted policy (no API calls)."""
from servicedesk.models import PolicyRule
from servicedesk.pipeline.lanes import is_approval_gate_field, required_fields_for
from servicedesk.pipeline.routing import canonical_team, category_for, team_for


def rule(**kw) -> PolicyRule:
    base = dict(
        rule_id="KB-X:r", article_id="KB-X", situation="s", scope="situation", triggers=[],
        category=None, owner_team=None, required_fields=[], needs_approval=False, approver=None,
        route_without_waiting=False, priority_hint=None, must_never=[], user_guidance=[], quotes=["q"],
    )
    base.update(kw)
    return PolicyRule(**base)


def test_short_team_name_maps_to_the_history_team():
    assert canonical_team("Security") == "Security Operations"
    assert canonical_team("Data Protection") == "Data Protection"
    assert canonical_team("Nonexistent Team") == "Nonexistent Team"


def test_category_from_named_team_when_article_names_none():
    r = rule(article_id="KB-002", owner_team="Security")
    assert category_for(r, [r]) == "Security"
    assert team_for(r, "Identity and Access Management") == "Security Operations"


def test_category_from_sibling_situations_of_the_same_article():
    battery = rule(rule_id="KB-005:battery", article_id="KB-005")
    display = rule(rule_id="KB-005:display", article_id="KB-005", category="Hardware")
    printer = rule(rule_id="KB-005:printer", article_id="KB-005", category="Hardware")
    assert category_for(battery, [battery, display, printer]) == "Hardware"


def test_approval_itself_is_never_asked_of_the_employee():
    restore = rule(needs_approval=True, required_fields=["restore scope", "data owner", "approval when required"])
    assert required_fields_for(restore, []) == ["restore scope", "data owner"]
    access = rule(needs_approval=True, required_fields=["requested role", "manager approval"])
    assert required_fields_for(access, []) == ["requested role", "manager approval"]
    assert is_approval_gate_field("approval") and not is_approval_gate_field("manager approval")
