"""Deterministic tests for the shared core (no API calls)."""
from servicedesk.config import KB_DIR
from servicedesk.guards import check_outgoing, redact_secrets
from servicedesk.kb import KBIndex, load_kb


def test_kb_loads_all_twelve_and_marks_kb011_inactive():
    arts = load_kb([KB_DIR])
    # The 12 provided articles; articles a user adds may come on top.
    assert {f"KB-{n:03d}" for n in range(1, 13)} <= {a.article_id for a in arts}
    kb011 = next(a for a in arts if a.article_id == "KB-011")
    assert not kb011.active and kb011.superseded_by == "KB-003"


def test_search_never_returns_deprecated_article():
    index = KBIndex(load_kb([KB_DIR]))
    hits = index.search("reinstall LegacyConnect VPN client gateway certificate", k=12)
    assert all(a.article_id != "KB-011" for a, _ in hits)
    assert hits[0][0].article_id == "KB-003"


def test_search_finds_expected_articles():
    index = KBIndex(load_kb([KB_DIR]))
    assert index.search("shared drive one folder access denied")[0][0].article_id == "KB-004"
    assert index.search("clicked phishing link entered password")[0][0].article_id in {"KB-008", "KB-002"}
    assert index.search("restore deleted project folder")[0][0].article_id == "KB-009"


def test_redacts_codes_and_passwords():
    text, found = redact_secrets("My MFA code is 482913 and my password is Hunter2!")
    assert "482913" not in text and "Hunter2!" not in text
    assert set(found) == {"one_time_code", "password"}


def test_outgoing_checks_allow_negated_warnings():
    assert check_outgoing("Never share your MFA code with anyone, including us.") == []
    assert check_outgoing("Do not disable the firewall.") == []


def test_outgoing_checks_catch_violations():
    assert "asks_for_code" in check_outgoing("Please send us the code from your app.")
    assert "advises_disabling_security" in check_outgoing("You can turn off the firewall while working.")
    assert "promises_safety" in check_outgoing("Your laptop is now safe.")
    assert "claims_access_granted" in check_outgoing("Access has been granted to the payroll app.")
