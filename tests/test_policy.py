"""Deterministic tests for the KB policy compiler (no API calls: a stub stands in for the model)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from servicedesk.kb import parse_article
from servicedesk.llm import LLMError, LLMResult, Usage
from servicedesk.models import PolicyRule, PolicyTable
from servicedesk.policy import compiler
from servicedesk.policy.checker import SupportReport, RuleVerdict, TableReview, check_quotes, check_safeguards
from servicedesk.policy.diff import diff_tables, match_rules
from servicedesk.policy.extractor import ExtractionOutput
from servicedesk.policy.text import quote_in_article
from servicedesk.policy.watcher import KBSource, detect_changes, load_articles

# ---------------------------------------------------------------- fixtures

ARTICLE_A = """# KB-101: Shared folders

- Owner: Workplace Infrastructure
- Last reviewed: 2026-08-02
- Status: Active
- Applies to: Shared folders

Capture the share path, exact error, and whether other folders on the same\r
share are accessible. Do not change permissions without data-owner approval.\r
Create an `Access` ticket for folder-level permission requests.
"""

ARTICLE_B = """# KB-102: Printers

- Owner: Workplace Technology
- Last reviewed: 2026-07-18
- Status: Active
- Applies to: Printers

Confirm the printer is online. Do not instruct users to assign an IP.
"""

ARTICLE_OLD = """# KB-103: Legacy client

- Owner: End User Computing
- Last reviewed: 2023-01-12
- Status: Deprecated
- Superseded by: KB-101
- Applies to: Retired client

Do not use these instructions.
"""


def rule(rule_id: str, **kw) -> PolicyRule:
    data = dict(
        rule_id=rule_id,
        article_id=rule_id.split(":")[0],
        situation=kw.pop("situation", rule_id.split(":")[1].replace("-", " ")),
        scope="situation",
        triggers=[],
        category=None,
        owner_team=None,
        required_fields=[],
        needs_approval=False,
        approver=None,
        route_without_waiting=False,
        priority_hint=None,
        must_never=[],
        user_guidance=[],
        quotes=[],
    )
    data.update(kw)
    return PolicyRule(**data)


def table(rules: list[PolicyRule], version: int = 1, hashes: dict | None = None, inactive=()) -> PolicyTable:
    return PolicyTable(
        version=version,
        compiled_at="2026-11-23T00:00:00+00:00",
        source="extracted",
        kb_hashes=hashes or {r.article_id: "h" for r in rules},
        inactive_articles=list(inactive),
        rules=rules,
    )


def write_kb(folder: Path, files: dict[str, str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8", newline="")
    return folder


FOLDER_RULE = rule(
    "KB-101:single-folder-denied",
    situation="One folder on a share is denied",
    category="Access",
    required_fields=["share path", "exact error"],
    needs_approval=True,
    approver="data owner",
    must_never=["change permissions without data-owner approval"],
    quotes=[
        "Capture the share path, exact error, and whether other folders on the same share are accessible.",
        "Do not change permissions without data-owner approval.",
        "Create an Access ticket for folder-level permission requests.",
    ],
)
PRINTER_RULE = rule(
    "KB-102:printer",
    situation="A user cannot print",
    must_never=["instruct users to assign an IP"],
    quotes=["Do not instruct users to assign an IP."],
)
NO_WAIT_RULE = rule(
    "KB-113:no-wait",
    situation="Any security or safety case",
    scope="cross_cutting",
    triggers=["safety", "security"],
    route_without_waiting=True,
    quotes=["Security and safety cases must not wait for missing details."],
)

# ---------------------------------------------------------------- (a) verbatim quotes


def test_quote_check_normalises_whitespace_crlf_and_backticks():
    text = ARTICLE_A
    assert quote_in_article("Capture the share path, exact error, and whether other folders on the same share are accessible.", text)
    assert quote_in_article("Create an `Access` ticket for folder-level permission requests.", text)
    assert quote_in_article("Create an Access ticket for folder-level permission requests.", text)  # backticks dropped
    assert quote_in_article("Do not change permissions   without\ndata-owner approval.", text)


def test_quote_check_rejects_paraphrase_and_invented_text():
    assert not quote_in_article("Never change permissions unless the data owner approves.", ARTICLE_A)
    assert not quote_in_article("Permissions may be changed without approval when urgent.", ARTICLE_A)
    assert not quote_in_article("", ARTICLE_A)


def test_quote_check_accepts_ellipsis_only_if_every_piece_is_verbatim_and_in_order():
    assert quote_in_article("Capture the share path ... Do not change permissions", ARTICLE_A)
    assert not quote_in_article("Do not change permissions ... Capture the share path", ARTICLE_A)
    assert not quote_in_article("Capture the share path ... grant access at once", ARTICLE_A)


def test_check_quotes_drops_rule_with_unsupported_quote(tmp_path):
    art = parse_article(write_kb(tmp_path, {"KB-101-a.md": ARTICLE_A}) / "KB-101-a.md")
    bad = FOLDER_RULE.model_copy(update={"rule_id": "KB-101:bad", "quotes": ["Permissions may be changed when urgent."]})
    noquote = FOLDER_RULE.model_copy(update={"rule_id": "KB-101:noquote", "quotes": [], "needs_approval": False, "must_never": []})
    kept, findings = check_quotes([FOLDER_RULE, bad, noquote], {"KB-101": art})
    assert [r.rule_id for r in kept] == ["KB-101:single-folder-denied"]
    assert {(f.kind, f.rule_id, f.action) for f in findings} == {
        ("unsupported_quote", "KB-101:bad", "dropped"),
        ("unsupported_quote", "KB-101:noquote", "dropped"),
    }
    by_id = {f.rule_id: f for f in findings}
    assert by_id["KB-101:bad"].blocking  # dropping a rule that carried a safeguard needs a decision
    assert not by_id["KB-101:noquote"].blocking
    assert by_id["KB-101:bad"].detail["bad_quotes"] == ["Permissions may be changed when urgent."]


# ---------------------------------------------------------------- (c) removed safeguards + diff


def test_removed_approval_is_flagged():
    live = table([FOLDER_RULE, PRINTER_RULE])
    weakened = FOLDER_RULE.model_copy(update={"needs_approval": False, "approver": None})
    findings = check_safeguards(live, [weakened, PRINTER_RULE])
    kinds = [(f.kind, f.detail.get("safeguard")) for f in findings]
    assert ("removed_safeguard", "needs_approval") in kinds
    assert all(f.blocking for f in findings)


def test_removed_rule_and_must_never_are_flagged():
    live = table([FOLDER_RULE, PRINTER_RULE])
    findings = check_safeguards(live, [PRINTER_RULE])
    safeguards = {f.detail.get("safeguard") for f in findings if f.kind == "removed_safeguard"}
    assert safeguards == {"needs_approval", "must_never"}
    assert all(f.article_id == "KB-101" for f in findings)


def test_dropped_must_never_entry_is_flagged_but_paraphrase_is_not():
    live = table([FOLDER_RULE])
    paraphrased = FOLDER_RULE.model_copy(update={"must_never": ["change folder permissions without the data owner's approval"]})
    assert check_safeguards(live, [paraphrased]) == []
    dropped = FOLDER_RULE.model_copy(update={"must_never": []})
    f = check_safeguards(live, [dropped])
    assert [x.detail["lost"] for x in f] == [["change permissions without data-owner approval"]]


def test_reworded_prohibition_with_synonyms_is_not_flagged():
    pw = PRINTER_RULE.model_copy(update={"must_never": ["ask for or store a password"]})
    reworded = pw.model_copy(update={"must_never": ["request or keep a password"]})
    assert check_safeguards(table([pw]), [reworded]) == []
    unrelated = pw.model_copy(update={"must_never": ["share a password"]})
    assert check_safeguards(table([pw]), [unrelated])


def test_removed_route_without_waiting_is_flagged_and_renamed_rule_is_not():
    live = table([NO_WAIT_RULE, PRINTER_RULE])
    gone = check_safeguards(live, [PRINTER_RULE])
    assert [(f.kind, f.detail["safeguard"]) for f in gone] == [("removed_safeguard", "route_without_waiting")]
    renamed = NO_WAIT_RULE.model_copy(update={"rule_id": "KB-113:security-safety-route-now"})
    assert check_safeguards(live, [renamed, PRINTER_RULE]) == []
    off = NO_WAIT_RULE.model_copy(update={"route_without_waiting": False})
    assert [f.detail["safeguard"] for f in check_safeguards(live, [off, PRINTER_RULE])] == ["route_without_waiting"]


def test_no_previous_version_means_no_safeguard_findings():
    assert check_safeguards(None, [FOLDER_RULE]) == []


def test_match_rules_pairs_renamed_rule_by_situation_words():
    old = [FOLDER_RULE, PRINTER_RULE]
    renamed = FOLDER_RULE.model_copy(update={"rule_id": "KB-101:folder-denied", "situation": "One folder on a share is denied to the user"})
    pairs = match_rules(old, [renamed, PRINTER_RULE])
    assert ("KB-101:single-folder-denied", "KB-101:folder-denied") in {(o.rule_id, n.rule_id) for o, n in pairs if o and n}
    assert all(o is not None and n is not None for o, n in pairs)


def test_diff_lists_added_removed_changed_and_field_changes():
    old = table([FOLDER_RULE, PRINTER_RULE], version=1)
    changed = FOLDER_RULE.model_copy(update={"required_fields": ["share path", "data classification"], "needs_approval": False})
    new = table([changed, NO_WAIT_RULE], version=2)
    d = diff_tables(old, new)
    assert [r.rule_id for r in d.added] == ["KB-113:no-wait"]
    assert [r.rule_id for r in d.removed] == ["KB-102:printer"]
    assert len(d.changed) == 1
    ch = d.changed[0]
    assert ch.added["required_fields"] == ["data classification"]
    assert ch.removed["required_fields"] == ["exact error"]
    assert ch.changes["needs_approval"] == (True, False)
    md = d.to_markdown()
    assert "KB-113:no-wait" in md and "needs_approval: true -> false" in md


def test_diff_quote_only_change_is_reworded_not_changed():
    old = table([FOLDER_RULE])
    reworded = FOLDER_RULE.model_copy(update={"quotes": ["Create an Access ticket for folder-level permission requests."]})
    d = diff_tables(old, table([reworded], version=2))
    assert not d.changed and len(d.reworded) == 1 and d.is_empty


# ---------------------------------------------------------------- watcher


def test_watcher_detects_added_changed_removed(tmp_path):
    kb = write_kb(tmp_path / "kb", {"KB-101-a.md": ARTICLE_A, "KB-102-b.md": ARTICLE_B, "KB-103-old.md": ARTICLE_OLD})
    arts = load_articles([kb])
    assert [a.article_id for a in arts] == ["KB-101", "KB-102", "KB-103"]
    prev = {a.article_id: a.sha256 for a in arts}
    assert not detect_changes(arts, prev).has_changes

    override = write_kb(tmp_path / "override", {"KB-102-b.md": ARTICLE_B.replace("online.", "online and the queue is running."),
                                                 "KB-104-new.md": ARTICLE_B.replace("KB-102", "KB-104")})
    src = KBSource("mut", (kb, override), remove=("KB-101",), allow_override=True)
    ch = detect_changes(load_articles(src), prev)
    assert ch.added == ["KB-104"]
    assert ch.changed == ["KB-102"]
    assert ch.removed == ["KB-101"]
    assert ch.unchanged == ["KB-103"]
    assert ch.touched == ["KB-101", "KB-102", "KB-104"]


def test_watcher_compares_against_a_policy_table(tmp_path):
    kb = write_kb(tmp_path / "kb", {"KB-101-a.md": ARTICLE_A})
    arts = load_articles([kb])
    live = table([FOLDER_RULE], hashes={"KB-101": "stale-hash"})
    assert detect_changes(arts, live).changed == ["KB-101"]


def test_unknown_kb_set_and_missing_folder_raise(tmp_path):
    with pytest.raises(ValueError):
        load_articles("no-such-set")
    with pytest.raises(FileNotFoundError):
        load_articles([tmp_path / "missing"])


# ---------------------------------------------------------------- approve / rollback


def _write_candidate(root: Path, kb_set: str, cand: PolicyTable, findings: list[dict], previous_version):
    d = root / kb_set
    d.mkdir(parents=True, exist_ok=True)
    (d / "candidate.json").write_text(cand.model_dump_json())
    (d / "findings.json").write_text(json.dumps({"findings": findings}))
    (d / "diff.md").write_text("# diff")
    (d / "report.json").write_text(json.dumps({"previous_version": previous_version}))


def test_approve_promotes_candidate_and_rollback_restores_previous(tmp_path):
    root = tmp_path / "policy"
    assert compiler.load_live("demo", root=root) is None
    _write_candidate(root, "demo", table([FOLDER_RULE], version=1), [], None)
    v1 = compiler.approve("demo", approved_by="tester", root=root)
    assert v1.version == 1
    assert compiler.load_live("demo", root=root).rules[0].rule_id == FOLDER_RULE.rule_id
    assert compiler.load_candidate("demo", root=root) is None  # candidate files consumed
    assert (root / "demo" / "v1.diff.md").exists()

    _write_candidate(root, "demo", table([FOLDER_RULE, PRINTER_RULE], version=2), [], 1)
    compiler.approve("demo", root=root)
    assert compiler.list_versions("demo", root=root) == [1, 2]
    assert len(compiler.load_live("demo", root=root).rules) == 2

    back = compiler.rollback("demo", root=root)
    assert back.version == 1
    assert compiler.load_live("demo", root=root).version == 1
    assert compiler.list_versions("demo", root=root) == [1, 2]  # nothing deleted
    events = [json.loads(x)["event"] for x in (root / "demo" / "history.jsonl").read_text().splitlines()]
    assert events == ["approve", "approve", "rollback"]

    # a new approval after a rollback gets a fresh version number, never overwrites v2
    _write_candidate(root, "demo", table([PRINTER_RULE], version=2), [], 1)
    v3 = compiler.approve("demo", root=root)
    assert v3.version == 3 and compiler.list_versions("demo", root=root) == [1, 2, 3]
    assert compiler.rollback("demo", 2, root=root).version == 2


def test_approve_refuses_blocking_findings_unless_acknowledged(tmp_path):
    root = tmp_path / "policy"
    finding = {"kind": "removed_safeguard", "article_id": "KB-101", "rule_id": "KB-101:x", "explanation": "approval gone", "blocking": True}
    info = {"kind": "possible_omission", "article_id": "KB-102", "explanation": "x", "blocking": False}
    _write_candidate(root, "demo", table([PRINTER_RULE]), [finding, info], None)
    with pytest.raises(compiler.PolicyApprovalError, match="removed_safeguard"):
        compiler.approve("demo", root=root)
    assert compiler.load_live("demo", root=root) is None
    compiler.approve("demo", acknowledge_findings=True, root=root)
    ptr = compiler.live_pointer("demo", root=root)
    assert ptr["version"] == 1 and [f["kind"] for f in ptr["acknowledged_findings"]] == ["removed_safeguard"]


def test_approve_refuses_stale_candidate_and_missing_candidate(tmp_path):
    root = tmp_path / "policy"
    with pytest.raises(compiler.PolicyApprovalError, match="no candidate"):
        compiler.approve("demo", root=root)
    _write_candidate(root, "demo", table([PRINTER_RULE]), [], None)
    compiler.approve("demo", root=root)
    _write_candidate(root, "demo", table([PRINTER_RULE], version=2), [], None)  # compiled before v1 went live
    with pytest.raises(compiler.PolicyApprovalError, match="recompile"):
        compiler.approve("demo", root=root)
    with pytest.raises(compiler.PolicyApprovalError):
        compiler.rollback("demo", root=root)  # only one version


# ---------------------------------------------------------------- compile flow with a stub model


class StubLLM:
    """Stands in for servicedesk.llm.LLM: canned structured outputs, no network."""

    model = "stub-model"

    def model_for(self, agent: str) -> str:
        return self.model

    def __init__(self, rules_by_article: dict[str, list[PolicyRule]], fail: set[str] = frozenset()):
        self.rules_by_article = rules_by_article
        self.fail = set(fail)
        self.calls: list[str] = []
        self.usage = Usage()

    def structured(self, *, call, system, user, schema, effort="low", max_tokens=8000):
        self.calls.append(call)
        aid = call.split(":", 1)[1] if ":" in call else None
        if aid in self.fail:
            self.usage.add(error=True)
            raise LLMError("overloaded", "stub failure", call=call)
        if schema is ExtractionOutput:
            parsed = ExtractionOutput(rules=self.rules_by_article[aid], injection_notes=[])
        elif schema is SupportReport:
            ids = [r.rule_id for r in self.rules_by_article[aid]]
            parsed = SupportReport(verdicts=[RuleVerdict(rule_id=i, supported=True, problems=[], missing_safeguards=[], omissions=[]) for i in ids])
        elif schema is TableReview:
            parsed = TableReview(issues=[])
        else:  # pragma: no cover
            raise AssertionError(schema)
        res = LLMResult(parsed=parsed, call=call, request_id="stub", model=self.model, stop_reason="end_turn",
                        input_tokens=1, output_tokens=1, latency_s=0.0)
        self.usage.add(res)
        return res


def test_compile_reuses_unchanged_articles_and_keeps_rules_when_extraction_fails(tmp_path):
    root = tmp_path / "policy"
    kb = write_kb(tmp_path / "kb", {"KB-101-a.md": ARTICLE_A, "KB-102-b.md": ARTICLE_B, "KB-103-old.md": ARTICLE_OLD})
    stub = StubLLM({"KB-101": [FOLDER_RULE], "KB-102": [PRINTER_RULE]})

    r1 = compiler.compile_policy([kb], name="demo", llm=stub, use_cache=False, root=root)
    assert r1.candidate.version == 1
    assert r1.candidate.inactive_articles == ["KB-103"]
    assert r1.rule_counts() == {"KB-101": 1, "KB-102": 1}
    assert not r1.blocking
    assert sorted(c for c in stub.calls if c.startswith("policy_extract")) == ["policy_extract:KB-101", "policy_extract:KB-102"]
    assert (root / "demo" / "diff.md").exists()
    compiler.approve("demo", root=root)

    # Unchanged KB: no model calls at all.
    stub.calls.clear()
    r2 = compiler.compile_policy([kb], name="demo", llm=stub, use_cache=False, root=root)
    assert stub.calls == [] and r2.diff.is_empty and r2.candidate.version == 2

    # KB-101 edited and its extraction fails: previous rules kept, finding recorded, old hash kept.
    (kb / "KB-101-a.md").write_text(ARTICLE_A.replace("exact error", "exact error message"), encoding="utf-8")
    stub.fail = {"KB-101"}
    stub.calls.clear()
    r3 = compiler.compile_policy([kb], name="demo", llm=stub, use_cache=False, root=root)
    assert [c for c in stub.calls if c.startswith("policy_extract")] == ["policy_extract:KB-101"]
    failed = [f for f in r3.findings if f.kind == "extraction_failed"]
    assert len(failed) == 1 and failed[0].action == "kept_previous" and failed[0].detail["error"]["kind"] == "overloaded"
    assert [r.rule_id for r in r3.candidate.rules if r.article_id == "KB-101"] == [FOLDER_RULE.rule_id]
    assert r3.candidate.kb_hashes["KB-101"] == compiler.load_live("demo", root=root).kb_hashes["KB-101"]
    assert r3.diff.kept_previous == ["KB-101"]
    with pytest.raises(compiler.PolicyApprovalError):
        compiler.approve("demo", root=root)

    # KB-101 deprecated: its rules disappear and the lost approval is flagged.
    (kb / "KB-101-a.md").write_text(ARTICLE_A.replace("Status: Active", "Status: Deprecated"), encoding="utf-8")
    stub.fail = set()
    r4 = compiler.compile_policy([kb], name="demo", llm=stub, use_cache=False, root=root)
    assert "KB-101" in r4.candidate.inactive_articles
    assert [r.rule_id for r in r4.diff.removed] == [FOLDER_RULE.rule_id]
    assert {f.detail.get("safeguard") for f in r4.findings if f.kind == "removed_safeguard"} == {"needs_approval", "must_never"}


class RejectOnceStub(StubLLM):
    """First support check rejects every rule of `reject`; the re-extraction returns `fixed` rules."""

    def __init__(self, rules_by_article, reject: str, fixed: list[PolicyRule]):
        super().__init__(rules_by_article)
        self.reject, self.fixed, self.support_calls = reject, fixed, 0

    def structured(self, *, call, system, user, schema, effort="low", max_tokens=8000):
        aid = call.split(":", 1)[1] if ":" in call else None
        if schema is ExtractionOutput and call.startswith("policy_reextract") and aid == self.reject:
            assert "found these problems" in user and "route_without_waiting: not in the article" in user
            self.calls.append(call)
            parsed = ExtractionOutput(rules=self.fixed, injection_notes=[])
            return LLMResult(parsed=parsed, call=call, request_id="stub", model=self.model, stop_reason="end_turn",
                             input_tokens=1, output_tokens=1, latency_s=0.0)
        if schema is SupportReport and aid == self.reject:
            self.calls.append(call)
            self.support_calls += 1
            first = self.support_calls == 1
            ids = [r.rule_id for r in (self.rules_by_article[aid] if first else self.fixed)]
            verdicts = [RuleVerdict(rule_id=i, supported=not first, problems=["route_without_waiting: not in the article"] if first else [],
                                    missing_safeguards=[], omissions=[]) for i in ids]
            return LLMResult(parsed=SupportReport(verdicts=verdicts), call=call, request_id="stub", model=self.model,
                             stop_reason="end_turn", input_tokens=1, output_tokens=1, latency_s=0.0)
        return super().structured(call=call, system=system, user=user, schema=schema, effort=effort, max_tokens=max_tokens)


def test_rejected_rule_is_re_extracted_with_checker_feedback_then_rechecked(tmp_path):
    kb = write_kb(tmp_path / "kb", {"KB-102-b.md": ARTICLE_B})
    wrong = PRINTER_RULE.model_copy(update={"route_without_waiting": True})
    stub = RejectOnceStub({"KB-102": [wrong]}, reject="KB-102", fixed=[PRINTER_RULE])
    res = compiler.compile_policy([kb], name="demo", llm=stub, use_cache=False, root=tmp_path / "policy")
    assert [r.route_without_waiting for r in res.candidate.rules] == [False]
    assert "policy_reextract:KB-102" in stub.calls and stub.support_calls == 2
    history = [f for f in res.findings if f.kind == "unsupported_rule"]
    assert len(history) == 1 and history[0].action == "re_extracted" and not history[0].blocking
    assert res.article_stats["KB-102"]["re_extracted"] is True


# ---------------------------------------------------------------- KB naming rules


def test_kb_naming_rules_accept_the_shipped_kb_and_normalise_status(tmp_path):
    from servicedesk.config import KB_ADDITIONS_DIR, KB_DIR, KB_MUTATIONS_DIR
    from servicedesk.kb import load_kb

    assert {f"KB-{n:03d}" for n in range(1, 14)} <= {a.article_id for a in load_kb([KB_DIR, KB_ADDITIONS_DIR])}
    for d in sorted(p for p in KB_MUTATIONS_DIR.iterdir() if p.is_dir()):
        load_kb([KB_DIR, d], allow_override=True)  # edited copies replace an article on purpose
    art = parse_article(write_kb(tmp_path, {"KB-101-a.md": ARTICLE_A.replace("Status: Active", "Status: active")})
                        / "KB-101-a.md")
    assert art.status == "Active" and art.active


@pytest.mark.parametrize("name, text, needle", [
    ("toner.md", ARTICLE_A, "not a valid KB file name"),
    ("KB-20-toner.md", ARTICLE_A.replace("KB-101", "KB-20"), "not a valid KB file name"),
    ("KB-101_Shared_Folders.md", ARTICLE_A, "not a valid KB file name"),
    ("KB-101-a.md", ARTICLE_A.replace("# KB-101: Shared folders", "Shared folders"), "missing the heading line"),
    ("KB-101-a.md", ARTICLE_A.replace("# KB-101:", "# KB-109:"), "file name says KB-101 but the heading says KB-109"),
    ("KB-101-a.md", ARTICLE_A.replace("- Owner: Workplace Infrastructure\n", ""), "missing metadata line(s): '- Owner: ...'"),
    ("KB-101-a.md", ARTICLE_A.replace("Status: Active", "Status: Draft"), "use Active or Deprecated"),
])
def test_kb_naming_rules_reject_bad_files_with_the_fix(tmp_path, name, text, needle):
    from servicedesk.kb import KBFormatError, load_kb

    kb = write_kb(tmp_path / "kb", {"KB-102-b.md": ARTICLE_B, name: text})
    with pytest.raises(KBFormatError) as e:
        load_kb([kb])
    assert needle in str(e.value) and name in str(e.value)


def test_an_article_id_may_appear_only_once(tmp_path):
    from servicedesk.kb import KBFormatError, load_kb

    same = write_kb(tmp_path / "same", {"KB-101-a.md": ARTICLE_A, "KB-101-copy.md": ARTICLE_A})
    with pytest.raises(KBFormatError, match="KB-101 appears twice"):
        load_kb([same], allow_override=True)  # never allowed inside one folder

    base = write_kb(tmp_path / "base", {"KB-101-a.md": ARTICLE_A})
    extra = write_kb(tmp_path / "extra", {"KB-101-new.md": ARTICLE_A.replace("exact error", "exact error text")})
    with pytest.raises(KBFormatError, match="base/KB-101-a.md and extra/KB-101-new.md"):
        load_kb([base, extra])
    assert "exact error text" in load_kb([base, extra], allow_override=True)[0].text  # KB change tests only


# ---------------------------------------------------------------- automatic recompile (ensure_current)


def test_ensure_current_compiles_approves_and_then_does_nothing_until_the_kb_changes(tmp_path):
    root = tmp_path / "policy"
    kb = write_kb(tmp_path / "kb", {"KB-101-a.md": ARTICLE_A, "KB-103-old.md": ARTICLE_OLD})
    stub = StubLLM({"KB-101": [FOLDER_RULE], "KB-102": [PRINTER_RULE]})
    log: list[str] = []

    first = compiler.ensure_current([kb], name="demo", llm=stub, root=root, use_cache=False, log=log.append)
    assert first.action == "approved" and first.table.version == 1
    assert compiler.live_pointer("demo", root=root)["approved_by"] == compiler.AUTO_APPROVER
    assert any("No approved policy" in m for m in log) and any("Approved automatically" in m for m in log)

    stub.calls.clear()
    same = compiler.ensure_current([kb], name="demo", llm=stub, root=root, use_cache=False, log=log.append)
    assert same.action == "unchanged" and same.table.version == 1 and stub.calls == []

    # A new article: only it goes to the model, and the new version goes live by itself.
    (kb / "KB-102-b.md").write_text(ARTICLE_B, encoding="utf-8")
    added = compiler.ensure_current([kb], name="demo", llm=stub, root=root, use_cache=False, log=log.append)
    assert added.action == "approved" and added.table.version == 2
    assert added.changes.added == ["KB-102"]
    assert [c for c in stub.calls if c.startswith("policy_extract")] == ["policy_extract:KB-102"]
    assert {r.rule_id for r in compiler.load_live("demo", root=root).rules} == {FOLDER_RULE.rule_id, PRINTER_RULE.rule_id}


def test_ensure_current_holds_a_change_the_checker_blocks_without_recompiling_it_again(tmp_path):
    root = tmp_path / "policy"
    kb = write_kb(tmp_path / "kb", {"KB-101-a.md": ARTICLE_A})
    stub = StubLLM({"KB-101": [FOLDER_RULE]})
    compiler.ensure_current([kb], name="demo", llm=stub, root=root, use_cache=False, log=lambda _m: None)

    # Deprecating KB-101 drops its approval rule: a removed safeguard, so a person must decide.
    (kb / "KB-101-a.md").write_text(ARTICLE_A.replace("Status: Active", "Status: Deprecated"), encoding="utf-8")
    log: list[str] = []
    held = compiler.ensure_current([kb], name="demo", llm=stub, root=root, use_cache=False, log=log.append)
    assert held.action == "held" and held.table.version == 1  # runs keep the approved version
    assert {f.kind for f in held.blocking} == {"removed_safeguard"}
    assert any("held for review" in m for m in log) and any("--acknowledge-findings" in m for m in log)

    stub.calls.clear()
    again = compiler.ensure_current([kb], name="demo", llm=stub, root=root, use_cache=False, log=lambda _m: None)
    assert again.action == "held" and stub.calls == []  # same KB, same verdict: no new model calls

    compiler.approve("demo", acknowledge_findings=True, root=root)  # a person accepts it
    after = compiler.ensure_current([kb], name="demo", llm=stub, root=root, use_cache=False, log=lambda _m: None)
    assert after.action == "unchanged" and after.table.version == 2 and after.table.rules == []


def test_ensure_current_with_nothing_approved_and_a_blocked_first_compile_returns_no_table(tmp_path):
    kb = write_kb(tmp_path / "kb", {"KB-101-a.md": ARTICLE_A})
    stub = StubLLM({"KB-101": [FOLDER_RULE]}, fail={"KB-101"})
    up = compiler.ensure_current([kb], name="demo", llm=stub, root=tmp_path / "policy", use_cache=False,
                                 log=lambda _m: None)
    assert up.action == "held" and up.table is None
    assert [f.kind for f in up.blocking] == ["extraction_failed"]


def test_load_policy_falls_back_to_gold_only_when_nothing_is_approved(monkeypatch):
    from servicedesk.evals import request_eval

    monkeypatch.setattr(compiler, "ensure_current",
                        lambda kb_set, log=print: compiler.PolicyUpdate(kb_set, None, "held"))
    log: list[str] = []
    table, source = request_eval.load_policy("base", "live", log=log.append)
    assert source == "gold" and table.rules and log and log[0].startswith("WARNING")


def test_a_run_checks_the_policy_once_and_reports_a_held_change_once(monkeypatch, capsys):
    from servicedesk import cli
    from servicedesk.evals import request_eval

    v1, checks = table([FOLDER_RULE]), []

    def held(kb_set, log=print, confirm=None):
        checks.append(kb_set)
        log(f"Policy change for '{kb_set}' held for review: 1 blocking checker finding(s):")
        return compiler.PolicyUpdate(kb_set, v1, "held")

    used = {}

    def run(kb_set, name, policy_source="live", policy_update=None, **_kw):
        used["policy"] = request_eval.load_policy(kb_set, policy_source, update=policy_update)
        return {"requests": {}}

    monkeypatch.setattr(compiler, "ensure_current", held)
    monkeypatch.setattr(cli, "_prepare_search", lambda kb_set, args: None)
    monkeypatch.setattr(request_eval, "run_requests", run)
    assert cli.main(["run", "--kb", "base", "--name", "x"]) == 0
    assert checks == ["base"] and used["policy"] == (v1, "live")  # the run used the check already made
    assert capsys.readouterr().out.count("held for review") == 1

    # Nothing approved yet: the check already made still leads to the hand-checked policy.
    log: list[str] = []
    gold, source = request_eval.load_policy("base", "live", log=log.append,
                                            update=compiler.PolicyUpdate("base", None, "held"))
    assert source == "gold" and gold.rules and log[0].startswith("WARNING") and checks == ["base"]


def test_cli_reports_a_bad_kb_file_without_a_traceback(monkeypatch, capsys):
    from servicedesk import cli
    from servicedesk.kb import KBFormatError
    from servicedesk.policy import compiler

    def bad(*_a, **_k):
        raise KBFormatError("knowledge-base/toner.md: not a valid KB file name.")

    # The KB check before every run loads the articles; it must not depend on a compiled policy existing.
    monkeypatch.setattr(compiler, "ensure_current", bad)
    assert cli.main(["run", "--kb", "base", "--name", "x"]) == 2
    assert "Error: knowledge-base/toner.md: not a valid KB file name." in capsys.readouterr().err


def test_a_person_at_the_terminal_can_accept_a_held_change(tmp_path):
    root = tmp_path / "policy"
    kb = write_kb(tmp_path / "kb", {"KB-101-a.md": ARTICLE_A})
    stub = StubLLM({"KB-101": [FOLDER_RULE]})
    compiler.ensure_current([kb], name="demo", llm=stub, root=root, use_cache=False, log=lambda _m: None)
    (kb / "KB-101-a.md").write_text(ARTICLE_A.replace("Status: Active", "Status: Deprecated"), encoding="utf-8")

    asked: list[str] = []
    declined = compiler.ensure_current([kb], name="demo", llm=stub, root=root, use_cache=False,
                                       log=lambda _m: None, confirm=lambda q: asked.append(q) or False)
    assert declined.action == "held" and declined.table.version == 1 and len(asked) == 1

    stub.calls.clear()  # the held candidate is reused: accepting it costs no model calls
    accepted = compiler.ensure_current([kb], name="demo", llm=stub, root=root, use_cache=False,
                                       log=lambda _m: None, confirm=lambda q: True)
    assert accepted.action == "approved" and accepted.approved_by == compiler.PERSON_APPROVER
    assert accepted.table.version == 2 and stub.calls == []
    assert compiler.live_pointer("demo", root=root)["acknowledged_findings"]


# ---------------------------------------------------------------- approved versions keep their articles


def _pipeline(policy: PolicyTable, dirs: list[Path], tmp_path: Path):
    from servicedesk.pipeline import Pipeline
    from servicedesk.stores import Stores

    return Pipeline(policy, dirs, Stores(tmp_path / "run"), llm=object(), retrieval="bm25")


def test_a_held_deprecation_changes_nothing_for_requests_until_a_person_accepts_it(tmp_path):
    root = tmp_path / "policy"
    kb = write_kb(tmp_path / "kb", {"KB-101-a.md": ARTICLE_A, "KB-102-b.md": ARTICLE_B})
    stub = StubLLM({"KB-101": [FOLDER_RULE], "KB-102": [PRINTER_RULE]})
    quiet = dict(name="demo", llm=stub, root=root, use_cache=False, log=lambda _m: None)
    v1 = compiler.ensure_current([kb], **quiet).table
    pinned = compiler.article_dirs([kb], v1, name="demo", root=root)
    assert pinned == [root / "demo" / "articles" / "v1"]
    assert not (root / "demo" / "articles" / "candidate").exists()  # moved, not copied

    (kb / "KB-101-a.md").write_text(ARTICLE_A.replace("Status: Active", "Status: Deprecated"), encoding="utf-8")
    held = compiler.ensure_current([kb], **quiet)
    assert held.action == "held" and held.table.version == 1
    dirs = compiler.article_dirs([kb], held.table, name="demo", root=root)
    assert dirs == pinned
    p = _pipeline(held.table, dirs, tmp_path)
    assert p.articles["KB-101"].active and FOLDER_RULE.rule_id in {r.rule_id for r in p.live_rules}
    assert "KB-101" in {a.article_id for a in p.retriever.articles}
    # What the pipeline did before article copies: the edited file on disk silently dropped the rule.
    assert FOLDER_RULE.rule_id not in {r.rule_id for r in _pipeline(held.table, [kb], tmp_path).live_rules}

    v2 = compiler.approve("demo", acknowledge_findings=True, root=root)  # a person accepts it
    p2 = _pipeline(v2, compiler.article_dirs([kb], v2, name="demo", root=root), tmp_path)
    assert not p2.articles["KB-101"].active and [r.rule_id for r in p2.live_rules] == [PRINTER_RULE.rule_id]

    back = compiler.rollback("demo", root=root)  # the older articles come back with the older rules
    p1 = _pipeline(back, compiler.article_dirs([kb], back, name="demo", root=root), tmp_path)
    assert p1.articles["KB-101"].active and FOLDER_RULE.rule_id in {r.rule_id for r in p1.live_rules}


def test_a_held_text_edit_keeps_the_approved_text_for_search_and_the_rule_picker(tmp_path):
    root = tmp_path / "policy"
    kb = write_kb(tmp_path / "kb", {"KB-102-b.md": ARTICLE_B})
    stub = StubLLM({"KB-102": [PRINTER_RULE]})
    quiet = dict(name="demo", llm=stub, root=root, use_cache=False, log=lambda _m: None)
    compiler.ensure_current([kb], **quiet)

    # The edit drops the must-never sentence: a removed safeguard, so the change is held.
    (kb / "KB-102-b.md").write_text(ARTICLE_B.replace(" Do not instruct users to assign an IP.", " Restart the spooler."),
                                    encoding="utf-8")
    stub.rules_by_article["KB-102"] = [rule("KB-102:printer", situation="A user cannot print",
                                            quotes=["Restart the spooler."])]
    held = compiler.ensure_current([kb], **quiet)
    assert held.action == "held"
    art = _pipeline(held.table, compiler.article_dirs([kb], held.table, name="demo", root=root), tmp_path).articles["KB-102"]
    assert "Do not instruct users to assign an IP." in art.text and "spooler" not in art.text
    assert art.sha256 == held.table.kb_hashes["KB-102"]


def test_a_kept_rule_keeps_its_approved_article_text_in_the_next_version(tmp_path):
    root = tmp_path / "policy"
    kb = write_kb(tmp_path / "kb", {"KB-101-a.md": ARTICLE_A})
    stub = StubLLM({"KB-101": [FOLDER_RULE]})
    quiet = dict(name="demo", llm=stub, root=root, use_cache=False, log=lambda _m: None)
    compiler.ensure_current([kb], **quiet)

    # KB-101 is edited but its extraction fails: the old rules (and so the old text) are kept.
    (kb / "KB-101-a.md").write_text(ARTICLE_A.replace("exact error", "exact error message"), encoding="utf-8")
    stub.fail = {"KB-101"}
    assert compiler.ensure_current([kb], **quiet).action == "held"
    v2 = compiler.approve("demo", acknowledge_findings=True, root=root)
    [d] = compiler.article_dirs([kb], v2, name="demo", root=root)
    assert d.name == "v2" and "exact error message" not in (d / "KB-101-a.md").read_text(encoding="utf-8")


def test_a_version_approved_before_article_copies_gets_one_only_if_the_kb_still_matches(tmp_path):
    root = tmp_path / "policy"
    kb = write_kb(tmp_path / "kb", {"KB-101-a.md": ARTICLE_A})
    hashes = {a.article_id: a.sha256 for a in load_articles([kb])}
    _write_candidate(root, "demo", table([FOLDER_RULE], hashes=hashes), [], None)  # no article copy
    v1 = compiler.approve("demo", root=root)
    assert compiler.article_dirs([kb], v1, name="demo", root=root) == [root / "demo" / "articles" / "v1"]

    _write_candidate(root, "demo", table([FOLDER_RULE], version=2, hashes=hashes), [], 1)
    v2 = compiler.approve("demo", root=root)
    (kb / "KB-101-a.md").write_text(ARTICLE_A.replace("exact error", "exact error text"), encoding="utf-8")
    notes: list[str] = []
    assert compiler.article_dirs([kb], v2, name="demo", root=root, log=notes.append) == [kb]
    assert notes and "no saved copy" in notes[0]


def test_the_hand_checked_policy_reads_the_kb_folders_as_they_are():
    from servicedesk.config import KB_ADDITIONS_DIR, KB_DIR

    assert compiler.article_dirs("base", compiler.load_gold("base")) == [KB_DIR]
    assert compiler.article_dirs("with-kb013", compiler.load_gold("with-kb013")) == [KB_DIR, KB_ADDITIONS_DIR]


def _demo_args(**kw):
    import argparse

    return argparse.Namespace(policy="live", yes=False, workers=1, **kw)


def test_demo_stops_at_step_1_when_the_base_policy_is_held(monkeypatch, capsys):
    from servicedesk import cli
    from servicedesk.evals import request_eval

    monkeypatch.setattr(compiler, "ensure_current", lambda kb_set, **_k: compiler.PolicyUpdate(kb_set, None, "held"))
    monkeypatch.setattr(request_eval, "run_requests", lambda *a, **k: pytest.fail("no run when the policy is held"))
    assert cli.cmd_demo(_demo_args()) == 2
    assert "--acknowledge-findings" in capsys.readouterr().out


def test_demo_stops_before_run_b_when_the_kb013_change_is_held(monkeypatch, capsys):
    from servicedesk import cli
    from servicedesk.evals import request_eval

    base = table([FOLDER_RULE], version=1)
    updates = {"base": compiler.PolicyUpdate("base", base, "unchanged"),
               "with-kb013": compiler.PolicyUpdate("with-kb013", base, "held")}
    runs: list[str] = []
    monkeypatch.setattr(compiler, "ensure_current", lambda kb_set, **_k: updates[kb_set])
    monkeypatch.setattr(request_eval, "run_requests", lambda kb_set, *a, **k: runs.append(kb_set) or {"requests": {}})
    assert cli.cmd_demo(_demo_args()) == 2
    assert runs == ["base"]  # run A happened, run B did not
    assert "Stopped before run B" in capsys.readouterr().out
