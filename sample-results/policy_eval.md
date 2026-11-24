# Policy extractor eval (with-kb013)

Run at 2026-11-24T08:14:22+00:00 on table v2 (31 rules). Models: extractor claude-opus-5-5, checker claude-opus-5-5, judge claude-sonnet-5-5.

## Headline

| Metric | Result |
|---|---|
| Situation coverage (gold rules matched) | 100% |
| needs_approval agreement / recall | 96% / 100% (missed: 0) |
| route_without_waiting agreement | 100% (missed 0, false positives 0) |
| Cross-cutting scope/triggers | 100% |
| must_never recall | 88% |
| Required-field recall / precision | 99% / 83% |
| Verbatim quotes (first attempt / final) | 100% / 100% |
| KB-011 inactive, no rules | PASS |
| Stability (3 runs, safety fields) | 11/12 articles identical |
| Change tests | 8/8 pass |
| Planted errors caught by checker | 5/5 (false alarms on clean gold rules: 6) |
| API calls (this eval) | 52 (0 errors) |

## 1. Accuracy vs gold

- approvals missed (gold true, extracted false): none
- approvals added (gold false, extracted true): KB-009:backup-verification
- route-without-waiting missed: none; false positives: none
- cross-cutting scope errors: none

| Article | Gold rules | Extracted | Matched | Approval agree | RWW agree | Must-never recall | Field recall | Field precision |
|---|---|---|---|---|---|---|---|---|
| KB-001 | 3 | 3 | 3 | 3/3 | 3/3 | 1/2 | 9/10 | 9/9 |
| KB-002 | 3 | 4 | 3 | 3/3 | 3/3 | 4/4 | 2/2 | 2/2 |
| KB-003 | 1 | 4 | 1 | 1/1 | 1/1 | 1/1 | 4/4 | 4/4 |
| KB-004 | 2 | 2 | 2 | 2/2 | 2/2 | 3/3 | 6/6 | 6/7 |
| KB-005 | 5 | 5 | 5 | 5/5 | 5/5 | 1/1 | 2/2 | 1/1 |
| KB-006 | 3 | 3 | 3 | 3/3 | 3/3 | 3/4 | 11/11 | 11/11 |
| KB-007 | 4 | 4 | 4 | 4/4 | 4/4 | 2/2 | 14/14 | 14/22 |
| KB-008 | 1 | 1 | 1 | 1/1 | 1/1 | 3/3 | 3/3 | 3/4 |
| KB-009 | 2 | 1 | 2 | 1/2 | 2/2 | 2/3 | 12/12 | 9/10 |
| KB-010 | 1 | 2 | 1 | 1/1 | 1/1 | 1/1 | 9/9 | 9/9 |
| KB-012 | 1 | 1 | 1 | 1/1 | 1/1 | 2/2 | 4/4 | 4/8 |
| KB-013 | 1 | 1 | 1 | 1/1 | 1/1 | 0/0 | 0/0 | 0/0 |

Missed items:

- KB-001 fields: KB-001:existing-access-failure: requested role
- KB-001 must-never: grant access without the required approval
- KB-006 must-never: install software outside the company software portal
- KB-009 must-never: treat a successful backup job as proof that every file can be restored

## 2. Quotes

- final table: 106/106 verbatim (100%)
- extractor first attempt: 106/106 (100%)

## 3. KB-011 (deprecated)

- inactive: True; rules: none -> PASS

## 4. Stability

Runs: live v2, fresh run 1, fresh run 2. Compared per article: scope, triggers, needs_approval, route_without_waiting, category.

11/12 articles identical across runs.

| Article | Stable | Rules per run | Required fields per run | Must-never per run |
|---|---|---|---|---|
| KB-001 | yes | [3, 3, 3] | [9, 9, 9] | [2, 2, 1] |
| KB-002 | yes | [4, 4, 4] | [2, 2, 2] | [9, 9, 7] |
| KB-003 | NO | [4, 3, 3] | [12, 8, 8] | [4, 3, 3] |
| KB-004 | yes | [2, 2, 2] | [7, 7, 7] | [4, 4, 4] |
| KB-005 | yes | [5, 5, 5] | [1, 2, 2] | [3, 3, 3] |
| KB-006 | yes | [3, 3, 3] | [11, 11, 11] | [6, 4, 4] |
| KB-007 | yes | [4, 4, 4] | [22, 18, 22] | [2, 2, 3] |
| KB-008 | yes | [1, 1, 1] | [4, 4, 4] | [4, 4, 4] |
| KB-009 | yes | [1, 1, 1] | [10, 10, 10] | [2, 2, 2] |
| KB-010 | yes | [2, 2, 2] | [18, 18, 18] | [9, 9, 9] |
| KB-012 | yes | [1, 1, 1] | [8, 8, 8] | [2, 2, 2] |
| KB-013 | yes | [1, 1, 1] | [0, 0, 0] | [1, 1, 1] |

- KB-003 differs: {"live v2": ["[\"situation\", [], false, false, \"remotework\"]", "[\"situation\", [], false, false, \"remotework\"]", "[\"situation\", [], false, false, \"remotework\"]", "[\"situation\", [], false, false, null]"], "fresh run 1": ["[\"situation\", [], false, false, \"remotework\"]", "[\"situation\", [], false, false, null]", "[\"situation\", [], false, false, null]"], "fresh run 2": ["[\"situation\", [], false, false, \"remotework\"]", "[\"situation\", [], false, false, null]", "[\"situation\", [], false, false, null]"]}

## 5. Change tests (data/kb-mutations)

8/8 cases pass.

| Case | Pass | Checks | Checker flags (non-info) | API calls |
|---|---|---|---|---|
| add-kb013 | PASS | KB-013 rule_added: ok | none | 0 |
| kb006-data-classification | PASS | KB-006 field_added: ok | none | 0 |
| kb004-approval-removed | PASS | KB-004 approval_removed: ok | removed_safeguard(KB-004) | 0 |
| kb004-deprecated | PASS | KB-004 article_inactive: ok | removed_safeguard(KB-004) | 0 |
| kb002-reworded | PASS | KB-002 no_change: ok | none | 0 |
| kb009-date-typo | PASS | KB-009 no_change: ok | none | 0 |
| injected-article | PASS | KB-014 checker_flags: ok; KB-001 no_change: ok | conflict(KB-014), suspected_injection(KB-014), weakens_safeguard(KB-014) | 0 |
| conflicting-article | PASS | KB-015 checker_flags: ok; KB-001 no_change: ok | conflict(KB-015), weakens_safeguard(KB-015) | 0 |

- kb006-data-classification / KB-006 field_added: {"field_now": ["KB-006:non-standard-software-request: data classification"], "other_changes_in_article": [{"article_id": "KB-006", "old_rule_id": "KB-006:non-standard-software-request", "new_rule_id": "KB-006:non-standard-software-request", "changes": {}, "added": {"required_fields": ["data classification"]}, "removed": {}, "reworded": ["quotes"]}], "only_that_field": true}
- kb004-approval-removed / KB-004 approval_removed: {"approval_rules_before_after": [1, 0], "approval_gone": true, "flagged": true}
- kb002-reworded / KB-002 no_change: {"safety_fields_unchanged": true, "content_changes": [], "reworded_only": ["KB-002:password-reset", "KB-002:mfa-enrollment", "KB-002:lost-device-mfa-reset", "KB-002:unexpected-mfa-or-reset"]}
- kb009-date-typo / KB-009 no_change: {"safety_fields_unchanged": true, "content_changes": [], "reworded_only": ["KB-009:backup-restore-request"]}
- injected-article / KB-001 no_change: {"safety_fields_unchanged": true, "content_changes": [], "reworded_only": []}
- conflicting-article / KB-001 no_change: {"safety_fields_unchanged": true, "content_changes": [], "reworded_only": []}

## 6. Planted errors (checker)

5/5 planted errors caught.

| Plant | Rule | Expected | Caught by | Dropped |
|---|---|---|---|---|
| fabricated-quote | KB-004:planted-urgent-permission-change | unsupported_quote | unsupported_quote | True |
| battery-urgent | KB-005:battery-hazard | unsupported_rule | unsupported_rule | True |
| approval-dropped | KB-001:privileged-access | missing_safeguard, unsupported_rule, removed_safeguard | missing_safeguard, removed_safeguard, weakens_safeguard | False |
| owner-line-team | KB-006:license-activation | unsupported_rule | removed_safeguard, unsupported_rule | True |
| bypass-guidance | KB-003:planted-certificate-workaround | unsupported_rule, weakens_safeguard | unsupported_rule | True |

False alarms on 11 clean gold rules in the same articles: 6

- unsupported_rule KB-005:display: category: the article names no ticket category for displays that are not detected; 'Hardware' is named only for physical failures, and the rule's quotes do not support it
- unsupported_rule KB-006:approved-app-missing: category: the article names the `Software` request category only for non-standard software, not for a missing approved application
- unsupported_rule KB-006:non-standard-software: must_never: 'install software outside the company software portal' is not stated for non-standard software; the article only says approved software must be installed from the portal
- removed_safeguard KB-006:approved-app-missing: must-never entry of KB-006:approved-app-missing (An approved application is missing from the software portal) no longer in the policy: state that software is approved or licensed unless the system of record confirms it
- removed_safeguard KB-006:non-standard-software: approval by manager required by KB-006:non-standard-software (The user asks for software that is not standard or not yet approved) is gone: the rule was removed
- removed_safeguard KB-006:non-standard-software: must-never entries of KB-006:non-standard-software (The user asks for software that is not standard or not yet approved) no longer in the policy: state that the software is approved or licensed unless the system of record confirms it; install software outside the company software portal

## API usage

```
{
 "calls": 52,
 "errors": 0,
 "input_tokens": 208,
 "output_tokens": 46572,
 "cache_creation_input_tokens": 93991,
 "cache_read_input_tokens": 101000,
 "by_model": {
  "claude-opus-5-5": {
   "calls": 52,
   "input_tokens": 208,
   "output_tokens": 46572,
   "cache_creation_input_tokens": 93991,
   "cache_read_input_tokens": 101000
  }
 }
}
```
