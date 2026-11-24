"""KB policy compiler: watcher (code) -> extractor agent -> checker agent -> diff -> approval.

See compiler.py for the public functions. Typical use:

    from servicedesk.policy import compile_policy, approve, load_live
    result = compile_policy("base")      # writes runtime/policy/base/candidate.json, findings.json, diff.md
    print(result.summary()); print(result.diff_md)
    approve("base", approved_by="alice", acknowledge_findings=True)
    table = load_live("base")

Runs call ensure_current(kb_set) instead: it recompiles when the KB changed and makes a
clean candidate live on its own (see compiler.ensure_current).
"""
from .checker import BLOCKING_KINDS, Finding
from .compiler import (
    AUTO_APPROVER,
    CompileResult,
    PERSON_APPROVER,
    PolicyApprovalError,
    PolicyUpdate,
    approve,
    compare,
    compile_policy,
    ensure_current,
    get_policy,
    list_versions,
    load_candidate,
    load_gold,
    load_live,
    load_version,
    policy_dir,
    rollback,
    status,
)
from .diff import PolicyDiff, diff_tables
from .watcher import KB_SETS, KBChanges, KBSource, detect_changes, load_articles, resolve_source, watch

__all__ = [
    "BLOCKING_KINDS",
    "Finding",
    "AUTO_APPROVER",
    "CompileResult",
    "PERSON_APPROVER",
    "PolicyApprovalError",
    "PolicyUpdate",
    "approve",
    "compare",
    "compile_policy",
    "ensure_current",
    "get_policy",
    "list_versions",
    "load_candidate",
    "load_gold",
    "load_live",
    "load_version",
    "policy_dir",
    "rollback",
    "status",
    "PolicyDiff",
    "diff_tables",
    "KB_SETS",
    "KBChanges",
    "KBSource",
    "detect_changes",
    "load_articles",
    "resolve_source",
    "watch",
]
