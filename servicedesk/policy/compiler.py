"""KB policy compiler: watcher -> extractor -> checker -> diff -> human approval -> live version.

Public functions (all take a KB set name: "base" or "with-kb013"; `compile`
also accepts a list of folders or a watcher.KBSource):

    compile(kb_set="base", **opts) -> CompileResult
        Load the KB set, detect added/changed/removed articles against the live
        version, extract rules for new or changed Active articles (unchanged
        articles reuse the live rules by hash; deprecated ones produce none),
        run the checker, and write the candidate to runtime/policy/<kb_set>/:
        candidate.json (PolicyTable, source "extracted", version live+1),
        findings.json (checker findings), diff.md (readable diff vs live) and
        report.json (per-article stats, structured diff, API usage). Never
        raises on a model error: the article gets an "extraction_failed"
        finding and keeps its previous rules if it had any.
        Options: llm=LLM(), use_cache=True (on-disk cache of model outputs),
        reuse_live=True, anchor=True (changed articles are re-extracted with
        their previous rules as a naming reference), repair=True (an article
        whose rules the checker rejects is re-extracted once with the checker's
        problems as feedback, then checked again), table_review=True,
        previous=<PolicyTable to diff against instead of the live one>,
        write=True, out_dir=None, workers=4, name=None (output folder name for
        a folder list).
    approve(kb_set, approved_by="reviewer", acknowledge_findings=False) -> PolicyTable
        Promote the pending candidate to vN.json and point live.json at it.
        Refuses if there are blocking findings (removed/weakened safeguard,
        conflict, failed extraction or check, suspected injection) unless
        acknowledge_findings=True, or if the live version changed since the
        compile. Earlier versions are kept.
    ensure_current(kb_set, log=print) -> PolicyUpdate
        What every run calls first: if any article was added, edited, deprecated
        or deleted since the live version (sha256 per file), compile, and make the
        candidate live automatically when the checker finds nothing blocking.
        Blocking findings hold the change for a person; runs keep the last
        approved version and its copy of the articles.
    rollback(kb_set, to_version=None) -> PolicyTable
        Point live.json at an earlier version (default: the one before live); its
        article copy goes with it.
    load_live(kb_set) -> PolicyTable | None
    load_version(kb_set, version) -> PolicyTable
    list_versions(kb_set) -> list[int]
    load_candidate(kb_set) -> (PolicyTable, list[Finding]) | None
    status(kb_set) -> dict        live version, pending candidate, KB changes since live
    compare(kb_set_a, kb_set_b) -> PolicyDiff   diff between two live tables (.to_markdown())
    load_gold(kb_set="with-kb013") -> PolicyTable   data/gold/policy_gold.json, filtered to
        the KB set's articles (base excludes KB-013)
    get_policy(kb_set, source="live"|"gold") -> PolicyTable   for the request pipeline
    article_dirs(kb_set, table) -> list[Path]   the article folders the pipeline loads with `table`

Every approved version keeps a copy of the article files it was compiled from
(runtime/policy/<kb_set>/articles/v<N>/; the compile writes articles/candidate/
next to candidate.json and approve moves it). The request pipeline reads its
articles from that copy, not from data/knowledge-base/: an article's text and
Status (what search and the rule picker see) always come from the same approved
version as its rules. A KB change the checker holds (an edit, a deprecation, a
deletion) therefore changes nothing in how requests are handled until a person
accepts it.

Every function takes an optional `root` (default runtime/policy) so tests can
use a temporary folder.
"""
from __future__ import annotations

import json
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from ..config import GOLD_DIR, POLICY_DIR, ROOT
from ..kb import Article, KBFormatError, load_kb
from ..llm import LLM, LLMError, usage_delta
from ..models import PolicyRule, PolicyTable
from ..stores import now_iso
from .checker import Finding, check_quotes, check_support, run_checker
from .diff import PolicyDiff, diff_tables
from .extractor import extract_article
from .watcher import KBChanges, KBSetLike, KBSource, detect_changes, load_articles, resolve_source

GOLD_FILE = GOLD_DIR / "policy_gold.json"
CANDIDATE_FILES = ("candidate.json", "findings.json", "diff.md", "report.json")

_LIVE = object()  # sentinel: "use the live version as previous"


class PolicyApprovalError(RuntimeError):
    """approve() or rollback() refused."""


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------


def policy_dir(kb_set: Union[str, KBSource], root: Optional[Path] = None) -> Path:
    name = kb_set.name if isinstance(kb_set, KBSource) else kb_set
    return Path(root or POLICY_DIR) / name


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json(path: Path, data) -> None:
    _write(path, json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _audit(d: Path, event: str, **detail) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now_iso(), "request_id": f"policy:{d.name}", "step": event, **detail}, ensure_ascii=False, default=str) + "\n")


def list_versions(kb_set: str, *, root: Optional[Path] = None) -> list[int]:
    d = policy_dir(kb_set, root)
    out = []
    for p in d.glob("v*.json"):
        stem = p.stem[1:]
        if stem.isdigit():
            out.append(int(stem))
    return sorted(out)


def load_version(kb_set: str, version: int, *, root: Optional[Path] = None) -> PolicyTable:
    p = policy_dir(kb_set, root) / f"v{version}.json"
    if not p.exists():
        raise FileNotFoundError(f"{kb_set}: no policy version {version} ({p})")
    return PolicyTable.model_validate_json(p.read_text(encoding="utf-8"))


def live_pointer(kb_set: str, *, root: Optional[Path] = None) -> Optional[dict]:
    p = policy_dir(kb_set, root) / "live.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_live(kb_set: str, *, root: Optional[Path] = None) -> Optional[PolicyTable]:
    """The approved policy table for a KB set, or None if nothing was approved yet."""
    ptr = live_pointer(kb_set, root=root)
    if not ptr:
        return None
    return load_version(kb_set, int(ptr["version"]), root=root)


def load_candidate(kb_set: str, *, root: Optional[Path] = None) -> Optional[tuple[PolicyTable, list[Finding]]]:
    d = policy_dir(kb_set, root)
    if not (d / "candidate.json").exists():
        return None
    table = PolicyTable.model_validate_json((d / "candidate.json").read_text(encoding="utf-8"))
    findings = []
    if (d / "findings.json").exists():
        findings = [Finding(**f) for f in json.loads((d / "findings.json").read_text(encoding="utf-8"))["findings"]]
    return table, findings


def articles_dir(kb_set: str, version: Optional[int] = None, *, root: Optional[Path] = None) -> Path:
    """Where a version's article copy is kept (version None: the pending candidate's)."""
    return policy_dir(kb_set, root) / "articles" / ("candidate" if version is None else f"v{version}")


def _copy_matches(folder: Path, table: PolicyTable) -> bool:
    """The folder holds exactly the article files `table` was compiled from (same ids, same sha256)."""
    if not folder.is_dir():
        return False
    try:
        return {a.article_id: a.sha256 for a in load_kb([folder])} == table.kb_hashes
    except (KBFormatError, OSError):
        return False


def _save_articles(dest: Path, table: PolicyTable, *sources: list[Article]) -> bool:
    """Write the article files `table` was compiled from (the text whose sha256 is in its kb_hashes)
    to `dest`, taking each from the first source that has that exact text. Writes nothing and
    returns False when some article's text is in none of the sources."""
    shutil.rmtree(dest, ignore_errors=True)
    have: dict[tuple[str, str], Article] = {}
    for arts in sources:
        for a in arts:
            have.setdefault((a.article_id, a.sha256), a)
    picked = [have.get((aid, h)) for aid, h in table.kb_hashes.items()]
    if any(a is None for a in picked):
        return False
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for a in picked:
        (tmp / a.path.name).write_bytes(a.text.encode("utf-8"))
    tmp.rename(dest)
    return True


def _previous_copy(kb_set: str, previous: Optional[PolicyTable], root: Optional[Path]) -> list[Article]:
    """The articles of the previous version's copy (for an article whose failed extraction kept its old rules)."""
    if previous is None or previous.source != "extracted":
        return []
    d = articles_dir(kb_set, previous.version, root=root)
    try:
        return load_kb([d]) if d.is_dir() else []
    except (KBFormatError, OSError):
        return []


# --------------------------------------------------------------------------
# compile
# --------------------------------------------------------------------------


@dataclass
class CompileResult:
    kb_set: str
    candidate: PolicyTable
    findings: list[Finding]
    diff: PolicyDiff
    diff_md: str
    changes: KBChanges
    previous_version: Optional[int]
    article_stats: dict[str, dict] = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    calls: list[dict] = field(default_factory=list)
    paths: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.blocking]

    def rule_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.candidate.rules:
            counts[r.article_id] = counts.get(r.article_id, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> str:
        c = self.diff.counts()
        return (
            f"{self.kb_set}: candidate v{self.candidate.version} ({len(self.candidate.rules)} rules, "
            f"inactive: {', '.join(self.candidate.inactive_articles) or 'none'}); {self.changes.summary()}; "
            f"diff vs v{self.previous_version}: +{c['added']} -{c['removed']} ~{c['changed']}; "
            f"findings: {len(self.findings)} ({len(self.blocking)} blocking); API calls: {self.usage.get('calls', 0)}"
        )

    def report(self) -> dict:
        return {
            "kb_set": self.kb_set,
            "candidate_version": self.candidate.version,
            "previous_version": self.previous_version,
            "compiled_at": self.candidate.compiled_at,
            "changes": self.changes.to_dict(),
            "rule_counts": self.rule_counts(),
            "article_stats": self.article_stats,
            "diff": self.diff.to_dict(),
            "findings_total": len(self.findings),
            "findings_blocking": len(self.blocking),
            "usage": self.usage,
            "calls": self.calls,
            "notes": self.notes,
        }


@dataclass
class ArticleOutcome:
    rules: list[PolicyRule]
    findings: list[Finding]
    calls: list[dict]
    stats: dict
    injection_notes: list[str]


def extract_and_verify(
    llm: LLM,
    article: Article,
    *,
    previous_rules: Optional[list[PolicyRule]] = None,
    use_cache: bool = True,
    repair: bool = True,
) -> ArticleOutcome:
    """Extractor + per-article checks (a) and (b) for one Active article.

    If the checker finds unsupported rules, the extractor gets one more try
    with the checker's problems as feedback, and the checker checks again.
    The checker itself never edits a rule: what still fails is dropped.
    Raises LLMError only if the first extraction call fails.
    """
    aid = article.article_id
    arts = {aid: article}
    ex = extract_article(llm, article, previous_rules=previous_rules, use_cache=use_cache)
    calls = list(ex.metas) if ex.origin == "llm" else []
    kept, f_quotes = check_quotes(ex.rules, arts)
    sup = check_support(llm, article, kept, use_cache=use_cache)
    calls += sup.metas if sup.origin == "llm" else []
    stats = {
        "origin": "extracted" if ex.origin == "llm" else "cache",
        "extracted_rules": len(ex.rules),
        "retried_for_quotes": ex.retried,
        "first_quotes_total": ex.first_quotes_total,
        "first_quotes_verbatim": ex.first_quotes_verbatim,
        "support_check": sup.origin,
        "unsupported_first_pass": [f.rule_id for f in sup.findings if f.kind == "unsupported_rule"],
        "re_extracted": False,
    }
    notes = list(ex.injection_notes)
    findings = f_quotes + sup.findings
    rules = sup.kept
    rejected = [f for f in f_quotes + sup.findings if f.kind in ("unsupported_rule", "unsupported_quote")]
    if rejected and repair:
        feedback = [f"{f.rule_id}: {f.explanation}" for f in rejected]
        try:
            ex2 = extract_article(
                llm, article, previous_rules=previous_rules, checker_feedback=feedback, previous_attempt=ex.rules, use_cache=use_cache
            )
        except LLMError as e:
            calls.append({"call": f"policy_reextract:{aid}", "error": e.to_dict()})
            stats["re_extract_error"] = e.kind
        else:
            calls += ex2.metas if ex2.origin == "llm" else []
            kept2, f_quotes2 = check_quotes(ex2.rules, arts)
            sup2 = check_support(llm, article, kept2, use_cache=use_cache)
            calls += sup2.metas if sup2.origin == "llm" else []
            history = [
                f.model_copy(update={"action": "re_extracted", "blocking": False, "explanation": "first pass: " + f.explanation})
                for f in rejected
            ]
            findings = history + f_quotes2 + sup2.findings
            rules = sup2.kept
            notes = list(dict.fromkeys(notes + ex2.injection_notes))
            stats.update(re_extracted=True, extracted_rules_second_pass=len(ex2.rules))
    stats["dropped_quote"] = sum(1 for f in findings if f.kind == "unsupported_quote" and f.action == "dropped")
    stats["dropped_unsupported"] = sum(1 for f in findings if f.kind == "unsupported_rule" and f.action == "dropped")
    return ArticleOutcome(rules, findings, calls, stats, notes)


def compile_policy(
    kb_set: KBSetLike = "base",
    *,
    previous=_LIVE,
    llm: Optional[LLM] = None,
    use_cache: bool = True,
    reuse_live: bool = True,
    anchor: bool = True,
    table_review: bool = True,
    write: bool = True,
    out_dir: Optional[Path] = None,
    root: Optional[Path] = None,
    workers: int = 4,
    name: Optional[str] = None,
    repair: bool = True,
) -> CompileResult:
    """Compile a candidate policy table for a KB set. See the module docstring."""
    src = resolve_source(kb_set, name=name)
    articles = load_articles(src)
    if previous is _LIVE:
        previous = load_live(src.name, root=root)
    changes = detect_changes(articles, previous)
    llm = llm or LLM()
    usage_before = llm.usage.as_dict()

    prev_by_article: dict[str, list[PolicyRule]] = {}
    for r in previous.rules if previous else []:
        prev_by_article.setdefault(r.article_id, []).append(r)

    findings: list[Finding] = []
    stats: dict[str, dict] = {}
    calls: list[dict] = []
    notes: list[str] = []
    rules_by_article: dict[str, list[PolicyRule]] = {}
    hashes: dict[str, str] = {}
    inactive: list[str] = []
    kept_previous: list[str] = []
    fresh: set[str] = set()
    injection_notes: dict[str, list[str]] = {}
    to_extract: list[Article] = []

    for a in articles:
        if not a.active:
            inactive.append(a.article_id)
            hashes[a.article_id] = a.sha256
            stats[a.article_id] = {"origin": "inactive", "status": a.status, "superseded_by": a.superseded_by}
            continue
        if reuse_live and previous is not None and a.article_id in changes.unchanged and prev_by_article.get(a.article_id):
            rules_by_article[a.article_id] = [r.model_copy(deep=True) for r in prev_by_article[a.article_id]]
            hashes[a.article_id] = a.sha256
            stats[a.article_id] = {"origin": "reused_live", "rules": len(rules_by_article[a.article_id])}
            continue
        to_extract.append(a)

    def _work(a: Article) -> ArticleOutcome:
        ref = prev_by_article.get(a.article_id) if anchor else None
        return extract_and_verify(llm, a, previous_rules=ref or None, use_cache=use_cache, repair=repair)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [(a, pool.submit(_work, a)) for a in to_extract]
        for a, fut in futures:
            aid = a.article_id
            try:
                out = fut.result()
            except Exception as e:  # LLMError or anything unexpected: never crash the compile
                err = e.to_dict() if isinstance(e, LLMError) else {"kind": "internal", "message": repr(e), "trace": traceback.format_exc(limit=3)}
                calls.append({"call": f"policy_extract:{aid}", "error": err})
                kept = prev_by_article.get(aid)
                if kept and previous is not None and aid in previous.kb_hashes:
                    rules_by_article[aid] = [r.model_copy(deep=True) for r in kept]
                    hashes[aid] = previous.kb_hashes[aid]  # old hash: the next compile retries this article
                    kept_previous.append(aid)
                findings.append(
                    Finding(
                        kind="extraction_failed",
                        article_id=aid,
                        explanation=f"extraction failed ({err.get('kind')}); "
                        + (f"the {len(kept)} rule(s) of the live version are kept" if kept else "the article has no rules in this candidate"),
                        action="kept_previous" if kept else "flagged",
                        blocking=True,
                        detail={"error": err},
                    )
                )
                stats[aid] = {"origin": "failed", "kept_previous": bool(kept)}
                continue
            rules_by_article[aid] = out.rules
            hashes[aid] = a.sha256
            fresh.add(aid)
            findings += out.findings
            calls += out.calls
            stats[aid] = {**out.stats, "anchored": bool(anchor and prev_by_article.get(aid))}
            if out.injection_notes:
                injection_notes[aid] = out.injection_notes
                findings.append(
                    Finding(
                        kind="suspected_injection",
                        article_id=aid,
                        explanation="the article contains text addressed to AI agents, not service-desk policy; it was "
                        "not turned into rules: " + " | ".join(out.injection_notes),
                        blocking=True,
                        detail={"notes": out.injection_notes},
                    )
                )

    run_table_review = table_review
    if table_review and previous is not None and reuse_live and not changes.has_changes and not fresh:
        run_table_review = False
        notes.append("No KB changes: whole-table review skipped (the live version was already reviewed).")

    all_rules = [r for aid in sorted(rules_by_article) for r in rules_by_article[aid]]
    check = run_checker(
        llm,
        articles,
        all_rules,
        previous,
        support_for=set(),  # per-article checks (a, b) already ran in extract_and_verify
        skip_quote_check=set(kept_previous),
        table_review=run_table_review,
        focus=changes.touched,
        notes=injection_notes,
        use_cache=use_cache,
        workers=workers,
    )
    findings += check.findings
    calls += check.metas
    for aid, st in check.stats.items():
        stats.setdefault(aid, {}).update(st)

    candidate = PolicyTable(
        version=(previous.version + 1) if previous else 1,
        compiled_at=now_iso(),
        source="extracted",
        kb_hashes=dict(sorted(hashes.items())),
        inactive_articles=sorted(inactive),
        rules=check.rules,
    )
    for aid in rules_by_article:
        stats.setdefault(aid, {})["final_rules"] = sum(1 for r in candidate.rules if r.article_id == aid)

    diff = diff_tables(previous, candidate, kept_previous)
    notes.insert(0, f"KB set: {src.name} ({', '.join(str(Path(d).name) for d in src.dirs)}"
                 + (f"; removed: {', '.join(src.remove)}" if src.remove else "") + ")")
    notes.insert(1, f"KB changes since v{previous.version if previous else '-'}: {changes.summary()}")
    diff_md = diff.to_markdown(title=f"Policy diff: {src.name}", findings=findings, notes=notes)
    result = CompileResult(
        kb_set=src.name,
        candidate=candidate,
        findings=findings,
        diff=diff,
        diff_md=diff_md,
        changes=changes,
        previous_version=previous.version if previous else None,
        article_stats=dict(sorted(stats.items())),
        usage=usage_delta(usage_before, llm.usage.as_dict()),
        calls=calls,
        notes=notes,
    )
    if write:
        d = Path(out_dir) if out_dir else policy_dir(src.name, root)
        d.mkdir(parents=True, exist_ok=True)
        _write(d / "candidate.json", candidate.model_dump_json(indent=2))
        _write_json(
            d / "findings.json",
            {
                "kb_set": src.name,
                "candidate_version": candidate.version,
                "previous_version": result.previous_version,
                "compiled_at": candidate.compiled_at,
                "blocking": len(result.blocking),
                "findings": [f.model_dump() for f in findings],
            },
        )
        _write(d / "diff.md", diff_md)
        _write_json(d / "report.json", result.report())
        # The article files this candidate was compiled from: approval keeps them with the version,
        # and requests read these, not the KB folder (see article_dirs).
        _save_articles(d / "articles" / "candidate", candidate, articles, _previous_copy(src.name, previous, root))
        result.paths = {n: str(d / n) for n in CANDIDATE_FILES}
        _audit(d, "policy_compile", candidate_version=candidate.version, previous_version=result.previous_version,
               changes=changes.to_dict(), findings=len(findings), blocking=len(result.blocking), usage=result.usage, calls=calls)
    return result


compile = compile_policy  # the name used in the spec; shadows the builtin only inside this module


# --------------------------------------------------------------------------
# approve / rollback
# --------------------------------------------------------------------------


def approve(
    kb_set: str,
    *,
    approved_by: str = "reviewer",
    acknowledge_findings: bool = False,
    root: Optional[Path] = None,
) -> PolicyTable:
    """Promote the pending candidate to a new live version (earlier versions are kept), together
    with the copy of the articles it was compiled from."""
    d = policy_dir(kb_set, root)
    pending = load_candidate(kb_set, root=root)
    if pending is None:
        raise PolicyApprovalError(f"{kb_set}: no candidate to approve; compile first")
    cand, findings = pending
    report = json.loads((d / "report.json").read_text(encoding="utf-8")) if (d / "report.json").exists() else {}
    live = load_live(kb_set, root=root)
    live_v = live.version if live else None
    if "previous_version" in report and report["previous_version"] != live_v:
        raise PolicyApprovalError(
            f"{kb_set}: the candidate was compiled against v{report['previous_version']} but live is now v{live_v}; recompile"
        )
    blocking = [f for f in findings if f.blocking]
    if blocking and not acknowledge_findings:
        listing = "; ".join(f"{f.kind} {f.rule_id or f.article_id or 'table'}" for f in blocking)
        raise PolicyApprovalError(
            f"{kb_set}: {len(blocking)} finding(s) need a reviewer decision ({listing}). Review findings.json/diff.md, "
            "then approve with acknowledge_findings=True"
        )
    new_v = max(list_versions(kb_set, root=root) + [0]) + 1
    if cand.version != new_v:
        cand = cand.model_copy(update={"version": new_v})
    _write(d / f"v{new_v}.json", cand.model_dump_json(indent=2))
    for src_name, dst_name in (("findings.json", "findings.json"), ("diff.md", "diff.md"), ("report.json", "report.json")):
        if (d / src_name).exists():
            shutil.copyfile(d / src_name, d / f"v{new_v}.{dst_name}")
    # Keep the articles the candidate was compiled from with the version, before it goes live.
    copy, pinned = articles_dir(kb_set, None, root=root), articles_dir(kb_set, new_v, root=root)
    shutil.rmtree(pinned, ignore_errors=True)
    if _copy_matches(copy, cand):
        copy.rename(pinned)
    shutil.rmtree(copy, ignore_errors=True)
    pointer = {
        "version": new_v,
        "file": f"v{new_v}.json",
        "approved_at": now_iso(),
        "approved_by": approved_by,
        "previous_live": live_v,
        "acknowledged_findings": [
            {"kind": f.kind, "article_id": f.article_id, "rule_id": f.rule_id, "explanation": f.explanation} for f in blocking
        ],
    }
    _write_json(d / "live.json", pointer)
    with open(d / "history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "approve", **pointer}, ensure_ascii=False) + "\n")
    for n in CANDIDATE_FILES:
        (d / n).unlink(missing_ok=True)
    _audit(d, "policy_approve", version=new_v, approved_by=approved_by, acknowledged=len(blocking))
    return cand


def rollback(kb_set: str, to_version: Optional[int] = None, *, by: str = "reviewer", root: Optional[Path] = None) -> PolicyTable:
    """Point live at an earlier approved version (default: the highest version below live)."""
    d = policy_dir(kb_set, root)
    ptr = live_pointer(kb_set, root=root)
    if not ptr:
        raise PolicyApprovalError(f"{kb_set}: nothing is live; nothing to roll back")
    live_v = int(ptr["version"])
    versions = list_versions(kb_set, root=root)
    if to_version is None:
        older = [v for v in versions if v < live_v]
        if not older:
            raise PolicyApprovalError(f"{kb_set}: no version older than live v{live_v}")
        to_version = max(older)
    if to_version not in versions:
        raise PolicyApprovalError(f"{kb_set}: version {to_version} does not exist (have {versions})")
    table = load_version(kb_set, to_version, root=root)
    pointer = {
        "version": to_version,
        "file": f"v{to_version}.json",
        "approved_at": now_iso(),
        "approved_by": by,
        "previous_live": live_v,
        "rolled_back_from": live_v,
        "acknowledged_findings": [],
    }
    _write_json(d / "live.json", pointer)
    with open(d / "history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "rollback", **pointer}, ensure_ascii=False) + "\n")
    _audit(d, "policy_rollback", from_version=live_v, to_version=to_version, by=by)
    return table


# --------------------------------------------------------------------------
# automatic recompile: keep the live policy in step with the KB
# --------------------------------------------------------------------------

AUTO_APPROVER = "auto (checker found nothing blocking)"
PERSON_APPROVER = "reviewer (accepted the checker's findings)"
_ensure_lock = threading.RLock()  # the web app can ask from several threads at once


@dataclass
class PolicyUpdate:
    """What ensure_current did. `table` is the policy runs should use (None: nothing approved yet)."""

    kb_set: str
    table: Optional[PolicyTable]
    action: str  # "unchanged" | "approved" | "held"
    changes: Optional[KBChanges] = None
    result: Optional[CompileResult] = None
    blocking: list[Finding] = field(default_factory=list)
    approved_by: Optional[str] = None


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _diff_lines(diff: PolicyDiff, limit: int = 8) -> list[str]:
    lines = [f"+ {r.rule_id}" for r in diff.added]
    lines += [f"- {r.rule_id}" for r in diff.removed]
    lines += [f"~ {ch.new_rule_id}: {', '.join(ch.changes) or 'changed'}" for ch in diff.changed]
    return lines[:limit] + ([f"... {len(lines) - limit} more in diff.md"] if len(lines) > limit else [])


def ensure_current(
    kb_set: KBSetLike,
    *,
    name: Optional[str] = None,
    llm: Optional[LLM] = None,
    root: Optional[Path] = None,
    use_cache: bool = True,
    confirm: Optional[Callable[[str], bool]] = None,
    log=print,
) -> PolicyUpdate:
    """Keep the live policy in step with the KB, so nobody has to recompile by hand.

    The watcher compares each article file's sha256 with the hashes the live version was
    compiled from. No change: the live table is returned with no model calls. An added,
    edited, deprecated or deleted article: the compiler runs (only new or changed
    articles go to the model), and the candidate goes live automatically when the
    checker finds nothing blocking. A blocking finding (a removed or weakened safeguard,
    a conflict, suspected injected text, a failed extraction) holds the change for a
    person: runs keep the last approved version and its copy of the articles (article_dirs),
    so the held change has no effect on requests, and an unchanged KB is not compiled
    again until the article is fixed or someone approves with --acknowledge-findings.

    `confirm` lets a person who is present (the CLI at a terminal) accept a held change
    on the spot: it gets the question and returns True to approve with the findings.
    """
    src = resolve_source(kb_set, name=name)
    d = policy_dir(src, root)
    with _ensure_lock:
        live = load_live(src.name, root=root)
        changes = detect_changes(load_articles(src), live)
        if live is not None and not changes.has_changes:
            return PolicyUpdate(src.name, live, "unchanged", changes)

        def held(blocking: list[Finding], result: Optional[CompileResult] = None) -> PolicyUpdate:
            log(f"Policy change for '{src.name}' held for review: {len(blocking)} blocking checker finding(s):")
            for f in blocking[:6]:
                expl = f.explanation if len(f.explanation) <= 200 else f.explanation[:197] + "..."
                log(f"  - {f.kind} ({f.rule_id or f.article_id or 'table'}): {expl}")
            if len(blocking) > 6:
                log(f"  ... {len(blocking) - 6} more in findings.json")
            if confirm is not None and confirm(f"Accept {'this finding' if len(blocking) == 1 else 'these findings'} "
                                               f"and make the recompiled '{src.name}' policy live?"):
                table = approve(src.name, approved_by=PERSON_APPROVER, acknowledge_findings=True, root=root)
                log(f"  Approved by you: '{src.name}' policy v{table.version} is live.")
                return PolicyUpdate(src.name, table, "approved", changes, result, blocking, PERSON_APPROVER)
            log("  Runs keep using " + (f"the approved policy v{live.version} and the articles as they were when it was "
                                        "approved: the change has no effect on requests until a person accepts it."
                                        if live else "the hand-checked policy until one is approved."))
            log(f"  Review {_rel(d / 'diff.md')} and findings.json, then fix the article, or accept the change with: "
                f"python -m servicedesk.cli policy approve --kb {src.name} --acknowledge-findings")
            return PolicyUpdate(src.name, live, "held", changes, result, blocking)

        pending = load_candidate(src.name, root=root)
        if pending is not None and pending[0].kb_hashes == changes.hashes:
            blocking = [f for f in pending[1] if f.blocking]
            if blocking:  # already compiled from this exact KB and held: don't spend calls again
                return held(blocking)

        if live is None:
            log(f"No approved policy for '{src.name}' yet: compiling it from the KB articles...")
        else:
            log(f"The KB changed since policy v{live.version} for '{src.name}' ({changes.summary()}): recompiling...")
        result = compile_policy(src, llm=llm, root=root, use_cache=use_cache)
        c = result.diff.counts()
        log(f"  Rules: +{c['added']} added, -{c['removed']} removed, ~{c['changed']} changed; "
            f"checker findings: {len(result.findings)} ({len(result.blocking)} blocking); "
            f"{result.usage.get('calls', 0)} model calls.")
        for line in _diff_lines(result.diff):
            log(f"    {line}")
        if result.blocking:
            return held(result.blocking, result)
        table = approve(src.name, approved_by=AUTO_APPROVER, root=root)
        log(f"  Approved automatically (nothing blocking): '{src.name}' policy v{table.version} is live. "
            f"Diff: {_rel(d / f'v{table.version}.diff.md')}")
        return PolicyUpdate(src.name, table, "approved", changes, result, approved_by=AUTO_APPROVER)


# --------------------------------------------------------------------------
# helpers for the CLI, UI, pipeline and evals
# --------------------------------------------------------------------------


def status(kb_set: str, *, root: Optional[Path] = None) -> dict:
    live = load_live(kb_set, root=root)
    pending = load_candidate(kb_set, root=root)
    try:
        changes = detect_changes(load_articles(kb_set), live).to_dict()
    except (ValueError, FileNotFoundError) as e:
        changes = {"error": str(e)}
    return {
        "kb_set": kb_set,
        "live_version": live.version if live else None,
        "live_rules": len(live.rules) if live else 0,
        "versions": list_versions(kb_set, root=root),
        "pending_candidate": pending[0].version if pending else None,
        "pending_blocking_findings": sum(1 for f in pending[1] if f.blocking) if pending else 0,
        "kb_changes_since_live": changes,
    }


def compare(kb_set_a: str, kb_set_b: str, *, root: Optional[Path] = None) -> PolicyDiff:
    """Diff the live tables of two KB sets (e.g. base vs with-kb013)."""
    a, b = load_live(kb_set_a, root=root), load_live(kb_set_b, root=root)
    if b is None:
        raise FileNotFoundError(f"{kb_set_b}: no live policy version")
    return diff_tables(a, b)


def load_gold(kb_set: KBSetLike = "with-kb013", *, gold_file: Optional[Path] = None) -> PolicyTable:
    """The hand-checked policy table, filtered to the KB set's articles (base excludes KB-013)."""
    g = PolicyTable.model_validate_json(Path(gold_file or GOLD_FILE).read_text(encoding="utf-8"))
    ids = {a.article_id for a in load_articles(kb_set)}
    return g.model_copy(
        update={
            "rules": [r for r in g.rules if r.article_id in ids],
            "kb_hashes": {k: v for k, v in g.kb_hashes.items() if k in ids},
            "inactive_articles": [a for a in g.inactive_articles if a in ids],
        }
    )


def get_policy(kb_set: str, source: str = "live", *, root: Optional[Path] = None) -> PolicyTable:
    """The table the request pipeline should use: the live extracted version, or gold."""
    if source == "gold":
        return load_gold(kb_set)
    if source != "live":
        raise ValueError("source must be 'live' or 'gold'")
    table = load_live(kb_set, root=root)
    if table is None:
        raise FileNotFoundError(f"{kb_set}: no approved policy yet; run compile and approve (or use source='gold')")
    return table


def article_dirs(kb_set: KBSetLike, table: Optional[PolicyTable], *, name: Optional[str] = None,
                 root: Optional[Path] = None, log=None) -> list[Path]:
    """The article folders the request pipeline loads together with `table`.

    For an approved (extracted) version: the copy of the articles it was compiled from, so
    search, the rule picker and each article's Status match its rules even while a later KB
    edit is held for review. The hand-checked gold table goes with the KB folders as they are.
    A version approved before copies were kept gets one now if the KB still matches it;
    otherwise the KB folders are used and `log` says so.
    """
    src = resolve_source(kb_set, name=name)
    if table is None or table.source != "extracted":
        return list(src.dirs)
    pinned = articles_dir(src.name, table.version, root=root)
    if _copy_matches(pinned, table):
        return [pinned]
    with _ensure_lock:
        if _copy_matches(pinned, table) or _save_articles(pinned, table, load_articles(src)):
            return [pinned]
    if log:
        log(f"Note: policy v{table.version} for '{src.name}' has no saved copy of the articles it was compiled "
            "from, and the KB has changed since; runs read the articles as they are now.")
    return list(src.dirs)
