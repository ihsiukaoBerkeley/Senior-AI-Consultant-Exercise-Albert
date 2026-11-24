"""Your own cases (data/custom_requests.jsonl, run --requests) and the API-key check. No API calls."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from servicedesk.config import CUSTOM_FILE, DIRECTORY_FILE, assumptions
from servicedesk.pipeline.emails import load_directory
from servicedesk.pipeline.inputs import CUSTOM_REQUESTER, RequestFileError, load_request_file


def write_lines(path, *rows):
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_minimal_case_gets_defaults_and_reply(tmp_path):
    f = write_lines(tmp_path / "cases.jsonl",
                    {"request_id": "CUS-001", "message": "Teams crashes when I share my screen."},
                    "",
                    {"request_id": "CUS-002", "message": "Email is down in Tokyo.", "channel": "chat",
                     "received_at": "2026-09-05T08:45:00Z", "requester_id": "REQ-101", "reply": "  Since 08:30.  "})
    reqs, replies = load_request_file(f)
    assert [r.request_id for r in reqs] == ["CUS-001", "CUS-002"]
    a, b = reqs
    assert (a.channel, a.requester_id, a.received_at, a.synthetic) == (
        "web", CUSTOM_REQUESTER, assumptions()["scenario_now"], True)
    assert (b.channel, b.requester_id, b.received_at) == ("chat", "REQ-101", "2026-09-05T08:45:00Z")
    assert replies == {"CUS-002": "Since 08:30."}


@pytest.mark.parametrize("line, needle", [
    ('{"request_id": "CUS-1", "message": "x"', "not valid JSON"),
    ('["CUS-1", "x"]', "expected a JSON object"),
    ('{"request_id": "CUS-1"}', "are required"),
    ('{"request_id": "CUS-1", "mesage": "x"}', "unknown field(s) mesage"),
    ('{"request_id": "NEW-001", "message": "x"}', "built-in request id"),
    ('{"request_id": "CUS 1", "message": "x"}', "may use only"),
    ('{"request_id": "CUS-1", "message": "x", "channel": "fax"}', "channel must be one of"),
    ('{"request_id": "CUS-1", "message": "x", "received_at": "yesterday"}', "not an ISO time"),
])
def test_bad_lines_name_the_line_and_the_fix(tmp_path, line, needle):
    f = write_lines(tmp_path / "cases.jsonl", {"request_id": "CUS-0", "message": "fine"}, line)
    with pytest.raises(RequestFileError) as e:
        load_request_file(f)
    assert "cases.jsonl line 2" in str(e.value) and needle in str(e.value)


def test_duplicate_ids_and_missing_file_are_rejected(tmp_path):
    f = write_lines(tmp_path / "cases.jsonl", {"request_id": "CUS-1", "message": "a"},
                    {"request_id": "CUS-1", "message": "b"})
    with pytest.raises(RequestFileError, match="duplicate request_id CUS-1"):
        load_request_file(f)
    with pytest.raises(RequestFileError, match="does not exist"):
        load_request_file(tmp_path / "missing.jsonl")


def test_shipped_example_file_is_valid_and_its_requester_exists():
    reqs, replies = load_request_file(CUSTOM_FILE)
    assert reqs and all(r.request_id.startswith("CUS-") for r in reqs)
    assert set(replies) <= {r.request_id for r in reqs}
    assert CUSTOM_REQUESTER in load_directory(DIRECTORY_FILE)


def _fake_run_setup(monkeypatch, tmp_path):
    import servicedesk.pipeline as pipeline_pkg
    from servicedesk.evals import request_eval
    from servicedesk.llm import Usage
    from servicedesk.models import IssueOutcome, RequestOutcome

    resumed = {}

    class FakePipeline:
        def __init__(self, policy, kb_dirs, stores, llm=None, retrieval=None):
            from servicedesk.retrieval import check_mode

            self.stores = stores
            self.retriever = SimpleNamespace(mode=check_mode(retrieval), articles=[], vector=None, embed_model=None)

        def run_batch(self, batch, workers=4):
            out = []
            for r in batch:
                o = RequestOutcome(request_id=r.request_id, status="awaiting_info",
                                   issues=[IssueOutcome(issue_index=1, summary="s", lane="awaiting_info")])
                self.stores.save_state(r.request_id, {"outcome": o.model_dump()})
                out.append(o)
            return out

        def resume(self, rid, text):
            resumed[rid] = text
            return RequestOutcome(request_id=rid, status="done",
                                  issues=[IssueOutcome(issue_index=1, summary="s", lane="ticket_routed")])

    class FakeLLM:
        usage = Usage()

    monkeypatch.setattr(pipeline_pkg, "Pipeline", FakePipeline)
    monkeypatch.setattr(request_eval, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(request_eval, "load_replies", lambda: {"NEW-001": "built-in reply"})
    return request_eval, FakeLLM, resumed


def test_run_with_request_file_processes_only_those_cases_and_uses_their_replies(tmp_path, monkeypatch):
    request_eval, FakeLLM, resumed = _fake_run_setup(monkeypatch, tmp_path)
    f = write_lines(tmp_path / "cases.jsonl",
                    {"request_id": "CUS-001", "message": "a", "reply": "my laptop is LT-1"},
                    {"request_id": "CUS-002", "message": "b"})
    snap = request_eval.run_requests("with-kb013", "mine", policy_source="gold", llm=FakeLLM(),
                                     log=lambda _m: None, request_file=f)
    assert set(snap["requests"]) == {"CUS-001", "CUS-002"}
    assert resumed == {"CUS-001": "my laptop is LT-1"}
    assert snap["requests"]["CUS-001"]["reply_applied"] is True
    assert snap["requests"]["CUS-002"]["reply_applied"] is False  # no reply given: stays paused


def test_bad_request_file_leaves_the_previous_run_intact(tmp_path, monkeypatch):
    request_eval, FakeLLM, _ = _fake_run_setup(monkeypatch, tmp_path)
    previous = tmp_path / "runs" / "mine" / "keep.txt"
    previous.parent.mkdir(parents=True)
    previous.write_text("earlier run")
    bad = write_lines(tmp_path / "cases.jsonl", '{"request_id": "CUS-1"}')
    with pytest.raises(RequestFileError):
        request_eval.run_requests("with-kb013", "mine", policy_source="gold", llm=FakeLLM(),
                                  log=lambda _m: None, request_file=bad)
    assert previous.read_text() == "earlier run"


def test_cli_reports_a_bad_request_file_without_a_traceback(tmp_path, monkeypatch, capsys):
    from servicedesk import cli

    request_eval, _, _ = _fake_run_setup(monkeypatch, tmp_path)
    bad = write_lines(tmp_path / "cases.jsonl", '{"request_id": "CUS-1", "message": "x", "channel": "fax"}')
    code = cli.main(["run", "--kb", "with-kb013", "--name", "mine", "--policy", "gold", "--requests", str(bad)])
    assert code == 2
    assert "Error: cases.jsonl line 1: channel must be one of" in capsys.readouterr().err


@pytest.mark.parametrize("value", [None, "", "your-anthropic-api-key"])
def test_missing_or_placeholder_key_fails_at_start_with_a_clear_message(monkeypatch, value):
    from servicedesk.llm import LLM, LLMError

    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    if value is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    with pytest.raises(LLMError) as e:
        LLM()
    assert e.value.kind == "auth" and ".env" in str(e.value)


def test_a_configured_key_builds_the_client_without_calling_the_api(monkeypatch):
    from servicedesk.llm import LLM

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    assert LLM().client is not None


@pytest.mark.parametrize("given, expected", [
    ("data owner", "the data owner"), ("manager", "the manager"), ("the group owner", "the group owner"),
    ("your manager", "your manager"), ("Security Operations", "Security Operations"), ("", "the approver"),
])
def test_approver_reads_as_a_phrase_in_the_confirmation(given, expected):
    from servicedesk.pipeline.emails import approver_phrase

    assert approver_phrase(given) == expected
