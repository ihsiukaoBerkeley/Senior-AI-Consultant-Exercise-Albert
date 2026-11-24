"""Model per agent and pricing per model (no API calls)."""
from __future__ import annotations

import pytest

from servicedesk import config
from servicedesk.evals.request_eval import usage_cost
from servicedesk.llm import LLM, LLMResult, Usage, agent_of


@pytest.mark.parametrize("call, agent", [
    ("screening:request", "screening"),
    ("screening:reply", "screening"),
    ("reading", "reading"),
    ("reply_reading", "reply_reading"),
    ("kb_pick", "kb_pick"),
    ("fill_fields", "fill_fields"),
    ("drafter", "drafter"),
    ("policy_extract:KB-001", "extractor"),
    ("policy_reextract:KB-007", "extractor"),
    ("policy_extract_quote_retry:KB-001", "extractor"),
    ("policy_check_support:KB-001", "checker"),
    ("policy_check_table", "checker"),
    ("policy_eval_judge:KB-001", "judge"),
])
def test_every_call_belongs_to_an_agent_in_the_model_table(call, agent):
    assert agent_of(call) == agent
    assert agent in config.AGENT_MODELS


def test_sd_model_moves_every_agent_and_sd_model_agent_moves_one(monkeypatch):
    for var in ("SD_MODEL", "SD_MODEL_DRAFTER", "SD_MODEL_READING"):
        monkeypatch.delenv(var, raising=False)
    assert config.model_for("drafter") == config.AGENT_MODELS["drafter"]
    monkeypatch.setenv("SD_MODEL", "model-x")
    assert config.model_for("drafter") == "model-x"
    monkeypatch.setenv("SD_MODEL_DRAFTER", "model-y")
    assert config.model_for("drafter") == "model-y"
    assert config.model_for("reading") == "model-x"


def test_a_model_given_to_the_client_is_used_for_every_agent(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    llm = LLM(model="model-z")
    assert {llm.model_for(a) for a in config.AGENT_MODELS} == {"model-z"}


def test_usage_is_counted_and_priced_per_model():
    u = Usage()
    for model in (config.OPUS, config.SONNET):
        u.add(LLMResult(parsed=None, call="drafter", request_id=None, model=model, stop_reason="end_turn",
                        input_tokens=0, output_tokens=1_000_000, latency_s=0.0))
    d = u.as_dict()
    assert d["output_tokens"] == 2_000_000
    assert d["by_model"][config.SONNET] == {"calls": 1, "input_tokens": 0, "output_tokens": 1_000_000,
                                            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    assert usage_cost(d) == pytest.approx(20.0 + 10.0)  # Opus $20 + Sonnet $10 per million output tokens


def test_an_empty_credit_balance_is_a_billing_error_so_runs_stop_with_a_clear_message():
    import anthropic
    import httpx

    from servicedesk.evals.request_eval import FATAL_KINDS
    from servicedesk.llm import _status_error

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": "invalid_request_error", "message":
            "Your credit balance is too low to access the Anthropic API."}}
    err = anthropic.BadRequestError("Error code: 400", response=httpx.Response(400, request=request, json=body),
                                    body=body)
    mapped = _status_error(err, "drafter")
    assert mapped.kind == "billing" and mapped.kind in FATAL_KINDS
