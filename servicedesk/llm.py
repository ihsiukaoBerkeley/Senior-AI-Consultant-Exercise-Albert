"""One place where the app talks to the Claude API.

Every model call in the app is a single structured-output request. This
module turns every way a call can go wrong into an `LLMError` with a `kind`,
so callers can fail safe (hand the case to a person) instead of crashing:

* The request errored (non-200). The SDK already retries 408/409/429/5xx and
  connection errors with backoff; whatever is left is mapped by error type.
* The request succeeded (200) but stopped early: `refusal` (after the
  server-side fallback also declined), `max_tokens` (retried once with a
  bigger budget), `model_context_window_exceeded`, or an unexpected stop.
* The output did not validate against the schema.

The request id is kept on every result and error so a failure can be traced.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Generic, Optional, Type, TypeVar

import anthropic
import pydantic
from pydantic import BaseModel

from .config import FALLBACK_BETA, model_for

T = TypeVar("T", bound=BaseModel)


def agent_of(call: str) -> str:
    """The agent a call name belongs to: 'screening:reply' -> 'screening',
    'policy_reextract:KB-007' -> 'extractor'."""
    name = call.split(":", 1)[0]
    if name.startswith(("policy_extract", "policy_reextract")):
        return "extractor"
    if name.startswith("policy_check"):
        return "checker"
    if name.startswith("policy_eval_judge"):
        return "judge"
    return name

RETRYABLE_KINDS = {"rate_limited", "overloaded", "server", "timeout", "connection"}


class LLMError(Exception):
    """A model call that produced no usable output."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        call: str = "",
        error_type: Optional[str] = None,
        status: Optional[int] = None,
        request_id: Optional[str] = None,
        stop_reason: Optional[str] = None,
        category: Optional[str] = None,
    ):
        super().__init__(f"[{kind}] {call}: {message}")
        self.kind = kind
        self.call = call
        self.error_type = error_type
        self.status = status
        self.request_id = request_id
        self.stop_reason = stop_reason
        self.category = category

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_KINDS

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "call": self.call,
            "message": str(self),
            "error_type": self.error_type,
            "status": self.status,
            "request_id": self.request_id,
            "stop_reason": self.stop_reason,
            "category": self.category,
        }


@dataclass
class LLMResult(Generic[T]):
    parsed: T
    call: str
    request_id: Optional[str]
    model: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    fallback_used: bool = False
    cache_creation_input_tokens: int = 0  # prompt tokens written to the cache (billed, not in input_tokens)
    cache_read_input_tokens: int = 0  # prompt tokens read from the cache (billed, not in input_tokens)

    def meta(self) -> dict:
        return {
            "call": self.call,
            "request_id": self.request_id,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": round(self.latency_s, 2),
            "fallback_used": self.fallback_used,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }


@dataclass
class Usage:
    calls: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    by_model: dict = field(default_factory=dict)  # model -> token counts, for pricing
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")

    def add(self, result: Optional[LLMResult] = None, error: bool = False) -> None:
        with self._lock:
            self.calls += 1
            if error:
                self.errors += 1
            if result:
                per = self.by_model.setdefault(result.model, {"calls": 0, **{k: 0 for k in self.TOKEN_KEYS}})
                per["calls"] += 1
                for k in self.TOKEN_KEYS:
                    setattr(self, k, getattr(self, k) + getattr(result, k))
                    per[k] += getattr(result, k)

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "by_model": {m: dict(v) for m, v in self.by_model.items()},
        }


def _status_error(e: anthropic.APIStatusError, call: str) -> LLMError:
    """Map a non-200 response to an LLMError, most specific class first."""
    common = dict(
        call=call,
        error_type=getattr(e, "type", None),
        status=e.status_code,
        request_id=getattr(e, "request_id", None),
    )
    if isinstance(e, anthropic.AuthenticationError):
        return LLMError("auth", "invalid or missing API key", **common)
    if isinstance(e, anthropic.PermissionDeniedError):
        return LLMError("permission", e.message, **common)
    if isinstance(e, anthropic.NotFoundError):
        return LLMError("not_found", f"model or endpoint not found: {e.message}", **common)
    if isinstance(e, anthropic.RateLimitError):
        return LLMError("rate_limited", e.message, **common)
    body = e.body if isinstance(e.body, dict) else {}
    detail = (body.get("error") or {}).get("message") or ""
    if "credit balance is too low" in f"{e.message} {detail}".lower():  # sent as a 400 invalid_request_error
        return LLMError("billing", detail or e.message, **common)
    if isinstance(e, anthropic.BadRequestError):
        return LLMError("bad_request", e.message, **common)
    status, etype = e.status_code, getattr(e, "type", None)
    if status == 402 or etype == "billing_error":
        return LLMError("billing", e.message, **common)
    if status == 413 or etype == "request_too_large":
        return LLMError("too_large", e.message, **common)
    if status == 529 or etype == "overloaded_error":
        return LLMError("overloaded", e.message, **common)
    if status == 504 or etype == "timeout_error":
        return LLMError("timeout", e.message, **common)
    if status >= 500:
        return LLMError("server", e.message, **common)
    return LLMError("api_error", e.message, **common)


def usage_delta(before: dict, after: dict) -> dict:
    """The usage between two Usage.as_dict() snapshots, per-model counts included."""
    out = {k: v - (before.get(k) or 0) for k, v in after.items() if k != "by_model"}
    old = before.get("by_model") or {}
    per = {m: {k: v - (old.get(m, {}).get(k) or 0) for k, v in counts.items()}
           for m, counts in (after.get("by_model") or {}).items()}
    out["by_model"] = {m: c for m, c in per.items() if c["calls"]}
    return out


PLACEHOLDER_KEYS = {"", "your-anthropic-api-key"}


def check_credentials() -> None:
    """Fail at start-up, with a clear message, when no API key is configured.

    Without this the SDK only fails at the first request, and the pipeline would hand
    every request to staff as an internal error.
    """
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if key in PLACEHOLDER_KEYS and not (os.environ.get("ANTHROPIC_AUTH_TOKEN") or "").strip():
        raise LLMError("auth", "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and put your key in it.",
                       call="init")


class LLM:
    """Thin wrapper: structured output, server-side refusal fallback, error mapping.

    Each call goes to its agent's model (config.AGENT_MODELS, agent taken from the call
    name); `model` puts every call on one model instead.
    """

    def __init__(self, model: Optional[str] = None, max_retries: int = 4, timeout: float = 180.0):
        check_credentials()
        self.override = model
        # The SDK retries connection errors, 408, 409, 429 and >=500 with backoff.
        self.client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout)
        self.usage = Usage()

    def model_for(self, agent: str) -> str:
        return self.override or model_for(agent)

    def structured(
        self,
        *,
        call: str,
        system: str,
        user: str,
        schema: Type[T],
        effort: str = "low",
        max_tokens: int = 8000,
    ) -> LLMResult[T]:
        budget = max_tokens
        model = self.model_for(agent_of(call))
        for attempt in range(2):
            started = time.monotonic()
            try:
                resp = self.client.beta.messages.parse(
                    model=model,
                    max_tokens=budget,
                    betas=[FALLBACK_BETA],
                    fallbacks="default",  # re-run a policy decline on Anthropic's recommended model
                    system=system,
                    output_config={"effort": effort},
                    output_format=schema,
                    cache_control={"type": "ephemeral"},
                    messages=[{"role": "user", "content": user}],
                )
            except anthropic.APIStatusError as e:
                self.usage.add(error=True)
                raise _status_error(e, call) from e
            except anthropic.APITimeoutError as e:  # subclass of APIConnectionError: check first
                self.usage.add(error=True)
                raise LLMError("timeout", "request timed out after retries", call=call) from e
            except anthropic.APIConnectionError as e:
                self.usage.add(error=True)
                raise LLMError("connection", "could not reach the API", call=call) from e
            except pydantic.ValidationError as e:
                self.usage.add(error=True)
                raise LLMError("invalid_output", f"output failed schema validation: {e}", call=call) from e

            request_id = getattr(resp, "_request_id", None)
            stop = resp.stop_reason
            if stop == "refusal":
                self.usage.add(error=True)
                details = getattr(resp, "stop_details", None)
                raise LLMError(
                    "refusal",
                    getattr(details, "explanation", None) or "the model declined",
                    call=call,
                    request_id=request_id,
                    stop_reason=stop,
                    category=getattr(details, "category", None),
                )
            if stop == "max_tokens" and attempt == 0:
                budget *= 2  # output was cut off: try once more with room to finish
                continue
            if stop in ("max_tokens", "model_context_window_exceeded"):
                self.usage.add(error=True)
                raise LLMError("truncated", "output was cut off", call=call, request_id=request_id, stop_reason=stop)
            if stop not in ("end_turn", "stop_sequence"):
                self.usage.add(error=True)
                raise LLMError("unexpected_stop", f"stop_reason={stop}", call=call, request_id=request_id, stop_reason=stop)
            if resp.parsed_output is None:
                self.usage.add(error=True)
                raise LLMError("invalid_output", "no parsed output", call=call, request_id=request_id, stop_reason=stop)

            iterations = getattr(resp.usage, "iterations", None) or []
            result = LLMResult(
                parsed=resp.parsed_output,
                call=call,
                request_id=request_id,
                model=resp.model,
                stop_reason=stop,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                latency_s=time.monotonic() - started,
                fallback_used=any(getattr(i, "type", None) == "fallback_message" for i in iterations),
                cache_creation_input_tokens=getattr(resp.usage, "cache_creation_input_tokens", None) or 0,
                cache_read_input_tokens=getattr(resp.usage, "cache_read_input_tokens", None) or 0,
            )
            self.usage.add(result)
            return result
        raise AssertionError("unreachable")
