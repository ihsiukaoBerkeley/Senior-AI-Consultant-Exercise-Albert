"""Paths, model settings and working assumptions."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "knowledge-base"            # the 12 provided articles
KB_ADDITIONS_DIR = DATA_DIR / "kb-additions"    # synthetic articles (KB-013), added only in run B
KB_MUTATIONS_DIR = DATA_DIR / "kb-mutations"    # edited KB copies for extractor change tests
GOLD_DIR = DATA_DIR / "gold"                    # hand-checked answer keys
RUNTIME_DIR = ROOT / "runtime"                  # git-ignored outputs: stores, policy versions, reports
POLICY_DIR = RUNTIME_DIR / "policy"

INCOMING_FILE = DATA_DIR / "incoming_requests.jsonl"
SYNTHETIC_FILE = DATA_DIR / "synthetic_requests.jsonl"
CUSTOM_FILE = DATA_DIR / "custom_requests.jsonl"      # cases a user adds (run --requests)
INCIDENTS_FILE = DATA_DIR / "current_incidents.csv"
HISTORY_FILE = DATA_DIR / "historical_tickets.csv"
DIRECTORY_FILE = DATA_DIR / "requester_directory.csv"
REPLIES_FILE = DATA_DIR / "simulated_replies.json"

OPUS = "claude-opus-5-5"
SONNET = "claude-sonnet-5-5"
DEFAULT_MODEL = OPUS

# The Claude model each agent runs on. SD_MODEL puts every agent on one model;
# SD_MODEL_<AGENT> (for example SD_MODEL_DRAFTER=claude-opus-5-5) changes one agent.
AGENT_MODELS = {
    # per request (servicedesk/pipeline/agents.py)
    "screening": OPUS,
    "reading": SONNET,
    "reply_reading": SONNET,
    "kb_pick": OPUS,
    "fill_fields": OPUS,
    "drafter": SONNET,      # code adds any ticket, incident or approval id the draft leaves out
    # policy compile (servicedesk/policy/), only when the KB changes
    "extractor": OPUS,
    "checker": OPUS,
    # policy eval only
    "judge": SONNET,
}


def model_for(agent: str) -> str:
    return (os.environ.get(f"SD_MODEL_{agent.upper()}") or os.environ.get("SD_MODEL")
            or AGENT_MODELS.get(agent, DEFAULT_MODEL))


# Approximate list prices, USD per million tokens: input, output, 5-minute cache write, cache read.
PRICES = {OPUS: (4.0, 20.0, 5.0, 0.20), SONNET: (2.0, 10.0, 2.5, 0.10)}

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Finding the articles for an issue (servicedesk/retrieval.py): "hybrid" (keyword + vector),
# "bm25" (keyword only) or "vector" (vector only). Anthropic offers no embedding model, so an
# open-source one runs locally (fastembed, ONNX), downloaded once into runtime/models.
RETRIEVAL_MODE = os.environ.get("SD_RETRIEVAL", "hybrid")
KB_CANDIDATES = 5                               # articles the rule picker reads for each issue
EMBED_MODEL = os.environ.get("SD_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MODELS_DIR = RUNTIME_DIR / "models"             # downloaded embedding model files
VECTOR_DIR = RUNTIME_DIR / "vector"             # cached article embeddings, by content hash


@lru_cache(maxsize=1)
def assumptions() -> dict:
    with open(ROOT / "config" / "working_assumptions.yaml") as f:
        return yaml.safe_load(f)
