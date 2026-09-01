"""One place where an Anthropic client is built and one place where it is called.

Both the rewrite and the generate layers needed the same three things: resolve
credentials, make one text call, and notice when the model ran out of budget
mid-answer. They had grown separate copies that disagreed in small ways, and
neither returned token usage, which a benchmark cannot do without.

Credential precedence is client, then argument, then environment. A caller that
passes its own client is never second-guessed, which is what lets the eval
harness replay recorded responses and lets a web request run under a key the
visitor supplied without that key ever touching the process environment.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from .errors import MissingAPIKey

# Extended thinking is on by default and its tokens count against max_tokens, so
# the budget has to cover reasoning as well as prose.
DEFAULT_MAX_TOKENS = 16384

_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


@dataclass
class LLMResponse:
    """One completed call: the prose, why it stopped, and what it cost."""

    text: str
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    model: str = ""


def resolve_client(api_key: str | None = None, client=None):
    """Return a client, or raise MissingAPIKey.

    `client` wins over `api_key`, which wins over ANTHROPIC_API_KEY.
    """
    if client is not None:
        return client
    # Secrets arrive newline-terminated more often than not: a file, a
    # `gcloud secrets versions add --data-file=-`, a heredoc. An HTTP header
    # cannot carry the newline, and the failure it produces is a connection
    # error with the key printed inside it, which is the worst of both worlds.
    key = (api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        raise MissingAPIKey(
            "No Anthropic credentials. Pass api_key=, pass client=, or set "
            "ANTHROPIC_API_KEY. The rules backend needs none of these."
        )
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - install-time failure
        raise MissingAPIKey(
            "The `anthropic` package is required. Install it with: "
            'pip install "cadence[llm]"'
        ) from exc
    return anthropic.Anthropic(api_key=key)


def have_credentials(api_key: str | None = None, client=None) -> bool:
    """Whether an llm call could be made, without building anything."""
    return bool(client is not None or api_key or os.environ.get("ANTHROPIC_API_KEY"))


def _usage_of(resp) -> dict:
    raw = getattr(resp, "usage", None)
    if raw is None:
        return dict.fromkeys(_USAGE_FIELDS, 0)
    return {f: int(getattr(raw, f, 0) or 0) for f in _USAGE_FIELDS}


def add_usage(*usages: dict) -> dict:
    """Sum usage dicts across the attempts of one generation."""
    total = dict.fromkeys(_USAGE_FIELDS, 0)
    for u in usages:
        for f in _USAGE_FIELDS:
            total[f] += int((u or {}).get(f, 0) or 0)
    return total


def call_text(client, model: str, system: str, prompt: str,
              max_tokens: int = DEFAULT_MAX_TOKENS) -> LLMResponse:
    """One text call. Raises rather than returning prose that was cut off.

    A truncated answer is worse than no answer here: the rewrite path parses the
    result as JSON, and the generate path measures it, so half an answer becomes
    a confident wrong number instead of an error.
    """
    started = time.monotonic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = getattr(resp, "latency_s", None)
    if latency is None:
        latency = time.monotonic() - started

    # `thinking` blocks are deliberately excluded; only prose is returned.
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    stop = getattr(resp, "stop_reason", None)
    if stop == "max_tokens":
        raise RuntimeError(
            f"Model hit the {max_tokens}-token cap (thinking tokens count toward it) "
            f"and returned {len(text)} characters of prose. Ask for shorter output."
        )
    return LLMResponse(
        text=text,
        stop_reason=stop,
        usage=_usage_of(resp),
        latency_s=round(float(latency), 3),
        model=getattr(resp, "model", "") or model,
    )
