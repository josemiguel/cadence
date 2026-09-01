"""Fakes shared across the suite. Nothing here touches the network.

Two ways to stand in for the Anthropic SDK, because the code has two paths worth
testing. `fake_client` builds a client object to pass in through the new
`client=` argument, which is how the web layer and the eval harness call it.
`install_fake_anthropic` patches the imported module instead, which is the only
way to exercise credential resolution from the environment.

Both record every call, in order, so a test can assert on what the second
correction prompt contained rather than only the last one.
"""

from __future__ import annotations

import sys
import types

import pytest


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Usage:
    def __init__(self, i=11, o=22):
        self.input_tokens = i
        self.output_tokens = o
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0


class _Resp:
    def __init__(self, text, stop_reason="end_turn", usage=None, model="fake-model"):
        self.content = [_Block(text)]
        self.stop_reason = stop_reason
        self.usage = usage if usage is not None else _Usage()
        self.model = model


class _FakeMessages:
    def __init__(self, payloads, calls, stop_reason):
        self._payloads = list(payloads)
        self._calls = calls
        self._stop = stop_reason

    def create(self, **kw):
        self._calls.append(kw)
        # A single payload answers every call; a list is consumed in order and
        # then repeats its last entry, so a test only specifies what it cares about.
        idx = min(len(self._calls) - 1, len(self._payloads) - 1)
        return _Resp(self._payloads[idx], stop_reason=self._stop)


class FakeClient:
    """Duck-types the one method this package calls on an Anthropic client."""

    def __init__(self, payloads, stop_reason="end_turn"):
        if isinstance(payloads, str):
            payloads = [payloads]
        self.calls: list[dict] = []
        self.api_key = None
        self.messages = _FakeMessages(payloads, self.calls, stop_reason)


@pytest.fixture
def fake_client():
    """Build a fake client from one payload or a list of them."""

    def build(payloads, stop_reason="end_turn"):
        return FakeClient(payloads, stop_reason=stop_reason)

    return build


@pytest.fixture
def install_fake_anthropic(monkeypatch):
    """Patch the `anthropic` module itself and set a key in the environment."""

    def install(payloads, key="test-key-not-real"):
        client = FakeClient(payloads)
        mod = types.ModuleType("anthropic")

        def _ctor(api_key=None, **_kw):
            client.api_key = api_key
            return client

        mod.Anthropic = _ctor
        monkeypatch.setitem(sys.modules, "anthropic", mod)
        if key is None:
            monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        else:
            monkeypatch.setenv("ANTHROPIC_API_KEY", key)
        return client

    return install
