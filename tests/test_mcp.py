"""The MCP server exposes the library and nothing else. No network."""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest

pytest.importorskip("mcp")
from cadence import mcp_server as m  # noqa: E402

CORPUS = [
    "The team met on Tuesday and wrote a note. It was short. The note said the "
    "pricing held, which nobody expected, and that the pilot would run again.",
    "She read it twice. The second reading changed her mind, because the numbers "
    "in the appendix did not match the summary, and the summary was the part "
    "everyone had already quoted.",
    "He asked for the appendix. Nobody had it. The meeting ended there, and the "
    "note went out unchanged, which is how the wrong number reached the board.",
    "The board asked for a correction. It arrived on Friday, three days late, "
    "and by then the number had been repeated in two other decks.",
]


def _run(value):
    return asyncio.run(value) if inspect.isawaitable(value) else value


def call(name, **kwargs) -> dict:
    result = _run(m.server.call_tool(name, kwargs))
    # Both SDK majors return the payload as a text content block of JSON.
    content = getattr(result, "content", result)
    if isinstance(content, tuple):
        content = content[0]
    text = content[0].text if isinstance(content, list) else content
    return json.loads(text) if isinstance(text, str) else text


def test_the_expected_tools_are_registered():
    names = sorted(t.name for t in _run(m.server.list_tools()))
    assert names == ["analyze_text", "check_fidelity", "measure_voice",
                     "repair_structure", "restyle", "score"]


def test_every_tool_has_a_docstring_a_model_can_read():
    for t in _run(m.server.list_tools()):
        assert t.description and len(t.description) > 40, t.name


def test_measure_voice_returns_a_spec_and_flags_thin_corpora():
    r = call("measure_voice", documents=CORPUS)
    assert r["sentences"] >= 10 and "SYNTACTIC TARGET PROFILE" in r["spec"]
    thin = call("measure_voice", documents=["One sentence. Two."])
    assert thin["enough_for_a_profile"] is False and thin["warnings"]


def test_the_profile_cache_reuses_identical_text():
    m._profiles.clear()
    call("measure_voice", documents=CORPUS)
    call("measure_voice", documents=list(CORPUS))
    assert len(m._profiles) == 1


def test_analyze_text_finds_structural_defects():
    r = call("analyze_text", text="The pricing read is the strongest element.")
    codes = {f["code"] for f in r["findings"]}
    assert "SUPERLATIVE_NO_SET" in codes and "NOMINALIZATION" in codes


def test_check_fidelity_catches_an_invented_name():
    r = call("check_fidelity", source="Revenue fell 12% to $4.2M.",
             output="Revenue fell 12% to $4.2M, Ann Lee said.")
    assert r["passed"] is False
    assert any(a["text"] == "ann lee" for a in r["introduced"])


def test_score_gives_voice_only_without_a_source():
    r = call("score", text="The pricing held.", documents=CORPUS)
    assert 0 < r["voice_similarity"] <= 100 and r["content_kept"] is None


def test_repair_structure_never_touches_content():
    r = call("repair_structure", text="She is a teacher, patient, and writing daily.")
    assert r["text"] == "She is a teacher, is patient, and is writing daily."
    assert r["fidelity"]["score"] == 100.0


def test_restyle_without_credentials_returns_a_fix_not_a_stack(monkeypatch):
    """The client is a model. It can act on a sentence; it cannot act on a trace."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = call("restyle", draft="The pricing held. It did.", documents=CORPUS)
    assert r["error"] == "missing_credentials"
    assert "ANTHROPIC_API_KEY" in r["fix"]
