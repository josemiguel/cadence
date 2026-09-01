"""Credential resolution, usage accounting, and the verify loop's prompts.

Three of these pin bugs that were live: the correction calls sent a system
prompt with the restyle rules and the remove-slop rules stripped out, the length
correction was written and never called, and the loop always ran to completion
however good the first attempt was.
"""

from __future__ import annotations

import pytest

from cadence.errors import MissingAPIKey
from cadence.generate import generate
from cadence.llm import add_usage, call_text, have_credentials, resolve_client
from cadence.profile import build_profile

CORPUS = [
    "The team met on Tuesday and wrote a note. It was short. The note said the "
    "pricing held, which nobody expected, and that the pilot would run again.",
    "She read it twice. The second reading changed her mind, because the numbers "
    "in the appendix did not match the summary, and the summary was the part "
    "everyone had already quoted.",
    "He asked for the appendix. Nobody had it. The meeting ended there, and the "
    "note went out unchanged, which is how the wrong number reached the board.",
]


@pytest.fixture
def profile():
    return build_profile(CORPUS, name="corpus")


# --- credentials -----------------------------------------------------------

def test_client_argument_beats_everything(fake_client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = fake_client("x")
    assert resolve_client(client=c) is c


def test_api_key_argument_is_used_without_an_environment_key(install_fake_anthropic):
    client = install_fake_anthropic("x", key=None)
    resolve_client(api_key="sk-passed-in")
    assert client.api_key == "sk-passed-in"


def test_missing_credentials_raise_a_catchable_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingAPIKey) as exc:
        resolve_client()
    # Callers written against the old API catch RuntimeError and fall back.
    assert isinstance(exc.value, RuntimeError)
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_have_credentials_does_not_build_a_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not have_credentials()
    assert have_credentials(api_key="sk-x")


# --- usage -----------------------------------------------------------------

def test_call_text_returns_usage_and_latency(fake_client):
    r = call_text(fake_client("hello"), "m", "sys", "prompt")
    assert r.text == "hello"
    assert r.usage["input_tokens"] == 11 and r.usage["output_tokens"] == 22
    assert r.latency_s >= 0.0


def test_truncated_output_raises_rather_than_returning_half_an_answer(fake_client):
    with pytest.raises(RuntimeError, match="token cap"):
        call_text(fake_client("half a sen", stop_reason="max_tokens"), "m", "s", "p")


def test_add_usage_sums_across_attempts():
    total = add_usage({"input_tokens": 3}, {"input_tokens": 4, "output_tokens": 5}, None)
    assert total["input_tokens"] == 7 and total["output_tokens"] == 5


def test_generation_usage_is_summed_over_the_whole_verify_loop(fake_client, profile):
    client = fake_client(["first attempt text.", "second one.", "third one."])
    r = generate("The pricing held.", mode="profile_verify", profile=profile,
                 task="compose", iterations=2, client=client)
    assert len(r.attempts) == len(client.calls)
    assert r.usage["input_tokens"] == 11 * len(client.calls)
    assert r.latency_s == pytest.approx(sum(a.latency_s for a in r.attempts))


# --- the verify loop -------------------------------------------------------

def test_corrections_keep_the_restyle_and_slop_rules(fake_client, profile):
    """The correction calls used to send a system prompt with both stripped out."""
    client = fake_client(["A short first attempt about pricing.",
                          "A second attempt about pricing that is longer."])
    generate("The pricing held and nobody expected it.", mode="profile_verify",
             profile=profile, task="restyle", iterations=1, client=client)
    assert len(client.calls) >= 2, "expected at least one correction call"
    correction_system = client.calls[1]["system"]
    assert "PRESERVE EVERY FACT" in correction_system
    assert "REMOVE-SLOP RULES" in correction_system


def test_length_correction_reaches_the_model_when_the_output_balloons(fake_client, profile):
    source = "The pricing held."
    ballooned = " ".join(["The pricing held and this is padding"] * 20)
    client = fake_client([ballooned, "The pricing held."])
    generate(source, mode="profile_verify", profile=profile, task="restyle",
             iterations=1, client=client)
    assert "LENGTH IS WRONG" in client.calls[1]["messages"][0]["content"]


def test_length_correction_is_absent_when_the_length_is_right(fake_client, profile):
    client = fake_client(["The pricing held.", "The pricing held again."])
    generate("The pricing held.", mode="profile_verify", profile=profile,
             task="restyle", iterations=1, client=client)
    assert "LENGTH IS WRONG" not in client.calls[1]["messages"][0]["content"]


def test_a_close_enough_first_attempt_stops_the_loop(fake_client, profile):
    """An attempt that is the corpus itself scores at the ceiling."""
    client = fake_client(["\n\n".join(CORPUS),
                          "a second attempt that should never be requested"])
    r = generate("\n\n".join(CORPUS), mode="profile_verify", profile=profile, task="compose",
                 iterations=2, client=client)
    assert r.similarity >= 90.0
    assert len(client.calls) == 1, "no correction call should have been made"


def test_the_loop_still_runs_when_the_first_attempt_is_far_off(fake_client, profile):
    client = fake_client(["Short. Terse. Clipped.", "Another. Also short.", "Third."])
    generate("The pricing held.", mode="profile_verify", profile=profile,
             task="compose", iterations=2, client=client)
    assert len(client.calls) == 3


# --- fidelity gates the verify loop ----------------------------------------

def test_an_attempt_that_invents_content_is_never_selected(fake_client, profile):
    """A higher structural score over a fabricated name is the failure mode."""
    source = "Halvorsen Dairy did not renew the 2024-03-11 contract."
    faithful = "The contract of 2024-03-11 was not renewed by Halvorsen Dairy."
    invented = ("Halvorsen Dairy did not renew the 2024-03-11 contract, which Ann Lee "
                "had signed, and the board met about it in Copenhagen afterwards.")
    client = fake_client([faithful, invented])
    r = generate(source, mode="profile_verify", profile=profile, task="restyle",
                 iterations=1, client=client)
    assert r.text == faithful
    assert r.fidelity is not None and r.fidelity.passed
    # The rejected attempt is still recorded, with the reason visible.
    assert len(r.attempts) == 2
    assert not r.attempts[1].fidelity.passed


def test_composing_from_a_topic_has_no_fidelity_score(fake_client, profile):
    r = generate("write about pricing", mode="profile", profile=profile,
                 task="compose", client=fake_client("Some prose about pricing."))
    assert r.fidelity is None, "there is no source to be faithful to"


def test_a_newline_terminated_key_is_cleaned_before_use(install_fake_anthropic):
    """Secret stores hand keys over with a trailing newline more often than not.

    Left in, it becomes an illegal HTTP header and surfaces as a connection
    error with the key printed inside the exception, which is the worst of
    both worlds.
    """
    client = install_fake_anthropic("x", key=None)
    resolve_client(api_key="sk-from-a-file\n")
    assert client.api_key == "sk-from-a-file"


def test_a_whitespace_only_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  \n")
    with pytest.raises(MissingAPIKey):
        resolve_client()


def test_restyle_instruction_forbids_reaching_length_by_adding(fake_client, profile):
    client = fake_client("The pricing held and nobody expected it.")
    generate("The pricing held. Nobody expected it.", mode="profile", profile=profile,
             task="restyle", client=client)
    prompt = client.calls[0]["messages"][0]["content"]
    assert "HARD LIMIT" in prompt and "merging or splitting" in prompt
    assert "2 sentences" in prompt
    assert "NEVER BY ADDING" in client.calls[0]["system"]


def test_when_no_attempt_passes_the_most_faithful_one_is_returned(fake_client, profile):
    """A visitor sees one attempt. It should be the least wrong, not the first."""
    source = "Halvorsen Dairy did not renew the 2024-03-11 contract. Revenue fell 12%."
    invented_first = ("Halvorsen Dairy did not renew the 2024-03-11 contract, which Ann Lee "
                      "had negotiated in Copenhagen, and revenue fell 12% across Denmark.")
    padded_second = (source + " This is a serious structural problem for the sector, "
                     "reflecting deeper competitive dynamics and raising questions.")
    client = fake_client([invented_first, padded_second, padded_second])
    r = generate(source, mode="profile_verify", profile=profile, task="restyle",
                 iterations=2, client=client)
    assert not r.fidelity.passed
    assert r.text == padded_second, "padding is less wrong than a fabricated person"
    assert not r.fidelity.introduced
