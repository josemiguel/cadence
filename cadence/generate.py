"""Generate prose against a syntactic profile, and measure whether it worked.

Three modes, built to be compared against each other:

tone            The baseline. Raw samples plus "write like this." What everyone
                already does, and what a structural spec has to beat to be worth
                anything.
profile         The spec ONLY -- numbers, no samples. If this matches the source
                voice, the style really was carried by structure, and it was
                carried without leaking the samples' content.
profile_verify  Generate, parse the output, measure the divergence, hand the
                deltas back, regenerate. The part tone prompting cannot do,
                because "sound more like this" has no error signal to iterate on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .fidelity import fidelity as check_fidelity
from .llm import add_usage, call_text, resolve_client
from .profile import SyntacticProfile, build_profile, divergence, spec_text
from .rewrite import DEFAULT_MODEL

MODES = ("tone", "profile", "profile_verify")

# What the input text IS.
#   restyle  the input is CONTENT: re-render it in the target structure, keeping
#            every fact and roughly the same length. Nothing may be invented.
#   compose  the input is a TOPIC: write something new about it.
TASKS = ("restyle", "compose")

# How far the output may drift from the source length before it is corrected.
LENGTH_TOLERANCE = 0.15

# Similarity above which another correction call is not worth its cost. The
# metric's own noise on a short passage is wider than the gap being chased.
EARLY_STOP_SIMILARITY = 90.0


@dataclass
class Attempt:
    iteration: int
    text: str
    similarity: float
    worst: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    fidelity: object = None


@dataclass
class GenerationResult:
    mode: str
    model: str
    text: str
    similarity: float
    divergence: dict = field(default_factory=dict)
    attempts: list = field(default_factory=list)
    task: str = "restyle"
    source_tokens: int = 0
    output_tokens: int = 0
    note: str = ""
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    fidelity: object = None

    @property
    def length_ratio(self) -> float:
        if not self.source_tokens:
            return 0.0
        return round(self.output_tokens / self.source_tokens, 2)


SYSTEM = """\
You write prose to a structural specification.

The specification describes SYNTAX -- sentence length, embedding depth, clause
counts, coordination, voice, function-word frequency. It says nothing about
subject matter. Write about the brief you are given, in the shape the spec
describes.

Rules:
1. Hit the RANGES, not the means. A spec saying "length 26 (range 15-37)" wants
   real variation across that range. Landing on 26 every sentence is the single
   most common way this fails -- it reads mechanical no matter how good the
   sentences are.
2. Match the spread as carefully as the average. Alternating long subordinated
   sentences with short ones IS the style; flattening the variance destroys it
   even when every mean is correct.
3. The spec is descriptive, never normative. If it shows heavy nominalisation,
   fragments, or agentless passives, those are the voice -- reproduce them. Do
   not "improve" them.
4. Write only the requested prose. No preamble, no commentary, no headings
   unless the brief asks for them."""

RESTYLE_RULES = """\

You are RESTYLING existing text, not writing new text. Additional rules, and
these outrank everything above:

A. PRESERVE EVERY FACT. Same claims, same numbers, same names, same hedges. You
   are changing how the sentences are built, not what they assert.
B. INVENT NOTHING. No new examples, no new figures, no elaboration, no added
   analysis. If the source is thin, the output is thin.
C. MATCH THE LENGTH. Stay within about 15% of the source's token count. Being
   structurally perfect at three times the length is a failure, not a success.
D. Drop nothing either. Every point in the source must survive."""


# Extended thinking is on by default and its tokens count against max_tokens, so
# the budget has to cover reasoning as well as prose. A verify-loop correction
# carries the spec, the previous attempt and the feedback, which provokes a lot
# of thinking for a short output -- 8192 was being exhausted before the prose
# began, on passages of barely 150 tokens.
GENERATION_MAX_TOKENS = 16384


def _token_count(text: str) -> int:
    """Content tokens, counted the same way the profile counts them."""
    from .syntax import parse
    return len([t for t in parse(text) if not t.is_space and not t.is_punct])


def _fidelity_of(brief: str, text: str, restyling: bool):
    """Content fidelity, when there is a source to be faithful to.

    Composing from a topic has no source, so there is nothing to check and a
    score would be an invented number rather than a missing one.
    """
    if not restyling or not text.strip():
        return None
    return check_fidelity(brief, text)


def _measure(text: str, target: SyntacticProfile) -> dict:
    """Divergence of one generated passage from the target profile."""
    candidate = build_profile([text], name="candidate")
    return divergence(target, candidate)


def _length_note(source_tokens: int, text: str) -> str:
    """Correction line when the output has drifted from the source length."""
    if not source_tokens:
        return ""
    got = _token_count(text)
    ratio = got / source_tokens
    if abs(ratio - 1.0) <= LENGTH_TOLERANCE:
        return ""
    verb = "far too long" if ratio > 1 else "too short"
    return (
        f"- LENGTH IS WRONG AND MATTERS MOST: you wrote {got} tokens against a "
        f"source of {source_tokens} ({ratio:.1f}x). That is {verb}. Cut to roughly "
        f"{source_tokens} tokens, keeping every fact and dropping the elaboration "
        "you added."
    ) if ratio > 1 else (
        f"- LENGTH IS WRONG: you wrote {got} tokens against a source of "
        f"{source_tokens} ({ratio:.1f}x), so material has been dropped. Restore it."
    )


def _feedback(target: SyntacticProfile, result: dict, text: str, top: int = 6) -> str:
    """Turn measured divergence into correction instructions with direction."""
    candidate = build_profile([text], name="candidate")
    lines = []
    stat_fields = {
        "sentence_length": (target.sentence_length, candidate.sentence_length,
                            "tokens per sentence"),
        "depth": (target.depth, candidate.depth, "embedding depth"),
        "finite_clauses": (target.finite_clauses, candidate.finite_clauses,
                           "finite clauses per sentence"),
        "subordination": (target.subordination, candidate.subordination,
                          "subordinate clauses per sentence"),
        "coordination": (target.coordination, candidate.coordination,
                         "coordinated items per sentence"),
        "pre_verb_weight": (target.pre_verb_weight, candidate.pre_verb_weight,
                            "tokens before the main verb"),
    }
    for key, score in list(result["by_feature"].items())[:top]:
        if score < 0.12:
            continue
        if key in stat_fields:
            t, c, label = stat_fields[key]
            direction = "raise" if c.mean < t.mean else "lower"
            lines.append(
                f"- {label}: you wrote mean {c.mean} (sd {c.sd}); target is {t.mean} "
                f"(sd {t.sd}, range {t.p10}-{t.p90}). {direction.capitalize()} it, and "
                f"{'widen' if c.sd < t.sd else 'narrow'} the variation between sentences."
            )
        elif key in target.rates:
            t, c = target.rates.get(key, 0.0), candidate.rates.get(key, 0.0)
            direction = "increase" if c < t else "reduce"
            lines.append(f"- {key.replace('_', ' ')}: you wrote {c}, target {t}. "
                         f"{direction.capitalize()}.")
        elif key == "function_words":
            lines.append(
                "- function-word mix is off. Match the target's frequencies for "
                "determiners, prepositions, conjunctions and pronouns."
            )
        elif key == "pos_mix":
            lines.append("- part-of-speech mix is off; compare against the target percentages.")
    return "\n".join(lines) or "- close on every measured feature; vary sentence rhythm more."


def generate(brief: str, mode: str = "profile", profile: SyntacticProfile | None = None,
             samples: list[str] | None = None, model: str = DEFAULT_MODEL,
             iterations: int = 2, max_tokens: int = GENERATION_MAX_TOKENS,
             task: str = "restyle", *, api_key: str | None = None,
             client=None) -> GenerationResult:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")

    restyling = task == "restyle"
    source_tokens = _token_count(brief) if restyling else 0
    from .slop import BANNED_PROFILE_FEATURES, PROMPT_RULES

    system = SYSTEM + (RESTYLE_RULES if restyling else "") + PROMPT_RULES
    # Where the profile measured something remove-slop bans, the ban wins and the
    # override is stated rather than left for the reader to notice.
    conflicts = []
    if profile is not None:
        for key, why in BANNED_PROFILE_FEATURES.items():
            if profile.rates.get(key, 0.0) > 0.2:
                conflicts.append(
                    f"corpus shows {key.replace('_', ' ')} at {profile.rates[key]}, but "
                    f"{why}, so it is not reproduced"
                )
    if restyling:
        instruction = (
            f"TEXT TO RESTYLE ({source_tokens} content tokens -- your output must be "
            f"within about 15% of that):\n\n{brief}\n\n"
            "Re-render this text so its SYNTAX matches the specification. Keep every "
            "fact, every number and every name. Add nothing. Match the length."
        )
    else:
        instruction = f"BRIEF -- write about this:\n{brief}"
    client = resolve_client(api_key=api_key, client=client)

    if mode == "tone":
        if not samples:
            raise ValueError("tone mode needs the raw samples.")
        joined = "\n\n---\n\n".join(s.strip() for s in samples)
        prompt = (
            "Here are samples of the writing to imitate:\n\n"
            f"{joined}\n\n"
            f"{instruction}\n\n"
            "Write only the prose."
        )
        resp = call_text(client, model, system, prompt, max_tokens)
        text = resp.text
        result = _measure(text, profile) if profile else {}
        return GenerationResult(
            mode="tone", model=model, text=text,
            similarity=result.get("similarity", 0.0), divergence=result,
            task=task, source_tokens=source_tokens, output_tokens=_token_count(text),
            usage=resp.usage, latency_s=resp.latency_s,
            fidelity=_fidelity_of(brief, text, restyling),
            note="Baseline: raw samples, no measurement. Content of the samples is "
                 "visible to the model and can leak into the output."
                 + ("  Overridden by remove-slop: " + "; ".join(conflicts) if conflicts else ""),
        )

    if profile is None:
        raise ValueError("profile mode needs a profile.")

    base_prompt = (
        f"{spec_text(profile)}\n\n"
        f"{instruction}\n\n"
        "Write only the prose."
    )
    resp = call_text(client, model, system, base_prompt, max_tokens)
    text = resp.text
    result = _measure(text, profile)
    fid = _fidelity_of(brief, text, restyling)
    attempts = [Attempt(1, text, result["similarity"],
                        list(result["by_feature"].items())[:4],
                        usage=resp.usage, latency_s=resp.latency_s, fidelity=fid)]

    if mode == "profile":
        return GenerationResult(
            mode="profile", model=model, text=text,
            similarity=result["similarity"], divergence=result, attempts=attempts,
            task=task, source_tokens=source_tokens, output_tokens=_token_count(text),
            usage=resp.usage, latency_s=resp.latency_s, fidelity=fid,
            note="Spec only -- the model never saw the samples, so nothing of their "
                 "content could leak. Any style match is carried by structure alone."
                 + ("  Overridden by remove-slop: " + "; ".join(conflicts) if conflicts else ""),
        )

    # Keep the best attempt, not merely the last -- iteration can overshoot.
    # When restyling, a length-compliant attempt beats a higher-scoring one:
    # style similarity is meaningless at three times the source length.
    def acceptable(candidate_text, candidate_fidelity):
        # An attempt that invented content is never an improvement, however
        # well it scored on structure. Style similarity over a fabricated fact
        # is the failure this tool is built to avoid, measured and rewarded.
        if candidate_fidelity is not None and not candidate_fidelity.passed:
            return False
        if not source_tokens:
            return True
        return abs(_token_count(candidate_text) / source_tokens - 1.0) <= LENGTH_TOLERANCE

    failures = []
    for i in range(2, iterations + 2):
        # Nothing left to correct. Spending another call to move a 94 to a 95
        # buys a rounding difference and risks losing a good attempt.
        if result["similarity"] >= EARLY_STOP_SIMILARITY and acceptable(text, fid):
            break
        corrections = _feedback(profile, result, text)
        # Length is the one deviation the structural feedback cannot express,
        # and it is the one that makes every other number meaningless.
        length_note = _length_note(source_tokens, text)
        if length_note:
            corrections = f"{length_note}\n{corrections}"
        prompt = (
            f"{base_prompt}\n\n"
            f"Your previous attempt:\n\n{text}\n\n"
            f"It scored {result['similarity']}/100 against the target. Measured "
            f"deviations:\n{corrections}\n\n"
            "Rewrite it correcting these deviations. Keep the content; change the "
            "structure. Write only the prose, at roughly the same total length."
        )
        # A failed correction must not discard the attempts that worked. The loop
        # is an improvement mechanism, not a precondition for returning anything.
        # The system prompt is the composed one: dropping the restyle rules and
        # the remove-slop rules here would licence the corrections to invent.
        try:
            resp = call_text(client, model, system, prompt, max_tokens)
            new_text = resp.text
            new_result = _measure(new_text, profile)
            new_fid = _fidelity_of(brief, new_text, restyling)
        except Exception as exc:
            failures.append(f"iteration {i}: {exc}")
            continue
        attempts.append(Attempt(i, new_text, new_result["similarity"],
                                list(new_result["by_feature"].items())[:4],
                                usage=resp.usage, latency_s=resp.latency_s,
                                fidelity=new_fid))

        better = new_result["similarity"] > result["similarity"]
        if acceptable(new_text, new_fid) and (better or not acceptable(text, fid)):
            text, result, fid = new_text, new_result, new_fid

    return GenerationResult(
        mode="profile_verify", model=model, text=text,
        similarity=result["similarity"], divergence=result, attempts=attempts,
        task=task, source_tokens=source_tokens, output_tokens=_token_count(text),
        usage=add_usage(*(a.usage for a in attempts)),
        latency_s=round(sum(a.latency_s for a in attempts), 3),
        fidelity=fid,
        note=(
            f"Generate -> parse -> measure -> correct, {len(attempts)} attempt(s). "
            "Best-scoring attempt kept, since iteration can overshoot."
            + (f" Skipped: {'; '.join(failures)}." if failures else "")
            + ("  Overridden by remove-slop: " + "; ".join(conflicts) if conflicts else "")
        ),
    )


def compare(brief: str, profile: SyntacticProfile, samples: list[str],
            model: str = DEFAULT_MODEL, iterations: int = 2,
            task: str = "restyle", *, api_key: str | None = None,
            client=None) -> list[GenerationResult]:
    """Run all three modes on one input so they can be read blind."""
    out = []
    for mode in MODES:
        try:
            out.append(generate(brief, mode=mode, profile=profile, samples=samples,
                                model=model, iterations=iterations, task=task,
                                api_key=api_key, client=client))
        except Exception as exc:  # one mode failing must not lose the others
            out.append(GenerationResult(mode=mode, model=model, text="",
                                        similarity=0.0, note=f"failed: {exc}"))
    return out
