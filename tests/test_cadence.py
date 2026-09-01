"""Regression tests.

Every expectation here was verified by hand before it was automated: the cases
come from a hand-checked analysis of note-register prose, so a failure means
the tool has drifted from an analysis a human checked.
"""

import pytest

from cadence.diagnostics import analyze
from cadence.rewrite import rewrite
from cadence.syntax import constituency, parse, pos_line

# --- POS and repairs -------------------------------------------------------

def test_flat_base_np_and_compound():
    """`founder case` is a flat noun-noun compound, not an adjective + noun."""
    sent = list(parse("The founder case is the strongest element.").sents)[0]
    assert pos_line(sent).startswith("The/DT founder/NN case/NN is/VBZ")
    brackets = constituency(sent).brackets()
    assert "(NP-SBJ\n    (DT The)\n    (NN founder)\n    (NN case))" in brackets


def test_iso_date_is_one_token():
    doc = parse("The read of record is still the 2026-08-24 one.")
    assert any(t.text == "2026-08-24" and t.tag_ == "CD" for t in doc)


def test_pronominal_one_is_nn_partitive_one_is_cd():
    """§5.8: the determiner discriminates the two `one`s."""
    doc = parse("The read of record is still the 2026-08-24 one.")
    assert [t.tag_ for t in doc if t.lower_ == "one"] == ["NN"]
    doc2 = parse("This is one of the most active areas in AI.")
    assert [t.tag_ for t in doc2 if t.lower_ == "one"] == ["CD"]


def test_compound_head_repair():
    """`the category read` must not parse as a clause."""
    doc = parse("The category read is that this is active.")
    read = [t for t in doc if t.lower_ == "read"][0]
    assert read.pos_ == "NOUN"
    assert read.dep_ == "nsubj"


def test_accusative_pronoun_is_not_a_subject():
    doc = parse("Liked him.")
    him = [t for t in doc if t.lower_ == "him"][0]
    assert him.dep_ == "dobj"


def test_hyphen_compound_tagging():
    """§5.7: noun-noun -> NN, adjectival prefix -> JJ."""
    doc = parse("Deal-team lead read no post-visit view.")
    tags = {t.text: t.tag_ for t in doc}
    assert tags["Deal-team"] == "NN"
    assert tags["post-visit"] == "JJ"


# --- Diagnostics -----------------------------------------------------------

def codes(text):
    return [f.code for f in analyze(text).findings]


def test_superlative_without_comparison_set():
    assert "SUPERLATIVE_NO_SET" in codes("The founder case is the strongest element.")


def test_superlative_with_vague_set():
    assert "SUPERLATIVE_VAGUE_SET" in codes(
        "At this stage the founder is most of what there is to underwrite."
    )


def test_superlative_with_real_set_is_not_flagged():
    """§4.6: `areas in AI` IS a comparison set. Not a finding."""
    got = codes("This is one of the most active areas in AI right now.")
    assert not [c for c in got if c.startswith("SUPERLATIVE")]


def test_agentless_passive():
    assert "AGENTLESS_PASSIVE" in codes("No view was being taken until the visit.")


def test_passive_with_agent_is_not_flagged():
    assert "AGENTLESS_PASSIVE" not in codes("The call was attached by Ann Lee.")


def test_null_subject_detected():
    assert "NULL_SUBJECT" in codes("Liked him.")


def test_coordinated_verbs_share_a_subject():
    """VP coordination is not a dropped subject."""
    assert "NULL_SUBJECT" not in codes(
        "This revision carries what the visit contained, and asserts no view."
    )


def test_unlike_coordination_across_full_chain():
    found = [f for f in analyze("She is a teacher, patient, and writing daily.").findings
             if f.code == "UCP"]
    assert found and found[0].auto
    assert "nominal" in found[0].label and "verbal" in found[0].label


def test_like_coordination_is_not_flagged():
    assert "UCP" not in codes("The founder read is positive and specific.")


def test_ambiguous_subordinator():
    assert "AMBIG_SUBORD" in codes("It stays open, since both companies sell there.")


def test_container_noun_needs_no_owner_to_be_flagged():
    assert "NOMINALIZATION" in codes("The read is still open.")


def test_concrete_event_noun_is_not_a_container():
    """A date modifier marks an event, not a packaged claim."""
    assert "NOMINALIZATION" not in codes("The 2026-08-26 call does not touch that.")


def test_finding_spans_preserve_word_order():
    """`token.subtree` is tree-ordered; spans must read as the original text."""
    a = analyze("At this stage the founder is most of what there is to underwrite.")
    sup = [f for f in a.findings if f.code == "SUPERLATIVE_VAGUE_SET"]
    assert sup, "expected a vague comparison set"
    assert sup[0].span == "of what there is to underwrite"


def test_finding_spans_preserve_original_spacing():
    a = analyze("A call recorded on 2026-08-26 with Ann Lee and the Bingley side.")
    passives = [f for f in a.findings if f.code == "AGENTLESS_PASSIVE"]
    assert passives
    assert " ( " not in passives[0].span and " ," not in passives[0].span


# --- Rewrite --------------------------------------------------------------

def test_copula_repetition_breaks_ucp():
    a = analyze("She is a teacher, patient, and writing daily.")
    r = rewrite(a, backend="rules")
    assert r.text == "She is a teacher, is patient, and is writing daily."


def test_rewrite_reports_gaps_and_never_fills_them():
    a = analyze("The founder case is the strongest element.")
    r = rewrite(a, backend="rules")
    assert r.gaps, "a superlative with no comparison set must be reported as a gap"
    # The rules backend must not invent the missing comparison set.
    assert "strongest element" in r.text


def test_llm_backend_requires_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    a = analyze("Liked him.")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        rewrite(a, backend="llm")


def test_auto_backend_falls_back_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert rewrite(analyze("Liked him."), backend="auto").backend == "rules"


# --- LLM backend plumbing (mocked; no network, no key) ---------------------

class _Block:
    type = "text"
    def __init__(self, text): self.text = text


class _Resp:
    def __init__(self, text): self.content = [_Block(text)]


class _FakeMessages:
    def __init__(self, payload, sink): self._payload, self._sink = payload, sink
    def create(self, **kw):
        self._sink.update(kw)
        return _Resp(self._payload)


class _FakeClient:
    def __init__(self, payload, sink): self.messages = _FakeMessages(payload, sink)


# A well-formed response that changes nothing, for tests that only care about
# what went out in the prompt.
EMPTY_PAYLOAD = '{"rewrite":"x","changes":[],"gaps":[],"inferences":[],"disputed":[]}'


def _install_fake(monkeypatch, payload):
    """Patch the anthropic module the backend imports, capturing the call kwargs."""
    import sys
    import types
    sink = {}
    mod = types.ModuleType("anthropic")
    mod.Anthropic = lambda api_key=None: _FakeClient(payload, sink)
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    return sink


def test_llm_backend_parses_response_and_passes_findings(monkeypatch):
    payload = """```json
{"rewrite": "Ann liked him.",
 "changes": [{"code": "NULL_SUBJECT", "before": "Liked him.",
              "after": "Ann liked him.", "rationale": "Subject restored."}],
 "gaps": ["comparison set absent"],
 "inferences": ["`Ann` inferred as the dropped subject"]}
```"""
    sink = _install_fake(monkeypatch, payload)
    from cadence.rewrite import rewrite
    r = rewrite(analyze("Liked him."), backend="llm")

    # Response parsing, including stripping the ``` fence.
    assert r.text == "Ann liked him."
    assert r.changes[0].code == "NULL_SUBJECT"
    assert r.inferences == ["`Ann` inferred as the dropped subject"]
    assert r.gaps == ["comparison set absent"]

    # The findings must actually reach the model as constraints.
    prompt = sink["messages"][0]["content"]
    assert "NULL_SUBJECT" in prompt
    assert "FINDINGS TO FIX" in prompt
    assert "NEVER FILL" in prompt
    assert "NEVER FILL AN INFORMATION GAP" in sink["system"]


def test_llm_backend_routes_gaps_separately(monkeypatch):
    """Gap-coded findings must arrive under the report-only heading, not as actions."""
    sink = _install_fake(
        monkeypatch, '{"rewrite": "x", "changes": [], "gaps": [], "inferences": []}')
    from cadence.rewrite import rewrite
    rewrite(analyze("The founder case is the strongest element."), backend="llm")
    prompt = sink["messages"][0]["content"]
    actions, gaps = prompt.split("GAPS -- REPORT ONLY, NEVER FILL:")
    assert "SUPERLATIVE_NO_SET" in gaps
    assert "SUPERLATIVE_NO_SET" not in actions


def test_llm_backend_rejects_non_json(monkeypatch):
    _install_fake(monkeypatch, "I think this reads pretty well already!")
    from cadence.rewrite import rewrite
    with pytest.raises(RuntimeError, match="valid JSON"):
        rewrite(analyze("Liked him."), backend="llm")


def test_auto_backend_selects_llm_when_key_present(monkeypatch):
    _install_fake(monkeypatch, '{"rewrite": "ok", "changes": [], "gaps": [], "inferences": []}')
    from cadence.rewrite import rewrite
    assert rewrite(analyze("Liked him."), backend="auto").backend.startswith("llm")


# --- Two-step pipeline: render, then rewrite against the render ------------

def test_step1_renders_only_flagged_sentences_by_default():
    from cadence.rewrite import render_trees
    a = analyze("Liked him. The cat sat on the mat.")
    t = render_trees(a)
    assert "--- S1" in t and "--- S2" not in t
    assert "--- S2" in render_trees(a, only_with_findings=False)


def test_step1_includes_all_three_layers():
    from cadence.rewrite import render_trees
    t = render_trees(analyze("Liked him."))
    assert "POS:" in t and "DEPENDENCIES:" in t and "CONSTITUENCY" in t
    assert "Liked/VBD" in t


def test_step2_receives_the_step1_render(monkeypatch):
    sink = _install_fake(monkeypatch, EMPTY_PAYLOAD)
    from cadence.rewrite import render_trees, rewrite
    a = analyze("Liked him.")
    r = rewrite(a, backend="llm")
    prompt = sink["messages"][0]["content"]
    assert "SYNTACTIC EVIDENCE" in prompt
    assert render_trees(a) in prompt, "the exact step 1 artefact must be what step 2 sees"
    assert r.trees == render_trees(a)
    assert "+trees" in r.backend


def test_tree_evidence_can_be_withheld(monkeypatch):
    sink = _install_fake(monkeypatch, EMPTY_PAYLOAD)
    from cadence.rewrite import rewrite
    r = rewrite(analyze("Liked him."), backend="llm", use_trees=False)
    assert "SYNTACTIC EVIDENCE" not in sink["messages"][0]["content"]
    assert r.trees == "" and "+trees" not in r.backend


def test_disputed_findings_are_parsed(monkeypatch):
    _install_fake(monkeypatch, """{"rewrite": "Liked him.", "changes": [], "gaps": [],
     "inferences": [],
     "disputed": [{"code": "NULL_SUBJECT", "sentence": 1,
                   "reason": "The parse shows an imperative, not a dropped subject."}]}""")
    from cadence.rewrite import rewrite
    r = rewrite(analyze("Liked him."), backend="llm")
    assert len(r.disputed) == 1
    assert r.disputed[0].code == "NULL_SUBJECT"
    assert r.disputed[0].sentence == 1
    assert "imperative" in r.disputed[0].reason


def test_system_prompt_licenses_disputes(monkeypatch):
    sink = _install_fake(monkeypatch, EMPTY_PAYLOAD)
    from cadence.rewrite import rewrite
    rewrite(analyze("Liked him."), backend="llm")
    assert "disputed" in sink["system"]
    assert "A wrong fix is worse than a missed one." in sink["system"]


def test_coordination_does_not_inflate_depth():
    """Confirmed by a model dispute: `conj`/`compound` are sibling structure.

    `with Ann Lee and Bob Ray` scored three levels of "nesting"
    that a reader never has to hold open, producing a false DEEP_NESTING.
    """
    from cadence.syntax import dep_depth, parse
    flat = dep_depth(list(parse("We met Ann.").sents)[0])
    coord = dep_depth(list(parse("We met Ann Lee, Bob Ray and Carla Diaz.").sents)[0])
    assert coord == flat, "coordinating more names must not deepen the tree"


# --- Style profile ---------------------------------------------------------

CORPUS_A = [
    "The market moved. We watched it move, and we did nothing, because doing "
    "nothing was the position we had argued for in March.",
    "Rates fell again. The desk had expected this, though not the speed of it, "
    "and the book was not positioned for speed.",
    "We met the founder twice. She was better the second time, which is the "
    "wrong direction for a founder to travel in a process this short.",
]
CORPUS_B = [
    "Short. Very short. Almost curt.",
    "No verbs sometimes. Just fragments. Like this.",
    "Brief again. Terse. Done.",
]


def test_profile_measures_distribution_not_just_mean():
    from cadence.profile import build_profile
    p = build_profile(CORPUS_A, name="a")
    assert p.documents == 3 and p.sentences >= 3
    assert p.sentence_length.mean > 0
    assert p.sentence_length.p90 >= p.sentence_length.p10
    assert p.depth.mean > 0


def test_profile_warns_on_thin_corpus():
    from cadence.profile import build_profile
    p = build_profile(["One sentence only."], name="thin")
    assert any("distribution" in w for w in p.warnings)
    assert any("Single document" in w for w in p.warnings)


def test_profile_rejects_empty_corpus():
    from cadence.profile import build_profile
    with pytest.raises(ValueError):
        build_profile(["   ", ""])


def test_distinct_styles_diverge_more_than_a_style_from_itself():
    from cadence.profile import build_profile, divergence
    a, b = build_profile(CORPUS_A, "a"), build_profile(CORPUS_B, "b")
    self_sim = divergence(a, a)["similarity"]
    cross_sim = divergence(a, b)["similarity"]
    assert self_sim == 100.0, "a profile must be identical to itself"
    assert cross_sim < self_sim, "different styles must score lower than self-match"


def test_divergence_penalises_flattened_variance():
    """Matching the mean while killing the spread is the robotic failure mode."""
    from cadence.profile import Stat, _stat_divergence
    target = Stat(mean=20, sd=8, p10=10, p50=20, p90=32, n=10)
    same_mean_no_spread = Stat(mean=20, sd=0, p10=20, p50=20, p90=20, n=10)
    assert _stat_divergence(target, same_mean_no_spread) > 0


def test_spec_text_states_ranges_not_only_means():
    from cadence.profile import build_profile, spec_text
    spec = spec_text(build_profile(CORPUS_A, "a"))
    assert "typical range" in spec
    assert "Hit the RANGES, not the means" in spec


def test_generate_requires_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from cadence.generate import generate
    from cadence.profile import build_profile
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        generate("a brief", mode="profile", profile=build_profile(CORPUS_A))


def test_profile_mode_never_sees_the_samples(monkeypatch):
    """The whole claim: structure transfers without the content transferring."""
    sink = _install_fake(monkeypatch, "generated prose here, of some length.")
    from cadence.generate import generate
    from cadence.profile import build_profile
    generate("write about tapirs", mode="profile", profile=build_profile(CORPUS_A),
             samples=CORPUS_A)
    prompt = sink["messages"][0]["content"]
    assert "SYNTACTIC TARGET PROFILE" in prompt
    for sample in CORPUS_A:
        assert sample not in prompt, "profile mode must not leak sample content"


def test_tone_mode_does_see_the_samples(monkeypatch):
    sink = _install_fake(monkeypatch, "generated prose here.")
    from cadence.generate import generate
    from cadence.profile import build_profile
    generate("write about tapirs", mode="tone", profile=build_profile(CORPUS_A),
             samples=CORPUS_A)
    assert CORPUS_A[0] in sink["messages"][0]["content"]


def test_verify_loop_survives_a_failed_correction(monkeypatch):
    """A failed iteration must not discard the attempts that already worked."""
    import sys
    import types

    from cadence.profile import build_profile
    calls = {"n": 0}

    class M:
        def create(self, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                class R:
                    content = [_Block("The market moved slowly, and we watched it move.")]
                    stop_reason = "end_turn"
                return R()
            raise RuntimeError("simulated API failure")

    class C:
        def __init__(self, api_key=None): self.messages = M()

    mod = types.ModuleType("anthropic")
    mod.Anthropic = C
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    from cadence.generate import generate
    r = generate("brief", mode="profile_verify", profile=build_profile(CORPUS_A),
                 iterations=2)
    assert r.text, "the good first attempt must survive"
    assert r.similarity > 0
    assert "Skipped" in r.note


# --- Restyle: content and length are the input, not a topic -----------------

def test_restyle_sends_the_source_and_a_length_target(monkeypatch):
    sink = _install_fake(monkeypatch, "Short restyled output.")
    from cadence.generate import generate
    from cadence.profile import build_profile
    src = "The company is early. We met the founder twice, and liked him."
    generate(src, mode="profile", profile=build_profile(CORPUS_A), task="restyle")
    prompt = sink["messages"][0]["content"]
    assert "TEXT TO RESTYLE" in prompt
    assert src in prompt
    assert "content tokens" in prompt
    assert "PRESERVE EVERY FACT" in sink["system"]
    assert "INVENT NOTHING" in sink["system"]


def test_compose_mode_has_no_restyle_rules(monkeypatch):
    sink = _install_fake(monkeypatch, "New prose about tapirs.")
    from cadence.generate import generate
    from cadence.profile import build_profile
    generate("tapirs", mode="profile", profile=build_profile(CORPUS_A), task="compose")
    assert "INVENT NOTHING" not in sink["system"]
    assert "BRIEF -- write about this" in sink["messages"][0]["content"]


def test_result_reports_length_ratio(monkeypatch):
    _install_fake(monkeypatch, "One two three four five six seven eight nine ten.")
    from cadence.generate import generate
    from cadence.profile import build_profile
    src = "The company is early. We met the founder twice, and liked him."
    r = generate(src, mode="profile", profile=build_profile(CORPUS_A), task="restyle")
    assert r.source_tokens > 0 and r.output_tokens > 0
    assert r.length_ratio == round(r.output_tokens / r.source_tokens, 2)


def test_length_correction_fires_when_output_balloons():
    from cadence.generate import _length_note
    note = _length_note(100, " ".join(["word"] * 700))
    assert "LENGTH IS WRONG" in note and "far too long" in note
    assert _length_note(100, " ".join(["word"] * 100)) == ""


# --- remove-slop layer -----------------------------------------------------

SLOPPY = ("In today's fast paced world, motivation isn't just about discipline. "
          "Here's the thing: most people don't realize the real power of it. "
          "The upshot: you need better goals. Structure matters.")


def test_slop_detects_the_skills_own_examples():
    got = {f.code for f in analyze(SLOPPY).findings}
    for code in ("SLOP_VAPID_OPENER", "SLOP_RHETORICAL", "SLOP_PATRONIZING",
                 "SLOP_SUMMARY_INTRO", "SLOP_INTENSIFIER", "SLOP_APPLAUSE"):
        assert code in got, f"{code} not detected"


def test_slop_is_not_optional():
    """One pass. Structural and slop findings arrive in the same report."""
    codes = {f.code for f in analyze(SLOPPY).findings}
    assert any(c.startswith("SLOP_") for c in codes)
    import inspect

    from cadence.diagnostics import analyze as fn
    assert "slop" not in inspect.signature(fn).parameters, "must not be a toggle"


def test_slop_flags_dashes_and_hyphens():
    got = {f.code for f in analyze("The read is thin — a real problem. "
                                   "The deal-team met twice.").findings}
    assert "SLOP_DASH" in got and "SLOP_HYPHEN" in got


def test_uniform_rhythm_detected_numerically():
    """Rule 6.3 is a claim about variance, so it is measured, not guessed."""
    uniform = " ".join(["The team reviewed the deal and wrote a note."] * 6)
    varied = ("The team met. After a long and genuinely difficult conversation "
              "about the numbers, which nobody had checked, they agreed to wait. "
              "Briefly. Then the analyst produced a revised model that changed "
              "the picture entirely and the discussion restarted. It ended.")
    assert "SLOP_UNIFORM_RHYTHM" in {f.code for f in analyze(uniform).findings}
    assert "SLOP_UNIFORM_RHYTHM" not in {f.code for f in analyze(varied).findings}


def test_slop_rules_always_reach_the_rewrite_prompt(monkeypatch):
    sink = _install_fake(monkeypatch, EMPTY_PAYLOAD)
    from cadence.rewrite import rewrite
    rewrite(analyze("Liked him."), backend="llm")
    assert "REMOVE-SLOP RULES" in sink["system"]


def test_slop_rules_always_reach_the_generator(monkeypatch):
    sink = _install_fake(monkeypatch, "Restyled output.")
    from cadence.generate import generate
    from cadence.profile import build_profile
    generate("Source text.", mode="profile", profile=build_profile(CORPUS_A),
             task="restyle")
    assert "REMOVE-SLOP RULES" in sink["system"]
    assert "PRESERVE EVERY FACT" in sink["system"], "restyle rules must survive too"


def test_banned_feature_is_measured_but_never_a_target(monkeypatch):
    """The profile stays a complete description; the ban governs the target."""
    _install_fake(monkeypatch, "Restyled output text here.")
    from cadence.generate import generate
    from cadence.profile import build_profile, spec_text
    dashy = ["The read is thin — a real problem. We waited — briefly — then moved.",
             "The deal stalled — nobody pushed — and it closed anyway.",
             "He was late — again — so the meeting slipped."]
    prof = build_profile(dashy)
    assert prof.rates["punct_predication_per_sentence"] > 0, "still measured"
    assert "NOT A TARGET" in spec_text(prof), "but excluded from the target"
    r = generate("Some source text.", mode="profile", profile=prof, task="restyle")
    assert "Overridden by remove-slop" in r.note


def test_uniform_corpus_does_not_become_a_uniformity_target():
    """Where profile and slop overlap on rhythm, the floor wins."""
    from cadence.profile import build_profile, spec_text
    spec = spec_text(build_profile(["The team met and wrote a note."] * 8))
    assert "Do NOT reproduce that uniformity" in spec


def test_adverbial_superlative_is_not_a_ranking():
    """`most recently` is a fixed adverbial; `most active` is a real ranking."""
    adverbial = codes("It has shipped since 2008, most recently the Mark IV engine.")
    assert not [c for c in adverbial if c.startswith("SUPERLATIVE")]
    assert "SUPERLATIVE_NO_SET" in codes("The founder case is the strongest element.")


def test_anaphora_severity_depends_on_a_local_antecedent():
    a = analyze("Longbourn is the concessionaire, and it has manufactured servers.")
    local = [f for f in a.findings if f.code == "PROP_ANAPHORA"]
    assert local and local[0].severity == "low"
    b = analyze("That is not incidental to the thesis.")
    loose = [f for f in b.findings if f.code == "PROP_ANAPHORA"]
    assert loose and loose[0].severity == "medium"


# --- Abstraction before evidence -------------------------------------------
# Fixtures are the four sentences a reader identified by hand while the ported
# string lists caught none of them. Each is the same move in a different surface
# form, which is why they are detected structurally rather than by matching.

EVIDENCE_TAIL = (" Okonkwo built and sold a Ghanaian cocoa export business on roughly "
                 "$20 to $25M raised. She was the first hire at Meridian Foundry, a Danish "
                 "shipyard whose $15M seed is documented.")


def test_count_announcement():
    got = codes("Three things are worth taking seriously." + EVIDENCE_TAIL)
    assert "SLOP_COUNT_ANNOUNCEMENT" in got


def test_count_announcement_generalises_past_the_wording():
    for opener in ("Two reasons stand out here and they are important.",
                   "Several factors are worth noting.",
                   "A few points are significant."):
        assert "SLOP_COUNT_ANNOUNCEMENT" in codes(opener + EVIDENCE_TAIL), opener


def test_negative_definition():
    got = codes("The founders are not generalists." + EVIDENCE_TAIL)
    assert "SLOP_NEGATIVE_DEFINITION" in got


def test_meta_evaluation_rates_its_own_claim():
    assert "SLOP_META_EVALUATION" in codes("The Longbourn route is real and checkable."
                                           + EVIDENCE_TAIL)


def test_contrast_theatric():
    got = codes("The framing is a coherent structural argument, not a slogan."
                + EVIDENCE_TAIL)
    assert "SLOP_CONTRAST_THEATRIC" in got


def test_contrast_theatric_is_not_double_reported():
    """One `not` must not surface as both a contrast and a negative definition."""
    got = codes("The framing is a coherent structural argument, not a slogan."
                + EVIDENCE_TAIL)
    assert not ("SLOP_NEGATIVE_DEFINITION" in got and "SLOP_CONTRAST_THEATRIC" in got)


def test_concrete_claims_are_not_flagged():
    """The detector must not fire on sentences that carry their own evidence."""
    clean = ("Okonkwo built and sold a Ghanaian cocoa export business on roughly "
             "$20 to $25M raised. Meridian Foundry has built Whitfield hulls in the "
             "Aarhus Free Trade Zone since 2008.")
    got = [c for c in codes(clean) if c in {
        "SLOP_COUNT_ANNOUNCEMENT", "SLOP_NEGATIVE_DEFINITION", "SLOP_META_EVALUATION",
        "SLOP_CONTRAST_THEATRIC", "SLOP_ABSTRACTION_FIRST"}]
    assert not got, f"false positives on concrete prose: {got}"
