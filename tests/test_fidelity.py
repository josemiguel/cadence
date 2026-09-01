"""The content check, including the cases where it must NOT fire.

A faithfulness metric that flags every legitimate rewrite is worse than none:
it trains the reader to ignore it. Half of these tests are false-positive
guards for exactly that reason.
"""

from __future__ import annotations

from cadence.diagnostics import analyze
from cadence.fidelity import NOVEL_LIMIT, fidelity, gap_sentence_map
from cadence.rewrite import rewrite

SOURCE = (
    "Halvorsen Dairy did not renew the 2024-03-11 contract. Revenue fell 12% to "
    "$4.2M, and the board was told in April."
)


def kinds(anchors):
    return {(a.kind, a.text) for a in anchors}


# --- the metric floor and ceiling ------------------------------------------

def test_identical_text_scores_full_marks():
    f = fidelity(SOURCE, SOURCE)
    assert f.score == 100.0 and f.passed
    assert not f.dropped and not f.introduced


def test_unrelated_text_scores_badly():
    f = fidelity(SOURCE, "The weather in Lisbon turned in the second week of June.")
    assert f.score < 40 and not f.passed


# --- dropping ---------------------------------------------------------------

def test_a_dropped_number_is_caught():
    out = "Halvorsen Dairy did not renew the 2024-03-11 contract. The board was told in April."
    f = fidelity(SOURCE, out)
    assert not f.passed
    assert ("number", "12.0") in kinds(f.dropped)


def test_a_dropped_date_is_caught():
    out = "Halvorsen Dairy did not renew the contract. Revenue fell 12% to $4.2M in April."
    f = fidelity(SOURCE, out)
    assert ("date", "2024-03-11") in kinds(f.dropped)


def test_a_drop_named_in_a_reported_gap_is_excused():
    out = "Halvorsen Dairy did not renew the 2024-03-11 contract. Revenue fell 12%."
    unexcused = fidelity(SOURCE, out)
    excused = fidelity(SOURCE, out, gaps=["figure 4200000.0 could not be placed"])
    assert not unexcused.passed
    assert ("number", "4200000.0") in kinds(excused.excused)
    assert ("number", "4200000.0") not in kinds(excused.dropped)


# --- introducing ------------------------------------------------------------

def test_an_invented_person_fails_outright():
    f = fidelity(SOURCE, SOURCE + " Ann Lee signed it.")
    assert not f.passed
    assert ("entity", "ann lee") in kinds(f.introduced)


def test_a_self_reported_inference_does_not_excuse_an_introduction():
    f = fidelity(SOURCE, SOURCE + " Ann Lee signed it.",
                 inferences=["`Ann Lee` inferred as the signatory"])
    assert not f.passed, "a model's own account must not overturn a measurement"
    assert "inference" in f.introduced[0].note


def test_an_introduction_on_a_gap_sentence_is_labelled_as_a_probable_fill():
    src = "No view was being taken until the visit."
    analysis = analyze(src)
    f = fidelity(src, "Ann Lee took no view until the visit.",
                 gap_sentences=gap_sentence_map(analysis))
    assert f.introduced
    assert "gap fill" in f.introduced[0].note


# --- number normalisation ---------------------------------------------------

def test_spelled_and_digit_numbers_are_the_same_number():
    assert fidelity(SOURCE, SOURCE.replace("12%", "twelve percent")).passed


def test_abbreviated_and_expanded_money_are_the_same_number():
    f = fidelity(SOURCE, SOURCE.replace("$4.2M", "4,200,000 dollars"))
    assert f.passed, [a.text for a in f.dropped + f.introduced]


def test_an_iso_date_is_one_anchor_not_three():
    f = fidelity(SOURCE, SOURCE)
    assert ("date", "2024-03-11") in kinds(f.preserved)


# --- negation ---------------------------------------------------------------

def test_a_dropped_negation_reverses_the_claim_and_fails():
    f = fidelity(SOURCE, SOURCE.replace("did not renew", "renewed"))
    assert not f.passed
    assert f.negations == (1, 0)


def test_a_negation_carried_by_the_verb_is_not_penalised():
    f = fidelity(SOURCE, SOURCE.replace("did not renew", "declined to renew"))
    assert f.passed, "`declined` carries the negation; the claim is unchanged"


# --- false-positive guards --------------------------------------------------

def test_a_synonym_swap_survives_the_free_allowance():
    assert fidelity(SOURCE, SOURCE.replace("Revenue", "Turnover")).passed


def test_a_derivational_change_is_not_novel_content():
    src = "The decision was taken on 2024-03-11."
    f = fidelity(src, "They decided on 2024-03-11.")
    assert "decide" not in f.novel_lemmas


def test_a_pronoun_resolved_to_a_name_from_the_source_is_not_invented():
    src = "Ann Lee read the note. Liked it."
    f = fidelity(src, "Ann Lee read the note. Ann Lee liked it.")
    assert not f.introduced


def test_a_sentence_initial_common_noun_is_not_a_fabricated_entity():
    src = "revenue fell in April."
    f = fidelity(src, "Revenue fell in April.")
    assert not f.introduced


# --- elaboration ------------------------------------------------------------

def test_heavy_elaboration_fails_on_the_novel_rate():
    padded = SOURCE + (" This is a serious structural problem for the category, "
                       "reflecting deeper competitive dynamics across the sector "
                       "and raising uncomfortable questions about the thesis.")
    f = fidelity(SOURCE, padded)
    assert f.novel_rate > NOVEL_LIMIT and not f.passed


def test_wholesale_dropping_shows_up_as_uncovered_sentences():
    f = fidelity(SOURCE, "Revenue fell 12% to $4.2M.")
    assert 1 in f.uncovered_sentences and f.coverage < 1.0


# --- integration ------------------------------------------------------------

def test_the_rules_backend_cannot_lose_content():
    """It repeats a copula and nothing else, so it must score full marks."""
    a = analyze("She is a teacher, patient, and writing daily.")
    r = rewrite(a, backend="rules")
    assert r.fidelity.score == 100.0 and r.fidelity.passed


def test_gap_findings_map_to_their_sentences():
    a = analyze("The pricing read is the strongest element. No view was being taken.")
    m = gap_sentence_map(a)
    assert m and all(isinstance(k, int) for k in m)


def test_notes_always_state_the_limit_of_the_measure():
    f = fidelity(SOURCE, SOURCE)
    assert any("floor" in n for n in f.notes)
