"""Detectors for the remove-slop rules.

Ported from skills/remove-slop (v2.0). The reason the rules are worth porting
here rather than leaving as prompt text: most of them are mechanically
checkable, and a rule you can measure is a rule you can hold a rewrite to.

Rule 6 is the one that benefits most. "Uniform rhythm is a stronger tell than
any single word" is a claim about VARIANCE, and variance is exactly what
profile.py already computes. So slop's structural rule is checked numerically
instead of by eye.

These are normative. They belong to the review path, not the restyle path,
where the corpus is the authority and its habits are signature rather than
defect. See CONFLICT_NOTE.
"""

from __future__ import annotations

import re
import statistics

from .languages import is_finite

# Rule 2 -- rhetorical framing.
RHETORICAL = [
    "zoom out", "step back", "big picture", "at a high level",
    "here's the thing", "here's where it gets interesting", "here's the kicker",
    "and that's the point", "so here's why this matters",
]
RHETORICAL_OPENERS = ["look,", "now,"]
VAPID_OPENERS = [
    "in today's fast paced world", "in today's fast-paced world", "in an era of",
    "as technology evolves",
]

# Rule 3 -- summary-remark intros.
SUMMARY_INTROS = [
    "the upshot", "in short", "in sum", "in summary", "overall", "put simply",
    "simply put", "bottom line", "the takeaway", "long story short", "all in all",
    "to sum up", "net net", "at the end of the day",
]

# Rule 5 -- patronizing and evaluative framing.
PATRONIZING = [
    "most people don't realize", "most people dont realize", "few understand",
    "what nobody tells you", "to be honest", "let me be blunt", "the hard truth is",
]
EVALUATIVE_SUPERLATIVES = [
    "the most important", "the strongest", "the single biggest", "the most critical",
]
SLOP_INTENSIFIERS = [
    "transformative", "delve", "profound landscape", "unlock", "real power",
    "key insight", "the irony", "journey not a destination",
]

# The abstraction-first family. Not in the skill's banned lists, because no list
# can hold it: the tell is that an evaluative abstraction stands where its own
# evidence would be stronger, and it is visible only by comparing a sentence
# against the ones that follow it.
EVALUATIVE_ADJECTIVES = {
    "coherent", "real", "checkable", "credible", "compelling", "solid", "serious",
    "strong", "important", "clear", "obvious", "notable", "significant",
    "interesting", "remarkable", "striking", "telling", "meaningful",
    "substantial", "genuine", "legitimate", "valid", "sound", "robust",
    "impressive", "unusual", "rare", "key", "critical", "essential", "worth",
}

# Nouns that name a claim rather than a thing, so rating them rates your own work.
CLAIM_NOUNS = {
    "argument", "framing", "route", "case", "story", "read", "point", "thesis",
    "claim", "logic", "rationale", "narrative", "pitch", "angle", "premise",
    "reasoning", "signal", "setup",
}

# Container nouns used to announce a count instead of making the points.
COUNT_CONTAINERS = {
    "things", "reasons", "points", "factors", "takeaways", "considerations",
    "observations", "elements", "aspects", "items", "questions", "issues",
    "themes", "reads",
}

# How much concrete material the following sentences must carry before a
# contentless opener counts as abstraction-before-evidence.
CONCRETE_LOOKAHEAD = 3
CONCRETE_THRESHOLD = 2

# Rule 4 -- applause lines: short declaratives used as emotional punctuation.
APPLAUSE_MAX_TOKENS = 6
APPLAUSE_NEIGHBOUR_RATIO = 2.0

# Rule 6.3 -- uniform rhythm. Coefficient of variation below this reads mechanical.
MIN_SENTENCES_FOR_RHYTHM = 5

# Dash connectors banned by rule 1. The ASCII hyphen is handled separately so
# that dates and identifiers are not flagged.
DASHES = {"—", "–", "--"}

# Profile features that remove-slop bans. They are still MEASURED -- the profile
# is a description and stays complete -- but they are never handed to a generator
# as something to reproduce. This is where the two layers meet: the profile owns
# structure, remove-slop owns surface machine tells, and where they overlap the
# ban wins and says so.
BANNED_PROFILE_FEATURES = {
    "punct_predication_per_sentence":
        "colon/dash predication is banned by remove-slop §1",
}

# remove-slop §6.3 sets a floor under rhythm variation. If a corpus is itself
# uniform, its variance is not a target worth hitting: reproducing it would
# reproduce the strongest machine tell there is.
VARIANCE_FLOOR_CV = MIN_LENGTH_CV = 0.25

# Injected into rewrite prompts when the rules are enabled.
PROMPT_RULES = """\

REMOVE-SLOP RULES. These are hard constraints on your output.

1. No hyphens or dashes in prose. Write hyphenated compounds as separate or
   closed words. Replace em dashes, en dashes and double hyphens with a period,
   comma, colon or parentheses.
2. No rhetorical framing: no camera metaphors (zoom out, step back, big
   picture), no staged reveals (here's the thing, here's the kicker), no
   rhetorical questions used to set up an answer, no "it's not X, it's Y"
   punchlines, no faux conversational openers, no vapid scene setting.
3. No summary-remark intros. Never open a sentence or paragraph with the
   upshot, in short, in sum, overall, put simply, bottom line, the takeaway,
   at the end of the day. State the conclusion directly.
4. No applause lines. Cut short punchy declaratives that work as emotional
   punctuation. If a sentence could stand alone as a LinkedIn post, fold it back
   into the argument.
5. No patronizing or evaluative framing: no "most people don't realize", no
   "the most important" or "the strongest" telling the reader what to think, no
   performative honesty, no slop intensifiers (transformative, delve, unlock,
   key insight).
6. Vary structure. Do not give every paragraph the same length or the same arc.
   Vary sentence length within paragraphs; uniform rhythm is a stronger tell
   than any single word. Keep qualifiers and asides (usually, sort of, I think)
   rather than polishing them out. Allow run on sentences joined with and, but,
   so where the voice supports them.
7. Never put an abstraction in front of its own evidence. Do not announce a
   count ("three things are worth taking seriously"), do not define by negation
   against a category nobody proposed ("the founders are not generalists"), do
   not rate your own claim ("a coherent structural argument"), and do not reject
   a strawman for emphasis ("not a slogan"). If concrete material follows,
   delete the abstraction and let it run. If a transition is genuinely needed,
   write the sharpest checkable summary of the evidence instead: "both founders
   have built and exited physical compute businesses" beats "the founders are
   not generalists" because a reader can check it."""


def _find(text_lower: str, needles: list[str]) -> list[str]:
    return [n for n in needles if n in text_lower]


def detect(doc, Finding) -> list:
    """Run every remove-slop check over a parsed document."""
    findings = []
    sents = [s for s in doc.sents if [t for t in s if not t.is_punct]]

    def add(code, label, origin, idx, span, detail, fix, severity="medium", auto=False):
        findings.append(Finding(code, label, origin, idx, span, detail, fix, auto, severity))

    # Rule 1 -- dashes and hyphenated compounds.
    for i, sent in enumerate(sents, 1):
        dashes = [t for t in sent if t.text in DASHES]
        hyphenated = [
            t for t in sent
            if ("-" in t.text and t.is_alpha is False
                and re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)+", t.text))
        ]
        if dashes:
            add("SLOP_DASH", f"{len(dashes)} dash connector(s)", "remove-slop §1", i,
                sent.text.strip()[:120],
                "Dashes used as sentence connectors. The rule bans them outright.",
                "Replace with a period, comma, colon or parentheses.", "medium", True)
        if hyphenated:
            add("SLOP_HYPHEN",
                f"hyphenated compound(s): {', '.join(t.text for t in hyphenated[:3])}",
                "remove-slop §1", i, sent.text.strip()[:120],
                "Hyphenated compounds are banned in prose.",
                "Write as separate or closed words.", "low", False)

    # Rules 2, 3, 5 -- banned lexical patterns.
    for i, sent in enumerate(sents, 1):
        low = sent.text.lower().strip()
        for hits, code, label, origin, fix in (
            (_find(low, RHETORICAL), "SLOP_RHETORICAL", "rhetorical framing", "remove-slop §2",
             "State the point instead of staging it."),
            (_find(low, VAPID_OPENERS), "SLOP_VAPID_OPENER", "vapid scene setting",
             "remove-slop §2",
             "Delete the opener and start with the claim."),
            (_find(low, PATRONIZING), "SLOP_PATRONIZING", "patronizing framing", "remove-slop §5",
             "Drop it and make the claim directly."),
            (_find(low, EVALUATIVE_SUPERLATIVES), "SLOP_EVALUATIVE", "evaluative superlative",
             "remove-slop §5", "Show why it matters instead of ranking it for the reader."),
            (_find(low, SLOP_INTENSIFIERS), "SLOP_INTENSIFIER", "slop intensifier",
             "remove-slop §5",
             "Replace with the specific thing you mean."),
        ):
            if hits:
                add(code, f"{label}: {', '.join(repr(h) for h in hits[:3])}", origin, i,
                    sent.text.strip()[:120],
                    f"Banned by remove-slop: {', '.join(hits)}.", fix)
        if any(low.startswith(p) for p in SUMMARY_INTROS) or any(
                low.startswith(p) for p in RHETORICAL_OPENERS):
            add("SLOP_SUMMARY_INTRO", "summary-remark or conversational opener",
                "remove-slop §3", i, sent.text.strip()[:120],
                "The sentence opens by announcing a summary instead of stating it.",
                "Delete the opener; state the conclusion directly.", "medium", True)

    # Rule 4 -- applause lines.
    lengths = [len([t for t in s if not t.is_punct]) for s in sents]
    for i, sent in enumerate(sents, 1):
        n = lengths[i - 1]
        if n == 0 or n > APPLAUSE_MAX_TOKENS:
            continue
        has_finite = any(is_finite(t) for t in sent)
        has_subordination = any(
            t.dep_ in {"advcl", "ccomp", "relcl", "acl", "xcomp"} for t in sent)
        prev_len = lengths[i - 2] if i >= 2 else 0
        if has_finite and not has_subordination and prev_len >= n * APPLAUSE_NEIGHBOUR_RATIO:
            add("SLOP_APPLAUSE", f"applause line ({n} tokens after a {prev_len}-token sentence)",
                "remove-slop §4", i, sent.text.strip(),
                "A short declarative landing after a long sentence reads as emotional "
                "punctuation rather than argument.",
                "Fold it back into the surrounding paragraph, or cut it.")

    # Rule 6.3 -- uniform rhythm, measured.
    usable = [n for n in lengths if n]
    if len(usable) >= MIN_SENTENCES_FOR_RHYTHM:
        mean = statistics.fmean(usable)
        sd = statistics.pstdev(usable)
        cv = sd / mean if mean else 0.0
        if cv < MIN_LENGTH_CV:
            add("SLOP_UNIFORM_RHYTHM", f"uniform sentence rhythm (variation {cv:.2f})",
                "remove-slop §6.3", 0,
                f"{len(usable)} sentences, mean {mean:.1f} tokens, sd {sd:.1f}",
                f"Sentence lengths vary by only {cv:.0%} of the mean. Uniform rhythm is "
                "a stronger machine tell than any single word.",
                f"Vary sentence length deliberately; aim for variation above "
                f"{MIN_LENGTH_CV:.0%}.", "high")

    # Rule 6.1/6.2 -- uniform paragraph length.
    paras = [p for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
    if len(paras) >= 4:
        plens = [len(p.split()) for p in paras]
        pmean = statistics.fmean(plens)
        pcv = statistics.pstdev(plens) / pmean if pmean else 0.0
        if pcv < 0.20:
            add("SLOP_UNIFORM_PARAGRAPHS", f"uniform paragraph length (variation {pcv:.2f})",
                "remove-slop §6.1", 0,
                f"{len(paras)} paragraphs, mean {pmean:.0f} words",
                "Every paragraph is close to the same length, which reads as a template.",
                "Let paragraph length follow the content.", "medium")

    findings.extend(_abstraction_first(sents, add))
    return findings


def _concreteness(sent) -> int:
    """Checkable material in a sentence: names, numbers, money, dates."""
    return len([
        t for t in sent
        if t.pos_ in {"PROPN", "NUM"} or t.ent_type_ in {"DATE", "MONEY", "CARDINAL",
                                                          "PERCENT", "ORG", "GPE", "PERSON"}
    ])


def _copular_predicate(sent):
    """(copula, complement, negated) for a simple copular clause, else Nones."""
    for tok in sent:
        if tok.lemma_ != "be" or tok.dep_ not in {"ROOT", "ccomp", "conj", "relcl"}:
            continue
        comp = next((c for c in tok.children if c.dep_ in {"attr", "acomp", "oprd"}), None)
        if comp is None:
            continue
        negated = any(c.dep_ == "neg" for c in tok.children) or any(
            c.dep_ == "neg" for c in comp.children)
        return tok, comp, negated
    return None, None, False


def _abstraction_first(sents, add):
    """The pattern the skill's lists cannot express.

    Fires when a sentence carries no checkable material, rates something instead,
    and is followed by sentences that do the work it claimed. The comparison
    against what follows is the whole detector: an evaluation is only slop when
    the evidence behind it is both present and sharper.
    """
    out = []
    concrete = [_concreteness(s) for s in sents]

    for i, sent in enumerate(sents):
        idx = i + 1
        cop, comp, negated = _copular_predicate(sent)
        following = sum(concrete[i + 1:i + 1 + CONCRETE_LOOKAHEAD])
        subj = next((t for t in sent if t.dep_ in {"nsubj", "nsubjpass"}), None)

        # Cardinality announcement: a count plus a container noun plus a rating.
        if subj is not None and subj.lower_ in COUNT_CONTAINERS and cop is not None:
            if any(c.pos_ == "NUM" or c.lower_ in {"few", "several", "some"}
                   for c in subj.children):
                add("SLOP_COUNT_ANNOUNCEMENT",
                    f"announces a count instead of making the point: {subj.text!r}",
                    "remove-slop §3 (extended)", idx, sent.text.strip(),
                    "The sentence tells the reader a list is coming and rates it in "
                    "advance. It carries no claim of its own, and the points that follow "
                    "introduce themselves.",
                    "Delete it and start on the first point.", "high", True)

        if cop is None or comp is None:
            continue

        # "X, not Y" trailing the complement is the contrast theatric; the same
        # `not` must not also be reported as a negative definition.
        trailing_contrast = bool([t for t in sent if t.lower_ == "not" and t.i > comp.i])

        # Negative definition: defined against a category nobody proposed.
        if negated and comp.pos_ in {"NOUN", "PROPN"} and not trailing_contrast:
            add("SLOP_NEGATIVE_DEFINITION",
                f"defined by negation: 'not {comp.text}'",
                "remove-slop §2.4 (extended)", idx, sent.text.strip(),
                f"States what the subject is not, against a category the text never "
                f"proposed. The reader was not thinking {comp.text!r} until told not to.",
                "Say what it is, in the sharpest checkable terms the evidence supports.",
                "high", False)

        # Contrast theatric: "X, not Y" as a punchline.
        if trailing_contrast and comp.pos_ in {"NOUN", "ADJ"}:
            add("SLOP_CONTRAST_THEATRIC",
                "contrast theatric: asserts, then rejects a strawman",
                "remove-slop §2.4", idx, sent.text.strip(),
                "The negated alternative is invented in order to be refused. Nobody "
                "proposed it, so refusing it defends against nothing and plants the "
                "word next to your claim.",
                "Drop the negated half and let the positive claim stand.", "high", True)

        # Meta-evaluation: rating a claim-noun instead of demonstrating it.
        rates_claim = (
            (subj is not None and subj.lower_ in CLAIM_NOUNS)
            or comp.lower_ in CLAIM_NOUNS
            or any(c.lower_ in CLAIM_NOUNS for c in comp.children)
        )
        evaluative = (
            comp.lower_ in EVALUATIVE_ADJECTIVES
            or any(c.lower_ in EVALUATIVE_ADJECTIVES for c in comp.children)
            or any(c.dep_ == "conj" and c.lower_ in EVALUATIVE_ADJECTIVES
                   for c in comp.children)
        )
        if rates_claim and evaluative:
            add("SLOP_META_EVALUATION",
                "rates its own argument instead of demonstrating it",
                "remove-slop §5 (extended)", idx, sent.text.strip(),
                "A claim is labelled good rather than shown to be good. The reader "
                "cannot check the label, only the evidence, so the label spends "
                "credibility without earning it.",
                "Cut the rating and let the evidence carry the judgement.", "high", False)

        # The general case: no checkable material, rating something, and the
        # following sentences supply exactly what it claimed.
        if (_concreteness(sent) == 0 and evaluative
                and following >= CONCRETE_THRESHOLD):
            add("SLOP_ABSTRACTION_FIRST",
                "abstraction placed in front of its own evidence",
                "remove-slop §6.4 (extended)", idx, sent.text.strip(),
                f"This sentence carries no checkable material, while the next "
                f"{CONCRETE_LOOKAHEAD} carry {following} concrete items. The "
                "abstraction states the conclusion of evidence the reader has not seen, "
                "usually in vaguer terms than the evidence supports.",
                "Delete it and let the facts run, or replace it with the sharpest "
                "checkable summary of them.", "high", True)
    return out
