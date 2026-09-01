"""Parse repairs for note-register English.

`en_core_web_sm` is trained on newswire and makes two errors that matter a lot
for note-register prose, both verified against a hand-checked corpus:

R1  ISO dates are shattered. `2026-08-24` becomes five tokens (`2026`, `-`,
    `08`, `-`, `24`) and the fragments pick up junk deps (`dobj`, `prep`).

R2  A noun-noun compound whose head noun is verb-ambiguous gets read as a
    clause. `the founder read is positive` parses with `read` as a VERB and
    `founder` as its subject, instead of `read` as the NP head with `founder`
    as a compound modifier. This is the single most common construction in the
    register (`founder read`, `category read`, `lead read`), so leaving it
    unrepaired makes every downstream diagnostic unreliable.

Both repairs are conservative and each one records itself on the Doc, so the
report can say which sentences were touched and how confident the parse is.
"""

from __future__ import annotations

import re

from spacy.tokens import Doc

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Hyphenated compounds (`Deal-team`, `post-visit`, `deal-team`) are single words
# in this register; the default tokenizer splits them on the hyphen infix.
HYPHEN_COMPOUND = re.compile(r"^[A-Za-z]+(?:-[A-Za-z]+)+$")

# Nouns that a newswire parser is prone to read as verbs in this register.
VERB_AMBIGUOUS_NOUNS = {
    "read", "call", "take", "view", "pass", "lead", "check", "flag", "focus",
    "raise", "spend", "build", "look", "sense", "hold", "run", "cut", "bet",
}

_FINITE_TAGS = {"VBZ", "VBD", "VBP", "MD"}

if not Doc.has_extension("repairs"):
    Doc.set_extension("repairs", default=None)


def install_tokenizer_rules(nlp) -> None:
    """R1: keep ISO dates and hyphenated compounds as single tokens.

    spaCy checks `token_match` before applying infix splits, so a whole-token
    pattern here survives the hyphen infix rule that would otherwise split it.
    """
    existing = nlp.tokenizer.token_match

    def token_match(text: str):
        if ISO_DATE.match(text) or HYPHEN_COMPOUND.match(text):
            return True
        return existing(text) if existing else None

    nlp.tokenizer.token_match = token_match


def _retag_iso_dates(doc: Doc, log: list[str]) -> None:
    for tok in doc:
        if ISO_DATE.match(tok.text) and tok.tag_ != "CD":
            log.append(f"R1 retag {tok.text!r}: {tok.tag_} -> CD")
            tok.tag_ = "CD"
            tok.pos_ = "NUM"


def _repair_compound_heads(doc: Doc, log: list[str]) -> None:
    """R2: recover `N N` subject NPs misread as `N V` clauses."""
    for tok in doc:
        if tok.dep_ not in {"nsubj", "csubj"}:
            continue
        if tok.pos_ not in {"VERB", "AUX"}:
            continue
        if tok.lower_ not in VERB_AMBIGUOUS_NOUNS:
            continue
        head = tok.head
        # The real verb must be finite and must follow the misparsed noun.
        if head.tag_ not in _FINITE_TAGS or head.i <= tok.i:
            continue

        log.append(f"R2 {tok.text!r} (i={tok.i}): {tok.tag_}/{tok.dep_} -> "
                   f"NN/nsubj (noun head of subject NP)")
        tok.tag_ = "NN"
        tok.pos_ = "NOUN"
        # It is no longer a clausal subject once it is a noun.
        tok.dep_ = "nsubj"

        # Dependents that were analysed as arguments of the phantom verb are
        # really modifiers inside the NP.
        for child in list(tok.children):
            if child.dep_ in {"nsubj", "csubj", "dobj"} and child.i < tok.i:
                log.append(f"   {child.text!r}: {child.dep_} -> compound")
                child.dep_ = "compound"
                # A determiner on the modifier belongs to the NP head.
                for grand in list(child.children):
                    if grand.dep_ == "det":
                        grand.head = tok


def _repair_nominal_subjects(doc: Doc, log: list[str]) -> None:
    """R2b: a nominal cannot take a subject.

    Even when the head noun is tagged correctly, the parser often leaves the
    preceding modifier attached as `nsubj` (`founder` -nsubj-> `read`). Any
    subject-like arc landing on a NOUN/PROPN head is really a compound.
    """
    for tok in doc:
        if tok.dep_ not in {"nsubj", "csubj", "dobj"}:
            continue
        head = tok.head
        if head.pos_ not in {"NOUN", "PROPN"} or head.i <= tok.i:
            continue
        log.append(f"R2b {tok.text!r}: {tok.dep_} -> compound (head {head.text!r} is nominal)")
        tok.dep_ = "compound"
        for grand in list(tok.children):
            if grand.dep_ == "det":
                grand.head = head


def _repair_pronominal_one(doc: Doc, log: list[str]) -> None:
    """R3: `one` is NN when pronominal, CD when a numeral.

    `the 2026-08-24 one` (anaphoric, takes a determiner) vs `one of the areas`
    (partitive numeral, no determiner). The determiner is the discriminator.
    """
    for tok in doc:
        if tok.lower_ != "one" or tok.tag_ != "CD":
            continue
        if any(c.dep_ == "det" for c in tok.children):
            log.append(f"R3 {tok.text!r} (i={tok.i}): CD -> NN (pronominal, has determiner)")
            tok.tag_ = "NN"
            tok.pos_ = "NOUN"


# English marks case on pronouns: these forms cannot be subjects.
ACCUSATIVE_PRONOUNS = {"him", "her", "them", "me", "us", "whom"}


def _repair_accusative_subjects(doc: Doc, log: list[str]) -> None:
    """R4: an accusative pronoun after its verb is an object, not a subject.

    `liked him` (subject dropped) parses as `him`-nsubj-`liked`, which hides the
    null subject the register is full of. Case morphology settles it: `him`
    cannot be a subject in English.
    """
    for tok in doc:
        if tok.dep_ != "nsubj" or tok.lower_ not in ACCUSATIVE_PRONOUNS:
            continue
        if tok.i <= tok.head.i:
            continue
        log.append(f"R4 {tok.text!r}: nsubj -> dobj (accusative case cannot be a subject)")
        tok.dep_ = "dobj"


# §5.7 of the hand analysis stipulates: noun-noun hyphenated compound -> NN,
# adjectival-prefix + noun -> JJ. The tagger gets these backwards about as often
# as it gets them right, so the stipulated rule is applied directly.
ADJECTIVAL_PREFIXES = (
    "post", "pre", "non", "anti", "multi", "semi", "quasi", "ex", "self", "co",
    "sub", "super", "inter", "intra", "cross", "over", "under", "near", "long",
    "short", "high", "low", "well", "ill", "re",
)


def _repair_hyphen_compounds(doc: Doc, log: list[str]) -> None:
    """R5: tag hyphenated premodifiers by the §5.7 rule."""
    for tok in doc:
        if not HYPHEN_COMPOUND.match(tok.text) or tok.tag_ not in {"JJ", "NN", "NNP"}:
            continue
        first = tok.text.split("-")[0].lower()
        want = "JJ" if first in ADJECTIVAL_PREFIXES else "NN"
        if tok.tag_ == want:
            continue
        log.append(f"R5 {tok.text!r}: {tok.tag_} -> {want} (§5.7 hyphen rule)")
        tok.tag_ = want
        tok.pos_ = "ADJ" if want == "JJ" else "NOUN"


def apply(doc: Doc) -> Doc:
    """Run every repair, recording what fired on `doc._.repairs`."""
    log: list[str] = []
    _retag_iso_dates(doc, log)
    _repair_compound_heads(doc, log)
    _repair_nominal_subjects(doc, log)
    _repair_pronominal_one(doc, log)
    _repair_accusative_subjects(doc, log)
    _repair_hyphen_compounds(doc, log)
    doc._.repairs = log
    return doc


def confidence(sent) -> tuple[str, list[str]]:
    """Coarse parse-confidence signal for one sentence.

    Not a probability -- a list of structural reasons to distrust the parse.
    """
    reasons = []
    if not any(t.tag_ in _FINITE_TAGS for t in sent):
        reasons.append("no finite verb: fragment, or the head noun was read as a verb")
    for tok in sent:
        if tok.lower_ in VERB_AMBIGUOUS_NOUNS and tok.pos_ in {"VERB", "AUX"}:
            subjects = {"nsubj", "nsubjpass"}
            if tok.dep_ == "ROOT" and not any(c.dep_ in subjects for c in tok.children):
                reasons.append(f"{tok.text!r} read as a verb with no subject: may be a noun head")
    if any(t.dep_ == "dep" for t in sent):
        reasons.append("parser emitted an unlabelled `dep` arc")
    level = "low" if len(reasons) >= 2 else ("medium" if reasons else "high")
    return level, reasons
