"""Did the rewrite keep the content? Measured, not asserted.

Every promise this package makes about not inventing facts was, until now, a
line in a prompt plus the model's own report of whether it had obeyed. A model
that fabricates is not a reliable witness to its own fabrication, so the report
proved nothing. This module checks the output against the source mechanically:
no second model, no network, only the parse both texts already need.

WHAT IS CHECKED. Numbers, dates and named entities are anchors: they either
survive into the output or they do not. Novel content lemmas catch elaboration
that adds no anchor. Sentence coverage catches wholesale dropping. Negation
counts catch the reversal that changes a claim while keeping every word in it.

WHAT IS DELIBERATELY NOT CHECKED. Meaning. Two sentences can share every anchor
and say different things, and no lexical measure will notice. The score is a
floor on faithfulness, not a certificate of it, and the report says so.

ASYMMETRY IS THE POINT. Dropping something can be legitimate: the rewrite is
allowed to report a gap rather than fill it, and an anchor named in a reported
gap is excused. Introducing something never is. A name in the output that
appears nowhere in the source is the failure this whole tool exists to prevent,
so it fails the gate outright, whatever the score says.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from . import languages
from .syntax import parse

# --- thresholds ------------------------------------------------------------
# Stated as constants because the benchmark writeup quotes them and the tests
# pin them. Changing one is a change to what the published number means.

#: Novel content lemmas below this share of the output are ordinary rewriting.
#: A restyle that changes clause structure inevitably reaches for a few words
#: the source did not use; measured runs sit around 5 to 12 per cent.
NOVEL_FREE_ALLOWANCE = 0.10

#: Above this share, the output is elaborating rather than restyling.
NOVEL_LIMIT = 0.25

#: Content-lemma overlap below which a source sentence counts as uncovered.
COVERAGE_JACCARD = 0.20

#: Share of source sentences that must be covered somewhere in the output.
COVERAGE_LIMIT = 0.80

#: Introduced anchors are capped so one bad rewrite cannot drive the score
#: below zero and lose the difference between bad and much worse.
INTRODUCED_PENALTY = 15.0
INTRODUCED_PENALTY_CAP = 45.0

#: Shared-prefix rule that treats `decision` and `decide` as the same lemma.
DERIVATION_PREFIX = 4
DERIVATION_COVERAGE = 0.6

# The English model's labels, then the ones the Spanish and Portuguese models use.
_ENTITY_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "EVENT", "NORP", "WORK_OF_ART",
    "PER", "MISC",
}
_NUMERIC_LABELS = {"CARDINAL", "MONEY", "PERCENT", "QUANTITY", "ORDINAL"}
_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
# The word lists live in `languages`, one set per language: negators, the
# relative dates that name no calendar point ("recently" surviving or not says
# nothing about whether a fact was kept), the verbs that carry a negation in
# their meaning so a rewrite can drop a `not` without changing the claim, and
# the spelled-out numbers. The English ones are kept here under their old
# names for callers that imported them.
_EN = languages.get("en")
_NEGATORS = set(_EN.negators)
_RELATIVE_DATES = set(_EN.relative_dates)
_LEXICAL_NEGATORS = set(_EN.lexical_negators)
_WORD_NUMBERS = dict(_EN.word_numbers)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NUMERIC = re.compile(r"\d")
_SUFFIX_MULTIPLIERS = {"k": 1_000, "m": 1_000_000, "bn": 1_000_000_000,
                       "b": 1_000_000_000, "tn": 1_000_000_000_000}


@dataclass
class Anchor:
    """One checkable item of content, and where it was found."""

    kind: str  # number | date | entity
    text: str  # normalised surface, which is what comparison uses
    sentence: int  # 1-based; source index, except for introduced anchors
    note: str = ""


@dataclass
class Fidelity:
    """What survived the rewrite, what did not, and what appeared from nowhere."""

    score: float
    passed: bool
    preserved: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    excused: list = field(default_factory=list)
    introduced: list = field(default_factory=list)
    novel_lemmas: list = field(default_factory=list)
    novel_rate: float = 0.0
    uncovered_sentences: list = field(default_factory=list)
    coverage: float = 1.0
    negations: tuple = (0, 0)
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["negations"] = list(self.negations)
        return d


# --- normalisation ---------------------------------------------------------

def _normalise_number(text: str, language=_EN) -> str:
    """One surface form for `two`, `2`, `$4.2M`, `4,200,000` and `twelve percent`.

    Works on a whole span rather than a token, because `$4.2M` reaches the
    parser as three tokens and the multiplier is the part that matters.
    """
    numbers = language.word_numbers
    raw = text.strip().lower().replace(",", "")
    m = re.search(r"(\d+\.?\d*)\s*(bn|tn|[kmb])?\b", raw)
    if m:
        value = float(m.group(1))
        suffix = m.group(2)
        if suffix:
            value *= _SUFFIX_MULTIPLIERS[suffix]
        return str(value)
    # No digits: a spelled-out number, possibly with a multiplier word.
    words = [w for w in languages.WORD_SPLIT.split(raw) if w in numbers]
    if not words:
        return raw
    value = float(numbers[words[0]])
    for w in words[1:]:
        mult = numbers[w]
        if mult >= 100:
            value *= mult
        else:
            value += mult
    return str(value)


def _lemma_key(token) -> str:
    return (token.lemma_ or token.text).lower()


def _matches_derivation(lemma: str, known: set[str]) -> bool:
    """`decision` counts as `decide` kept, `company` does not count as `firm`."""
    if lemma in known:
        return True
    for other in known:
        shorter, longer = sorted((lemma, other), key=len)
        need = max(DERIVATION_PREFIX, int(len(shorter) * DERIVATION_COVERAGE))
        if len(shorter) >= DERIVATION_PREFIX and longer.startswith(shorter[:need]):
            return True
    return False


# --- extraction ------------------------------------------------------------

def _anchors(doc, language=_EN) -> list[Anchor]:
    out: list[Anchor] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, text: str, sent_index: int):
        key = (kind, text)
        if text and key not in seen:
            seen.add(key)
            out.append(Anchor(kind=kind, text=text, sentence=sent_index))

    sent_index = {}
    for i, sent in enumerate(doc.sents, start=1):
        for tok in sent:
            sent_index[tok.i] = i

    date_tokens = set()
    for ent in doc.ents:
        if ent.label_ != "DATE":
            continue
        for tok in ent:
            if tok.is_punct or tok.is_space:
                continue
            low = tok.text.lower()
            if low in language.relative_dates or tok.is_stop:
                continue
            date_tokens.add(tok.i)

    # Numeric entities are read as whole spans. `$4.2M` arrives as three tokens
    # and the multiplier lives in the last of them, so a per-token reading
    # records 4.2 and loses six orders of magnitude.
    numeric_span_tokens = set()
    for ent in doc.ents:
        if ent.label_ not in _NUMERIC_LABELS:
            continue
        text = ent.text.strip()
        if not (_NUMERIC.search(text) or any(w in language.word_numbers
                                             for w in languages.WORD_SPLIT.split(text.lower()))):
            continue
        add("number", _normalise_number(text, language), sent_index.get(ent.start, 0))
        numeric_span_tokens.update(t.i for t in ent)

    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        i = sent_index.get(tok.i, 0)
        if _ISO_DATE.match(tok.text):
            add("date", tok.text, i)
            continue
        if tok.i in date_tokens:
            add("date", tok.text.lower(), i)
            continue
        if tok.i in numeric_span_tokens:
            continue
        numeric = (tok.like_num or tok.tag_ == "CD" or tok.pos_ == "NUM"
                   or tok.ent_type_ in _NUMERIC_LABELS)
        if numeric and (_NUMERIC.search(tok.text) or tok.text.lower() in language.word_numbers):
            add("number", _normalise_number(tok.text, language), i)

    # Entities are token sets: a later mention that shortens `Meridian Foundry`
    # to `Meridian` still refers, so any surviving token counts as preserved.
    for ent in doc.ents:
        if ent.label_ not in _ENTITY_LABELS:
            continue
        toks = [t.text.lower() for t in ent if not t.is_punct and not t.is_space]
        if toks:
            add("entity", " ".join(toks), sent_index.get(ent.start, 0))
    covered = {t.i for ent in doc.ents for t in ent}
    for tok in doc:
        if tok.pos_ == "PROPN" and tok.i not in covered and not tok.is_stop:
            add("entity", tok.text.lower(), sent_index.get(tok.i, 0))
    return out


def _entity_vocabulary(doc) -> set[str]:
    """Every word in the text, lowercased. Used only to judge introductions.

    Deliberately every token, not only entity tokens: a name the source states
    as a common noun, or a sentence-initial word the tagger mislabels, must not
    read as fabricated.
    """
    return {t.text.lower() for t in doc if not t.is_space and not t.is_punct}


def _negation_count(doc, language=_EN) -> int:
    n = 0
    for tok in doc:
        if tok.dep_ == "neg" or tok.text.lower() in language.negators:
            n += 1
    return n


def _content_lemmas(doc) -> list[str]:
    return [_lemma_key(t) for t in doc
            if t.pos_ in _CONTENT_POS and not t.is_stop and not t.is_punct
            and len(_lemma_key(t)) > 2 and _lemma_key(t) not in {"be", "have", "do"}]


def _sentence_lemma_sets(doc) -> list[set[str]]:
    return [{_lemma_key(t) for t in sent
             if t.pos_ in _CONTENT_POS and not t.is_stop and len(_lemma_key(t)) > 2}
            for sent in doc.sents]


# --- the check -------------------------------------------------------------

def fidelity(source: str, output: str, gaps: list[str] | None = None,
             inferences: list[str] | None = None,
             gap_sentences: dict | None = None, lang: str | None = None) -> Fidelity:
    """Compare an output against its source. No model is consulted.

    `lang` is the language both texts are in: `en`, `es` or `pt`, default
    English. It picks the parser and the negator and number word lists.

    `gaps` and `inferences` are the rewrite's own report. They can excuse a
    drop, because reporting a gap instead of filling it is the correct
    behaviour. They can never excuse an introduction: a self-report does not
    get to overturn a measurement.

    `gap_sentences` maps a source sentence index to the gap codes found there,
    so an introduction landing on one can be labelled as a probable gap fill.
    """
    gaps_text = " ".join(gaps or []).lower()
    inferences_text = " ".join(inferences or []).lower()
    gap_sentences = gap_sentences or {}
    notes: list[str] = []

    language = languages.get(lang)
    src_doc = parse(source, language.code)
    out_doc = parse(output, language.code)

    src_anchors = _anchors(src_doc, language)
    out_anchors = _anchors(out_doc, language)
    out_vocab = _entity_vocabulary(out_doc)
    src_vocab = _entity_vocabulary(src_doc)
    out_by_kind = {k: {a.text for a in out_anchors if a.kind == k}
                   for k in ("number", "date", "entity")}

    preserved: list[Anchor] = []
    dropped: list[Anchor] = []
    excused: list[Anchor] = []
    for a in src_anchors:
        if a.kind == "entity":
            kept = any(part in out_vocab for part in a.text.split())
        else:
            kept = a.text in out_by_kind[a.kind]
        if kept:
            preserved.append(a)
        elif a.text and a.text in gaps_text:
            excused.append(Anchor(a.kind, a.text, a.sentence,
                                  "dropped, but named in a reported gap"))
        else:
            dropped.append(a)

    introduced: list[Anchor] = []
    for a in out_anchors:
        if a.kind == "entity":
            novel = all(part not in src_vocab for part in a.text.split())
        else:
            novel = a.text not in {s.text for s in src_anchors if s.kind == a.kind}
        if not novel:
            continue
        note = ""
        codes = gap_sentences.get(a.sentence) or gap_sentences.get(str(a.sentence))
        if codes:
            note = f"possible gap fill: source S{a.sentence} carried {', '.join(codes)}"
        elif a.text and a.text in inferences_text:
            note = "the model declared this an inference, not a fact from the source"
        introduced.append(Anchor(a.kind, a.text, a.sentence, note))

    src_lemmas = set(_content_lemmas(src_doc))
    out_lemmas = _content_lemmas(out_doc)
    novel_lemmas = sorted({lem for lem in out_lemmas
                           if not _matches_derivation(lem, src_lemmas)})
    novel_rate = round(len(novel_lemmas) / len(out_lemmas), 4) if out_lemmas else 0.0

    src_sets = _sentence_lemma_sets(src_doc)
    out_sets = _sentence_lemma_sets(out_doc)
    uncovered = []
    for i, s in enumerate(src_sets, start=1):
        if not s:
            continue
        best = max((len(s & o) / len(s | o) for o in out_sets if o), default=0.0)
        if best < COVERAGE_JACCARD:
            uncovered.append(i)
    measurable = len([s for s in src_sets if s])
    coverage = round(1 - len(uncovered) / measurable, 4) if measurable else 1.0

    negations = (_negation_count(src_doc, language), _negation_count(out_doc, language))
    neg_delta = abs(negations[0] - negations[1])
    # A restyle may carry a negation lexically -- `did not renew` becoming
    # `declined to renew` -- which drops the count without changing the claim.
    # One unit of drift is excused only when such a verb actually appears.
    neg_excused = 0
    if neg_delta:
        src_low = {t.lemma_.lower() for t in src_doc}
        out_low = {t.lemma_.lower() for t in out_doc}
        lexical = language.lexical_negators
        if (out_low | src_low) & lexical - (out_low & src_low & lexical):
            neg_excused = 1
    neg_penalised = max(0, neg_delta - neg_excused)

    anchored = len(preserved) + len(dropped)
    anchor_recall = len(preserved) / anchored if anchored else 1.0

    score = 100.0 * anchor_recall
    score -= min(INTRODUCED_PENALTY * len(introduced), INTRODUCED_PENALTY_CAP)
    score -= 150.0 * max(0.0, novel_rate - NOVEL_FREE_ALLOWANCE)
    score -= 40.0 * (1.0 - coverage)
    score -= 10.0 * neg_penalised
    score = round(max(0.0, min(100.0, score)), 1)

    passed = (not introduced and not dropped and novel_rate <= NOVEL_LIMIT
              and coverage >= COVERAGE_LIMIT and neg_penalised == 0)

    if introduced:
        notes.append(f"{len(introduced)} item(s) appear in the output that are "
                     "nowhere in the source. This is the failure the tool exists "
                     "to prevent; check each one.")
    if dropped:
        notes.append(f"{len(dropped)} anchor(s) from the source are missing and were "
                     "not reported as gaps.")
    if excused:
        notes.append(f"{len(excused)} missing anchor(s) were named in reported gaps, "
                     "which is the intended behaviour, not a defect.")
    if neg_penalised:
        notes.append(f"Negation count moved from {negations[0]} to {negations[1]} "
                     "with no negative verb accounting for it. A dropped or added "
                     "negation reverses a claim while keeping every other word.")
    elif neg_delta:
        notes.append(f"Negation count moved from {negations[0]} to {negations[1]}, "
                     "but a negative verb in one text and not the other accounts "
                     "for it. Not penalised.")
    notes.append("Anchors, lemmas and coverage are lexical. Two passages can share "
                 "every anchor and still say different things, so this is a floor "
                 "on faithfulness rather than proof of it.")

    return Fidelity(
        score=score, passed=passed, preserved=preserved, dropped=dropped,
        excused=excused, introduced=introduced, novel_lemmas=novel_lemmas,
        novel_rate=novel_rate, uncovered_sentences=uncovered, coverage=coverage,
        negations=negations, notes=notes,
    )


def gap_sentence_map(analysis) -> dict:
    """Source sentences carrying a gap finding, so fills can be labelled."""
    from .rewrite import GAP_CODES

    out: dict[int, list[str]] = {}
    for f in analysis.findings:
        if f.code in GAP_CODES:
            out.setdefault(f.sent_index, []).append(f.code)
    return out
