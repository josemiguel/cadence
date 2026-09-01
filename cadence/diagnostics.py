"""Structural diagnostics for note-register prose.

Every detector here is a port of a finding from the hand analysis in
text.sample.out. Section references in `origin` point back to it.

A finding is a defect that a rewrite can act on. An observation is a register
fact -- true of the text, not wrong with it. Keeping them apart matters: dates
used as attributive modifiers are worth knowing about and are not a problem.

The hard rule the whole tool obeys: a detector may report that information is
missing. Nothing here, and nothing downstream, may invent it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import repair
from .syntax import (
    chunks,
    deepest_path,
    dep_depth,
    has_finite_verb,
    parse,
)

_FINITE_TAGS = {"VBZ", "VBD", "VBP", "MD"}
_NOMINAL = {"NOUN", "PROPN", "PRON", "NUM"}

# Nouns that package a claim so it can be referenced without being asserted.
CONTAINER_NOUNS = {
    "read", "view", "take", "sense", "call", "assessment", "impression",
    "feeling", "conviction", "hunch", "concern", "worry", "thesis", "thinking",
    "understanding", "belief", "sentiment", "judgment", "judgement", "opinion",
    "position", "stance", "conclusion", "sceptism", "skepticism",
}

# Subordinators with two readings that share one POS tag.
AMBIGUOUS_SUBORDINATORS = {
    "since": ("causal", "because", "temporal", "after"),
    "as": ("causal", "because", "temporal", "while"),
    "while": ("temporal", "during", "concessive", "whereas"),
}

PROMISCUOUS_PREPS = {"with", "in", "on", "at", "for", "from", "by", "within"}

# Dependency depth (head-to-leaf), NOT constituency depth -- the two differ.
DEPTH_LIMIT = 8


def span_of(tok) -> str:
    """Original text covered by a token's subtree.

    `token.subtree` yields tokens in TREE order, so joining it scrambles the
    words ("of there is what to underwrite"). Slicing the source text by the
    subtree's character extent keeps word order and the original spacing.
    """
    left, right = tok.left_edge, tok.right_edge
    return tok.doc.text[left.idx:right.idx + len(right.text)].strip()


@dataclass
class Finding:
    code: str
    label: str
    origin: str
    sent_index: int
    span: str
    detail: str
    fix: str
    auto: bool
    severity: str
    # Structured payload for rewriters: character offsets and head tokens, so a
    # rewriter never has to re-discover by regex what the parser already knew.
    data: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(
            code=self.code, label=self.label, origin=self.origin,
            sentence=self.sent_index, span=self.span, detail=self.detail,
            fix=self.fix, auto=self.auto, severity=self.severity, data=self.data,
        )


@dataclass
class Observation:
    code: str
    label: str
    origin: str
    detail: str

    def as_dict(self):
        return dict(code=self.code, label=self.label, origin=self.origin, detail=self.detail)


@dataclass
class SentenceView:
    index: int
    text: str
    depth: int
    confidence: str
    confidence_reasons: list[str]
    repairs: list[str] = field(default_factory=list)


@dataclass
class Analysis:
    text: str
    doc: object
    sentences: list[SentenceView]
    findings: list[Finding]
    observations: list[Observation]
    metrics: dict


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def _fragments(sent, idx):
    if not has_finite_verb(list(sent)):
        yield Finding(
            "FRAG", "Fragment: no finite verb", "§4.2", idx, sent.text.strip(),
            "Nothing in this sentence is a finite verb, so no clause is asserted.",
            "Give the intended claim a subject and a finite verb.",
            False, "high",
        )
        return
    parts = chunks(sent)
    for pos, (delim, toks) in enumerate(parts):
        if not delim or not toks or delim not in {":", "—", "--", "–"}:
            continue
        left = parts[pos - 1][1] if pos else []
        left_verbless = bool(left) and not has_finite_verb(left)
        right_verbless = not has_finite_verb(toks)
        if not (left_verbless or right_verbless):
            continue
        side = "label" if left_verbless else "content"
        verbless = " ".join(t.text for t in (left if left_verbless else toks))
        yield Finding(
            "PUNCT_COPULA", f"`{delim}` predicating ({side} side has no finite verb)",
            "§4.2", idx, verbless.strip()[:120],
            f"The {side} side of this `{delim}` contains no finite verb, so the "
            "punctuation is carrying the predication. Punctuation cannot be a head, "
            "which is why the parser has to fall back on a fragment here.",
            "Promote to a heading, or supply the verb the punctuation stands in for.",
            True, "medium",
        )


def _agentless_passives(sent, idx):
    for tok in sent:
        is_full = any(c.dep_ == "auxpass" for c in tok.children)
        is_reduced = (
            tok.tag_ == "VBN"
            and tok.dep_ in {"acl", "relcl", "advcl", "pcomp"}
            and not any(c.dep_ in {"nsubj", "auxpass"} for c in tok.children)
        )
        if not (is_full or is_reduced):
            continue
        if any(c.dep_ == "agent" for c in tok.children):
            continue
        kind = "passive" if is_full else "reduced passive"
        yield Finding(
            "AGENTLESS_PASSIVE", f"Agentless {kind}: {tok.text!r}", "§4.6", idx,
            span_of(tok),
            f"{tok.text!r} is {kind} with no by-phrase, so the agent is suppressed "
            "rather than merely omitted. This is a hedging device, not a style tic.",
            "Name the agent and recast active, or confirm the agent is genuinely unknown.",
            False, "high",
        )


def _null_subjects(sent, idx):
    for tok in sent:
        if tok.tag_ not in _FINITE_TAGS:
            continue
        # A `conj` verb inherits its subject from the first coordinate; that is
        # ordinary VP coordination, not a dropped subject.
        if tok.dep_ not in {"ROOT", "parataxis"}:
            continue
        subject_deps = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
        if any(c.dep_ in subject_deps for c in tok.children):
            continue
        yield Finding(
            "NULL_SUBJECT", f"Finite verb with no subject: {tok.text!r}", "§4.3", idx,
            span_of(tok),
            f"{tok.text!r} is finite but has no subject. Diary-drop: recoverable from "
            "context, absent from the syntax.",
            "Restore the subject.",
            False, "high",
        )


def _coarse_category(tok):
    if tok.pos_ in _NOMINAL:
        return "nominal"
    if tok.pos_ == "ADJ":
        return "adjectival"
    if tok.pos_ in {"VERB", "AUX"}:
        return "verbal"
    if tok.pos_ == "ADP":
        return "prepositional"
    if tok.pos_ == "ADV":
        return "adverbial"
    return tok.pos_.lower()


def _conj_chain(tok):
    """Full coordination chain from its head.

    spaCy chains coordination (`a` -conj-> `b` -conj-> `c`) rather than making
    every coordinate a sibling, so looking only at direct children sees adjacent
    pairs and misses both the real category mix and the chain head that carries
    the copula.
    """
    chain, frontier = [tok], [tok]
    while frontier:
        node = frontier.pop()
        for child in node.children:
            if child.dep_ == "conj":
                chain.append(child)
                frontier.append(child)
    return chain


def _unlike_coordination(sent, idx):
    for tok in sent:
        # Only report from the head of a chain, never from inside it.
        if tok.dep_ == "conj":
            continue
        group = _conj_chain(tok)
        if len(group) < 2:
            continue
        cats = {_coarse_category(t) for t in group}
        if len(cats) < 2:
            continue
        copula = None
        if tok.dep_ in {"attr", "acomp", "oprd"} and tok.head.lemma_ == "be":
            copula = tok.head
        doc = sent.doc
        start_char = min(g.left_edge.idx for g in group)
        last = max(group, key=lambda g: g.right_edge.i).right_edge
        end_char = last.idx + len(last.text)
        span = doc.text[start_char:end_char]
        fix = (
            f"Repeat the copula {copula.text!r} before each coordinate to turn this into "
            "like-category VP coordination."
            if copula else
            "Make the coordinates the same category, or split them into separate clauses."
        )
        yield Finding(
            "UCP", "Unlike coordination (" + " + ".join(sorted(cats)) + ")", "§4.4", idx,
            span.strip(),
            "Coordinates differ in category: "
            + ", ".join(f"{t.text!r} ({_coarse_category(t)})" for t in group)
            + ". PTB needs a UCP node for this; most readers need a rewrite.",
            fix, copula is not None, "medium",
            data={
                "copula": copula.text if copula else None,
                "coord_start": start_char,
                "coord_end": end_char,
                "coordinates": [t.text for t in group],
            },
        )


def _nominalizations(sent, idx):
    for tok in sent:
        if tok.lower_ not in CONTAINER_NOUNS or tok.pos_ != "NOUN":
            continue
        has_owner = any(c.dep_ in {"poss", "nmod"} for c in tok.children) or any(
            c.dep_ == "prep" and c.lower_ == "of" and
            any(g.ent_type_ == "PERSON" for g in c.subtree)
            for c in tok.children
        )
        if has_owner:
            continue
        # Only count it when the noun is actually packaging a claim -- as the head
        # of a subject NP or of a copular predicate. An event reading (`a call
        # recorded on the 26th`) is a concrete noun, not a container.
        claim_position = tok.dep_ in {"nsubj", "nsubjpass", "attr", "acomp"} or (
            tok.dep_ == "pobj" and tok.head.lower_ == "of"
        )
        if not claim_position:
            continue
        # A date or numeral modifier marks a concrete event (`the 2026-08-26 call`),
        # not a packaged claim.
        if any(c.tag_ == "CD" for c in tok.children):
            continue
        yield Finding(
            "NOMINALIZATION", f"Claim packaged as a noun: {tok.text!r}", "§4.1", idx,
            span_of(tok),
            f"{tok.text!r} is a container noun with no named holder. The claim can be "
            "referred to, dated and revised without anyone asserting it.",
            "Name whose it is, or convert it to a finite verb.",
            False, "medium",
        )


def _comparison_set(tok):
    """Find the comparison set licensing a superlative, if the syntax supplies one.

    Returns (kind, node): ("prep", prep_token) | ("than", None) | (None, None).
    Kind and node are kept separate on purpose -- returning either a Token or a
    sentinel string from one slot silently breaks `!=`, because comparing a
    spaCy Token to a str does not behave like ordinary inequality.
    """
    for node in [tok, tok.head, tok.head.head]:
        for child in node.children:
            if child.dep_ == "prep" and child.lower_ in {"of", "in", "among", "amongst"}:
                return "prep", child
    if any(t.lower_ == "than" for t in tok.sent):
        return "than", None
    return None, None


def _superlatives(sent, idx):
    for tok in sent:
        is_sup = tok.tag_ in {"JJS", "RBS"} or (
            tok.lower_ in {"most", "least"} and tok.dep_ == "advmod"
        )
        if not is_sup:
            continue
        # A superlative modifying an ADVERB is a fixed adverbial ("most recently",
        # "most notably"), not a ranking, and expects no comparison set. One
        # modifying an ADJECTIVE ("most active") does.
        if tok.pos_ == "ADV" and tok.head.pos_ == "ADV":
            continue
        kind, cset = _comparison_set(tok)
        if kind is None:
            yield Finding(
                "SUPERLATIVE_NO_SET", f"Superlative with no comparison set: {tok.text!r}",
                "§4.5", idx, " ".join(t.text for t in tok.head.subtree),
                f"{tok.text!r} ranks something against a set the sentence never gives.",
                "Name what it is being compared against, or drop the superlative.",
                False, "high",
            )
        elif kind == "prep":
            # A clausal complement of `of` arrives as `pcomp`, not `pobj`.
            pobj = next((c for c in cset.children if c.dep_ in {"pobj", "pcomp"}), None)
            vague = pobj is not None and (pobj.pos_ == "VERB" or pobj.tag_ in {"WP", "WDT"})
            if vague:
                yield Finding(
                    "SUPERLATIVE_VAGUE_SET",
            f"Comparison set present but not enumerable: {tok.text!r}",
                    "§4.6", idx, span_of(cset),
                    f"The set is given as a free relative "
            f"({' '.join(t.text for t in cset.subtree)!r}), "
                    "which names a set without letting a reader list its members.",
                    "Enumerate the set, or say how short it is.",
                    False, "medium",
                )


def _propositional_anaphora(sent, idx):
    for tok in sent:
        bare = not any(c.dep_ not in {"punct"} for c in tok.children)
        if (
            tok.lower_ in {"this", "that", "it"}
            and tok.dep_ in {"nsubj", "nsubjpass"}
            and bare
        ):
            # A nominal earlier in the same sentence is a candidate antecedent, so
            # the referent is recoverable and this is a clarity nit, not a hole.
            local_antecedent = next(
                (t for t in sent
                 if t.i < tok.i and t.pos_ in {"NOUN", "PROPN"}
                 and t.dep_ in {"nsubj", "nsubjpass", "dobj", "pobj", "attr"}),
                None,
            )
            if local_antecedent is not None:
                detail = (
                    f"{tok.text!r} is a bare subject. {local_antecedent.text!r} earlier in "
                    "the same sentence is a candidate antecedent, so the reference is "
                    "recoverable; naming it is a clarity choice, not a missing fact."
                )
                severity = "low"
            else:
                detail = (
                    f"{tok.text!r} is a bare subject with no nominal antecedent in its own "
                    "sentence. In this register such pronouns often point at a whole "
                    "proposition, so the referent is not recoverable from the syntax alone."
                )
                severity = "medium"
            yield Finding(
                "PROP_ANAPHORA", f"Unnamed reference: {tok.text!r}", "§4.8", idx,
                sent.text.strip(), detail,
                "Replace with the noun it stands for.",
                False, severity,
            )
        if tok.lower_ == "which" and tok.dep_ in {"nsubj", "dobj"}:
            verb = tok.head
            host = verb.head
            comma_before = tok.i > 0 and sent.doc[tok.i - 1].text == ","
            if comma_before and host.pos_ in {"NOUN", "PROPN"}:
                yield Finding(
                    "RELCL_ATTACH", "Non-restrictive `which`: sentential or NP-modifying?",
                    "§5.2", idx, span_of(verb),
                    f"The parser attached this relative to {host.text!r}, but a "
                    "comma-marked `which` in this position often modifies the whole "
                    "preceding clause instead. The two readings differ in what is claimed.",
                    "Name the subject of the relative to close the attachment.",
                    False, "high",
                )


def _anaphoric_one(sent, idx):
    for tok in sent:
        if tok.lower_ == "one" and tok.tag_ == "NN":
            yield Finding(
                "ANAPHORIC_ONE", "Pronominal `one`", "§5.8", idx,
                span_of(tok),
                "`one` stands in for a noun mentioned earlier; the reader has to "
                "reconstruct which.",
                "Repeat the noun.",
                False, "low",
            )


def _np_negation(sent, idx):
    for tok in sent:
        if tok.lower_ == "no" and tok.dep_ == "det":
            yield Finding(
                "NP_NEGATION", f"Negation inside the NP: `no {tok.head.text}`", "§4.6", idx,
                " ".join(t.text for t in tok.head.subtree),
                "The absence is asserted by a determiner rather than by the verb, which "
                "reads as weaker than a plain denial.",
                "Consider negating the verb if a denial is meant.",
                False, "low",
            )


def _deep_nesting(sent, idx):
    depth = dep_depth(sent)
    if depth <= DEPTH_LIMIT:
        return
    path = deepest_path(sent)
    yield Finding(
        "DEEP_NESTING", f"Dependency depth {depth}", "§4 (S2)", idx,
        " > ".join(f"{t.text}" for t in path),
        f"The deepest head-to-leaf dependency chain runs {depth} levels "
        f"(threshold {DEPTH_LIMIT}). Readers lose the thread well before that.",
        "Split at a clause boundary on the deep path.",
        True, "medium",
    )


def _pp_attachment(sent, idx):
    for tok in sent:
        if tok.dep_ != "prep" or tok.lower_ not in PROMISCUOUS_PREPS:
            continue
        head = tok.head
        if head.pos_ not in {"VERB", "AUX"}:
            continue
        rivals = [
            t for t in sent
            if head.i < t.i < tok.i and t.pos_ in {"NOUN", "PROPN"} and t.dep_ != "punct"
        ]
        if not rivals:
            continue
        alt = rivals[-1]
        yield Finding(
            "PP_ATTACH", f"PP attachment ambiguity: `{tok.text} ...`", "§5.1", idx,
            span_of(tok),
            f"Attached to the verb {head.text!r}, but {alt.text!r} sits between them and "
            f"could host it. Reading A: it modifies {head.text!r}. Reading B: it modifies "
            f"{alt.text!r}. These say different things.",
            "Move the phrase next to its intended host, or recast as a relative clause.",
            False, "high",
        )


def _shared_gap(sent, idx):
    for tok in sent:
        if tok.tag_ not in {"WDT", "WP"} or tok.dep_ not in {"dobj", "nsubj"}:
            continue
        verb = tok.head
        pool = [verb] + [d for d in verb.subtree if d.pos_ == "VERB" and d.i != verb.i]
        objectless = [
            v for v in pool
            if v.pos_ == "VERB"
            and any(c.dep_ == "conj" for c in v.children) or
            (v.dep_ == "conj" and not any(c.dep_ == "dobj" for c in v.children))
        ]
        if len(objectless) >= 1 and any(v.dep_ == "conj" for v in objectless):
            span = span_of(verb)
            yield Finding(
                "SHARED_GAP", "One filler, two extraction sites", "§S7", idx, span.strip(),
                f"{tok.text!r} serves as the object of more than one coordinated verb. "
                "Legitimate, but it makes the reader hold a gap open across the coordination.",
                "Check the coordinated verbs are not near-synonyms; say it once if they are.",
                False, "low",
            )
            return


def _ambiguous_subordinators(sent, idx):
    for tok in sent:
        entry = AMBIGUOUS_SUBORDINATORS.get(tok.lower_)
        if not entry or tok.dep_ != "mark":
            continue
        r1, g1, r2, g2 = entry
        yield Finding(
            "AMBIG_SUBORD", f"Ambiguous subordinator: {tok.text!r}", "§5.3", idx,
            " ".join(t.text for t in tok.head.subtree),
            f"{tok.text!r} can be {r1} or {r2}, and both readings carry the same POS tag, "
            "so the tag layer cannot record which you meant.",
            f"Use {g1!r} for the {r1} reading or {g2!r} for the {r2} reading.",
            False, "medium",
        )


DETECTORS = (
    _fragments, _agentless_passives, _null_subjects, _unlike_coordination,
    _nominalizations, _superlatives, _propositional_anaphora, _anaphoric_one,
    _np_negation, _deep_nesting, _pp_attachment, _shared_gap,
    _ambiguous_subordinators,
)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def _observations(doc) -> list[Observation]:
    out = []
    containers = [t for t in doc if t.lower_ in CONTAINER_NOUNS and t.pos_ == "NOUN"]
    if containers:
        forms = ", ".join(sorted({t.lower_ for t in containers}))
        out.append(Observation(
            "CONTAINER_INVENTORY", f"{len(containers)} container noun(s): {forms}", "§4.1",
            "The text is organised around named claims rather than assertions. Legitimate "
            "for record-keeping, evasive when it replaces asserting.",
        ))
    dates = [t for t in doc if t.tag_ == "CD" and t.dep_ in {"nummod", "compound", "npadvmod"}]
    if dates:
        out.append(Observation(
            "DATE_ATTRIBUTIVE", f"{len(dates)} date(s) used as attributive modifiers", "§4.7",
            "Dates premodify nouns here (`the 2026-08-26 call`), individuating which event "
            "is meant. Not a defect.",
        ))
    passives = [t for t in doc if any(c.dep_ == "auxpass" for c in t.children)]
    finite = [t for t in doc if t.tag_ in _FINITE_TAGS]
    if passives and finite:
        out.append(Observation(
            "PASSIVE_RATIO", f"{len(passives)} passive(s) against {len(finite)} finite verb(s)",
            "§4.6", "Voice is doing hedging work; see the agentless-passive findings.",
        ))
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def analyze(text: str) -> Analysis:
    """Parse, repair, and run every detector: structural and remove-slop alike.

    The two are one pass on purpose. A dash used as a copula is a structural
    finding (PUNCT_COPULA) and a slop finding (SLOP_DASH) simultaneously, and
    splitting them into separate modes would have made the caller reconcile two
    reports of the same sentence.
    """
    doc = parse(text)
    sents, findings = [], []
    for idx, sent in enumerate(doc.sents, start=1):
        level, reasons = repair.confidence(sent)
        sents.append(SentenceView(idx, sent.text.strip(), dep_depth(sent), level, reasons))
        for detector in DETECTORS:
            findings.extend(detector(sent, idx))

    from .slop import detect as detect_slop
    findings.extend(detect_slop(doc, Finding))

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (order[f.severity], f.sent_index))

    counts = {}
    for f in findings:
        counts[f.code] = counts.get(f.code, 0) + 1

    metrics = dict(
        sentences=len(sents),
        tokens=len([t for t in doc if not t.is_space]),
        max_depth=max((s.depth for s in sents), default=0),
        findings=len(findings),
        auto_fixable=sum(1 for f in findings if f.auto),
        by_code=counts,
        repairs=list(doc._.repairs or []),
    )
    return Analysis(text, doc, sents, findings, _observations(doc), metrics)
