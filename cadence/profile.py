"""Syntactic fingerprinting: measure a corpus, then use it as a writing target.

The diagnostics module measures prose to find defects. This module runs the same
kind of measurements over a corpus you *like* and turns them into a target. Same
machinery, opposite sign.

That sign flip matters. `NOMINALIZATION x5` is a bug report when auditing and a
fingerprint when imitating -- heavy nominalisation may be exactly someone's
voice. Nothing here is normative; every number is descriptive.

Two design decisions worth stating:

SPREAD IS PART OF THE VOICE. Every feature is a distribution, not a mean. Real
writing alternates a long subordinated sentence with a short one, and a
generator that hits the mean every time reads like a machine. Targets are
ranges (p10-p90), and the spread is scored as its own feature.

A TREE IS A TOKEN, NOT A TYPE. One sentence's tree is that sentence. A style is
a distribution over many, so a profile needs a corpus, and `build_profile`
refuses to pretend otherwise on thin input.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field

from .syntax import dep_depth, parse

_FINITE_TAGS = {"VBZ", "VBD", "VBP", "MD"}
_SUBORDINATE_DEPS = {"advcl", "ccomp", "xcomp", "relcl", "acl", "csubj", "pcomp"}
_CLOSED_CLASS = {"DET", "ADP", "PRON", "CCONJ", "SCONJ", "AUX", "PART"}
_PUNCT_PREDICATION = {":", ";", "—", "--", "–"}

# Enough sentences that a distribution means something.
MIN_SENTENCES = 12


@dataclass
class Stat:
    """One feature as a distribution."""
    mean: float
    sd: float
    p10: float
    p50: float
    p90: float
    n: int

    @classmethod
    def of(cls, values: list[float]) -> Stat:
        if not values:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        vs = sorted(values)
        def pct(p):
            if len(vs) == 1:
                return float(vs[0])
            i = min(len(vs) - 1, max(0, int(round(p * (len(vs) - 1)))))
            return float(vs[i])
        return cls(
            mean=round(statistics.fmean(vs), 2),
            sd=round(statistics.pstdev(vs), 2) if len(vs) > 1 else 0.0,
            p10=pct(0.10), p50=pct(0.50), p90=pct(0.90), n=len(vs),
        )

    def target(self) -> str:
        return f"{self.mean} (typical range {self.p10}-{self.p90}, sd {self.sd})"


@dataclass
class SyntacticProfile:
    name: str
    documents: int
    sentences: int
    tokens: int
    sentence_length: Stat
    depth: Stat
    finite_clauses: Stat
    subordination: Stat
    coordination: Stat
    pre_verb_weight: Stat
    rates: dict = field(default_factory=dict)
    pos_mix: dict = field(default_factory=dict)
    function_words: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def _sentence_features(sent):
    toks = [t for t in sent if not t.is_space and not t.is_punct]
    finite = [t for t in sent if t.tag_ in _FINITE_TAGS]
    subord = [t for t in sent if t.dep_ in _SUBORDINATE_DEPS]
    coord = [t for t in sent if t.dep_ == "conj"]
    root = sent.root
    pre_verb = len([t for t in toks if t.i < root.i])
    passives = [t for t in sent if any(c.dep_ == "auxpass" for c in t.children)]
    agentless = [t for t in passives if not any(c.dep_ == "agent" for c in t.children)]
    return dict(
        length=len(toks),
        depth=dep_depth(sent),
        finite=len(finite),
        subord=len(subord),
        coord=len(coord),
        pre_verb=pre_verb,
        fragment=1 if not finite else 0,
        passive=len(passives),
        agentless=len(agentless),
        punct_pred=len([t for t in sent if t.text in _PUNCT_PREDICATION]),
        pronoun=len([t for t in sent if t.pos_ == "PRON"]),
    )


def build_profile(texts: list[str], name: str = "corpus") -> SyntacticProfile:
    """Measure a corpus. Pass several documents; one sentence is not a style."""
    docs = [parse(t) for t in texts if t and t.strip()]
    if not docs:
        raise ValueError("No usable text in the corpus.")

    rows, all_tokens = [], []
    for doc in docs:
        for sent in doc.sents:
            if not [t for t in sent if not t.is_space and not t.is_punct]:
                continue
            rows.append(_sentence_features(sent))
        all_tokens.extend([t for t in doc if not t.is_space])

    if not rows:
        raise ValueError("No usable sentences in the corpus.")

    n_sent = len(rows)
    n_tok = len([t for t in all_tokens if not t.is_punct])

    warnings = []
    if n_sent < MIN_SENTENCES:
        warnings.append(
            f"Only {n_sent} sentences. A profile is a distribution; below about "
            f"{MIN_SENTENCES} the spread is noise, and the generator will imitate "
            "accidents of this sample rather than a style."
        )
    if len(docs) < 2:
        warnings.append(
            "Single document. Features specific to this one piece will be read as "
            "style. Several samples by the same author give a cleaner signal."
        )

    def col(key):
        return [float(r[key]) for r in rows]

    def per_100(c):
        return round(100.0 * c / n_tok, 2) if n_tok else 0.0

    def per_sent(c):
        return round(c / n_sent, 3)
    total_finite = sum(r["finite"] for r in rows) or 1

    pos_counts = Counter(t.pos_ for t in all_tokens if not t.is_punct)
    pos_mix = ({k: round(100.0 * v / n_tok, 2) for k, v in pos_counts.most_common(14)}
               if n_tok else {})

    fw = Counter(
        t.lower_ for t in all_tokens
        if t.pos_ in _CLOSED_CLASS and t.is_alpha
    )
    function_words = {k: per_100(v) for k, v in fw.most_common(25)}

    return SyntacticProfile(
        name=name,
        documents=len(docs),
        sentences=n_sent,
        tokens=n_tok,
        sentence_length=Stat.of(col("length")),
        depth=Stat.of(col("depth")),
        finite_clauses=Stat.of(col("finite")),
        subordination=Stat.of(col("subord")),
        coordination=Stat.of(col("coord")),
        pre_verb_weight=Stat.of(col("pre_verb")),
        rates=dict(
            fragment_rate=per_sent(sum(r["fragment"] for r in rows)),
            passive_per_finite=round(sum(r["passive"] for r in rows) / total_finite, 3),
            agentless_passive_per_finite=round(sum(r["agentless"] for r in rows) / total_finite, 3),
            punct_predication_per_sentence=per_sent(sum(r["punct_pred"] for r in rows)),
            pronouns_per_100_tokens=per_100(sum(r["pronoun"] for r in rows)),
            subordination_per_sentence=per_sent(sum(r["subord"] for r in rows)),
            coordination_per_sentence=per_sent(sum(r["coord"] for r in rows)),
        ),
        pos_mix=pos_mix,
        function_words=function_words,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------

def _stat_divergence(a: Stat, b: Stat) -> float:
    """Normalised distance between two distributions, on mean AND spread."""
    scale = max(abs(a.mean), 1.0)
    mean_d = abs(a.mean - b.mean) / scale
    spread_scale = max(a.sd, 1.0)
    sd_d = abs(a.sd - b.sd) / spread_scale
    # Spread is weighted meaningfully: matching the mean while flattening the
    # variance is the classic way to sound robotic.
    return 0.65 * mean_d + 0.35 * sd_d


def _dist_divergence(a: dict, b: dict) -> float:
    """Total-variation distance over a shared key space."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    total_a = sum(a.values()) or 1.0
    total_b = sum(b.values()) or 1.0
    return 0.5 * sum(abs(a.get(k, 0.0) / total_a - b.get(k, 0.0) / total_b) for k in keys)


def divergence(target: SyntacticProfile, candidate: SyntacticProfile) -> dict:
    """How far `candidate` sits from `target`. Lower is closer; 0 is identical."""
    parts = {
        "sentence_length": _stat_divergence(target.sentence_length, candidate.sentence_length),
        "depth": _stat_divergence(target.depth, candidate.depth),
        "finite_clauses": _stat_divergence(target.finite_clauses, candidate.finite_clauses),
        "subordination": _stat_divergence(target.subordination, candidate.subordination),
        "coordination": _stat_divergence(target.coordination, candidate.coordination),
        "pre_verb_weight": _stat_divergence(target.pre_verb_weight, candidate.pre_verb_weight),
    }
    for key in target.rates:
        t, c = target.rates.get(key, 0.0), candidate.rates.get(key, 0.0)
        parts[key] = abs(t - c) / max(abs(t), 0.5)
    parts["pos_mix"] = _dist_divergence(target.pos_mix, candidate.pos_mix)
    parts["function_words"] = _dist_divergence(target.function_words, candidate.function_words)

    parts = {k: round(v, 3) for k, v in parts.items()}
    overall = statistics.fmean(parts.values()) if parts else 0.0
    # Squash to a 0-100 similarity; 0 divergence -> 100.
    similarity = round(100.0 * math.exp(-1.4 * overall), 1)
    return {
        "overall_divergence": round(overall, 3),
        "similarity": similarity,
        "by_feature": dict(sorted(parts.items(), key=lambda kv: -kv[1])),
    }


# ---------------------------------------------------------------------------
# Prompt-facing rendering
# ---------------------------------------------------------------------------

def _variance_directive(p: SyntacticProfile) -> str:
    """Rhythm target, floored at the remove-slop minimum.

    Where the corpus is itself uniform, its variance is not worth reproducing:
    it is the strongest machine tell there is, so the floor wins.
    """
    from .slop import VARIANCE_FLOOR_CV

    cv = (p.sentence_length.sd / p.sentence_length.mean) if p.sentence_length.mean else 0.0
    if cv < VARIANCE_FLOOR_CV:
        needed = round(VARIANCE_FLOOR_CV * p.sentence_length.mean, 1)
        return (
            f"RHYTHM: the corpus varies sentence length by only {cv:.0%} of its mean, "
            f"which is below the floor remove-slop sets. Do NOT reproduce that "
            f"uniformity. Vary sentence length by at least {needed} tokens (sd), even "
            "though the corpus does not."
        )
    return (
        f"RHYTHM: vary sentence length by about {p.sentence_length.sd} tokens (sd), "
        f"matching the corpus at {cv:.0%} of its mean."
    )


def spec_text(p: SyntacticProfile) -> str:
    """The profile as a target spec a generator can aim at."""
    lines = [
        f"SYNTACTIC TARGET PROFILE ({p.name}: {p.documents} documents, "
        f"{p.sentences} sentences, {p.tokens} tokens)",
        "",
        "Hit the RANGES, not the means. Varying within the range is required --",
        "landing on the mean every sentence is the main way this goes wrong.",
        "",
        _variance_directive(p),
        "",
        f"  sentence length (tokens)      {p.sentence_length.target()}",
        f"  embedding depth               {p.depth.target()}",
        f"  finite clauses per sentence   {p.finite_clauses.target()}",
        f"  subordinate clauses/sentence  {p.subordination.target()}",
        f"  coordinated items/sentence    {p.coordination.target()}",
        f"  tokens before the main verb   {p.pre_verb_weight.target()}",
        "",
        "RATES",
    ]
    from .slop import BANNED_PROFILE_FEATURES

    labels = {
        "fragment_rate": "sentences with no finite verb",
        "passive_per_finite": "passives per finite verb",
        "agentless_passive_per_finite": "agentless passives per finite verb",
        "punct_predication_per_sentence": "colon/dash predications per sentence",
        "pronouns_per_100_tokens": "pronouns per 100 tokens",
        "subordination_per_sentence": "subordinate clauses per sentence",
        "coordination_per_sentence": "coordinated items per sentence",
    }
    for key, label in labels.items():
        value = p.rates.get(key, 0.0)
        if key in BANNED_PROFILE_FEATURES:
            lines.append(
                f"  {label:36} {value}  <- MEASURED BUT NOT A TARGET: "
                f"{BANNED_PROFILE_FEATURES[key]}. Do not reproduce it."
            )
        else:
            lines.append(f"  {label:36} {value}")
    lines += ["", "PART-OF-SPEECH MIX (% of tokens)"]
    lines.append("  " + ", ".join(f"{k} {v}%" for k, v in p.pos_mix.items()))
    lines += ["", "FUNCTION-WORD FREQUENCY (per 100 tokens) -- the strongest authorship signal"]
    lines.append("  " + ", ".join(f"{k} {v}" for k, v in list(p.function_words.items())[:18]))
    return "\n".join(lines)
