"""Rewriting driven by the diagnostics, never by taste.

Two backends:

rules   Deterministic string surgery. Only transforms that follow mechanically
        from the parse are applied -- currently copula repetition to break an
        unlike coordination, and subordinator disambiguation when the caller
        opts in. Everything else becomes a brief.

llm     Sends the text plus the findings to Claude as explicit constraints. The
        findings are the instructions; the model is not asked for an opinion
        about style.

The invariant both backends hold, and the reason the brief exists: a rewrite may
fix structure. It may not invent content. Where the analysis found missing
information, that stays missing and gets reported.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .diagnostics import Analysis
from .fidelity import fidelity as check_fidelity
from .fidelity import gap_sentence_map
from .llm import DEFAULT_MAX_TOKENS, call_text, have_credentials, resolve_client

DEFAULT_MODEL = "claude-opus-5"

# Findings that mark absent information. A rewrite must surface these, never
# resolve them -- resolving them would mean fabricating facts.
GAP_CODES = {
    "SUPERLATIVE_NO_SET", "SUPERLATIVE_VAGUE_SET", "AGENTLESS_PASSIVE",
    "PP_ATTACH", "RELCL_ATTACH",
}


@dataclass
class Dispute:
    """A finding the model believes rests on a misparse."""
    code: str
    sentence: int
    reason: str


@dataclass
class Change:
    code: str
    before: str
    after: str
    rationale: str


@dataclass
class RewriteResult:
    backend: str
    text: str
    changes: list[Change] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    inferences: list[str] = field(default_factory=list)
    brief: list[str] = field(default_factory=list)
    disputed: list[Dispute] = field(default_factory=list)
    trees: str = ""
    note: str = ""
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    fidelity: object = None


# ---------------------------------------------------------------------------
# Step 1 -- render the tree
# ---------------------------------------------------------------------------

def render_trees(analysis, only_with_findings: bool = True) -> str:
    """Step 1 of the pipeline: render the syntactic evidence.

    This is the artefact step 2 rewrites against. Kept as its own function, and
    its own CLI command, so the evidence can be read on its own before anything
    acts on it -- and so what the model receives is exactly what you can inspect.

    By default only sentences carrying a finding are rendered: a clean sentence
    contributes no evidence to a rewrite and costs tokens.
    """
    from .syntax import constituency, dep_tree, pos_line

    flagged = {f.sent_index for f in analysis.findings}
    blocks = []
    for view, sent in zip(analysis.sentences, analysis.doc.sents, strict=False):
        if only_with_findings and view.index not in flagged:
            continue
        blocks.append("\n".join([
            f"--- S{view.index} (dependency depth {view.depth}, "
            f"parse confidence {view.confidence}) ---",
            view.text,
            "",
            "POS:",
            pos_line(sent),
            "",
            "DEPENDENCIES:",
            dep_tree(sent),
            "",
            "CONSTITUENCY (projected from the dependency parse):",
            constituency(sent).brackets(),
        ]))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Shared: the instruction set derived from findings
# ---------------------------------------------------------------------------

def build_brief(analysis: Analysis) -> tuple[list[str], list[str]]:
    """Turn findings into (actionable instructions, gaps to report only)."""
    actions, gaps = [], []
    for f in analysis.findings:
        line = f"[{f.code}] S{f.sent_index}: {f.label} -- {f.fix} (span: {f.span[:80]!r})"
        if f.code in GAP_CODES:
            gaps.append(line)
        else:
            actions.append(line)
    return actions, gaps


# ---------------------------------------------------------------------------
# Rules backend
# ---------------------------------------------------------------------------

_SUBORD_GLOSS = {"since": "because", "as": "because", "while": "whereas"}


def _repeat_copula(text: str, analysis: Analysis) -> list[Change]:
    """Break an unlike coordination by repeating the copula before each coordinate.

    `is a teacher, patient, and writing daily`
      -> `is a teacher, is patient, and is writing daily`

    Turns a UCP into like-category VP coordination without touching content.
    """
    changes = []
    for f in analysis.findings:
        if f.code != "UCP" or not f.auto:
            continue
        copula = f.data.get("copula")
        start, end = f.data.get("coord_start"), f.data.get("coord_end")
        if not copula or start is None or end is None:
            continue
        original = text[start:end]
        # Insert the copula before each coordinate after the first.
        rebuilt = re.sub(r",\s+and\s+", f", and {copula} ", original)
        rebuilt = re.sub(r",\s+(?!and\b|is\b)", f", {copula} ", rebuilt)
        if rebuilt == original:
            continue
        changes.append(Change(
            "UCP", original, rebuilt,
            f"Repeated {copula!r} before each coordinate: unlike coordination becomes "
            "like-category VP coordination.",
        ))
    return changes


def _disambiguate_subordinators(text: str, analysis: Analysis) -> list[Change]:
    changes = []
    for f in analysis.findings:
        if f.code != "AMBIG_SUBORD":
            continue
        m = re.search(r"'([^']+)'", f.label)
        if not m:
            continue
        word = m.group(1).lower()
        gloss = _SUBORD_GLOSS.get(word)
        if not gloss:
            continue
        changes.append(Change(
            "AMBIG_SUBORD", word, gloss,
            f"{word!r} carries two readings under one POS tag; {gloss!r} fixes the "
            "causal reading in the wording.",
        ))
    return changes


def rewrite_rules(analysis: Analysis, apply_subordinators: bool = False) -> RewriteResult:
    text = analysis.text
    changes = _repeat_copula(text, analysis)
    if apply_subordinators:
        changes += _disambiguate_subordinators(text, analysis)

    out = text
    for change in changes:
        if change.code == "UCP":
            out = out.replace(change.before, change.after, 1)
        elif change.code == "AMBIG_SUBORD":
            out = re.sub(rf"\b{re.escape(change.before)}\b", change.after, out, count=1)

    actions, gaps = build_brief(analysis)
    applied = {c.code for c in changes}
    remaining = [a for a in actions if a.split("]")[0].lstrip("[") not in applied]

    return RewriteResult(
        backend="rules",
        text=out,
        changes=changes,
        gaps=gaps,
        brief=remaining,
        note=(
            "The rules backend applies only transforms that follow mechanically from "
            "the parse. Everything in the brief needs a judgement a parser cannot make; "
            "run the llm backend, or work the brief by hand."
        ),
    )


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You rewrite prose to fix specific, already-diagnosed syntactic defects. A parser \
found them; your job is to act on that list and nothing else.

Absolute rules:

1. STRUCTURE AND GRAMMAR ONLY. Do not add, drop, or strengthen any fact. Every \
claim in your rewrite must be traceable to the original.
2. NEVER FILL AN INFORMATION GAP. Some findings report missing information -- an \
unnamed comparison set, a suppressed agent, an unnamed referent. If the text does \
not supply it, you MUST leave it missing and list it under "gaps". Inventing a \
plausible filler is the worst thing you can do here.
3. Anything you infer that the text does not state (resolving a pronoun to a name, \
assigning a suppressed agent) goes in "inferences" so a human can confirm it. If \
you cannot infer it safely, leave the original wording.
4. Fix only what the findings name. Do not restyle prose you were not asked about.
5. The syntactic evidence is a real parse, but the parser is trained on newswire
and does get this register wrong. Use the trees to check each finding before you
act on it. If a finding rests on a misparse, do NOT "fix" it -- put it in
"disputed" with the tree evidence that contradicts it, and leave that wording
alone. A wrong fix is worse than a missed one.

Return ONLY valid JSON:
{"rewrite": "...", "changes": [{"code": "...", "before": "...", "after": "...", \
"rationale": "..."}], "gaps": ["..."], "inferences": ["..."], \
"disputed": [{"code": "...", "sentence": 1, "reason": "..."}]}"""


def rewrite_llm(analysis: Analysis, model: str = DEFAULT_MODEL,
                use_trees: bool = True, *, api_key: str | None = None,
                client=None) -> RewriteResult:
    client = resolve_client(api_key=api_key, client=client)

    actions, gaps = build_brief(analysis)
    trees = render_trees(analysis) if use_trees else ""
    parts = ["TEXT TO REWRITE:", analysis.text.strip(), ""]
    if trees:
        parts += [
            "SYNTACTIC EVIDENCE (step 1 -- the parse each finding was derived from):",
            trees, "",
        ]
    parts += [
        "FINDINGS TO FIX (act on each, after checking it against the parse):",
        *(actions or ["(none)"]), "",
        "GAPS -- REPORT ONLY, NEVER FILL:",
        *(gaps or ["(none)"]), "",
        "Rewrite the text. Fix the findings the parse supports, leave the gaps "
        "open, and dispute any finding the parse contradicts.",
    ]
    prompt = "\n".join(parts)

    from .slop import PROMPT_RULES
    system = SYSTEM_PROMPT + PROMPT_RULES

    # The response carries the full rewrite plus a change log, and extended
    # thinking counts toward this cap as well, so it needs real headroom. 4096
    # truncated mid-JSON on anything longer than a short paragraph.
    resp = call_text(client, model, system, prompt, max_tokens=DEFAULT_MAX_TOKENS)
    raw = resp.text
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model did not return valid JSON: {exc}\n{raw[:400]}") from exc

    return RewriteResult(
        backend=f"llm ({model}){' +trees' if use_trees else ''}",
        text=data.get("rewrite", "").strip(),
        changes=[
            Change(c.get("code", "?"), c.get("before", ""), c.get("after", ""),
                   c.get("rationale", ""))
            for c in data.get("changes", [])
        ],
        gaps=data.get("gaps", []) or gaps,
        inferences=data.get("inferences", []),
        disputed=[
            Dispute(d.get("code", "?"), d.get("sentence", 0), d.get("reason", ""))
            for d in data.get("disputed", [])
        ],
        trees=trees,
        usage=resp.usage,
        latency_s=resp.latency_s,
        note=(
            "Step 1 rendered the parse; step 2 rewrote against it. Findings were "
            "passed as constraints, gaps as report-only, and the model could "
            "dispute findings the parse contradicts."
        ),
    )


def rewrite(analysis: Analysis, backend: str = "auto", model: str = DEFAULT_MODEL,
            apply_subordinators: bool = False, use_trees: bool = True, *,
            api_key: str | None = None, client=None) -> RewriteResult:
    """Rewrite via the named backend. `auto` uses the LLM when credentials exist."""
    if backend == "auto":
        backend = "llm" if have_credentials(api_key=api_key, client=client) else "rules"
    if backend == "llm":
        result = rewrite_llm(analysis, model=model, use_trees=use_trees,
                             api_key=api_key, client=client)
    else:
        result = rewrite_rules(analysis, apply_subordinators=apply_subordinators)
    # Measured after the fact, against the source. The model's own account of
    # what it kept is not evidence of what it kept.
    result.fidelity = check_fidelity(
        analysis.text, result.text, gaps=result.gaps,
        inferences=result.inferences, gap_sentences=gap_sentence_map(analysis),
    )
    return result
