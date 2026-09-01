# cadence

Parse prose, measure the shape of its sentences, then rewrite a draft to match.

Ask a model to write like you and it copies your topics and your favourite
words, which are the easiest things to copy and the least of what makes writing
recognisable. What actually distinguishes one writer from another is closer to
the skeleton: how long a sentence runs before its main verb arrives, how deep
clauses embed, whether ideas get joined or subordinated, how often a claim
becomes a noun rather than staying a verb.

Those are measurable, which means a style can be a specification rather than a
mood, and a specification can be checked after the fact.

## The rule that shapes everything

> A rewrite may fix structure. It may not invent content.

Where the analysis finds *missing information*, a superlative with no comparison
set, a passive with a suppressed agent, a pronoun with no recoverable referent,
that stays missing and is reported as a **gap**. Filling it would mean
fabricating facts, which is the one failure mode this tool is built to avoid.

That promise used to be a line in a prompt plus the model's own report of
whether it had obeyed. A model that fabricates is not a reliable witness to its
own fabrication, so `fidelity.py` now checks it mechanically.

## Install

```bash
pip install "cadence-writer[llm]"   # the import name is `cadence`
cadence download-model
```

The spaCy model is a separate step because PyPI rejects packages that declare a
dependency by URL, and the model is only distributed that way.

## Use

```bash
cadence analyze notes.txt                    # trees and findings, no rewrite
cadence report notes.txt                     # everything, including fidelity
cadence rewrite notes.txt --backend rules    # deterministic, no API call
cadence profile samples/*.txt                # the specification for a corpus
cadence restyle draft.txt --corpus samples/*.txt
cadence score out.txt --corpus samples/*.txt --source-text draft.txt
```

```python
import cadence

profile = cadence.build_profile(my_documents, name="me")
result = cadence.generate(draft, mode="profile", profile=profile,
                          task="restyle", api_key=key)

result.similarity          # voice match, 0 to 100
result.fidelity.score      # content kept, 0 to 100
result.fidelity.introduced # anything invented, which should be empty
```

## As an MCP server

```bash
pip install "cadence-writer[mcp]"
cadence mcp
```

That serves the library over stdio for Claude Desktop, Claude Code, or any other
MCP client, so a model can measure a voice, restyle a draft, and check fidelity
as tools. Nothing leaves the machine except the model call the restyle tool
makes. In Claude Code:

```bash
claude mcp add cadence -- cadence mcp
```

Tools: `measure_voice`, `analyze_text`, `restyle`, `check_fidelity`, `score`,
`repair_structure`. Each is one library function with its arguments flattened;
where a tool takes a corpus, it takes the documents themselves, so the server
holds no state a client has to manage.

## Two numbers, and what neither can tell you

**Voice match** compares sentence shape against the corpus: length, embedding
depth, clause mix, subordination, function words. It cannot tell you whether
the result sounds right.

**Content kept** compares names, numbers, dates, negations and coverage between
the draft and the rewrite. It cannot tell you the meaning survived. Two
passages can share every anchor and still say different things.

Both are floors. They catch the failures worth catching and they are honest
about the ones they miss.

## What the diagnostics find

| Code | What it finds |
|---|---|
| `FRAG` / `PUNCT_COPULA` | Punctuation carrying predication; no finite verb on one side |
| `AGENTLESS_PASSIVE` | Passive with the agent suppressed, not merely omitted |
| `NULL_SUBJECT` | Finite verb with no subject (diary-drop) |
| `UCP` | Unlike coordination (`a teacher, patient, and writing daily`) |
| `NOMINALIZATION` | A claim packaged as a noun so it need never be asserted |
| `SUPERLATIVE_NO_SET` | Superlative ranking against a set the text never gives |
| `SUPERLATIVE_VAGUE_SET` | Set given, but as a free relative you cannot enumerate |
| `PROP_ANAPHORA` | `this` / `that` / `it` pointing at a proposition |
| `RELCL_ATTACH` / `PP_ATTACH` | Attachment ambiguities that change what is claimed |
| `AMBIG_SUBORD` | `since` / `as` / `while` — two readings, one POS tag |
| `DEEP_NESTING`, `SHARED_GAP`, `NP_NEGATION`, `ANAPHORIC_ONE` | see `diagnostics.py` |
| `SLOP_DASH` / `SLOP_HYPHEN` | dashes as connectors, hyphenated compounds |
| `SLOP_RHETORICAL` / `SLOP_VAPID_OPENER` | staged reveals, camera metaphors, scene setting |
| `SLOP_SUMMARY_INTRO` | "the upshot", "in short", "bottom line" leading a sentence |
| `SLOP_APPLAUSE` | short declarative landing after a long one, as punctuation |
| `SLOP_PATRONIZING` / `SLOP_EVALUATIVE` / `SLOP_INTENSIFIER` | telling the reader what to think |
| `SLOP_UNIFORM_RHYTHM` / `SLOP_UNIFORM_PARAGRAPHS` | measured variance below the floor |

Findings are defects. **Observations** are register facts (dates used as
attributive modifiers, container-noun inventory) — true of the text, not wrong
with it. They are reported separately on purpose.

## Layers

```
syntax.py       PTB tags, dependency trees, constituency projection
repair.py       parse repairs for note-register English (R1–R5)
diagnostics.py  the detectors
profile.py      corpus measurement, divergence, the generation spec
generate.py     constrained generation, and the verify loop
rewrite.py      rules backend (deterministic) + Claude backend
fidelity.py     did the rewrite keep the content, measured not asserted
scores.py       the two numbers, in one call
llm.py          credentials, one call, token usage
```

### On the constituency layer

The dependency parse is real. The constituency tree is a **head projection** of
it, not a native PTB parse. It reproduces PTB conventions that follow from
dependencies (flat base NPs, `-SBJ`/`-PRD` function tags, PP/ADJP/SBAR labels)
and cannot reproduce what only a constituency treebank knows (traces, empty
categories, licensed `UCP` nodes). It is labelled as projected everywhere it
appears.

### On remove-slop

The remove-slop rule set (`slop.py`) is not a mode or a checkbox. It runs in the
same pass as the structural detectors, and its rules are appended to every
rewrite and generation prompt. A dash used as a copula is a structural finding
(`PUNCT_COPULA`) and a slop finding (`SLOP_DASH`) at the same time; splitting
them would have forced the reader to reconcile two reports of one sentence.

Most of the rules are mechanically checkable, which is why they belong here
rather than in prompt text alone. "Uniform rhythm is a stronger tell than any
single word" is a claim about **variance**, and variance is what `profile.py`
already computes, so it is measured rather than eyeballed.

**Where the two layers meet.** A style profile is descriptive and remove-slop is
normative, so they can disagree. The resolution is fixed, not a preference:

1. The profile owns **structure** (length, depth, clause mix, subordination).
2. remove-slop owns **surface machine tells** (banned phrases, dashes, applause).
3. Where they overlap, the ban wins, and the override is printed rather than
   left for the reader to notice. Banned features are still *measured* — the
   profile stays a complete description — but they are marked `NOT A TARGET`
   in the spec and never handed to a generator to reproduce.
4. On rhythm the two agree, with a floor: if a corpus is itself uniform, the
   spec says *do not reproduce that uniformity*, because copying it would copy
   the strongest machine tell there is.

### On the parse repairs

`en_core_web_sm` is trained on newswire and fails on this register in specific,
reproducible ways. Five repairs run before any diagnostic, and every one that
fires is reported:

- **R1** ISO dates and hyphenated compounds kept as single tokens
- **R2** `the category read is…` — head noun misread as a verb
- **R2b** subject-like arc landing on a nominal head → `compound`
- **R3** `one` — `NN` when pronominal, `CD` when a partitive numeral
- **R4** accusative pronoun parsed as a subject (`liked him`)
- **R5** hyphenated premodifier tagging: noun-noun → `NN`, prefix+noun → `JJ`

Sentences also carry a coarse parse-confidence signal — a list of structural
reasons to distrust the parse, not a probability.

## Tests

```bash
pytest -q
```

Nothing in the suite touches the network. Every expectation was verified by hand
before it was automated, so a failure means the tool has drifted from an
analysis a human checked.

## License

MIT.
