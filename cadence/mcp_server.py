"""A local MCP server, so an agent can call cadence the way a person would.

    pip install "cadence-writer[mcp]"
    cadence mcp

That starts a server on stdio. Register it with Claude Desktop, Claude Code, or
any other MCP client and the tools below appear there. Nothing leaves the
machine except the model call the restyle tool makes, and that uses whatever
ANTHROPIC_API_KEY the process was started with.

The tools are thin. Every one of them is a function from the library with its
arguments flattened to strings and lists, because the point of exposing cadence
over MCP is to let a model use the measurement rather than to invent a second
API for it. Where a tool takes a corpus, it takes the documents themselves: the
server holds no state a client has to manage, only a cache keyed by the text so
that measuring the same writing twice in a session costs one parse.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict

from .diagnostics import analyze
from .errors import MissingAPIKey
from .fidelity import fidelity as fidelity_of
from .generate import generate
from .profile import MIN_SENTENCES, SyntacticProfile, build_profile, spec_text
from .rewrite import DEFAULT_MODEL, rewrite
from .scores import score as score_text

# The SDK renamed its server class between major versions. Both spellings are
# accepted so the extra can float rather than pin a version a user may not have.
try:
    from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2
except ImportError:  # pragma: no cover - depends on the installed SDK
    try:
        from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x
    except ImportError as exc:
        raise ImportError(
            'The MCP server needs the `mcp` package. Install it with: '
            'pip install "cadence-writer[mcp]"'
        ) from exc

server = _Server(
    "cadence",
    instructions=(
        "cadence measures the shape of a writer's sentences and rewrites drafts to "
        "match. Use measure_voice on a few pieces someone wrote to get their "
        "profile, restyle to rewrite a draft in that voice, and check_fidelity to "
        "confirm a rewrite kept every name, number and date. Scores are floors, "
        "not verdicts: read the prose."
    ),
)

_profiles: dict[str, SyntacticProfile] = {}


def _profile_for(documents: list[str], name: str = "corpus",
                 language: str = "en") -> SyntacticProfile:
    """Build a profile, or reuse the one built from identical text this session."""
    docs = [d.strip() for d in documents if d and d.strip()]
    if not docs:
        raise ValueError("documents is empty; pass at least one piece of writing.")
    key = hashlib.sha256((language + "\x00" + "\x00".join(docs)).encode("utf-8")).hexdigest()
    prof = _profiles.get(key)
    if prof is None:
        prof = build_profile(docs, name=name, lang=language)
        _profiles[key] = prof
    return prof


@server.tool()
def measure_voice(documents: list[str], name: str = "corpus", language: str = "en") -> dict:
    """Measure the shape of someone's writing.

    Pass several pieces the same person wrote. `language` is en, es or pt and
    must match the writing. Returns the profile as numbers, the prose
    specification a model can write against, and any warnings about the corpus
    being too thin to trust. Below about twelve sentences the spread is noise,
    and the warnings will say so.
    """
    prof = _profile_for(documents, name, language)
    return {
        "documents": prof.documents,
        "sentences": prof.sentences,
        "tokens": prof.tokens,
        "enough_for_a_profile": prof.sentences >= MIN_SENTENCES,
        "warnings": prof.warnings,
        "spec": spec_text(prof),
        "profile": prof.as_dict(),
        "language": prof.language,
    }


@server.tool()
def analyze_text(text: str, language: str = "en") -> dict:
    """Find structural defects in a passage without calling a model.

    Returns findings a parser can point at (unlike coordination, agentless
    passives, superlatives with no comparison set, machine-register tells) and
    observations that are facts about the register rather than defects.
    `language` is en, es or pt; the machine-register lists are English.
    """
    a = analyze(text, lang=language)
    return {
        "metrics": a.metrics,
        "findings": [f.as_dict() for f in a.findings],
        "observations": [o.as_dict() for o in a.observations],
    }


@server.tool()
def restyle(draft: str, documents: list[str], model: str = DEFAULT_MODEL,
            iterations: int = 2, language: str = "en") -> dict:
    """Rewrite a draft so its sentence shapes match the writer's.

    The model sees the measured specification, never the samples, so nothing
    from them can leak into the output. Runs a verify loop: generate, parse,
    measure, correct. Returns the text with two scores: voice match against the
    profile and content fidelity against the draft, plus anything introduced or
    dropped. `language` (en, es, pt) is the language of both the draft and the
    documents. Needs ANTHROPIC_API_KEY in the environment the server started in.
    """
    prof = _profile_for(documents, language=language)
    try:
        r = generate(draft, mode="profile_verify", profile=prof, task="restyle",
                     model=model, iterations=iterations, lang=language)
    except MissingAPIKey as exc:
        # A structured answer, not a raised error: the client is a model, and a
        # model can act on "set this variable" where it cannot act on a stack.
        return {"error": "missing_credentials", "detail": str(exc),
                "fix": "Start `cadence mcp` with ANTHROPIC_API_KEY in its environment."}
    fid = r.fidelity
    return {
        "text": r.text,
        "voice_match": r.similarity,
        "content_kept": None if fid is None else fid.score,
        "fidelity_passed": None if fid is None else fid.passed,
        "introduced": [] if fid is None else [asdict(a) for a in fid.introduced],
        "dropped": [] if fid is None else [asdict(a) for a in fid.dropped],
        "length_ratio": r.length_ratio,
        "attempts": len(r.attempts),
        "model": r.model,
        "usage": r.usage,
        "note": r.note,
    }


@server.tool()
def check_fidelity(source: str, output: str, language: str = "en") -> dict:
    """Did a rewrite keep the content? Measured against the source, no model.

    Compares names, numbers, dates, negations and sentence coverage. An
    introduced item is a failure whatever the score says. A dropped item that
    was reported as a gap is excused. The score is a floor on faithfulness, not
    proof of it: two passages can share every anchor and say different things.
    `language` is en, es or pt, the language of both texts.
    """
    return fidelity_of(source, output, lang=language).as_dict()


@server.tool()
def score(text: str, documents: list[str], source: str | None = None,
          language: str = "en") -> dict:
    """Both numbers for a passage: voice match, and content kept if a source is given."""
    prof = _profile_for(documents, language=language)
    return score_text(text, prof, source=source).as_dict()


@server.tool()
def repair_structure(text: str, language: str = "en") -> dict:
    """Apply the deterministic repairs only, with no model.

    Repeats a copula across an unlike coordination and nothing else. Every other
    finding comes back as a brief for a human, and gaps are reported rather than
    filled. Fidelity is always 100 on this path, and the tool says so.
    """
    a = analyze(text, lang=language)
    r = rewrite(a, backend="rules")
    return {
        "text": r.text,
        "changes": [asdict(c) for c in r.changes],
        "brief": r.brief,
        "gaps": r.gaps,
        "fidelity": None if r.fidelity is None else r.fidelity.as_dict(),
    }


def main() -> int:
    """Serve on stdio until the client disconnects."""
    server.run(transport="stdio")
    return 0
