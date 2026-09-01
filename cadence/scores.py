"""The two numbers a writer is shown, computed in one call.

They answer different questions and neither substitutes for the other. Voice
match asks whether the sentences are shaped like the ones in your corpus.
Content kept asks whether the facts survived. A rewrite can score 95 on the
first while inventing a name, and 100 on the second while sounding nothing like
you, so showing one number would hide exactly the failure the other catches.

Both are floors, not verdicts. Structure is not voice, and anchors are not
meaning. `caveats` says so in the same object that carries the numbers, because
a caveat kept in the documentation is a caveat nobody reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .fidelity import Fidelity, fidelity
from .profile import SyntacticProfile, build_profile, divergence

VOICE_CAVEAT = (
    "Voice match compares sentence shape: length, depth, clause mix, function "
    "words. It cannot tell you whether the writing sounds right. Read it."
)
FIDELITY_CAVEAT = (
    "Content kept compares names, numbers, dates and coverage. Two passages can "
    "share all of those and still say different things."
)


@dataclass
class Scores:
    """Voice distance and content fidelity for one piece of text."""

    voice_similarity: float
    voice_divergence: float
    by_feature: dict = field(default_factory=dict)
    fidelity: Fidelity | None = None
    caveats: list = field(default_factory=list)

    @property
    def content_kept(self) -> float | None:
        return None if self.fidelity is None else self.fidelity.score

    def as_dict(self) -> dict:
        return {
            "voice_similarity": self.voice_similarity,
            "voice_divergence": self.voice_divergence,
            "by_feature": self.by_feature,
            "content_kept": self.content_kept,
            "fidelity": None if self.fidelity is None else self.fidelity.as_dict(),
            "caveats": list(self.caveats),
        }


def score(text: str, profile: SyntacticProfile, source: str | None = None,
          gaps: list[str] | None = None,
          inferences: list[str] | None = None) -> Scores:
    """Score one passage against a target profile, and against its source.

    `source` is the text this one was rewritten from. Without it there is
    nothing to be faithful to, and `fidelity` is left as None rather than
    filled with a number that would mean nothing.
    """
    measured = divergence(profile, build_profile([text], name="candidate"))
    caveats = [VOICE_CAVEAT]
    fid = None
    if source is not None:
        fid = fidelity(source, text, gaps=gaps, inferences=inferences)
        caveats.append(FIDELITY_CAVEAT)
    return Scores(
        voice_similarity=measured["similarity"],
        voice_divergence=measured["overall_divergence"],
        by_feature=measured["by_feature"],
        fidelity=fid,
        caveats=caveats,
    )
