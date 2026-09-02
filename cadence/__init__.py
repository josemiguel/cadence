"""cadence -- parse prose, measure its shape, then rewrite it from the parse.

Two things live here. A diagnostic layer that finds structural defects a parser
can point at, and a profile layer that measures a corpus you like and uses the
measurement as a writing target. They are the same machinery with the sign
flipped: heavy nominalisation is a defect when auditing and a fingerprint when
imitating.

One rule holds across both. A rewrite may fix structure; it may not invent
content. Missing information is reported as a gap and left missing.

Importing this package does not load the spaCy model. The first call to
`parse`, `analyze` or `build_profile` does.
"""

__version__ = "0.3.0"

from .demo import DEMO_TEXT
from .diagnostics import Analysis, Finding, Observation, analyze
from .errors import CadenceError, MissingAPIKey, ModelNotInstalled
from .fidelity import Anchor, Fidelity, fidelity
from .generate import GenerationResult, compare, generate
from .languages import LANGUAGES, Language
from .languages import guess as guess_language
from .profile import SyntacticProfile, build_profile, divergence, spec_text
from .report import text_report
from .rewrite import DEFAULT_MODEL, RewriteResult, render_trees, rewrite
from .scores import Scores, score
from .syntax import parse

__all__ = [
    "DEFAULT_MODEL",
    "DEMO_TEXT",
    "Analysis",
    "Anchor",
    "CadenceError",
    "Fidelity",
    "Finding",
    "GenerationResult",
    "LANGUAGES",
    "Language",
    "MissingAPIKey",
    "ModelNotInstalled",
    "Observation",
    "RewriteResult",
    "Scores",
    "SyntacticProfile",
    "__version__",
    "analyze",
    "build_profile",
    "compare",
    "divergence",
    "fidelity",
    "generate",
    "guess_language",
    "parse",
    "render_trees",
    "rewrite",
    "score",
    "spec_text",
    "text_report",
]
