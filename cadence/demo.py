"""The sample text the web UI and the README use.

Written for this repo, not drawn from anyone's real notes. It is note-register
prose chosen to fire a wide spread of detectors in three sentences, so the
default view of the tool shows what the tool actually does. `tests/test_demo.py`
pins the codes it must produce, which is what keeps it from going stale as the
detectors change.
"""

from __future__ import annotations

DEMO_TEXT = (
    "The pricing read is the strongest element. Liked her — a teacher, patient, "
    "and writing daily. No decision was taken before the 2025-04-09 review, "
    "which cuts both ways."
)

# Codes DEMO_TEXT is chosen to produce. Asserted in tests/test_demo.py.
DEMO_CODES = (
    "AGENTLESS_PASSIVE",
    "NOMINALIZATION",
    "NULL_SUBJECT",
    "PUNCT_COPULA",
    "RELCL_ATTACH",
    "SLOP_DASH",
    "SUPERLATIVE_NO_SET",
    "UCP",
)
