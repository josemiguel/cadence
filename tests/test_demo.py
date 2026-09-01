"""The demo text must keep firing the detectors it was written to show.

It is the first thing anyone sees in the web UI and the README. If a detector
change silences one of these, the default view quietly stops demonstrating the
tool, and nothing else in the suite would catch that.
"""

from cadence.demo import DEMO_CODES, DEMO_TEXT
from cadence.diagnostics import analyze


def test_demo_text_fires_its_advertised_codes():
    got = {f.code for f in analyze(DEMO_TEXT).findings}
    missing = [c for c in DEMO_CODES if c not in got]
    assert not missing, f"demo text no longer fires: {missing}"


def test_demo_text_keeps_the_iso_date_observation():
    obs = {o.code for o in analyze(DEMO_TEXT).observations}
    assert "DATE_ATTRIBUTIVE" in obs
