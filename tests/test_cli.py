"""Every subcommand parses and runs. Nothing here reaches the network."""

from __future__ import annotations

import json

import pytest

from cadence.cli import EXIT_BAD_INPUT, EXIT_NO_MODEL, build_parser, main

CORPUS = (
    "The team met on Tuesday and wrote a note. It was short. The note said the "
    "pricing held, which nobody expected, and that the pilot would run again.\n"
    "She read it twice. The second reading changed her mind, because the numbers "
    "in the appendix did not match the summary, and the summary was the part "
    "everyone had already quoted.\n"
    "He asked for the appendix. Nobody had it. The meeting ended there, and the "
    "note went out unchanged, which is how the wrong number reached the board."
)
DRAFT = "Halvorsen Dairy did not renew the 2024-03-11 contract. Revenue fell 12%."


@pytest.fixture
def files(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS)
    draft = tmp_path / "draft.txt"
    draft.write_text(DRAFT)
    return corpus, draft


def run(argv, capsys):
    code = main(argv)
    return code, capsys.readouterr()


def test_every_subcommand_is_registered():
    actions = [a for a in build_parser()._actions if hasattr(a, "choices") and a.choices]
    names = set(actions[0].choices)
    assert names == {"analyze", "render", "rewrite", "report", "profile",
                     "restyle", "compose", "score", "download-model"}


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "cadence" in capsys.readouterr().out


def test_analyze_prints_a_report(files, capsys):
    code, out = run(["analyze", str(files[1])], capsys)
    assert code == 0
    assert "SYNTACTIC ANALYSIS" in out.out


def test_analyze_json_round_trips(files, capsys):
    code, out = run(["analyze", str(files[1]), "--json"], capsys)
    assert code == 0
    payload = json.loads(out.out)
    assert {"metrics", "findings", "observations"} <= set(payload)


def test_render_prints_trees(files, capsys):
    code, out = run(["render", str(files[1])], capsys)
    assert code == 0 and out.out.strip()


def test_rewrite_falls_back_to_rules_without_a_key(files, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code, out = run(["rewrite", str(files[1])], capsys)
    assert code == 0
    assert "backend: rules" in out.out


def test_report_includes_the_fidelity_block(files, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code, out = run(["report", str(files[1]), "--no-trees"], capsys)
    assert code == 0
    assert "CONTENT FIDELITY" in out.out


def test_profile_prints_a_spec(files, capsys):
    code, out = run(["profile", str(files[0])], capsys)
    assert code == 0
    assert "SYNTACTIC TARGET PROFILE" in out.out


def test_profile_json_is_a_profile(files, capsys):
    code, out = run(["profile", str(files[0]), "--json"], capsys)
    assert code == 0
    assert "sentence_length" in json.loads(out.out)


def test_score_reports_voice_only_without_a_source(files, capsys):
    code, out = run(["score", str(files[1]), "--corpus", str(files[0])], capsys)
    assert code == 0
    assert "voice match" in out.out
    assert "not measured" in out.out


def test_score_reports_both_numbers_with_a_source(files, capsys):
    code, out = run(["score", str(files[1]), "--corpus", str(files[0]),
                     "--source-text", str(files[1])], capsys)
    assert code == 0
    assert "voice match" in out.out and "content kept" in out.out
    assert "100.0/100 [PASS]" in out.out


def test_score_json_carries_the_caveats(files, capsys):
    code, out = run(["score", str(files[1]), "--corpus", str(files[0]), "--json"], capsys)
    payload = json.loads(out.out)
    assert code == 0
    assert payload["caveats"] and payload["content_kept"] is None


def test_writing_to_a_file(files, tmp_path, capsys):
    dest = tmp_path / "out.txt"
    code, _ = run(["analyze", str(files[1]), "-o", str(dest)], capsys)
    assert code == 0 and dest.read_text().strip()


def test_an_empty_input_is_bad_input(tmp_path, capsys):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n")
    code, out = run(["analyze", str(empty)], capsys)
    assert code == EXIT_BAD_INPUT
    assert "empty" in out.err


def test_a_missing_file_is_bad_input(capsys):
    code, out = run(["analyze", "/nonexistent/nope.txt"], capsys)
    assert code == EXIT_BAD_INPUT


def test_a_missing_model_has_its_own_exit_code(files, capsys, monkeypatch):
    """A user with no model must not read the same failure as a bad argument."""
    import spacy

    from cadence.syntax import _load

    _load.cache_clear()
    monkeypatch.setattr(spacy, "load", lambda *a, **k: (_ for _ in ()).throw(OSError("no model")))
    code, out = run(["analyze", str(files[1])], capsys)
    _load.cache_clear()
    assert code == EXIT_NO_MODEL
    assert "download-model" in out.err
