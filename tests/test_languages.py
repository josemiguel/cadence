"""Spanish and Portuguese go through the same counters as English.

The parsers differ: Universal Dependencies labels and morphology in place of
Penn tags. `languages.normalise` is what makes that invisible downstream, and
these tests check it from the outside, through the measurements a user sees.
The Spanish and Portuguese models are separate installs, so tests that need
one skip rather than fail where it is missing.
"""

from __future__ import annotations

import pytest

import cadence
from cadence import languages
from cadence.fidelity import fidelity
from cadence.profile import build_profile, spec_text
from cadence.syntax import parse


def _have(code: str) -> bool:
    import spacy.util

    return spacy.util.is_package(languages.get(code).model)


needs_es = pytest.mark.skipif(not _have("es"), reason="es_core_news_sm not installed")
needs_pt = pytest.mark.skipif(not _have("pt"), reason="pt_core_news_sm not installed")

ES = (
    "La junta pidió una corrección porque las cifras no coincidían. El informe fue "
    "enviado por el equipo el 2025-04-30, y nadie lo leyó hasta el lunes. Marisol Vega "
    "dirige el área de datos desde febrero. Su equipo creció de 6 a 9 personas. Dos de "
    "los tres tableros previstos se entregaron a tiempo. El tercero se retrasó porque "
    "el proveedor incumplió la fecha. Ella recomienda confirmar el nombramiento. La "
    "decisión se tomará en septiembre. Nadie discute que el trimestre fue sólido. El "
    "presupuesto sigue siendo de 220 mil dólares. Quedan 34 mil sin gastar. La junta "
    "volverá a reunirse el jueves."
)

PT = (
    "A diretoria pediu uma correção porque os números não batiam. O relatório foi "
    "enviado pela equipe em 2025-04-30, e ninguém o leu até segunda-feira. Marisol Vega "
    "dirige a área de dados desde fevereiro. Sua equipe cresceu de 6 para 9 pessoas. "
    "Dois dos três painéis previstos foram entregues no prazo. O terceiro atrasou "
    "porque o fornecedor perdeu a data. Ela recomenda confirmar a nomeação. A decisão "
    "será tomada em setembro. Ninguém discute que o trimestre foi sólido. O orçamento "
    "continua em 220 mil dólares. Restam 34 mil por gastar. A diretoria volta a se "
    "reunir na quinta-feira."
)


# --- the registry, no model needed ------------------------------------------

def test_the_three_languages_and_their_codes():
    assert languages.CODES == ("en", "es", "pt")
    assert languages.get("pt-BR").name == "Portuguese"
    assert languages.get("ES").native_name == "Español"
    assert languages.get(None).code == "en"
    with pytest.raises(ValueError):
        languages.get("fr")


def test_guessing_the_language_from_function_words():
    assert cadence.guess_language("The board asked for a correction because the numbers "
                                  "did not match.") == "en"
    assert cadence.guess_language(ES) == "es"
    assert cadence.guess_language(PT) == "pt"
    assert cadence.guess_language("") == "en"
    assert cadence.guess_language("2025-04-30") == "en"


def test_english_is_unchanged_by_the_language_layer():
    """The English model's labels pass through untouched: `normalise` is a no-op."""
    doc = parse("The report was sent by the team, and nobody read it.")
    assert any(t.dep_ == "auxpass" for t in doc)
    assert any(t.dep_ == "agent" for t in doc)
    prof = build_profile(["The report was sent by the team. Nobody read it until Monday."])
    assert prof.language == "en"
    assert "LANGUAGE: English" in spec_text(prof)


def test_the_english_profile_dict_round_trips_without_a_language():
    """Profiles saved before languages existed load as English."""
    prof = cadence.SyntacticProfile(
        name="old", documents=1, sentences=1, tokens=5,
        sentence_length=cadence.profile.Stat.of([5.0]), depth=cadence.profile.Stat.of([2.0]),
        finite_clauses=cadence.profile.Stat.of([1.0]), subordination=cadence.profile.Stat.of([0.0]),
        coordination=cadence.profile.Stat.of([0.0]), pre_verb_weight=cadence.profile.Stat.of([1.0]))
    assert prof.language == "en"


# --- Spanish -----------------------------------------------------------------

@needs_es
def test_spanish_parse_carries_the_labels_the_counters_read():
    doc = parse("El informe fue enviado por el equipo, y nadie lo leyó.", "es")
    assert doc.lang_ == "es"
    deps = {t.dep_ for t in doc}
    assert "auxpass" in deps, deps        # aux:pass, rewritten
    assert "agent" in deps, deps          # obl:agent, rewritten
    assert "nsubjpass" in deps, deps
    assert not any(":" in d for d in deps), deps
    finite = [t.text for t in doc if languages.is_finite(t)]
    assert "fue" in finite and "leyó" in finite


@needs_es
def test_spanish_profile_measures_clauses_and_names_its_language():
    prof = build_profile([ES], name="es", lang="es")
    assert prof.language == "es"
    assert prof.sentences >= 12
    assert prof.finite_clauses.mean >= 1.0
    assert prof.subordination.mean > 0          # porque ... coincidían
    assert prof.rates["passive_per_finite"] > 0  # fue enviado
    assert "LANGUAGE: Spanish" in spec_text(prof)
    assert set(prof.function_words) & {"el", "la", "de", "que"}


@needs_es
def test_spanish_fidelity_reads_negation_and_numbers():
    source = "El proveedor no entregó el tablero el 2025-04-30. Costó 220 mil dólares."
    same = "El 2025-04-30 el proveedor no entregó el tablero, que costó 220 mil dólares."
    flipped = "El proveedor entregó el tablero el 2025-04-30. Costó 220 mil dólares."
    kept = fidelity(source, same, lang="es")
    assert kept.negations == (1, 1)
    assert kept.passed, kept.notes
    lost = fidelity(source, flipped, lang="es")
    assert lost.negations == (1, 0)
    assert not lost.passed
    assert {a.text for a in kept.preserved if a.kind == "date"} == {"2025-04-30"}


@needs_es
def test_spanish_spelled_numbers_normalise():
    from cadence.fidelity import _normalise_number

    es = languages.get("es")
    assert _normalise_number("doscientos", es) == "doscientos"  # not in the closed list
    assert _normalise_number("dos mil", es) == "2000.0"
    assert _normalise_number("tres millones", es) == "3000000.0"
    assert _normalise_number("veinte", es) == "20.0"


@needs_es
def test_generate_tells_the_model_the_output_language(fake_client):
    prof = build_profile([ES], name="es", lang="es")
    client = fake_client("La junta pidió una corrección porque las cifras no coincidían "
                         "con el informe enviado el 2025-04-30.")
    draft = "Las cifras no coincidían con el informe que se envió el 2025-04-30."
    result = cadence.generate(draft, mode="profile", profile=prof, task="restyle",
                              client=client)
    call = client.calls[0]
    assert "OUTPUT LANGUAGE: Spanish" in call["system"]
    assert "LANGUAGE: Spanish" in call["messages"][0]["content"]
    assert result.fidelity is not None


# --- Portuguese --------------------------------------------------------------

@needs_pt
def test_portuguese_parse_and_profile():
    doc = parse("O relatório não foi enviado pela equipe, e ninguém o leu.", "pt")
    deps = {t.dep_ for t in doc}
    assert {"auxpass", "agent", "nsubjpass"} <= deps, deps
    assert [t.text for t in doc if t.dep_ == "neg"] == ["não"], [(t.text, t.dep_) for t in doc]
    prof = build_profile([PT], name="pt", lang="pt")
    assert prof.language == "pt"
    assert prof.sentences >= 12
    assert prof.subordination.mean > 0
    assert prof.rates["passive_per_finite"] > 0
    assert "LANGUAGE: Portuguese" in spec_text(prof)


@needs_pt
def test_portuguese_fidelity_counts_nao():
    source = "O fornecedor não entregou o painel em 2025-04-30."
    flipped = "O fornecedor entregou o painel em 2025-04-30."
    f = fidelity(source, flipped, lang="pt")
    assert f.negations == (1, 0)
    assert not f.passed


@needs_pt
def test_score_uses_the_profile_language():
    prof = build_profile([PT], name="pt", lang="pt")
    s = cadence.score("A diretoria pediu uma correção porque os números não batiam.",
                      prof, source="Os números não batiam e a diretoria pediu correção.")
    assert 0 < s.voice_similarity <= 100
    assert s.fidelity is not None and s.fidelity.negations == (1, 1)


# --- the CLI and the MCP server take the language too ------------------------

def test_download_model_accepts_a_language_list(monkeypatch):
    from cadence import cli

    got = []
    import spacy.cli

    monkeypatch.setattr(spacy.cli, "download", lambda m: got.append(m))
    assert cli.main(["download-model", "--lang", "es", "pt"]) == 0
    assert got == ["es_core_news_sm", "pt_core_news_sm"]
    got.clear()
    assert cli.main(["download-model", "--lang", "all"]) == 0
    assert got == ["en_core_web_sm", "es_core_news_sm", "pt_core_news_sm"]
    got.clear()
    assert cli.main(["download-model"]) == 0
    assert got == ["en_core_web_sm"]


@needs_es
def test_mcp_tools_take_a_language():
    from cadence import mcp_server

    out = mcp_server.measure_voice([ES], name="es", language="es")
    assert out["language"] == "es"
    assert "LANGUAGE: Spanish" in out["spec"]
    fid = mcp_server.check_fidelity("No llegó el 2025-04-30.", "Llegó el 2025-04-30.",
                                    language="es")
    assert fid["negations"] == [1, 0]
