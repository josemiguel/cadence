"""The languages cadence measures, and what differs between them.

Every measurement in this package is a count over a dependency parse: finite
verbs, subordinate arcs, tokens before the main verb, coordinated items. The
counts are the same in English, Spanish and Portuguese. What differs is the
parser. spaCy's English model uses Penn tags and ClearNLP dependency labels;
its Spanish and Portuguese models use Universal Dependencies labels, with the
tag column repeating the part of speech and finiteness living in morphology.

This module holds the per-language facts and one function, `normalise`, that
rewrites a Universal Dependencies parse into the labels every downstream count
already reads. The alternative, teaching every counter two label sets, would
have spread the difference across the whole package and guaranteed that one
counter forgot.

Word lists here are small and closed-class on purpose: negators, relative
dates, spelled-out numbers, and the verbs that carry a negation in their
meaning. Nothing about style is language-specific; a profile is a profile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT = "en"


@dataclass(frozen=True)
class Language:
    code: str
    name: str  # in English, which is the language the prompts are written in
    native_name: str
    model: str
    negators: frozenset
    relative_dates: frozenset
    lexical_negators: frozenset
    word_numbers: dict
    dep_map: dict
    #: Lemma of the auxiliary that forms the periphrastic passive, and the
    #: prepositions that introduce its agent. Empty where the parser labels
    #: passives itself, as the English model does.
    passive_aux: frozenset = frozenset()
    agent_markers: frozenset = frozenset()


# Universal Dependencies labels rewritten to the ClearNLP ones the English
# model emits and the counters read. Labels not listed pass through unchanged;
# `advmod` carrying a negation is handled separately, by word and morphology.
_UD_TO_CLEAR = {
    "nsubj:pass": "nsubjpass",
    "csubj:pass": "csubjpass",
    "aux:pass": "auxpass",
    "obl:agent": "agent",
    "acl:relcl": "relcl",
    "obj": "dobj",
    "expl:pv": "expl",
    "expl:pass": "expl",
    "expl:impers": "expl",
}

_EN_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000,
}

_ES_WORD_NUMBERS = {
    "cero": 0, "uno": 1, "una": 1, "un": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11,
    "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciséis": 16,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100,
    "mil": 1000, "millón": 1_000_000, "millon": 1_000_000, "millones": 1_000_000,
}

_PT_WORD_NUMBERS = {
    "zero": 0, "um": 1, "uma": 1, "dois": 2, "duas": 2, "três": 3, "tres": 3,
    "quatro": 4, "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
    "onze": 11, "doze": 12, "treze": 13, "catorze": 14, "quatorze": 14, "quinze": 15,
    "dezesseis": 16, "dezasseis": 16, "dezessete": 17, "dezassete": 17, "dezoito": 18,
    "dezenove": 19, "dezanove": 19, "vinte": 20, "trinta": 30, "quarenta": 40,
    "cinquenta": 50, "sessenta": 60, "setenta": 70, "oitenta": 80, "noventa": 90,
    "cem": 100, "cento": 100, "mil": 1000, "milhão": 1_000_000, "milhao": 1_000_000,
    "milhões": 1_000_000, "milhoes": 1_000_000, "bilhão": 1_000_000_000,
    "bilhao": 1_000_000_000, "bilhões": 1_000_000_000, "bilhoes": 1_000_000_000,
}

LANGUAGES: dict[str, Language] = {
    "en": Language(
        code="en", name="English", native_name="English", model="en_core_web_sm",
        negators=frozenset({"no", "not", "never", "none", "nothing", "nobody", "neither",
                            "nor", "without", "n't"}),
        relative_dates=frozenset({"today", "yesterday", "tomorrow", "now", "later",
                                  "recently", "currently", "soon", "then", "earlier",
                                  "lately", "ago"}),
        lexical_negators=frozenset({"decline", "refuse", "fail", "deny", "lack", "absent",
                                    "unable", "reject", "omit", "avoid", "cease", "forgo",
                                    "withhold", "exclude", "miss"}),
        word_numbers=_EN_WORD_NUMBERS,
        dep_map={},
    ),
    "es": Language(
        code="es", name="Spanish", native_name="Español", model="es_core_news_sm",
        negators=frozenset({"no", "nunca", "jamás", "nada", "nadie", "ninguno", "ninguna",
                            "ningún", "ni", "tampoco", "sin"}),
        relative_dates=frozenset({"hoy", "ayer", "mañana", "ahora", "luego", "recientemente",
                                  "actualmente", "pronto", "entonces", "antes", "después",
                                  "últimamente"}),
        lexical_negators=frozenset({"negar", "rechazar", "rehusar", "fallar", "carecer",
                                    "faltar", "omitir", "evitar", "cesar", "excluir",
                                    "ausente", "incapaz", "descartar"}),
        word_numbers=_ES_WORD_NUMBERS,
        dep_map=_UD_TO_CLEAR,
        passive_aux=frozenset({"ser"}),
        agent_markers=frozenset({"por"}),
    ),
    "pt": Language(
        code="pt", name="Portuguese", native_name="Português", model="pt_core_news_sm",
        negators=frozenset({"não", "nunca", "jamais", "nada", "ninguém", "nenhum", "nenhuma",
                            "nem", "tampouco", "sem"}),
        relative_dates=frozenset({"hoje", "ontem", "amanhã", "agora", "depois", "recentemente",
                                  "atualmente", "logo", "então", "antes", "ultimamente"}),
        lexical_negators=frozenset({"negar", "recusar", "rejeitar", "falhar", "carecer",
                                    "faltar", "omitir", "evitar", "cessar", "excluir",
                                    "ausente", "incapaz", "descartar"}),
        word_numbers=_PT_WORD_NUMBERS,
        dep_map=_UD_TO_CLEAR,
        passive_aux=frozenset({"ser"}),
        agent_markers=frozenset({"por", "pelo", "pela", "pelos", "pelas"}),
    ),
}

CODES = tuple(LANGUAGES)

#: Letters a spelled-out number can contain, in any of the three languages.
WORD_SPLIT = re.compile(r"[^a-záéíóúàâêôãõçñüï]+")


def get(code: str | None) -> Language:
    """The language for a code such as `es`, `pt-BR` or `en_GB`; None means English."""
    if not code:
        return LANGUAGES[DEFAULT]
    key = str(code).strip().lower().replace("_", "-").split("-")[0]
    try:
        return LANGUAGES[key]
    except KeyError:
        raise ValueError(
            f"unsupported language {code!r}; cadence measures {', '.join(CODES)}"
        ) from None


def normalise(doc, language: Language) -> None:
    """Rewrite a parse's labels into the ones the counters read. In place.

    English needs nothing. For the Universal Dependencies models the label map
    above is applied, and an adverb that is a negator becomes `neg`, which is
    what the English model calls it and what the fidelity check counts.
    """
    if not language.dep_map and language.code == DEFAULT:
        return
    for tok in doc:
        dep = language.dep_map.get(tok.dep_)
        if dep is not None:
            tok.dep_ = dep
        elif tok.dep_ == "advmod" and (tok.lower_ in language.negators
                                       or "Neg" in tok.morph.get("Polarity")):
            tok.dep_ = "neg"
    if language.passive_aux:
        _label_periphrastic_passives(doc, language)


def _label_periphrastic_passives(doc, language: Language) -> None:
    """`fue enviado por el equipo`: mark what the small models leave as plain `aux`.

    The Spanish and Portuguese models label some passives with `aux:pass` and
    `nsubj:pass` and leave others as `aux` under a participle. The form itself
    is unambiguous, `ser` plus a past participle, so it is read from morphology:
    the auxiliary becomes `auxpass`, the participle's subject `nsubjpass`, and
    an oblique introduced by the agent preposition becomes `agent`.
    """
    for tok in doc:
        if tok.dep_ != "aux" or tok.lemma_.lower() not in language.passive_aux:
            continue
        head = tok.head
        if not is_participle(head) or head.pos_ not in ("VERB", "AUX"):
            continue
        tok.dep_ = "auxpass"
        for kid in head.children:
            if kid.dep_ == "nsubj":
                kid.dep_ = "nsubjpass"
            elif kid.dep_ == "csubj":
                kid.dep_ = "csubjpass"
            # A passive participle takes no direct object, so an `obj` carrying the
            # agent preposition is the agent the parser mislabelled.
            elif kid.dep_ in ("obl", "nmod", "dobj") and any(
                    c.dep_ == "case" and c.lower_ in language.agent_markers for c in kid.children):
                kid.dep_ = "agent"


def is_finite(tok) -> bool:
    """A finite verb, whichever way the model marks it.

    The English model says so with a Penn tag; the Universal Dependencies models
    say so in morphology. Both are checked, so a caller never asks which.
    """
    if tok.tag_ in ("VBZ", "VBD", "VBP", "MD"):
        return True
    return tok.pos_ in ("VERB", "AUX") and "Fin" in tok.morph.get("VerbForm")


def is_participle(tok) -> bool:
    return tok.tag_ == "VBN" or "Part" in tok.morph.get("VerbForm")


def _stop_words(code: str) -> frozenset:
    if code == "en":
        from spacy.lang.en.stop_words import STOP_WORDS
    elif code == "es":
        from spacy.lang.es.stop_words import STOP_WORDS
    else:
        from spacy.lang.pt.stop_words import STOP_WORDS
    return frozenset(STOP_WORDS)


_TOKEN = re.compile(r"[a-záéíóúàâêôãõçñüï']+")


def guess(text: str, default: str = DEFAULT) -> str:
    """Which of the three languages a text is written in, by function words.

    Function words are the strongest authorship signal and also the strongest
    language signal: a few hundred closed-class words cover half of any text.
    No model is loaded. Ties and empty input fall back to `default`.
    """
    words = _TOKEN.findall((text or "").lower())
    if len(words) < 3:
        return default
    scores = {}
    for code in CODES:
        stops = _stop_words(code)
        scores[code] = sum(1 for w in words if w in stops)
    best = max(scores.values())
    if best == 0:
        return default
    winners = [c for c, s in scores.items() if s == best]
    return default if default in winners else winners[0]
