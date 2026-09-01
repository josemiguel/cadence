"""Syntactic layer: PTB tags, dependency trees, projected constituency trees.

The dependency layer is a real parse (spaCy, en_core_web_sm). The constituency
layer is a HEAD PROJECTION of that parse, not a native PTB parse -- every
bracketing here is derived from dependency structure by projecting each head
over its dependents. It reproduces PTB conventions where they follow from
dependencies (flat base NPs, PP/ADJP/SBAR labels, -SBJ/-PRD function tags) and
cannot reproduce what only a constituency treebank knows (traces, empty
categories, UCP as a licensed node). Labelled as projected wherever it is shown.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

MODEL = "en_core_web_sm"

# Dependency labels that make their head a clause rather than a phrase.
_SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
_FINITE_TAGS = {"VBZ", "VBD", "VBP", "MD"}

# dep_ -> PTB function tag
_FUNCTION_TAGS = {
    "nsubj": "-SBJ",
    "nsubjpass": "-SBJ",
    "csubj": "-SBJ",
    "attr": "-PRD",
    "acomp": "-PRD",
    "oprd": "-PRD",
    "advcl": "-ADV",
}

_TEMPORAL_ENTS = {"DATE", "TIME"}


@functools.lru_cache(maxsize=1)
def _load():
    import spacy

    from . import repair

    try:
        nlp = spacy.load(MODEL)
        repair.install_tokenizer_rules(nlp)
        return nlp
    except OSError as exc:  # pragma: no cover - install-time failure
        from .errors import ModelNotInstalled

        raise ModelNotInstalled(
            f"spaCy model {MODEL!r} is missing. Install it with:\n"
            f"    cadence download-model\n"
            f"or, equivalently:\n"
            f"    python -m spacy download {MODEL}"
        ) from exc


def parse(text: str):
    """Parse text, apply note-register repairs, return the Doc."""
    from . import repair

    return repair.apply(_load()(text.strip()))


# --------------------------------------------------------------------------
# POS layer
# --------------------------------------------------------------------------

# PTB writes brackets as -LRB-/-RRB-; spaCy hands back the literal character.
_BRACKET_TAGS = {"(": "-LRB-", ")": "-RRB-", "[": "-LRB-", "]": "-RRB-"}


def ptb_tag(tok) -> str:
    """PTB tag for a token, with PTB's bracket convention applied."""
    return _BRACKET_TAGS.get(tok.text, tok.tag_)


def pos_line(sent, width: int = 76) -> str:
    """Inline `word/TAG` rendering, wrapped."""
    items = [f"{t.text}/{ptb_tag(t)}" for t in sent if not t.is_space]
    lines, cur = [], ""
    for item in items:
        if cur and len(cur) + len(item) + 1 > width:
            lines.append(cur)
            cur = item
        else:
            cur = f"{cur} {item}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Dependency tree
# --------------------------------------------------------------------------

def dep_tree(sent, show_punct: bool = False) -> str:
    """ASCII dependency tree rooted at the sentence root."""
    root = sent.root
    lines = [f"{root.text}/{ptb_tag(root)} --- ROOT"]
    kids = _dep_children(root, show_punct)
    for i, kid in enumerate(kids):
        _dep_branch(kid, "", i == len(kids) - 1, lines, show_punct)
    return "\n".join(lines)


def _dep_children(tok, show_punct: bool):
    kids = [c for c in tok.children if show_punct or c.dep_ != "punct"]
    return sorted(kids, key=lambda t: t.i)


def _dep_branch(tok, prefix: str, last: bool, lines: list[str], show_punct: bool):
    connector = "\\-- " if last else "|-- "
    lines.append(f"{prefix}{connector}{tok.text}/{ptb_tag(tok)}  {tok.dep_}")
    kids = _dep_children(tok, show_punct)
    child_prefix = prefix + ("    " if last else "|   ")
    for i, kid in enumerate(kids):
        _dep_branch(kid, child_prefix, i == len(kids) - 1, lines, show_punct)


# Arcs that do not embed. Coordination is sibling structure and a compound is
# internal to one NP; counting them as depth inflates the metric on any
# coordinated name (`with Ann Lee and Bob Ray` scored three levels
# of "nesting" that a reader never has to hold open).
_NON_EMBEDDING_DEPS = {"conj", "compound", "flat", "appos", "cc", "punct", "dep"}


def _arc_cost(tok) -> int:
    return 0 if tok.dep_ in _NON_EMBEDDING_DEPS else 1


def dep_depth(sent) -> int:
    """Maximum embedding depth, counting only arcs that actually nest."""

    def depth(tok):
        kids = [c for c in tok.children if c.dep_ != "punct"]
        return max([depth(k) + _arc_cost(k) for k in kids] + [1])

    return depth(sent.root)


def deepest_path(sent) -> list:
    """The chain of tokens realising dep_depth, for reporting where nesting bites."""

    def walk(tok):
        kids = [c for c in tok.children if c.dep_ != "punct"]
        if not kids:
            return [tok]
        best = max(kids, key=lambda k: depth_of(k) + _arc_cost(k))
        return [tok] + walk(best)

    def depth_of(tok):
        kids = [c for c in tok.children if c.dep_ != "punct"]
        return max([depth_of(k) + _arc_cost(k) for k in kids] + [1])

    return walk(sent.root)


# --------------------------------------------------------------------------
# Constituency projection
# --------------------------------------------------------------------------

@dataclass
class Node:
    label: str
    children: list = field(default_factory=list)

    def brackets(self, indent: int = 0) -> str:
        pad = " " * indent
        if len(self.children) == 1 and isinstance(self.children[0], Leaf):
            return f"{pad}({self.label} {self.children[0].render()})"
        parts = [f"{pad}({self.label}"]
        for child in self.children:
            if isinstance(child, Leaf):
                parts.append(f"{' ' * (indent + 2)}{child.render()}")
            else:
                parts.append(child.brackets(indent + 2))
        return "\n".join(parts) + ")"


@dataclass
class Leaf:
    tag: str
    word: str

    def render(self) -> str:
        return f"({self.tag} {self.word})"


def _phrase_label(tok) -> str:
    pos = tok.pos_
    if pos in {"NOUN", "PROPN", "PRON", "NUM"}:
        return "NP"
    if pos == "ADJ":
        return "ADJP"
    if pos == "ADV":
        return "ADVP"
    if pos == "ADP":
        return "PRT" if tok.dep_ == "prt" else "PP"
    if pos == "PART":
        return "PRT" if tok.dep_ == "prt" else "VP"
    if pos == "DET":
        return "NP"
    if pos == "SCONJ":
        return "SBAR"
    if pos == "INTJ":
        return "INTJ"
    if pos in {"VERB", "AUX"}:
        has_subject = any(c.dep_ in _SUBJECT_DEPS for c in tok.children)
        if has_subject and tok.tag_ in _FINITE_TAGS:
            return "S"
        if has_subject:
            return "S"
        return "VP"
    return "X"


def _function_tag(tok) -> str:
    tag = _FUNCTION_TAGS.get(tok.dep_, "")
    if tok.dep_ == "npadvmod" and tok.ent_type_ in _TEMPORAL_ENTS:
        return "-TMP"
    if tok.dep_ == "mark":
        return ""
    return tag


def _label_for(tok, is_root: bool) -> str:
    label = _phrase_label(tok)
    if is_root:
        return label
    return label + _function_tag(tok)


# Dependency labels whose dependent always earns its own phrase node, even as a
# single token. Everything else that is childless projects as a bare leaf, which
# is what keeps base NPs flat the way PTB keeps them flat.
_ALWAYS_PHRASE = {
    "nsubj", "nsubjpass", "csubj", "csubjpass", "dobj", "iobj", "dative",
    "attr", "acomp", "oprd", "pobj", "prep", "agent", "advcl", "ccomp",
    "xcomp", "pcomp", "relcl", "acl", "appos", "conj", "npadvmod", "expl",
    "parataxis",
}

_VERBAL = {"VERB", "AUX"}


def _projects_phrase(tok) -> bool:
    if tok.dep_ == "punct":
        return False
    if next(tok.children, None) is not None:
        return True
    return tok.dep_ in _ALWAYS_PHRASE


def constituency(sent) -> Node:
    """Project a constituency tree from the dependency parse.

    Clause heads get an S over a VP, so the copula and its predicative
    complement sit inside VP as PTB puts them. Sentence-final punctuation is
    lifted to the top node rather than buried in the VP.
    """
    toks = [t for t in sent if not t.is_space]
    final_punct_i = toks[-1].i if toks and toks[-1].dep_ == "punct" else None

    def node_for(tok):
        return build(tok) if _projects_phrase(tok) else Leaf(ptb_tag(tok), tok.text)

    def build(tok, is_root=False):
        kids = [k for k in sorted(tok.children, key=lambda t: t.i) if k.i != final_punct_i]
        subj = next((k for k in kids if k.dep_ in _SUBJECT_DEPS), None)
        label = _label_for(tok, is_root)
        head_leaf = Leaf(ptb_tag(tok), tok.text)

        if subj is not None and tok.pos_ in _VERBAL:
            s_items, vp_items = [], [(tok.i, head_leaf)]
            for kid in kids:
                target = s_items if (kid is subj or kid.i < subj.i) else vp_items
                target.append((kid.i, node_for(kid)))
            s_items.sort(key=lambda pair: pair[0])
            vp_items.sort(key=lambda pair: pair[0])
            return Node(label, [n for _, n in s_items] + [Node("VP", [n for _, n in vp_items])])

        items = [(tok.i, head_leaf)] + [(k.i, node_for(k)) for k in kids]
        items.sort(key=lambda pair: pair[0])
        return Node(label, [n for _, n in items])

    tree = build(sent.root, is_root=True)
    if final_punct_i is not None:
        punct = sent.doc[final_punct_i]
        tree.children.append(Leaf(ptb_tag(punct), punct.text))
    return tree


# --------------------------------------------------------------------------
# Chunking on predication punctuation
# --------------------------------------------------------------------------

_CHUNK_PUNCT = {":", ";", "--", "---", "—", "–"}


def chunks(sent) -> list[tuple[str, list]]:
    """Split a sentence on `:` `;` and dashes.

    Returns (delimiter_before, tokens) pairs. These are the sites where the
    source text was using punctuation to predicate; each chunk is then checked
    for a finite verb of its own.
    """
    out, cur, delim = [], [], ""
    for tok in sent:
        if tok.text in _CHUNK_PUNCT and tok.dep_ == "punct":
            if cur:
                out.append((delim, cur))
            cur, delim = [], tok.text
            continue
        cur.append(tok)
    if cur:
        out.append((delim, cur))
    return out


def has_finite_verb(tokens) -> bool:
    return any(t.tag_ in _FINITE_TAGS for t in tokens)
