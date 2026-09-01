"""Text report, in the shape of the hand-written analysis it was ported from."""

from __future__ import annotations

from .diagnostics import Analysis
from .rewrite import RewriteResult
from .syntax import constituency, dep_tree, pos_line

RULE = "=" * 78
THIN = "-" * 78


def _header(analysis: Analysis) -> list[str]:
    m = analysis.metrics
    return [
        RULE, "SYNTACTIC ANALYSIS", RULE,
        "Scheme : Penn Treebank POS + projected constituency + dependencies",
        f"Metrics: {m['sentences']} sentences | {m['tokens']} tokens | "
        f"max dependency depth {m['max_depth']}",
        f"Findings: {m['findings']} ({m['auto_fixable']} mechanically fixable)",
        "",
    ]


def _repairs(analysis: Analysis) -> list[str]:
    repairs = analysis.metrics["repairs"]
    if not repairs:
        return []
    out = [THIN, "PARSE REPAIRS APPLIED", THIN,
           "The parser is trained on newswire and mis-analyses this register in known",
           "ways. Each repair below was applied before any diagnostic ran.", ""]
    out += [f"  {r}" for r in repairs]
    return out + [""]


def _sentences(analysis: Analysis) -> list[str]:
    out = [THIN, "SENTENCE-BY-SENTENCE", THIN]
    for view, sent in zip(analysis.sentences, analysis.doc.sents, strict=False):
        out += ["", f"S{view.index}  (dependency depth {view.depth}, "
                    f"parse confidence {view.confidence})", f'"{view.text}"', ""]
        if view.confidence_reasons:
            out += ["  parse warnings:"] + [f"    - {r}" for r in view.confidence_reasons] + [""]
        out += ["POS", pos_line(sent), "", "CONSTITUENCY (projected)",
                constituency(sent).brackets(), "", "DEPENDENCIES", dep_tree(sent), ""]
    return out


def _findings(analysis: Analysis) -> list[str]:
    out = [THIN, "FINDINGS", THIN]
    if not analysis.findings:
        out.append("None.")
        return out
    for f in analysis.findings:
        out += ["", f"[{f.severity.upper():6}] {f.code}  ({f.origin})  S{f.sent_index}",
                f"  {f.label}", f"  span : {f.span[:150]}",
                f"  why  : {f.detail}", f"  fix  : {f.fix}",
                f"  auto : {'yes' if f.auto else 'no -- needs a judgement a parser cannot make'}"]
    return out


def _observations(analysis: Analysis) -> list[str]:
    if not analysis.observations:
        return []
    out = ["", THIN, "OBSERVATIONS (register facts, not defects)", THIN]
    for o in analysis.observations:
        out += ["", f"{o.code} ({o.origin}): {o.label}", f"  {o.detail}"]
    return out


def _rewrite(result: RewriteResult) -> list[str]:
    out = ["", RULE, f"REWRITE -- backend: {result.backend}", RULE, ""]
    out += ["RULE: structure and grammar only. No fact added, dropped, or",
            "strengthened. Information gaps are reported, never filled.", "",
            THIN, "TEXT", THIN, "", result.text, ""]
    if result.changes:
        out += [THIN, "CHANGES", THIN]
        for c in result.changes:
            out += ["", f"[{c.code}]", f"  before: {c.before[:160]}",
                    f"  after : {c.after[:160]}", f"  why   : {c.rationale}"]
    if result.disputed:
        out += ["", THIN, "DISPUTED -- findings the model says rest on a misparse", THIN,
                "These were NOT acted on. Each is a claim that the parser, not the",
                "prose, is what went wrong. Worth checking: a confirmed dispute is a",
                "detector bug.", ""]
        for d in result.disputed:
            out += [f"  [{d.code}] S{d.sentence}", f"    {d.reason}"]
    if result.inferences:
        out += ["", THIN, "INFERENCES -- confirm or correct these", THIN]
        out += [f"  - {i}" for i in result.inferences]
    if result.gaps:
        out += ["", THIN, "GAPS -- content the rewrite could not fix", THIN,
                "These are missing facts, not grammar. Only a human can close them.", ""]
        out += [f"  - {g}" for g in result.gaps]
    if result.brief:
        out += ["", THIN, "BRIEF -- findings needing judgement", THIN]
        out += [f"  - {b}" for b in result.brief]
    out += _fidelity(result.fidelity)
    if result.note:
        out += ["", f"NOTE: {result.note}"]
    return out


def _fidelity(fid) -> list[str]:
    """The mechanical check on the no-invention rule, next to the rule itself."""
    if fid is None:
        return []
    verdict = "PASS" if fid.passed else "FAIL"
    out = ["", THIN, f"CONTENT FIDELITY -- {fid.score}/100  [{verdict}]", THIN,
           "Measured against the source. Not the model's account of itself.", ""]
    out += [f"  anchors kept      : {len(fid.preserved)}",
            f"  novel word rate   : {fid.novel_rate}",
            f"  sentence coverage : {fid.coverage}",
            f"  negations         : {fid.negations[0]} -> {fid.negations[1]}"]
    if fid.introduced:
        out += ["", "  INTRODUCED -- present in the output, absent from the source:"]
        for a in fid.introduced:
            out.append(f"    - [{a.kind}] {a.text}" + (f"  ({a.note})" if a.note else ""))
    if fid.dropped:
        out += ["", "  DROPPED -- in the source, missing from the output, not a gap:"]
        out += [f"    - [{a.kind}] {a.text}" for a in fid.dropped]
    if fid.excused:
        out += ["", "  EXCUSED -- missing, but reported as a gap, which is correct:"]
        out += [f"    - [{a.kind}] {a.text}" for a in fid.excused]
    if fid.uncovered_sentences:
        out += ["", "  UNCOVERED source sentences: "
                + ", ".join(f"S{i}" for i in fid.uncovered_sentences)]
    out += [""] + [f"  note: {n}" for n in fid.notes]
    return out


def text_report(analysis: Analysis, result: RewriteResult | None = None,
                trees: bool = True) -> str:
    lines = _header(analysis) + _repairs(analysis)
    if trees:
        lines += _sentences(analysis)
    lines += _findings(analysis) + _observations(analysis)
    if result is not None:
        lines += _rewrite(result)
    lines += ["", RULE]
    return "\n".join(lines)
