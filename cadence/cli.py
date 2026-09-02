"""Command line entry point.

One subcommand per question someone actually asks. The previous version was a
single flag namespace shared by six commands, which meant `--brief` appeared in
the help for `analyze` and `--all-sentences` for `profile`, and nothing in the
parser knew which combinations were meaningful.

Exit codes are distinct because scripts branch on them: 2 for bad input, 3 for
a missing spaCy model, 4 for missing credentials. The last two are both
recoverable by the user doing one specific thing, and a single failure code
would not tell them which.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, languages
from .diagnostics import analyze
from .errors import MissingAPIKey, ModelNotInstalled
from .generate import compare, generate
from .profile import build_profile, spec_text
from .report import text_report
from .rewrite import DEFAULT_MODEL, render_trees, rewrite
from .scores import score as score_text

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_NO_MODEL = 3
EXIT_NO_KEY = 4


def _read(source: str) -> str:
    if source == "-":
        return sys.stdin.read()
    return Path(source).read_text()


def _read_all(sources: list[str]) -> list[str]:
    docs = [d for d in (_read(s) for s in sources) if d.strip()]
    if not docs:
        raise ValueError("input is empty")
    return docs


def _emit(text: str, out: str | None) -> int:
    if out:
        Path(out).write_text(text)
        print(f"wrote {out}", file=sys.stderr)
    else:
        print(text)
    return EXIT_OK


def _corpus(args):
    docs = _read_all(args.corpus)
    prof = build_profile(docs, name=args.name or Path(args.corpus[0]).stem, lang=args.lang)
    return docs, prof


# --- commands --------------------------------------------------------------

def _cmd_analyze(args) -> int:
    a = analyze(_read_all([args.source])[0], lang=args.lang)
    if args.json:
        return _emit(json.dumps({
            "metrics": a.metrics,
            "sentences": [vars(s) for s in a.sentences],
            "findings": [f.as_dict() for f in a.findings],
            "observations": [o.as_dict() for o in a.observations],
        }, indent=2, ensure_ascii=False), args.out)
    return _emit(text_report(a, None, trees=not args.no_trees), args.out)


def _cmd_render(args) -> int:
    a = analyze(_read_all([args.source])[0], lang=args.lang)
    return _emit(render_trees(a, only_with_findings=not args.all_sentences), args.out)


def _rewrite_result(args, analysis):
    """Rewrite, falling back to the rules backend if credentials are missing.

    A fallback is right here and wrong in the library: a person at a terminal
    wants the deterministic repairs rather than an error, but a caller that
    asked for the llm backend needs to know it did not get it.
    """
    try:
        return rewrite(analysis, backend=args.backend, model=args.model,
                       apply_subordinators=args.apply_subordinators,
                       use_trees=not args.no_tree_evidence)
    except MissingAPIKey as exc:
        print(f"warning: {exc}\nfalling back to the rules backend.", file=sys.stderr)
        return rewrite(analysis, backend="rules",
                       apply_subordinators=args.apply_subordinators)


def _cmd_rewrite(args) -> int:
    a = analyze(_read_all([args.source])[0], lang=args.lang)
    r = _rewrite_result(args, a)
    if args.json:
        return _emit(json.dumps(_rewrite_payload(a, r), indent=2, ensure_ascii=False), args.out)
    return _emit(text_report(a, r, trees=False), args.out)


def _cmd_report(args) -> int:
    a = analyze(_read_all([args.source])[0], lang=args.lang)
    r = _rewrite_result(args, a)
    if args.json:
        return _emit(json.dumps(_rewrite_payload(a, r), indent=2, ensure_ascii=False), args.out)
    return _emit(text_report(a, r, trees=not args.no_trees), args.out)


def _rewrite_payload(a, r) -> dict:
    return {
        "metrics": a.metrics,
        "findings": [f.as_dict() for f in a.findings],
        "observations": [o.as_dict() for o in a.observations],
        "rewrite": {
            "backend": r.backend, "text": r.text,
            "changes": [vars(c) for c in r.changes],
            "gaps": r.gaps, "inferences": r.inferences, "brief": r.brief,
            "disputed": [vars(d) for d in r.disputed], "note": r.note,
            "usage": r.usage,
            "fidelity": None if r.fidelity is None else r.fidelity.as_dict(),
        },
    }


def _cmd_profile(args) -> int:
    _, prof = _corpus(args)
    if args.json:
        return _emit(json.dumps(prof.as_dict(), indent=2, ensure_ascii=False), args.out)
    out = spec_text(prof)
    if prof.warnings:
        out += "\n\nWARNINGS\n" + "\n".join(f"  - {w}" for w in prof.warnings)
    return _emit(out, args.out)


def _generation(args, task: str) -> int:
    docs, prof = _corpus(args)
    brief = _read_all([args.source])[0]
    if args.mode == "compare":
        results = compare(brief, prof, docs, model=args.model,
                          iterations=args.iterations, task=task, lang=args.lang)
    else:
        results = [generate(brief, mode=args.mode, profile=prof, samples=docs,
                            model=args.model, iterations=args.iterations, task=task,
                            lang=args.lang)]
    if args.json:
        return _emit(json.dumps([{
            "mode": r.mode, "similarity": r.similarity, "text": r.text,
            "divergence": r.divergence, "note": r.note, "usage": r.usage,
            "latency_s": r.latency_s, "length_ratio": r.length_ratio,
            "fidelity": None if r.fidelity is None else r.fidelity.as_dict(),
            "attempts": [{"iteration": a.iteration, "similarity": a.similarity,
                          "worst": a.worst} for a in r.attempts],
        } for r in results], indent=2, ensure_ascii=False), args.out)

    blocks = []
    for r in results:
        head = f"MODE: {r.mode}   voice match {r.similarity}/100"
        if r.fidelity is not None:
            verdict = "PASS" if r.fidelity.passed else "FAIL"
            head += f"   content kept {r.fidelity.score}/100 [{verdict}]"
        if r.source_tokens:
            head += f"   length {r.output_tokens}/{r.source_tokens} ({r.length_ratio}x)"
        blocks += ["=" * 78, head, "=" * 78, "", r.text, "", f"note: {r.note}"]
        if r.fidelity is not None and not r.fidelity.passed:
            for anchor in r.fidelity.introduced:
                blocks.append(f"  INTRODUCED [{anchor.kind}] {anchor.text}")
            for anchor in r.fidelity.dropped:
                blocks.append(f"  DROPPED    [{anchor.kind}] {anchor.text}")
        if len(r.attempts) > 1:
            blocks.append("attempts: " + ", ".join(
                f"#{a.iteration}={a.similarity}" for a in r.attempts))
        blocks.append("")
    return _emit("\n".join(blocks), args.out)


def _cmd_restyle(args) -> int:
    return _generation(args, "restyle")


def _cmd_compose(args) -> int:
    return _generation(args, "compose")


def _cmd_score(args) -> int:
    _, prof = _corpus(args)
    text = _read_all([args.source])[0]
    source = _read(args.source_text) if args.source_text else None
    s = score_text(text, prof, source=source)
    if args.json:
        return _emit(json.dumps(s.as_dict(), indent=2, ensure_ascii=False), args.out)
    lines = [f"voice match   {s.voice_similarity}/100"]
    if s.fidelity is not None:
        verdict = "PASS" if s.fidelity.passed else "FAIL"
        lines.append(f"content kept  {s.fidelity.score}/100 [{verdict}]")
    else:
        lines.append("content kept  not measured (pass --source-text to compare)")
    lines += [""] + [f"note: {c}" for c in s.caveats]
    return _emit("\n".join(lines), args.out)


def _cmd_mcp(args) -> int:
    try:
        from .mcp_server import main as serve
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    return serve()


def _cmd_download_model(args) -> int:
    from spacy.cli import download

    codes = list(languages.CODES) if "all" in args.lang else list(dict.fromkeys(args.lang))
    for code in codes:
        model = languages.get(code).model
        print(f"downloading spaCy model {model} ({languages.get(code).name})...",
              file=sys.stderr)
        download(model)
    return EXIT_OK


# --- parser ----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="cadence",
        description="Parse prose, measure its shape, rewrite it from the parse.",
    )
    ap.add_argument("--version", action="version", version=f"cadence {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name, help_text, func):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("-o", "--out", help="write to this file instead of stdout")
        p.set_defaults(func=func)
        return p

    def add_lang(p):
        p.add_argument("--lang", default=languages.DEFAULT, choices=list(languages.CODES),
                       help="language of the text and corpus (default en)")

    def add_text(p):
        p.add_argument("source", help="file to read, or - for stdin")
        add_lang(p)

    def add_corpus(p):
        p.add_argument("--corpus", nargs="+", required=True,
                       help="files of writing to measure and match")
        p.add_argument("--name", help="a label for the corpus")

    def add_llm(p):
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--iterations", type=int, default=2,
                       help="verify-loop corrections (default 2)")
        p.add_argument("--mode", default="compare",
                       choices=["compare", "tone", "profile", "profile_verify"],
                       help="which mode(s) to run (default: compare all three)")
        p.add_argument("--json", action="store_true")

    def add_rewrite_flags(p):
        p.add_argument("--backend", choices=["auto", "rules", "llm"], default="auto",
                       help="auto uses the llm when credentials exist (default)")
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--apply-subordinators", action="store_true",
                       help="rules backend: also disambiguate since/as/while")
        p.add_argument("--no-tree-evidence", action="store_true",
                       help="withhold the step 1 parse from the model")
        p.add_argument("--json", action="store_true")

    p = add("analyze", "trees and findings, no rewrite", _cmd_analyze)
    add_text(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-trees", action="store_true")

    p = add("render", "step 1 alone: the parse", _cmd_render)
    add_text(p)
    p.add_argument("--all-sentences", action="store_true",
                   help="include sentences with no findings")

    p = add("rewrite", "step 1 then step 2, without the trees", _cmd_rewrite)
    add_text(p)
    add_rewrite_flags(p)

    p = add("report", "everything: trees, findings, rewrite, fidelity", _cmd_report)
    add_text(p)
    add_rewrite_flags(p)
    p.add_argument("--no-trees", action="store_true")

    p = add("profile", "measure a corpus and print the spec", _cmd_profile)
    p.add_argument("corpus", nargs="+", help="files to measure")
    p.add_argument("--name", help="a label for the corpus")
    p.add_argument("--json", action="store_true")
    add_lang(p)

    p = add("restyle", "re-render a text in the corpus's structure", _cmd_restyle)
    add_text(p)
    add_corpus(p)
    add_llm(p)

    p = add("compose", "write about a topic in the corpus's structure", _cmd_compose)
    add_text(p)
    add_corpus(p)
    add_llm(p)

    p = add("score", "voice match, and content kept against a source", _cmd_score)
    add_text(p)
    add_corpus(p)
    p.add_argument("--source-text", metavar="FILE",
                   help="the text this one was rewritten from, for the fidelity score")
    p.add_argument("--json", action="store_true")

    p = add("download-model", "install the spaCy model(s) this package needs",
            _cmd_download_model)
    p.add_argument("--lang", nargs="+", default=[languages.DEFAULT],
                   choices=[*languages.CODES, "all"],
                   help="which language model(s) to install (default en; `all` for every one)")

    add("mcp", "serve cadence as a local MCP server on stdio", _cmd_mcp)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ModelNotInstalled as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NO_MODEL
    except MissingAPIKey as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NO_KEY
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT


if __name__ == "__main__":
    raise SystemExit(main())
