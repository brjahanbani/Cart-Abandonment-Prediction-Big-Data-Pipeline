"""
build.py
=========

Builds the manuscript targets and checks the arXiv target is unbranded.

    python paper/build.py                 # all targets + checks + arXiv bundle
    python paper/build.py arxiv           # one target
    python paper/build.py --allow-latex-errors

Targets (outputs land in paper/build/<target>/paper.pdf, gitignored):
  ijacsa-nonblind  paper/ijacsa/IJACSA_Final_NonBlind_Reference_Copy.tex, unchanged
  ijacsa-anon      paper/ijacsa/IJACSA_Submission_Anonymized_Babak_Cart_Abandonment.tex, unchanged
  arxiv            paper/arxiv/main.tex (non-blind source minus IJACSA branding)

Checks run on the arxiv target (the script exits non-zero if any fails):
  1. Source: main.tex body (\\begin{document} onward) is identical to the
     non-blind IJACSA body, apart from the removed submission-portal comment.
  2. Source: no "ijacsa" / "thesai" anywhere in main.tex.
  3. PDF (needs pdftotext): no IJACSA header, thesai footer, "Vol. XX" or
     "|Page" text, and the word multiset equals the non-blind build's once
     the branding text is removed.

The arXiv upload bundle is paper/build/arxiv-upload.zip: main.tex, main.bbl
(arXiv does not run BibTeX), IEEEtran.cls.
"""

import argparse
import collections
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
IJACSA = PAPER / "ijacsa"
BUILD = PAPER / "build"
NONBLIND_TEX = IJACSA / "IJACSA_Final_NonBlind_Reference_Copy.tex"
ANON_TEX = IJACSA / "IJACSA_Submission_Anonymized_Babak_Cart_Abandonment.tex"
ARXIV_TEX = PAPER / "arxiv" / "main.tex"
SUPPORT = [IJACSA / "IEEEtran.cls", IJACSA / "references_revised.bib"]

TARGETS = {"ijacsa-nonblind": NONBLIND_TEX, "ijacsa-anon": ANON_TEX, "arxiv": ARXIV_TEX}

# Text the IJACSA template puts on every page.
BRANDING_PDF = re.compile(
    r"\(IJACSA\)[^\n]*|International Journal of Advanced Computer Science and Applications,?"
    r"|Vol\. XX, No\. X, 2026|www\.ijacsa\.thesai\.org|\d+\s*\|\s*Page|P a g e",
)
FORBIDDEN_PDF = re.compile(r"ijacsa|thesai|Vol\. XX|\|\s*Page|P a g e", re.IGNORECASE)
# Noise an outdated local LaTeX kernel leaks into the PDF via fancyhdr hooks.
KERNEL_NOISE = re.compile(r"para ?before|para ?begin|para ?end|para after")

MIKTEX_BIN = Path("C:/Program Files/MiKTeX/miktex/bin/x64")


def tool(name):
    found = shutil.which(name)
    if found:
        return found
    candidate = MIKTEX_BIN / f"{name}.exe"
    return str(candidate) if candidate.exists() else None


def build(target, tex_path):
    """Build one target in its own directory, jobname 'paper' (short paths
    keep Windows under MAX_PATH). Returns (pdf_path, n_latex_errors)."""
    out = BUILD / target
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shutil.copy(tex_path, out / "paper.tex")
    for f in SUPPORT:
        shutil.copy(f, out / f.name)

    pdflatex, bibtex = tool("pdflatex"), tool("bibtex")
    if not pdflatex or not bibtex:
        sys.exit("pdflatex/bibtex not found on PATH or in the default MiKTeX location")

    def latex():
        subprocess.run([pdflatex, "-interaction=nonstopmode", "paper.tex"],
                       cwd=out, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    latex()
    subprocess.run([bibtex, "paper"], cwd=out, stdout=subprocess.DEVNULL)
    latex()
    latex()

    log = (out / "paper.log").read_text(encoding="latin-1")
    n_errors = sum(1 for line in log.splitlines() if line.startswith("!"))
    pdf = out / "paper.pdf"
    status = "ok" if pdf.exists() else "NO PDF"
    if n_errors:
        status += f", {n_errors} LaTeX errors (see {out / 'paper.log'})"
    print(f"  {target:16s} {status}")
    return pdf, n_errors


def body(tex):
    return tex.split(r"\begin{document}", 1)[1]


def check_source():
    failures = []
    arxiv = ARXIV_TEX.read_text(encoding="utf-8")
    nonblind = NONBLIND_TEX.read_text(encoding="utf-8")

    # The only body change allowed is dropping the IJACSA portal comment.
    expected = re.sub(r"\n% NON-BLIND / camera-ready.*?camera-ready manuscript\.\n\n", "\n",
                      body(nonblind), flags=re.S)
    if body(arxiv) != expected:
        failures.append("arxiv/main.tex body differs from the non-blind IJACSA body")
    if re.search(r"ijacsa|thesai", arxiv, re.IGNORECASE):
        failures.append("arxiv/main.tex still mentions ijacsa/thesai")
    return failures


def pdf_words(pdf, pdftotext):
    text = subprocess.run([pdftotext, pdf.name, "-"], cwd=pdf.parent,
                          capture_output=True, check=True).stdout.decode("utf-8", "ignore")
    text = text.replace("-\n", "")
    return text, collections.Counter(
        re.findall(r"[A-Za-z0-9.%@]+", KERNEL_NOISE.sub("", BRANDING_PDF.sub("", text))))


def check_pdf(arxiv_pdf, nonblind_pdf):
    pdftotext = tool("pdftotext")
    if not pdftotext:
        print("  (pdftotext not found — PDF checks skipped)")
        return []
    failures = []
    arxiv_text, arxiv_words = pdf_words(arxiv_pdf, pdftotext)
    hits = sorted(set(FORBIDDEN_PDF.findall(arxiv_text)))
    if hits:
        failures.append(f"arXiv PDF contains branding text: {hits}")
    if nonblind_pdf.exists():
        _, nonblind_words = pdf_words(nonblind_pdf, pdftotext)
        extra, missing = arxiv_words - nonblind_words, nonblind_words - arxiv_words
        # Page numbers move from "N | Page" to a bare "N"; nothing else may change.
        extra = {w: n for w, n in extra.items() if not w.isdigit()}
        if extra or missing:
            failures.append(f"arXiv PDF words differ from the IJACSA build: "
                            f"extra={extra} missing={dict(missing)}")
    return failures


def bundle():
    out = BUILD / "arxiv"
    zip_path = BUILD / "arxiv-upload.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out / "paper.tex", "main.tex")
        z.write(out / "paper.bbl", "main.bbl")
        z.write(out / "IEEEtran.cls", "IEEEtran.cls")
    print(f"  arXiv bundle: {zip_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*", default=list(TARGETS),
                        help=f"any of: {', '.join(TARGETS)} (default: all)")
    parser.add_argument("--allow-latex-errors", action="store_true",
                        help="don't fail on '!' lines in the LaTeX log (outdated local TeX)")
    args = parser.parse_args()
    unknown = set(args.targets) - set(TARGETS)
    if unknown:
        parser.error(f"unknown target(s): {', '.join(sorted(unknown))}")

    print("Building:")
    results = {t: build(t, TARGETS[t]) for t in args.targets}

    failures = [f"{t}: {n} LaTeX errors" for t, (_, n) in results.items()
                if n and not args.allow_latex_errors]
    if "arxiv" in results:
        print("Checking arXiv target:")
        failures += check_source()
        failures += check_pdf(results["arxiv"][0], BUILD / "ijacsa-nonblind" / "paper.pdf")
        if (BUILD / "arxiv" / "paper.bbl").exists():
            bundle()

    for f in failures:
        print(f"  FAIL: {f}")
    if failures:
        sys.exit(1)
    print("  all checks passed")


if __name__ == "__main__":
    main()
