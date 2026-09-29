# Manuscript source

"When Offline Performance Misleads: A Leakage Audit of Completed-Session
Features for Cart-Abandonment Prediction" — Babak Rad Jahanbani, Ahmet Sayar.
Submitted to IJACSA (not accepted as of 2026-09-29). Preprint target: arXiv.

| Folder | Contents |
|---|---|
| `ijacsa/` | The IJACSA-template source exactly as provided by the author on 2026-09-29. **Do not edit** — it is the record of what was submitted. |
| `ijacsa/submitted-pdf/` | The two PDFs sent to IJACSA. Gitignored (journal-branded review copies); kept locally so the build can be checked against them. |
| `arxiv/main.tex` | The only derived file: the non-blind source with the IJACSA branding removed. |
| `build.py` | Builds all targets into `build/` (gitignored), checks the arXiv target, and writes `build/arxiv-upload.zip`. |

## Provenance of `ijacsa/`

Verified 2026-09-29 by building both sources and comparing the extracted PDF
text (as a word multiset, ignoring header/footer) against the submitted PDFs:

- `IJACSA_Submission_Anonymized_Babak_Cart_Abandonment.tex` — identical text to
  the submitted anonymized PDF.
- `IJACSA_Final_NonBlind_Reference_Copy.tex` — identical text to the non-blind
  reference PDF except one word: the corresponding-author email is
  `babakrad.jahanbani@bahcesehir.edu.tr` in the source and
  `babakrad.jahanbani@bau.edu.tr` in the PDF.

## What `arxiv/main.tex` changes

Relative to `ijacsa/IJACSA_Final_NonBlind_Reference_Copy.tex`, preamble only:

- IJACSA running header ("(IJACSA) International Journal of Advanced Computer
  Science and Applications, Vol. XX, No. X, 2026") → removed.
- `www.ijacsa.thesai.org` footer, footer rule and "N | P a g e" → a plain
  centred page number.
- Template placeholders in the `plain` page style ("first page center
  header/footer") → the same centred page number.
- The source comment about the double-blind submission portal → removed
  (arXiv publishes the `.tex`, comments included).

Page geometry (`\headheight`, `\footskip`, the template's `\thispagestyle`
override) is kept, so pagination matches the submitted PDF. `build.py`
enforces that the document body is otherwise byte-identical.

## Building

    python paper/build.py                       # all targets
    python paper/build.py arxiv                 # one target
    python paper/build.py --allow-latex-errors  # see note below

Needs `pdflatex` and `bibtex` (PATH, or MiKTeX's default install location);
`pdftotext` enables the PDF-level checks.

**Local TeX note (2026-09-29):** the MiKTeX install on the author's machine
has a 2021 LaTeX kernel but 2025–2026 `microtype`/`fancyhdr`, which call
kernel hooks the old kernel lacks. Every target — including the untouched
IJACSA source — then logs ~83 "Undefined control sequence" errors and prints
stray "para/beforepara/…" text in the header/footer. This is a local
toolchain problem, not a manuscript problem (the Overleaf-built PDFs are
clean). Updating MiKTeX fixes it; until then use `--allow-latex-errors`, and
judge the final appearance from arXiv's own compiled preview.

## Uploading to arXiv

Upload `build/arxiv-upload.zip` (`main.tex`, `main.bbl`, `IEEEtran.cls`).
arXiv does not run BibTeX, so `main.bbl` must be included; rebuild it after
any change to the references.
