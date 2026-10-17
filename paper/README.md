# Manuscript source

"When Offline Performance Misleads: A Leakage Audit of Completed-Session
Features for Cart-Abandonment Prediction" — Babak Rad Jahanbani, Ahmet Sayar.
Corresponding author: babakrad.jahanbani@bahcesehir.edu.tr (ORCID [0009-0001-6910-9747](https://orcid.org/0009-0001-6910-9747)).

## Status (as of 2026-10-17)

| Venue | Status |
|---|---|
| IJACSA (thesai.org) | Manuscript **MS-17-10-0048**, **accepted for publication** (acceptance email October 2026). Registration fee paid (split equally between both authors; the university is expected to reimburse in January 2027). **Camera-ready submitted** (deadline 16 Oct 2026, 15:40) and confirmed **in the production queue** on 2026-10-17. Source: `camera-ready/`; reviewer responses: `camera-ready/RESPONSE_TO_REVIEWERS.md`. Still to come: publication notice → DOI. |
| arXiv | **Submitted 2026-09-21** by Prof. Ahmet Sayar, category cs.AI, submission ID `submit/8109238`. The public `arXiv:YYMM.NNNNN` identifier is not recorded here yet — add it once announced. |

## Contents

| Path | What it is | Edit? |
|---|---|---|
| `ijacsa/` | IJACSA-template source (non-blind + anonymized), `IEEEtran.cls`, `references_revised.bib`, as provided by the author 2026-09-29. | **No** — record of the IJACSA submission |
| `ijacsa/submitted-pdf/` | The two PDFs sent to IJACSA. Gitignored (journal-branded review copies); kept locally so builds can be checked against them. | No |
| `arxiv/` | **The arXiv submission of 2026-09-21**, as provided by the author: source `IJACSA_Final_NonBlind.tex` + `.bbl` + `IEEEtran.cls` + `references_revised.bib`; `arXiv.pdf` (arXiv's compiled PDF, with the `arXiv:submit/8109238 [cs.AI] 21 Sep 2026` stamp); `arXiv_Preprint_Cart_Abandonment_Leakage_Audit.pdf` (the same source compiled with pdfTeX 1.40.25, no stamp). | **No** — record of the arXiv submission |
| `camera-ready/` | Working copy of the camera-ready revision (2026-10-14): `IJACSA_CameraReady_MS-17-10-0048.tex`, `references_revised.bib`, `IEEEtran.cls`, and `RESPONSE_TO_REVIEWERS.md` (every reviewer comment → change → location). Baseline commit `fc209dc` = unchanged copy of `ijacsa/` non-blind source, so `git diff fc209dc -- paper/camera-ready` shows every revision. | Superseded by `Submitted Files/` |
| `camera-ready/Submitted Files/` | **What was uploaded to IJACSA** (deadline 16 Oct 2026, 15:40; in production queue 2026-10-17), as provided by the author: `.tex`, `.bib`, `.bbl`, `IEEEtran.cls`, and the Overleaf-built PDF. Verified 2026-10-17: `.tex` and `.bib` byte-identical to the working copy; `IEEEtran.cls` identical apart from line endings; PDF built with pdfTeX 1.40.25, 10 pages, no unresolved references, author names exact, contains the corrected 1,354 (3.1%) and no 1,745. The PDF is gitignored (journal-branded), like `ijacsa/submitted-pdf/`. | **No** — record of the camera-ready submission |
| `arxiv-v2/main.tex` | A cleaner unbranded variant prepared 2026-09-29, after the arXiv submission, from the *pre-review* source. Superseded for content: an arXiv v2 should be built from `camera-ready/Submitted Files/` with the IJACSA header/footer removed, so it carries the reviewer revisions and the corrected counts. | Yes |
| `build.py` | Builds `ijacsa-nonblind`, `ijacsa-anon` and `arxiv` (= `arxiv-v2/main.tex`) into `build/` (gitignored), checks the arXiv-v2 target, writes `build/arxiv-upload.zip`. | Yes |

## Provenance

### `ijacsa/` vs the PDFs sent to IJACSA

Verified 2026-09-29 by building both sources and comparing extracted PDF text
(word multiset, header/footer ignored):

- `IJACSA_Submission_Anonymized_Babak_Cart_Abandonment.tex` — identical text
  to the submitted anonymized PDF.
- `IJACSA_Final_NonBlind_Reference_Copy.tex` — identical except the
  corresponding-author email: `@bahcesehir.edu.tr` in the source,
  `@bau.edu.tr` in the non-blind reference PDF. **`@bahcesehir.edu.tr` is
  correct** (confirmed by the author 2026-09-30; it is the address on the
  IJACSA author account and in the arXiv version). The non-blind PDF was a
  reference copy; IJACSA received the anonymized one.

### `arxiv/` vs `ijacsa/`

`diff` of `arxiv/IJACSA_Final_NonBlind.tex` against
`ijacsa/IJACSA_Final_NonBlind_Reference_Copy.tex` (2026-09-30) — three
changes, all in the preamble or comments; the body is identical:

1. IJACSA running header ("(IJACSA) International Journal of Advanced
   Computer Science and Applications, Vol. XX, No. X, 2026") removed.
2. `www.ijacsa.thesai.org` footer removed.
3. The "do not upload to the double-blind portal" comment replaced by
   "Full-attribution version. Used for arXiv preprint posting…".

`IEEEtran.cls` and `references_revised.bib` are byte-identical to `ijacsa/`.
`arXiv.pdf` has the same body text as a build of `arxiv-v2/main.tex` (only
hyphenation and the arXiv stamp differ) and contains no IJACSA or thesai text.

**One template remnant on arXiv:** the source keeps the IJACSA footer rule and
`\fancyfoot[R]{\thepage \ $|$ P a g e }`, so every page of `arXiv.pdf` shows a
horizontal rule and "N | Page" at the bottom right next to a centred page
number. It is not branding. `arxiv-v2/main.tex` removes it, together with the
unused template placeholders ("first page center header/footer").

### `arxiv-v2/main.tex` vs `ijacsa/`

Preamble only: IJACSA header removed; thesai footer, footer rule and
"N | P a g e" replaced by a centred page number (also in the `plain` page
style); the double-blind-portal comment removed. Page geometry
(`\headheight`, `\footskip`, the template's `\thispagestyle` override) is
kept, so pagination matches the submitted PDF. `build.py` enforces that the
document body is otherwise byte-identical.

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
toolchain problem, not a manuscript problem (the PDFs submitted to IJACSA and
compiled by arXiv are clean). Updating MiKTeX fixes it; until then use
`--allow-latex-errors`, and judge the final appearance from arXiv's own
compiled preview.

## Replacing the arXiv version (v2), if ever needed

Upload `build/arxiv-upload.zip` (`main.tex`, `main.bbl`, `IEEEtran.cls`) as a
**replacement** of the existing submission — never as a new submission — from
the submitting account (Prof. Sayar). arXiv does not run BibTeX, so `main.bbl`
must be included; rebuild it after any change to the references.
