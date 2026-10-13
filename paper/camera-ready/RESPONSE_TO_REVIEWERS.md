# Response to reviewers — IJACSA MS-17-10-0048 (camera-ready)

"When Offline Performance Misleads: A Leakage Audit of Completed-Session
Features for Cart-Abandonment Prediction" — Babak Rad Jahanbani, Ahmet Sayar.

We thank both reviewers. All published results are unchanged (renumbered Tables V and VI in this
version, Figures 3–4); new evidence is added and labelled by purpose. Section
and table numbers below refer to the camera-ready version.

## Reviewer 1

| # | Comment | Response |
|---|---|---|
| 1 | Controlled ablation is credible. | Thank you. Unchanged. |
| 2 | Eq. 2 holds by construction; contribution beyond a post-mortem is thin. | We agree the identity is trivial once written. §VI-A now says so explicitly and argues its significance: it passed through an ordinary pipeline unnoticed and produced a publishable-looking score. We cite concurrent work that finds comparable post-session descriptors on other clickstream benchmarks (Alzghoul et al., 2026). The new point-in-time evaluation (§V-D) also adds substance beyond the post-mortem. |
| 3 | Counts-only is still completed-session; Eq. 4 is never evaluated; a prefix experiment at the first add-to-cart is needed. | **Added §V-D and Table VIII.** Eq. 4 is evaluated at the first add-to-cart (k = 0; 43,745 sessions; 26.88% positive) with 13 prefix-only features, four models, visitor-grouped and chronological splits. ROC-AUC 0.57–0.65 (LR 0.630 / 0.642; XGBoost 0.622 / 0.633). Example construction is formalized in §IV-C. |
| 4 | The 1,745 ambiguous sessions are not checked against item IDs; no measured effect. | **Corrected and verified (§IV-C).** The timing groups overlap in 23 sessions, so the distinct count is 1,722 (3.9%), not 1,745 (4.0%). An item-level check finds 753 of them, and 1,354 sessions overall (3.1% of cart sessions), combining a purchase with a carted item never bought. Re-running the ablation (Table VI) without these sessions leaves every conclusion unchanged (counts-only falls from 0.606 to 0.578–0.595). The abstract now reports the item-verified 1,354 (3.1%). |
| 5 | LR/XGB convergence at 0.606 reflects a feature ceiling, not proof. | **Agreed; claim softened** (§V-E, abstract, conclusion). The convergence is now described as consistent with a feature ceiling; it shows only that the large advantage depended on the removed features. |
| 6 | Random splits ignore repeat visitors and drift; add visitor-grouped and chronological splits. | **Added §V-C and Table VII.** The full ablation is repeated under both splits; the pattern is unchanged, and visitor-grouped standard deviations are larger (up to ±0.017 vs ±0.0035), which we state. Threats to validity updated. |
| 7 | Add references to the Introduction. | **Added** (Esmeli 2021, Requena 2020, Sakar 2019, Rausch 2022, Waldmann 2025, Kaufman 2012, Kapoor 2023). |
| — | Proofreading. | Full language pass: long sentences split, tense made consistent, active voice where clearer. |

## Reviewer 2

| # | Comment | Response |
|---|---|---|
| 1 | Checklist validated only on the motivating dataset. | Validation on independent datasets was not possible within the camera-ready window. §VI-B now states that the checklist was derived from and applied to one artifact, that its completeness, reliability and false-positive behaviour are unevaluated, and that it is a structured starting point rather than a validated instrument. |
| 2 | Operational feature-lineage specification needed. | **Added Table II**: every variable (completed-session and point-in-time) with construction formula, source events, and earliest availability. |
| 3 | Eq. 4 lacks a horizon and rules for repeated carts, products, transactions. | **Formalized in §IV-C ("Example construction")**: anchor at the first add-to-cart, k-th-event cutoffs, observation window ≤ cutoff, outcome window = rest of the session (30-min inactivity, finite), risk-set entry and exit, later carts as features, session-level label after the cutoff regardless of item or number of transactions, the 172 pre-cart purchases excluded. |
| 4 | Inconsistent protocols across models. | **Clarified (§III-D, Table IX settings)**: LR and XGBoost carry the primary evidence under all splits; DT/RF are a seed-42 reproduction; all four models are evaluated under identical conditions in §V-D; the LSTM is development context only and supports no conclusion. |
| 5 | Quantify cost and explain integration into streaming/ML pipelines. | **Added §VI-D**: 43,917 sessions → 91,036 examples for six cutoffs (2.1×), about 13 min single-threaded vs about 10 min for the completed-session table; integration via event-time session windows, watermarks, cutoff-stamped feature vectors, late labels, and point-in-time joins. |
| 6 | Single dataset; narrow generalizations or validate elsewhere. | **Narrowed** (§VI-E, conclusion): claims restricted to the audited artifact; prevalence and magnitude elsewhere not measured. |
| 7 | Classify experiments by purpose. | **Added**: Table IV now labels every evaluation as diagnostic, diagnostic check, or decision-time estimate; §IV and §V state which conclusions each supports. |
| — | Shorter sentences, consistent tense, active voice. | Applied throughout. |

## Reproducibility of the additions

All new numbers are produced by committed scripts with saved outputs:

- `2-Completed-Session/reanalysis_split_robustness.py` → `5-Results/split_robustness_summary.csv`
- `2-Completed-Session/label_ambiguity_check.py` → `5-Results/label_ambiguity_check.json`
- `3-Decision-Time/build_prefix_dataset.py`, `train_prefix_models.py` → `5-Results/curve_all_eligible.csv` (k = 0 rows)
- `2-Completed-Session/build_cart_session_table.py` → `5-Results/completed-session__descriptive-stats.json`
