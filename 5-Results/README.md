Metric outputs land here, one JSON file per run.

| Written by | Files |
|---|---|
| `2-Completed-Session/reanalysis_controlled_audit.py` | 42 `completed-session__*.json` — paper Tables IV–V (4 feature configs × {LR, XGBoost} × 5 seeds, plus DT/RF at seed 42) |
| `2-Completed-Session/build_cart_session_table.py` | `completed-session__descriptive-stats.json` — every descriptive number in the paper, paper vs computed, with git commit, timestamp, library versions and input hashes |
| `3-Decision-Time/train_prefix_models.py` | 288 `decision-time__*.json`, plus `curve_all_eligible.csv` and `curve_fixed_cohort.csv` (not in the submitted paper) |

Filename protocol (required, not optional — files leave the repo, folders do not):

    <lineage>__<model>__<split>__<k>__<seed>.json

    lineage : completed-session | decision-time
    model   : e.g. logistic-regression, xgboost, decision-tree, random-forest
    split   : e.g. visitor-grouped, temporal, stratified-80-20
    k       : cutoff_k for decision-time runs (e.g. k0, k3, k10)
    seed    : e.g. s42

As actually written (2026-07-29), two slots carry an extra qualifier:

- completed-session runs have no cutoff, so the `<k>` slot holds the feature
  configuration instead: `completed-session__xgboost__stratified-80-20__timing-only__s42.json`
- decision-time runs append the population to the split:
  `decision-time__xgboost__temporal-all-eligible__k3__s42.json`
  (`all-eligible` or `fixed-cohort`)

The JSONs record metrics and run parameters only — no library versions, git
commit or timestamp; see `docs/results_provenance.md` for the environment the
runs used.
