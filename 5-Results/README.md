Metric outputs land here, one JSON file per run. No training script currently
writes JSON (they print metrics to stdout) — this folder is empty until that
changes.

Filename protocol (required, not optional — files leave the repo, folders do not):

    <lineage>__<model>__<split>__<k>__<seed>.json

    lineage : completed-session | decision-time
    model   : e.g. logistic-regression, xgboost, lstm
    split   : e.g. visitor-grouped, random
    k       : cutoff_k for decision-time runs (e.g. k0, k3, k10); use k-na
              for completed-session runs, which have no cutoff concept
    seed    : e.g. s42

Example: decision-time__xgboost__visitor-grouped__k3__s42.json
