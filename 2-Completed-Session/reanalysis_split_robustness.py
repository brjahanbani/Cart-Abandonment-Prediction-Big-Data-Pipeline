"""
reanalysis_split_robustness.py
===============================

Camera-ready addition (reviewer R1, comment 6): the Table V ablation under
splits that respect repeat visitors and time.

Everything except the train/test split is imported unchanged from
reanalysis_controlled_audit.py (feature configurations, seeds, the
training-only 99th-percentile cap, model settings, metrics), so the only
difference from published Table V is how sessions are divided:

  stratified-80-20 : the published protocol (re-run here for side-by-side
                     comparison; matches docs/results_provenance.md)
  visitor-grouped  : GroupShuffleSplit on visitorid, 80/20, one split per seed
                     — no visitor appears in both train and test
  temporal         : sessions ordered by start time; earliest 80% train,
                     latest 20% test. The split is fixed; seeds only change
                     model randomness (LR is deterministic, so its sd is 0)

Published Table V is not modified. Outputs:
  5-Results/completed-session__<model>__<split>__<config>__s<seed>.json
  5-Results/split_robustness_summary.csv

Run from this folder:  python reanalysis_split_robustness.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from reanalysis_controlled_audit import (
    DATA_PATH,
    FEATURE_CONFIGS,
    RESULTS_DIR,
    SEEDS,
    build_lr,
    build_xgb,
    cap_at_99th,
    fit_and_score,
    save_result_json,
    slugify,
)

EVENTS_PATH = "../1-Data/events_clean.csv"
SPLITS = ["stratified-80-20", "visitor-grouped", "temporal"]


def split_indices(df, split, seed):
    """Return boolean train/test masks over df's rows."""
    n = len(df)
    if split == "stratified-80-20":
        idx_tr, idx_te = train_test_split(np.arange(n), test_size=0.2,
                                          stratify=df["has_transaction"], random_state=seed)
    elif split == "visitor-grouped":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        idx_tr, idx_te = next(gss.split(df, groups=df["visitorid"]))
    elif split == "temporal":
        order = np.argsort(df["start_ts"].values, kind="stable")
        cut = int(round(0.8 * n))
        idx_tr, idx_te = order[:cut], order[cut:]
    else:
        raise ValueError(split)
    tr = np.zeros(n, dtype=bool)
    tr[idx_tr] = True
    te = np.zeros(n, dtype=bool)
    te[idx_te] = True
    return tr, te


def make_xy(df, feature_cols, tr, te):
    X_train, X_test = df.loc[tr, feature_cols], df.loc[te, feature_cols]
    X_train, X_test = cap_at_99th(X_train, X_test, feature_cols)
    return (X_train.fillna(0), X_test.fillna(0),
            df.loc[tr, "has_transaction"].astype(int), df.loc[te, "has_transaction"].astype(int))


def main():
    df = pd.read_csv(DATA_PATH)
    starts = pd.read_csv(EVENTS_PATH, usecols=["timestamp", "session_key"]).groupby("session_key")["timestamp"].min()
    df["start_ts"] = df["session_key"].map(starts)
    assert df["start_ts"].notna().all(), "every cart session must have a start time"
    print(f"Loaded {len(df):,} cart sessions, {df['visitorid'].nunique():,} visitors")

    rows = []
    for split in SPLITS:
        for config, cols in FEATURE_CONFIGS.items():
            for seed in SEEDS:
                tr, te = split_indices(df, split, seed)
                X_train, X_test, y_train, y_test = make_xy(df, cols, tr, te)

                scaler = StandardScaler()
                lr = fit_and_score(build_lr(seed), scaler.fit_transform(X_train), y_train,
                                   scaler.transform(X_test), y_test)
                xgb = fit_and_score(build_xgb(seed, y_train), X_train, y_train, X_test, y_test)

                for model, name, m in (("LR", "logistic-regression", lr), ("XGB", "xgboost", xgb)):
                    payload = {"config": config, "features": cols, "seed": seed, "split": split,
                               "model": model, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                               "test_positive_rate": float(y_test.mean()), **m}
                    if split != "stratified-80-20":   # published-protocol JSONs already exist
                        save_result_json("completed-session", name, split, slugify(config), seed, payload)
                    rows.append(payload)
            print(f"  {split:<18} {config:<20} done")

    runs = pd.DataFrame(rows)
    summary = (runs.groupby(["split", "config", "model"])
               .agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", lambda s: s.std(ddof=1)),
                    pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", lambda s: s.std(ddof=1)),
                    n_test=("n_test", "first"), test_positive_rate=("test_positive_rate", "mean"))
               .reset_index())
    order = {c: i for i, c in enumerate(FEATURE_CONFIGS)}
    summary["o"] = summary["config"].map(order)
    summary = summary.sort_values(["o", "model", "split"]).drop(columns="o")
    out = RESULTS_DIR / "split_robustness_summary.csv"
    summary.to_csv(out, index=False)

    print()
    print(f"{'config':<20} {'model':<4} " + "  ".join(f"{s:>22}" for s in SPLITS))
    for (config, model), g in summary.groupby(["config", "model"], sort=False):
        cells = []
        for s in SPLITS:
            r = g[g["split"] == s].iloc[0]
            cells.append(f"{r.roc_auc_mean:.4f}±{r.roc_auc_sd:.4f} / {r.pr_auc_mean:.3f}")
        print(f"{config:<20} {model:<4} " + "  ".join(f"{c:>22}" for c in cells))
    print(f"\n(ROC-AUC mean±sd / PR-AUC mean, 5 seeds)  Saved: {out}")


if __name__ == "__main__":
    main()
