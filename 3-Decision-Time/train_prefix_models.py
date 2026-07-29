"""
train_prefix_models.py
========================

Modeling harness for the point-in-time prefix dataset
(1-Data/decision-time/prefix_dataset.parquet). This is deliberately separate
from anything under 2-Completed-Session/ — it evaluates a different lineage
(point-in-time cutoffs, not whole-session features) and must not be merged
with that frozen evidence.

WHY NOT A PLAIN STRATIFIED SPLIT
----------------------------------
`prefix_dataset.parquet` has multiple rows per session (one per cutoff k),
and the SAME visitor can appear in multiple sessions. A plain stratified
random split would let one visitor's session end up in both train and test,
letting the model implicitly memorize a visitor's identity/behaviour rather
than learning from features alone. We therefore never stratify — we always
either group-split by visitorid (no visitor crosses the train/test boundary)
or split by absolute time (train on earlier sessions, test on later ones,
which is what a real deployment would face).

TWO SPLIT PROTOCOLS, BOTH RUN
------------------------------
  A) visitor-grouped : GroupShuffleSplit on visitorid, 80/20.
  B) temporal        : sort by each session's own start time, earliest 80%
                        train / latest 20% test. The split boundary is fixed
                        by time (not reseeded); only the models' own
                        internal randomness varies across seeds.

`prefix_dataset.parquet` does not store absolute timestamps (only relative
seconds-since-* features, plus hour-of-day/day-of-week from the cutoff). To
get each session's start time for the temporal split, we re-run the SAME
shared cleaning + sessionization logic used to build the prefix dataset
(0-Shared/clean_events.py) and take each session's first event timestamp.
This does not change or re-derive any feature — it only supplies a sort key
for the temporal split.

TWO CURVES
-----------
  1) all_eligible : every session that is still in the risk set at cutoff k
     (this is the natural population from prefix_dataset.parquet — it
     shrinks as k grows, both because sessions convert early and because
     short sessions run out of events).
  2) fixed_cohort : ONLY the sessions that survive all the way to k=10,
     scored at every k. Same session_keys at every k, so any change in
     performance across k in this curve is purely information gain from
     watching the same sessions longer — it cannot be explained by the
     population itself changing.

Curve 1 alone is not interpretable on its own, because k=0 and k=10 are
scoring different, self-selected populations (see docs/prefix_rebuild_log.md
for why the positive rate is non-monotonic in k). Curve 2 isolates the
"more information over time" effect.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "0-Shared"))
from clean_events import clean_and_sessionize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PREFIX_PATH = REPO_ROOT / "1-Data" / "decision-time" / "prefix_dataset.parquet"
RAW_PATH = REPO_ROOT / "1-Data" / "raw" / "1-events.csv"
RESULTS_DIR = REPO_ROOT / "5-Results"

SEEDS = [11, 22, 33, 42, 55]
TABLE_SEED = 42  # DT/RF, "once ... for continuity with the report"
CUTOFFS = [0, 1, 2, 3, 5, 10]
FIXED_COHORT_K = 10
N_BOOTSTRAP = 1000
CI_ALPHA = 0.05  # 95% CI

ID_COLS = ["session_key", "visitorid", "cutoff_k", "y"]
FEATURE_COLS = [
    "view_count", "addtocart_count", "unique_items_count", "view_to_cart_ratio",
    "seconds_since_first_event", "seconds_since_first_cart",
    "seconds_since_last_cart", "seconds_since_previous_event",
    "inter_event_interval_mean", "events_after_cart",
    "repeated_views_of_cart_items", "hour_of_day", "day_of_week",
]


# ============================================================================
# Session metadata: start time (for the temporal split) + the "transaction
# strictly between first and last addtocart" count the user asked for.
# ============================================================================

def compute_session_metadata():
    """Re-run the shared cleaning + sessionization pipeline to get, per cart
    session: its start timestamp (for the temporal split) and whether a
    transaction occurred strictly between its first and last addtocart event.

    Returns (session_start_ts: dict[session_key -> int ms],
             n_txn_between_first_last_cart: int)
    """
    print("Deriving session start times (re-running shared clean+sessionize; "
          "this does not change any feature, only supplies a sort key)...")
    _, df_clean, _ = clean_and_sessionize(str(RAW_PATH))

    has_cart = (
        df_clean.groupby("session_key")["event"]
        .apply(lambda s: (s == "addtocart").any())
    )
    cart_session_keys = set(has_cart[has_cart].index)
    df_cart = df_clean[df_clean["session_key"].isin(cart_session_keys)]

    session_start_ts = {}
    n_txn_between = 0

    for session_key, g in df_cart.groupby("session_key", sort=False):
        ts = g["timestamp"].values
        events = g["event"].values

        session_start_ts[session_key] = int(ts.min())

        cart_ts = ts[events == "addtocart"]
        txn_ts = ts[events == "transaction"]
        if len(cart_ts) >= 2 and len(txn_ts) > 0:
            first_cart, last_cart = cart_ts.min(), cart_ts.max()
            if np.any((txn_ts > first_cart) & (txn_ts < last_cart)):
                n_txn_between += 1

    print(f"Cart sessions considered: {len(cart_session_keys):,}")
    print(f"Cart sessions with a transaction STRICTLY BETWEEN their first and "
          f"last addtocart event: {n_txn_between:,}")
    print("(Separate from the 172 sessions dropped for a transaction occurring "
          "before the first addtocart — see build_prefix_dataset.py.)")
    print()

    return session_start_ts, n_txn_between


# ============================================================================
# Splitting
# ============================================================================

def visitor_grouped_split(df_k, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(df_k, groups=df_k["visitorid"]))
    return df_k.iloc[train_idx], df_k.iloc[test_idx]


def temporal_split(df_k):
    """Deterministic: earliest 80% of sessions (by session start time) train,
    latest 20% test. Not reseeded — the split boundary doesn't depend on the
    model seed, only the models' own internal randomness does."""
    df_sorted = df_k.sort_values("session_start_ts", kind="mergesort")
    n_train = int(len(df_sorted) * 0.8)
    return df_sorted.iloc[:n_train], df_sorted.iloc[n_train:]


# ============================================================================
# Fitting + metrics
# ============================================================================

def cap_at_99th(train_df, test_df, cols):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in cols:
        cap = train_df[col].quantile(0.99)
        train_df[col] = train_df[col].clip(upper=cap)
        test_df[col] = test_df[col].clip(upper=cap)
    return train_df, test_df


def prepare_xy(train_df, test_df):
    assert "visitorid" not in FEATURE_COLS and "session_key" not in FEATURE_COLS
    X_train = train_df[FEATURE_COLS]
    X_test = test_df[FEATURE_COLS]
    assert "visitorid" not in X_train.columns and "session_key" not in X_train.columns, (
        "visitorid / session_key leaked into the feature matrix"
    )
    X_train, X_test = cap_at_99th(X_train, X_test, FEATURE_COLS)
    y_train = train_df["y"].values
    y_test = test_df["y"].values
    return X_train.fillna(0), X_test.fillna(0), y_train, y_test


def fit_and_score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "test_positive_rate": float(np.mean(y_test)),
    }, probs


def build_lr(seed):
    return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)


def build_xgb(seed, y_train):
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    return XGBClassifier(
        n_estimators=100,
        scale_pos_weight=n_neg / max(n_pos, 1),
        tree_method="hist",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )


def bootstrap_auc_ci(y_test, probs, n_boot=N_BOOTSTRAP, alpha=CI_ALPHA, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_test)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = y_test[idx], probs[idx]
        if len(np.unique(y_b)) < 2:
            continue  # skip degenerate resamples (all one class)
        boot_aucs.append(roc_auc_score(y_b, p_b))
    boot_aucs = np.array(boot_aucs)
    lo = np.percentile(boot_aucs, 100 * (alpha / 2))
    hi = np.percentile(boot_aucs, 100 * (1 - alpha / 2))
    return float(lo), float(hi), len(boot_aucs)


def slugify(name):
    return name.replace("_", "-")


def save_result_json(model, split_desc, k, seed, payload):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"decision-time__{model}__{split_desc}__k{k}__s{seed}.json"
    with open(RESULTS_DIR / fname, "w") as f:
        json.dump(payload, f, indent=2)
    return fname


# ============================================================================
# Main sweep: one population (all_eligible or fixed_cohort) x one split type
# x one cutoff k x one seed x one model
# ============================================================================

def run_sweep(df, population_name, split_type):
    """Returns a list of per-(k, seed, model) result rows for this
    (population, split_type) combination, and writes one JSON per run."""
    rows = []
    split_desc = f"{split_type}-{population_name.replace('_', '-')}"

    for k in CUTOFFS:
        df_k = df[df["cutoff_k"] == k]
        n_total = len(df_k)
        pos_rate = df_k["y"].mean() if n_total > 0 else float("nan")

        for seed in SEEDS:
            if split_type == "visitor-grouped":
                train_df, test_df = visitor_grouped_split(df_k, seed)
            else:
                train_df, test_df = temporal_split(df_k)

            X_train, X_test, y_train, y_test = prepare_xy(train_df, test_df)

            # --- Logistic Regression (training-only standardization) -----
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)
            lr = build_lr(seed)
            lr_metrics, lr_probs = fit_and_score(lr, X_train_sc, y_train, X_test_sc, y_test)
            rows.append({"population": population_name, "split": split_type, "k": k,
                         "seed": seed, "model": "LR", "n_total": n_total,
                         "pos_rate": pos_rate, **lr_metrics})
            save_result_json("logistic-regression", split_desc, k, seed,
                              {"population": population_name, "split": split_type,
                               "k": k, "seed": seed, "model": "LogisticRegression",
                               "n_total": n_total, "positive_rate": float(pos_rate),
                               **lr_metrics})

            # --- XGBoost (raw capped features) ----------------------------
            xgb = build_xgb(seed, y_train)
            xgb_metrics, xgb_probs = fit_and_score(xgb, X_train, y_train, X_test, y_test)
            rows.append({"population": population_name, "split": split_type, "k": k,
                         "seed": seed, "model": "XGB", "n_total": n_total,
                         "pos_rate": pos_rate, **xgb_metrics})
            save_result_json("xgboost", split_desc, k, seed,
                              {"population": population_name, "split": split_type,
                               "k": k, "seed": seed, "model": "XGBoost",
                               "n_total": n_total, "positive_rate": float(pos_rate),
                               **xgb_metrics})

            # --- Bootstrap CI at the reference seed only -------------------
            if seed == TABLE_SEED:
                lr_lo, lr_hi, lr_nboot = bootstrap_auc_ci(y_test, lr_probs, seed=seed)
                xgb_lo, xgb_hi, xgb_nboot = bootstrap_auc_ci(y_test, xgb_probs, seed=seed)
                for m, lo, hi in [("LR", lr_lo, lr_hi), ("XGB", xgb_lo, xgb_hi)]:
                    rows.append({"population": population_name, "split": split_type,
                                 "k": k, "seed": "ci", "model": m,
                                 "roc_auc_ci_lo": lo, "roc_auc_ci_hi": hi})

            # --- DT/RF once at TABLE_SEED, per (population, split, k) ------
            if seed == TABLE_SEED:
                dt = DecisionTreeClassifier(class_weight="balanced", max_depth=8,
                                             random_state=seed)
                dt_metrics, _ = fit_and_score(dt, X_train, y_train, X_test, y_test)
                rows.append({"population": population_name, "split": split_type, "k": k,
                             "seed": seed, "model": "DT", "n_total": n_total,
                             "pos_rate": pos_rate, **dt_metrics})
                save_result_json("decision-tree", split_desc, k, seed,
                                  {"population": population_name, "split": split_type,
                                   "k": k, "seed": seed, "model": "DecisionTree",
                                   "n_total": n_total, "positive_rate": float(pos_rate),
                                   **dt_metrics})

                rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                             random_state=seed, n_jobs=-1)
                rf_metrics, _ = fit_and_score(rf, X_train, y_train, X_test, y_test)
                rows.append({"population": population_name, "split": split_type, "k": k,
                             "seed": seed, "model": "RF", "n_total": n_total,
                             "pos_rate": pos_rate, **rf_metrics})
                save_result_json("random-forest", split_desc, k, seed,
                                  {"population": population_name, "split": split_type,
                                   "k": k, "seed": seed, "model": "RandomForest",
                                   "n_total": n_total, "positive_rate": float(pos_rate),
                                   **rf_metrics})

        print(f"  [{population_name} / {split_type}] k={k}: N={n_total:,}, "
              f"positive_rate={pos_rate:.4f} — done")

    return rows


def summarize_curve(rows, population_name):
    """Aggregate per-(k, model) mean/sd AUC across seeds 11-55, attach the
    seed=42 bootstrap CI, and return a tidy DataFrame for CSV export."""
    df_rows = pd.DataFrame(rows)
    df_rows = df_rows[df_rows["population"] == population_name]

    summary_records = []
    for split_type in df_rows["split"].unique():
        for k in CUTOFFS:
            sub = df_rows[(df_rows["split"] == split_type) & (df_rows["k"] == k)]
            for model in ["LR", "XGB", "DT", "RF"]:
                model_runs = sub[(sub["model"] == model) & (sub["seed"].isin(SEEDS))]
                if len(model_runs) == 0:
                    continue
                n = model_runs["n_total"].iloc[0]
                pos_rate = model_runs["pos_rate"].iloc[0]
                mean_auc = model_runs["roc_auc"].mean()
                sd_auc = model_runs["roc_auc"].std(ddof=1) if len(model_runs) > 1 else 0.0
                mean_pr_auc = model_runs["pr_auc"].mean()

                ci_row = sub[(sub["model"] == model) & (sub["seed"] == "ci")]
                ci_lo = ci_row["roc_auc_ci_lo"].iloc[0] if len(ci_row) else np.nan
                ci_hi = ci_row["roc_auc_ci_hi"].iloc[0] if len(ci_row) else np.nan

                summary_records.append({
                    "population": population_name, "split": split_type, "k": k,
                    "model": model, "n": n, "positive_rate": pos_rate,
                    "mean_roc_auc": mean_auc, "sd_roc_auc": sd_auc,
                    "ci_lo_seed42": ci_lo, "ci_hi_seed42": ci_hi,
                    "mean_pr_auc": mean_pr_auc,
                })
    return pd.DataFrame(summary_records)


def print_table(summary_df, population_name):
    print()
    print("=" * 100)
    print(f"SUMMARY — population={population_name}")
    print("=" * 100)
    for split_type in summary_df["split"].unique():
        print(f"\n--- split={split_type} ---")
        header = f"{'k':>3} {'model':<5} {'N':>7} {'pos_rate':>9} {'mean_AUC':>9} {'sd_AUC':>7} {'95% CI (seed42)':>18} {'mean_PR_AUC':>12}"
        print(header)
        print("-" * len(header))
        sub = summary_df[summary_df["split"] == split_type].sort_values(["k", "model"])
        for _, r in sub.iterrows():
            ci_str = (f"[{r['ci_lo_seed42']:.4f}, {r['ci_hi_seed42']:.4f}]"
                      if pd.notna(r["ci_lo_seed42"]) else "n/a")
            print(f"{int(r['k']):>3} {r['model']:<5} {int(r['n']):>7} {r['positive_rate']:>9.4f} "
                  f"{r['mean_roc_auc']:>9.4f} {r['sd_roc_auc']:>7.4f} {ci_str:>18} {r['mean_pr_auc']:>12.4f}")


def main():
    print("=" * 78)
    print("train_prefix_models.py — decision-time lineage modeling")
    print("=" * 78)

    df = pd.read_parquet(PREFIX_PATH)
    print(f"Loaded {len(df):,} rows from {PREFIX_PATH.name}")
    assert set(df["cutoff_k"].unique()) == set(CUTOFFS), (
        f"Unexpected cutoff_k values: {sorted(df['cutoff_k'].unique())}"
    )
    print()

    session_start_ts, n_txn_between = compute_session_metadata()
    df["session_start_ts"] = df["session_key"].map(session_start_ts)
    missing_ts = df["session_start_ts"].isna().sum()
    assert missing_ts == 0, (
        f"{missing_ts} rows have a session_key with no recovered start time — "
        "the raw/clean population no longer matches prefix_dataset.parquet."
    )

    # Fixed cohort: session_keys with a row at cutoff_k == FIXED_COHORT_K.
    fixed_cohort_keys = set(df.loc[df["cutoff_k"] == FIXED_COHORT_K, "session_key"])
    df_fixed = df[df["session_key"].isin(fixed_cohort_keys)].copy()
    print(f"Fixed cohort size (sessions surviving to k={FIXED_COHORT_K}): "
          f"{len(fixed_cohort_keys):,} session_keys, "
          f"{len(df_fixed):,} rows across all k (should be {len(fixed_cohort_keys) * len(CUTOFFS):,})")
    print()

    all_rows = []
    for split_type in ["visitor-grouped", "temporal"]:
        print(f"### all_eligible / {split_type} ###")
        all_rows += run_sweep(df, "all_eligible", split_type)
        print(f"### fixed_cohort / {split_type} ###")
        all_rows += run_sweep(df_fixed, "fixed_cohort", split_type)

    curve_all_eligible = summarize_curve(all_rows, "all_eligible")
    curve_fixed_cohort = summarize_curve(all_rows, "fixed_cohort")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    curve_all_eligible.to_csv(RESULTS_DIR / "curve_all_eligible.csv", index=False)
    curve_fixed_cohort.to_csv(RESULTS_DIR / "curve_fixed_cohort.csv", index=False)

    print_table(curve_all_eligible, "all_eligible")
    print_table(curve_fixed_cohort, "fixed_cohort")

    print()
    print("=" * 78)
    print("PAPER PLACEHOLDER NUMBER")
    print("=" * 78)
    print(f"Cart sessions with a transaction strictly BETWEEN their first and "
          f"last addtocart event: {n_txn_between:,}")
    print("(Separate from the 172 sessions dropped for a transaction occurring "
          "before the first addtocart in build_prefix_dataset.py.)")

    print()
    print(f"Saved: {RESULTS_DIR / 'curve_all_eligible.csv'}")
    print(f"Saved: {RESULTS_DIR / 'curve_fixed_cohort.csv'}")
    print(f"Per-run metric JSON written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
