"""
reanalysis_controlled_audit.py
================================

Reconstruction of a script that was never committed to this repo, written
from the protocol described in the workshop paper (Tables 4 and 5). Those
tables are the paper's PRIMARY EVIDENCE and, before this file existed, traced
to no script anywhere in the repo — this is why the "seed-42" numbers looked
ambiguous in docs/results_provenance.md: the real source was simply missing,
not hidden among the other three lineages.

This script lives in 2-Completed-Session/ because it reads the same frozen
`cart_session_features.csv` as the rest of that lineage and is meant to
reproduce ALREADY-PUBLISHED numbers, not to explore new ones. Per the
2-Completed-Session/README.md warning: do not fix bugs here, do not adjust
anything to force a closer match to the paper. If the numbers disagree,
report the disagreement.

ASSUMPTION FLAGGED EXPLICITLY: the prompt that produced this script mentions
"Equation 3" without stating its formula. The only other row-level identity
available in cart_session_features.csv is the relationship between
has_transaction and transaction_count, so Equation 3 is assumed here to be:

    has_transaction == 1  if and only if  transaction_count > 0

If this is not the paper's actual Equation 3, that assertion will need to be
corrected — it is NOT invented to make anything else in this script pass.

FOUR FEATURE CONFIGURATIONS (exactly as specified):
    target_proxy       : event_count, view_count, addtocart_count
    original_completed : view_count, addtocart_count, view_to_cart_ratio,
                          unique_items_count, session_age_seconds,
                          inter_event_interval_mean, recency_last_cart_seconds
    timing_only        : session_age_seconds, inter_event_interval_mean,
                          recency_last_cart_seconds
    counts_only        : view_count, addtocart_count, view_to_cart_ratio,
                          unique_items_count

PROTOCOL:
    - Stratified 80/20 train/test split, seeds 11, 22, 33, 42, 55.
    - 99th-percentile cap fitted on the TRAINING partition only, then
      applied (clipped) to both train and test.
    - Logistic Regression: class_weight='balanced', max_iter=2000,
      StandardScaler fitted on the training partition only.
    - XGBoost: 100 trees, scale_pos_weight = n_neg/n_pos (from the training
      partition), tree_method='hist'.
    - Seed 42 only, original_completed only: also fit DecisionTree
      (class_weight='balanced', max_depth=8) and RandomForest (100 trees,
      class_weight='balanced') — this is Table 4.
    - Metrics: ROC-AUC and PR-AUC (primary), plus accuracy/precision/recall/F1
      at a 0.5 threshold (secondary).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = "../1-Data/completed-session/cart_session_features.csv"
RESULTS_DIR = Path("../5-Results")

SEEDS = [11, 22, 33, 42, 55]
TABLE4_SEED = 42
TABLE4_CONFIG = "original_completed"

FEATURE_CONFIGS = {
    "target_proxy": ["event_count", "view_count", "addtocart_count"],
    "original_completed": [
        "view_count", "addtocart_count", "view_to_cart_ratio",
        "unique_items_count", "session_age_seconds",
        "inter_event_interval_mean", "recency_last_cart_seconds",
    ],
    "timing_only": [
        "session_age_seconds", "inter_event_interval_mean",
        "recency_last_cart_seconds",
    ],
    "counts_only": [
        "view_count", "addtocart_count", "view_to_cart_ratio",
        "unique_items_count",
    ],
}

# Published values to diff against (workshop paper Tables 4 and 5).
PUBLISHED_TABLE5_ROC_AUC = {
    ("target_proxy", "LR"): (0.9944, 0.0004),
    ("target_proxy", "XGB"): (0.9997, 0.0001),
    ("original_completed", "LR"): (0.8306, 0.0036),
    ("original_completed", "XGB"): (0.9947, 0.0005),
    ("timing_only", "LR"): (0.8006, 0.0032),
    ("timing_only", "XGB"): (0.8688, 0.0035),
    ("counts_only", "LR"): (0.6061, 0.0035),
    ("counts_only", "XGB"): (0.6065, 0.0024),
}
PUBLISHED_TABLE5_PR_AUC = {
    ("target_proxy", "LR"): 0.9525,
    ("target_proxy", "XGB"): 0.9992,
    ("original_completed", "LR"): 0.6123,
    ("original_completed", "XGB"): 0.9872,
    ("timing_only", "LR"): 0.5476,
    ("timing_only", "XGB"): 0.6530,
    ("counts_only", "LR"): 0.3895,
    ("counts_only", "XGB"): 0.3943,
}
PUBLISHED_TABLE4_ROC_AUC = {
    "LR": 0.825,
    "DT": 0.934,
    "RF": 0.984,
    "XGB": 0.995,
}

DIFF_TOLERANCE = 0.002


def cap_at_99th(train_df, test_df, cols):
    """Fit the 99th-percentile cap on the TRAINING partition only, then
    clip both partitions to it. Returns new (train, test) copies."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in cols:
        cap = train_df[col].quantile(0.99)
        train_df[col] = train_df[col].clip(upper=cap)
        test_df[col] = test_df[col].clip(upper=cap)
    return train_df, test_df


def fit_and_score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs),
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }


def make_split(df, feature_cols, seed):
    """One stratified 80/20 split for a given seed, with the 99th-percentile
    cap fitted on the training partition only."""
    X = df[feature_cols]
    y = df["has_transaction"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_train, X_test = cap_at_99th(X_train, X_test, feature_cols)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    return X_train, X_test, y_train, y_test


def build_lr(seed):
    return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)


def build_xgb(seed, y_train):
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    return XGBClassifier(
        n_estimators=100,
        scale_pos_weight=n_neg / n_pos,
        tree_method="hist",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )


def slugify(name):
    return name.replace("_", "-")


def save_result_json(lineage, model, split, k, seed, payload):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{lineage}__{model}__{split}__{k}__s{seed}.json"
    with open(RESULTS_DIR / fname, "w") as f:
        json.dump(payload, f, indent=2)
    return fname


def main():
    print("=" * 78)
    print("reanalysis_controlled_audit.py — reconstructing workshop Tables 4 & 5")
    print("=" * 78)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} cart sessions from {DATA_PATH}")

    # --- Equation 2: transaction_count = event_count - view_count - addtocart_count
    eq2_lhs = df["transaction_count"]
    eq2_rhs = df["event_count"] - df["view_count"] - df["addtocart_count"]
    assert (eq2_lhs == eq2_rhs).all(), (
        "Equation 2 failed: transaction_count != event_count - view_count - "
        "addtocart_count for at least one row."
    )
    print("Equation 2 confirmed: transaction_count == event_count - view_count - addtocart_count")

    # --- Equation 3 (assumption — see module docstring): has_transaction is
    # exactly the indicator of transaction_count > 0.
    eq3_lhs = df["has_transaction"].astype(int)
    eq3_rhs = (df["transaction_count"] > 0).astype(int)
    assert (eq3_lhs == eq3_rhs).all(), (
        "Equation 3 (assumed: has_transaction == (transaction_count > 0)) failed "
        "for at least one row. This assumption may not match the paper's actual "
        "Equation 3 — see module docstring."
    )
    print("Equation 3 (assumed form) confirmed: has_transaction == (transaction_count > 0)")
    print()

    # ------------------------------------------------------------------
    # Table 5: 4 configs x {LR, XGB} x 5 seeds
    # ------------------------------------------------------------------
    all_runs = {}  # (config, model) -> list of per-seed metric dicts

    for config_name, feature_cols in FEATURE_CONFIGS.items():
        for seed in SEEDS:
            X_train, X_test, y_train, y_test = make_split(df, feature_cols, seed)

            # --- Logistic Regression (training-only standardization) -----
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            lr = build_lr(seed)
            lr_metrics = fit_and_score(lr, X_train_sc, y_train, X_test_sc, y_test)
            all_runs.setdefault((config_name, "LR"), []).append(lr_metrics)
            save_result_json(
                "completed-session", "logistic-regression", "stratified-80-20",
                slugify(config_name), seed,
                {"config": config_name, "features": feature_cols, "seed": seed,
                 "model": "LogisticRegression", **lr_metrics},
            )

            # --- XGBoost (raw capped features, no scaling) ---------------
            xgb = build_xgb(seed, y_train)
            xgb_metrics = fit_and_score(xgb, X_train, y_train, X_test, y_test)
            all_runs.setdefault((config_name, "XGB"), []).append(xgb_metrics)
            save_result_json(
                "completed-session", "xgboost", "stratified-80-20",
                slugify(config_name), seed,
                {"config": config_name, "features": feature_cols, "seed": seed,
                 "model": "XGBoost", **xgb_metrics},
            )

            # --- Table 4 extra models: seed 42, original_completed only ---
            if seed == TABLE4_SEED and config_name == TABLE4_CONFIG:
                dt = DecisionTreeClassifier(
                    class_weight="balanced", max_depth=8, random_state=seed
                )
                dt_metrics = fit_and_score(dt, X_train, y_train, X_test, y_test)
                all_runs.setdefault((config_name, "DT"), []).append(dt_metrics)
                save_result_json(
                    "completed-session", "decision-tree", "stratified-80-20",
                    slugify(config_name), seed,
                    {"config": config_name, "features": feature_cols, "seed": seed,
                     "model": "DecisionTree", **dt_metrics},
                )

                rf = RandomForestClassifier(
                    n_estimators=100, class_weight="balanced",
                    random_state=seed, n_jobs=-1,
                )
                rf_metrics = fit_and_score(rf, X_train, y_train, X_test, y_test)
                all_runs.setdefault((config_name, "RF"), []).append(rf_metrics)
                save_result_json(
                    "completed-session", "random-forest", "stratified-80-20",
                    slugify(config_name), seed,
                    {"config": config_name, "features": feature_cols, "seed": seed,
                     "model": "RandomForest", **rf_metrics},
                )

        print(f"Completed config: {config_name}")

    print()
    print("=" * 78)
    print("TABLE 5 COMPARISON — ROC-AUC (mean +/- sd across 5 seeds)")
    print("=" * 78)
    print(f"{'config':<20} {'model':<5} {'reproduced':<18} {'published':<18} {'diff':<8} flag")
    table5_diffs = []
    for config_name in FEATURE_CONFIGS:
        for model in ["LR", "XGB"]:
            runs = all_runs[(config_name, model)]
            aucs = np.array([r["roc_auc"] for r in runs])
            mean, sd = aucs.mean(), aucs.std(ddof=1)
            pub_mean, pub_sd = PUBLISHED_TABLE5_ROC_AUC[(config_name, model)]
            diff = mean - pub_mean
            flag = "DIFFERS" if abs(diff) > DIFF_TOLERANCE else "ok"
            if flag == "DIFFERS":
                table5_diffs.append((config_name, model, "roc_auc", mean, pub_mean, diff))
            print(f"{config_name:<20} {model:<5} {mean:.4f}+/-{sd:.4f}      "
                  f"{pub_mean:.4f}+/-{pub_sd:.4f}      {diff:+.4f}  {flag}")

    print()
    print("=" * 78)
    print("TABLE 5 COMPARISON — PR-AUC (mean across 5 seeds)")
    print("=" * 78)
    print(f"{'config':<20} {'model':<5} {'reproduced':<12} {'published':<12} {'diff':<8} flag")
    for config_name in FEATURE_CONFIGS:
        for model in ["LR", "XGB"]:
            runs = all_runs[(config_name, model)]
            pr_aucs = np.array([r["pr_auc"] for r in runs])
            mean = pr_aucs.mean()
            pub = PUBLISHED_TABLE5_PR_AUC[(config_name, model)]
            diff = mean - pub
            flag = "DIFFERS" if abs(diff) > DIFF_TOLERANCE else "ok"
            if flag == "DIFFERS":
                table5_diffs.append((config_name, model, "pr_auc", mean, pub, diff))
            print(f"{config_name:<20} {model:<5} {mean:.4f}       {pub:.4f}       {diff:+.4f}  {flag}")

    print()
    print("=" * 78)
    print(f"TABLE 4 COMPARISON — ROC-AUC, seed={TABLE4_SEED}, config={TABLE4_CONFIG}")
    print("=" * 78)
    table4_diffs = []
    model_map = {"LR": "LR", "DT": "DT", "RF": "RF", "XGB": "XGB"}
    for model in ["LR", "DT", "RF", "XGB"]:
        runs = [r for r in all_runs[(TABLE4_CONFIG, model)]]
        # For Table 4 we want the specific seed=42 run only.
        # LR/XGB have 5 seeds stored in order matching SEEDS list; DT/RF have 1.
        if model in ("LR", "XGB"):
            idx = SEEDS.index(TABLE4_SEED)
            value = runs[idx]["roc_auc"]
        else:
            value = runs[0]["roc_auc"]
        pub = PUBLISHED_TABLE4_ROC_AUC[model]
        diff = value - pub
        flag = "DIFFERS" if abs(diff) > DIFF_TOLERANCE else "ok"
        if flag == "DIFFERS":
            table4_diffs.append((model, value, pub, diff))
        print(f"{model:<5} reproduced={value:.4f}  published={pub:.4f}  diff={diff:+.4f}  {flag}")

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if not table5_diffs and not table4_diffs:
        print("No values differ from the published tables by more than "
              f"+/-{DIFF_TOLERANCE}.")
    else:
        print(f"Values differing from published by more than +/-{DIFF_TOLERANCE}:")
        for config_name, model, metric, mine, pub, diff in table5_diffs:
            print(f"  Table 5  {config_name:<20} {model:<5} {metric:<10} "
                  f"reproduced={mine:.4f}  published={pub:.4f}  diff={diff:+.4f}")
        for model, mine, pub, diff in table4_diffs:
            print(f"  Table 4  {model:<5} reproduced={mine:.4f}  published={pub:.4f}  diff={diff:+.4f}")

    print()
    print(f"Metric JSON files written to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
