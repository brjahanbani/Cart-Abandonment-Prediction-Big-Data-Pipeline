"""
label_ambiguity_check.py
=========================

Camera-ready addition (reviewer R1, comment 4): verify the §IV-C label
ambiguity against item identifiers, and measure its effect on Table V.

Timing groups (same rules as build_prefix_dataset.py / train_prefix_models.py):
  before  : a transaction strictly before the session's first addtocart
  between : a transaction strictly between the first and last addtocart
The distinct union is 1,722 sessions (the paper printed 172 + 1,573 = 1,745;
23 sessions are in both groups — see docs/results_provenance.md).

Item-level verification: a session is a VERIFIED mixed outcome when it
contains at least one purchase AND at least one carted item that is never
purchased in the session ("bought A, abandoned B"). This is checked for the
timing groups and, for scale, for every purchased cart session.

Effect: Table V (stratified 80/20, 5 seeds, unchanged settings) is re-run
with (a) the 1,722 timing-ambiguous sessions removed and (b) every verified
mixed-outcome session removed, and compared with the full population.

Outputs: 5-Results/label_ambiguity_check.json
Run from this folder:  python label_ambiguity_check.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from reanalysis_controlled_audit import (
    DATA_PATH,
    FEATURE_CONFIGS,
    RESULTS_DIR,
    SEEDS,
    build_lr,
    build_xgb,
    fit_and_score,
    make_split,
)

EVENTS_PATH = "../1-Data/events_clean.csv"


def classify_sessions(ev):
    rows = []
    for key, g in ev.groupby("session_key", sort=False):
        ts, et, it = g["timestamp"].values, g["event"].values, g["itemid"].values
        cart_ts, txn_ts = ts[et == "addtocart"], ts[et == "transaction"]
        carted, bought = set(it[et == "addtocart"]), set(it[et == "transaction"])
        has_txn = len(txn_ts) > 0
        rows.append({
            "session_key": key,
            "purchased": has_txn,
            "before": has_txn and txn_ts.min() < cart_ts.min(),
            "between": has_txn and len(cart_ts) >= 2
                       and bool(np.any((txn_ts > cart_ts.min()) & (txn_ts < cart_ts.max()))),
            # bought A, abandoned B: a purchase plus a carted item never bought in the session
            "mixed_outcome": has_txn and len(carted - bought) > 0,
            "n_carted_items": len(carted),
        })
    return pd.DataFrame(rows).set_index("session_key")


def table5(df, label):
    out = {}
    for config, cols in FEATURE_CONFIGS.items():
        for model in ("LR", "XGB"):
            aucs, prs = [], []
            for seed in SEEDS:
                X_train, X_test, y_train, y_test = make_split(df, cols, seed)
                if model == "LR":
                    sc = StandardScaler()
                    m = fit_and_score(build_lr(seed), sc.fit_transform(X_train), y_train,
                                      sc.transform(X_test), y_test)
                else:
                    m = fit_and_score(build_xgb(seed, y_train), X_train, y_train, X_test, y_test)
                aucs.append(m["roc_auc"])
                prs.append(m["pr_auc"])
            out[f"{config}/{model}"] = {"roc_auc_mean": float(np.mean(aucs)),
                                        "roc_auc_sd": float(np.std(aucs, ddof=1)),
                                        "pr_auc_mean": float(np.mean(prs))}
    print(f"  Table V re-run: {label} ({len(df):,} sessions, positive rate {df['has_transaction'].mean():.4f})")
    return out


def main():
    table = pd.read_csv(DATA_PATH)
    ev = pd.read_csv(EVENTS_PATH, usecols=["timestamp", "event", "itemid", "session_key"])
    ev = ev[ev["session_key"].isin(set(table["session_key"]))]
    cls = classify_sessions(ev).reindex(table["session_key"])

    timing = cls["before"] | cls["between"]
    stats = {
        "cart_sessions": len(cls),
        "purchased_sessions": int(cls["purchased"].sum()),
        "before_first_cart": int(cls["before"].sum()),
        "between_first_and_last_cart": int(cls["between"].sum()),
        "before_and_between_overlap": int((cls["before"] & cls["between"]).sum()),
        "timing_ambiguous_distinct": int(timing.sum()),
        "timing_ambiguous_pct": round(100 * timing.mean(), 2),
        "timing_ambiguous_verified_mixed": int((timing & cls["mixed_outcome"]).sum()),
        "timing_ambiguous_all_carted_items_bought": int((timing & ~cls["mixed_outcome"]).sum()),
        "purchased_with_mixed_outcome": int(cls["mixed_outcome"].sum()),
        "purchased_with_mixed_outcome_pct_of_purchased":
            round(100 * cls["mixed_outcome"].sum() / cls["purchased"].sum(), 2),
        "purchased_with_mixed_outcome_pct_of_all": round(100 * cls["mixed_outcome"].mean(), 2),
    }
    for k, v in stats.items():
        print(f"  {k:<48} {v}")

    keep_a = ~timing.values
    keep_b = ~cls["mixed_outcome"].values
    effect = {
        "full": table5(table, "full population"),
        "without_timing_ambiguous": table5(table[keep_a].reset_index(drop=True), "without timing-ambiguous"),
        "without_mixed_outcome": table5(table[keep_b].reset_index(drop=True), "without all mixed-outcome"),
    }

    print(f"\n{'config/model':<26} {'full':>8} {'-timing':>8} {'-mixed':>8}   (ROC-AUC mean, 5 seeds)")
    for key in effect["full"]:
        print(f"{key:<26} " + " ".join(f"{effect[v][key]['roc_auc_mean']:>8.4f}"
                                       for v in ("full", "without_timing_ambiguous", "without_mixed_outcome")))

    out = RESULTS_DIR / "label_ambiguity_check.json"
    out.write_text(json.dumps({"stats": stats, "table5_effect": effect}, indent=2) + "\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
