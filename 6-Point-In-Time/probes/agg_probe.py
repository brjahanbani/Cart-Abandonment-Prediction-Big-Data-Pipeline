"""Step-3 probe 2: aggregate features with wrong time windows (feature-store style).
Item conversion rate and visitor purchase history, computed three ways:
  asof     : only events strictly before the cart timestamp (point-in-time correct)
  full_loo : whole dataset, excluding the scored session itself (future, no self)
  full     : whole dataset including the scored session (future + self)
k=0 decision point. Read-only. Runs from any directory."""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])  # repo root; data paths below are relative to it

t0 = time.time()
SEEDS = [11, 22, 33, 42, 55]
pref = pd.read_parquet("1-Data/decision-time/prefix_dataset.parquet")
pref = pref[pref["cutoff_k"] == 0].copy()
ev = pd.read_csv("1-Data/events_clean.csv", usecols=["timestamp", "visitorid", "event", "itemid", "session_key"])
ev = ev.sort_values("timestamp", kind="stable").reset_index(drop=True)

fc = (ev[(ev["event"] == "addtocart") & ev["session_key"].isin(set(pref["session_key"]))]
      .groupby("session_key").first()[["timestamp", "itemid"]].rename(columns={"timestamp": "cart_ts"}))
start = ev.groupby("session_key")["timestamp"].min().rename("start_ts")
pref = pref.merge(fc, left_on="session_key", right_index=True).merge(start, left_on="session_key", right_index=True)

# --- event-level cumulative counts (as-of = strictly before) --------------
is_cart = (ev["event"] == "addtocart").astype(int)
is_txn = (ev["event"] == "transaction").astype(int)
ev["item_carts_before"] = is_cart.groupby(ev["itemid"]).cumsum() - is_cart
ev["item_txns_before"] = is_txn.groupby(ev["itemid"]).cumsum() - is_txn
ev["vis_txns_before"] = is_txn.groupby(ev["visitorid"]).cumsum() - is_txn
# events with identical timestamps: cumsum order is arbitrary among ties; ties are rare and
# would only shift counts by events at the very same millisecond.
first_cart_rows = ev[(ev["event"] == "addtocart")].groupby("session_key").head(1).set_index("session_key")
asof = first_cart_rows.loc[pref["session_key"], ["item_carts_before", "item_txns_before", "vis_txns_before"]]
pref["item_cr_asof"] = (asof["item_txns_before"].values + 1) / (asof["item_carts_before"].values + 2)
pref["vis_txns_asof"] = asof["vis_txns_before"].values

# --- whole-dataset totals --------------------------------------------------
item_carts = ev[ev["event"] == "addtocart"].groupby("itemid").size()
item_txns = ev[ev["event"] == "transaction"].groupby("itemid").size()
vis_txns = ev[ev["event"] == "transaction"].groupby("visitorid").size()
# the scored session's own contribution
own = ev[ev["session_key"].isin(set(pref["session_key"]))]
own_item_carts = own[own["event"] == "addtocart"].groupby(["session_key", "itemid"]).size()
own_item_txns = own[own["event"] == "transaction"].groupby(["session_key", "itemid"]).size()
own_txns = own[own["event"] == "transaction"].groupby("session_key").size()

ic = pref["itemid"].map(item_carts).fillna(0).values
it = pref["itemid"].map(item_txns).fillna(0).values
idx = pd.MultiIndex.from_arrays([pref["session_key"], pref["itemid"]])
oc = own_item_carts.reindex(idx).fillna(0).values
ot = own_item_txns.reindex(idx).fillna(0).values
vt = pref["visitorid"].map(vis_txns).fillna(0).values
vo = pref["session_key"].map(own_txns).fillna(0).values

pref["item_cr_full"] = (it + 1) / (ic + 2)
pref["item_cr_full_loo"] = (it - ot + 1) / (ic - oc + 2)
pref["vis_txns_full"] = vt
pref["vis_txns_full_loo"] = vt - vo
print(f"features built {time.time()-t0:.0f}s")

BASE = ["view_count", "addtocart_count", "unique_items_count", "view_to_cart_ratio",
        "seconds_since_first_event", "seconds_since_first_cart", "seconds_since_last_cart",
        "seconds_since_previous_event", "inter_event_interval_mean", "events_after_cart",
        "repeated_views_of_cart_items", "hour_of_day", "day_of_week"]


def auc(feats, split, seed):
    df = pref
    if split == "temporal":
        r = df["start_ts"].rank(method="first"); tr = (r <= 0.8 * len(df)).values; te = ~tr
    else:
        i_tr, i_te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(df, groups=df["visitorid"]))
        tr = np.zeros(len(df), bool); tr[i_tr] = True; te = ~tr
    ytr, yte = df["y"].values[tr], df["y"].values[te]
    m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, tree_method="hist",
                      scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(), random_state=seed,
                      eval_metric="logloss", n_jobs=-1)
    m.fit(df[feats].values[tr], ytr)
    return roc_auc_score(yte, m.predict_proba(df[feats].values[te])[:, 1])


rows = []
for window in ["asof", "full_loo", "full"]:
    agg = [f"item_cr_{window}", f"vis_txns_{window}"]
    for name, feats in (("agg_only", agg), ("base+agg", BASE + agg)):
        for split in ["visitor-grouped", "temporal"]:
            a = [auc(feats, split, s) for s in SEEDS]
            rows.append({"window": window, "features": name, "split": split,
                         "xgb_roc": np.mean(a), "sd": np.std(a)})
for split in ["visitor-grouped", "temporal"]:
    a = [auc(BASE, split, s) for s in SEEDS]
    rows.append({"window": "none", "features": "base", "split": split, "xgb_roc": np.mean(a), "sd": np.std(a)})
print(pd.DataFrame(rows).round(4).to_string(index=False))
print(f"total {time.time()-t0:.0f}s")
