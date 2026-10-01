"""Step-3 probe: does a non-point-in-time item-property join inflate AUC?
k=0 decision point (first addtocart). Read-only; prints a table. Runs from any directory."""
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])  # repo root; data paths below are relative to it

t0 = time.time()
PROPS = ["available", "790", "categoryid"]
SEEDS = [11, 22, 33, 42, 55]

opts = pcsv.ConvertOptions(column_types={"timestamp": pa.int64(), "itemid": pa.int64(),
                                         "property": pa.string(), "value": pa.string()})
tabs = []
for i in (1, 2):
    t = pcsv.read_csv(f"1-Data/raw/1-item_properties_part{i}.csv", convert_options=opts)
    tabs.append(t.filter(pc.is_in(t["property"], value_set=pa.array(PROPS))))
p = pa.concat_tables(tabs).to_pandas()
p["num"] = np.where(p["property"] == "790",
                    pd.to_numeric(p["value"].str.lstrip("n"), errors="coerce"),
                    pd.to_numeric(p["value"], errors="coerce"))
p = p.sort_values("timestamp")
print(f"properties loaded ({len(p):,} rows) {time.time()-t0:.0f}s")

# k=0 rows + the first-cart item and its timestamp
pref = pd.read_parquet("1-Data/decision-time/prefix_dataset.parquet")
pref = pref[pref["cutoff_k"] == 0].copy()
ev = pd.read_csv("1-Data/events_clean.csv", usecols=["timestamp", "event", "itemid", "session_key"])
ev = ev[ev["session_key"].isin(set(pref["session_key"]))]
start = ev.groupby("session_key")["timestamp"].min().rename("start_ts")
fc = (ev[ev["event"] == "addtocart"].sort_values("timestamp")
      .groupby("session_key").first()[["timestamp", "itemid"]].rename(columns={"timestamp": "cart_ts"}))
pref = pref.merge(fc, left_on="session_key", right_index=True).merge(start, left_on="session_key", right_index=True)
pref = pref.sort_values("cart_ts").reset_index(drop=True)


def attach(df, mode):
    out = df.copy()
    for prop in PROPS:
        g = p[p["property"] == prop][["timestamp", "itemid", "num"]].rename(
            columns={"timestamp": "pts", "num": prop})
        if mode == "latest":
            v = g.sort_values("pts").groupby("itemid")[prop].last()
            out[prop] = out["itemid"].map(v)
        else:
            m = pd.merge_asof(out[["cart_ts", "itemid"]].reset_index(), g.sort_values("pts"),
                              left_on="cart_ts", right_on="pts", by="itemid", direction=mode)
            out[prop] = m.set_index("index")[prop]
        out[prop + "_missing"] = out[prop].isna().astype(int)
    out["log_price"] = np.log1p(out.pop("790"))
    out = out.drop(columns=["categoryid"])  # high-cardinality id; not used as a raw feature
    return out


BASE = ["view_count", "addtocart_count", "unique_items_count", "view_to_cart_ratio",
        "seconds_since_first_event", "seconds_since_first_cart", "seconds_since_last_cart",
        "seconds_since_previous_event", "inter_event_interval_mean", "events_after_cart",
        "repeated_views_of_cart_items", "hour_of_day", "day_of_week"]
ITEM = ["available", "available_missing", "log_price", "790_missing", "categoryid_missing"]


def run(df, feats, split, seed):
    if split == "temporal":
        order = df["start_ts"].rank(method="first")
        tr = order <= 0.8 * len(df)
        te = ~tr
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        i_tr, i_te = next(gss.split(df, groups=df["visitorid"]))
        tr = df.index.isin(i_tr); te = df.index.isin(i_te)
    Xtr, Xte = df.loc[tr, feats], df.loc[te, feats]
    ytr, yte = df.loc[tr, "y"], df.loc[te, "y"]
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, tree_method="hist",
                        scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(), random_state=seed,
                        eval_metric="logloss", n_jobs=-1)
    xgb.fit(Xtr, ytr)
    px = xgb.predict_proba(Xte)[:, 1]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(Xtr.fillna(Xtr.median()), ytr)
    pl = lr.predict_proba(Xte.fillna(Xtr.median()))[:, 1]
    return roc_auc_score(yte, px), average_precision_score(yte, px), roc_auc_score(yte, pl)


rows = []
for mode in ["none", "backward", "nearest", "latest"]:
    df = pref if mode == "none" else attach(pref, mode)
    for featset in (["base"] if mode == "none" else ["base+item", "item_only"]):
        feats = BASE if featset == "base" else (BASE + ITEM if featset == "base+item" else ITEM)
        for split in ["visitor-grouped", "temporal"]:
            res = np.array([run(df, feats, split, s) for s in SEEDS])
            rows.append({"join": mode, "features": featset, "split": split,
                         "xgb_roc": res[:, 0].mean(), "xgb_roc_sd": res[:, 0].std(),
                         "xgb_pr": res[:, 1].mean(), "lr_roc": res[:, 2].mean()})
            print(rows[-1])
print(pd.DataFrame(rows).round(4).to_string(index=False))
print(f"positive rate {pref['y'].mean():.4f}, n={len(pref):,}, total {time.time()-t0:.0f}s")
