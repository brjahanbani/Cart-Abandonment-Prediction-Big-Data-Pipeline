"""Step-3 exploration: profile Retail Rocket item_properties against cart sessions.
Read-only; prints findings. Runs from any directory."""
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])  # repo root; data paths below are relative to it

t0 = time.time()
RAW = "1-Data/raw/"
opts = pcsv.ConvertOptions(column_types={"timestamp": pa.int64(), "itemid": pa.int64(),
                                         "property": pa.string(), "value": pa.string()})
parts = [pcsv.read_csv(RAW + f"1-item_properties_part{i}.csv", convert_options=opts) for i in (1, 2)]
props = pa.concat_tables(parts)
print(f"rows={props.num_rows:,}  load {time.time()-t0:.0f}s")

ts = props["timestamp"].to_numpy()
print("timestamp range:", pd.to_datetime(ts.min(), unit="ms"), "->", pd.to_datetime(ts.max(), unit="ms"))
uts = np.unique(ts)
print(f"distinct snapshot timestamps: {len(uts)}")
dts = pd.to_datetime(uts, unit="ms")
print("weekday counts:", pd.Series(dts.day_name()).value_counts().to_dict())
print("hour counts:", pd.Series(dts.hour).value_counts().head(5).to_dict())
print("first 12 snapshot times:", [str(d) for d in dts[:12]])
print(f"unique items: {pc.count_distinct(props['itemid']).as_py():,}")
prop_counts = pd.Series(props["property"].to_numpy(zero_copy_only=False)).value_counts()
print(f"distinct properties: {len(prop_counts):,}")
print("top properties:", prop_counts.head(12).to_dict())

# Keep only the properties of interest as pandas.
KEEP = ["categoryid", "available", "790", "888", "283", "776", "678", "364"]
mask = pc.is_in(props["property"], value_set=pa.array(KEEP))
p = props.filter(mask).to_pandas()
del props, parts
print(f"rows for {KEEP}: {len(p):,}")
for prop, g in p.groupby("property"):
    per_item = g.groupby("itemid")["value"].nunique()
    snaps = g.groupby("itemid").size()
    print(f"  {prop:>10}: items={len(per_item):,}  mean snapshots/item={snaps.mean():.1f}  "
          f"items whose value changes={(per_item > 1).mean()*100:.1f}%  "
          f"example values={g['value'].head(3).tolist()}")

# --- Cart sessions: items added to cart, at the add-to-cart time -----------
ev = pd.read_csv("1-Data/events_clean.csv", usecols=["timestamp", "visitorid", "event", "itemid", "session_key"])
cart_keys = set(pd.read_csv("1-Data/completed-session/cart_session_features.csv", usecols=["session_key"])["session_key"])
ev = ev[ev["session_key"].isin(cart_keys)]
lab = pd.read_csv("1-Data/completed-session/cart_session_features.csv", usecols=["session_key", "has_transaction"])
carts = ev[ev["event"] == "addtocart"].merge(lab, on="session_key")
# first addtocart per session = the decision point used by the prefix dataset (k=0)
first = carts.sort_values("timestamp").groupby("session_key").first().reset_index()
print(f"\ncart sessions: {len(first):,}; first-cart items: {first['itemid'].nunique():,}")
ev_items = set(ev["itemid"])


def asof_vs_latest(prop):
    g = p[p["property"] == prop].sort_values(["itemid", "timestamp"])
    have = first["itemid"].isin(set(g["itemid"]))
    print(f"\n[{prop}] coverage of first-cart items: {have.mean()*100:.1f}%")
    q = first[have].sort_values("timestamp")
    asof = pd.merge_asof(q, g.rename(columns={"timestamp": "pts"}).sort_values("pts"),
                         left_on="timestamp", right_on="pts", by="itemid", direction="backward")
    latest = g.groupby("itemid").last()["value"].rename("latest")
    earliest = g.groupby("itemid").first()
    asof = asof.merge(latest, left_on="itemid", right_index=True)
    no_snapshot_before = asof["value"].isna()
    print(f"  no snapshot at/before the cart time (as-of missing): {no_snapshot_before.mean()*100:.1f}%")
    ok = asof[~no_snapshot_before]
    differ = ok["value"] != ok["latest"]
    print(f"  as-of value != latest value: {differ.mean()*100:.1f}% of sessions with an as-of value")
    lag_days = (ok["timestamp"] - ok["pts"]) / 86_400_000
    print(f"  staleness of as-of snapshot (days): median {lag_days.median():.1f}, p90 {lag_days.quantile(.9):.1f}")
    print("  differ rate by class:", ok.groupby("has_transaction").apply(
        lambda d: round((d['value'] != d['latest']).mean() * 100, 1), include_groups=False).to_dict())
    return asof


a = asof_vs_latest("available")
for col, name in (("value", "as-of"), ("latest", "latest")):
    d = a.dropna(subset=[col])
    tab = d.groupby(col)["has_transaction"].agg(["mean", "size"])
    print(f"  P(purchase | {name} available=v):", {k: (round(r['mean'], 3), int(r['size'])) for k, r in tab.iterrows()})
asof_vs_latest("categoryid")
asof_vs_latest("790")
print(f"\ntotal {time.time()-t0:.0f}s")
