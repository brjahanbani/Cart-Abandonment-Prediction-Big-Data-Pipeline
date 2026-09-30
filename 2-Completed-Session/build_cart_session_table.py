"""
build_cart_session_table.py
============================

Raw Retail Rocket events -> 1-Data/completed-session/cart_session_features.csv,
plus every descriptive number the paper quotes, in one committed script.

WHY THIS SCRIPT EXISTS
-----------------------
The 43,917-row cart-session table is the input to the paper's primary evidence
(reanalysis_controlled_audit.py, Tables IV-V), but until now the only code that
produced it was an auto-saved Jupyter checkpoint
(.ipynb_checkpoints/stage1_eda-checkpoint.ipynb, cell 23: filter
session_features.csv to addtocart_count > 0). Several descriptive numbers in
the paper (visitor count, Fig. 2 medians, zero-recency shares, label-ambiguity
counts, bot-event composition, the row-for-row check) were computed ad hoc and
printed by no script. This script closes both gaps.

HOW IT STAYS FAITHFUL TO THE FROZEN LINEAGE
--------------------------------------------
Like everything in 2-Completed-Session/, it depends on no code outside this
folder. The loading, sessionization, bot rules and the per-session feature
function are copied verbatim from stage1_data_cleaning_feature_engineering.py
(Sections 1, 3, 5, 6). The only difference is efficiency: features are built
for cart sessions only instead of all 1,761,640 sessions, which cannot change
any value (every feature depends only on its own session's events). The
rebuilt table is then compared BYTE FOR BYTE with the committed file.

OUTPUTS
--------
  1-Data/completed-session/cart_session_features.csv
      written only if it does not exist yet. If it exists and the rebuild
      differs, the rebuild goes to cart_session_features.rebuilt.csv and the
      script exits non-zero; the committed table is never overwritten.
  5-Results/completed-session__descriptive-stats.json
      every descriptive number, each paper-quoted value next to the computed
      one, plus git commit, UTC timestamp, library versions and input hashes.

Run from this folder, like the other scripts here:

    cd 2-Completed-Session
    python build_cart_session_table.py

Needs 1-Data/raw/1-events.csv (gitignored; Retail Rocket dataset on Kaggle).
The row-for-row check additionally uses 1-Data/events_clean.csv if present.
Runtime: about 10 minutes (bot CV per visitor and per-session features).
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RAW_PATH = REPO / "1-Data" / "raw" / "1-events.csv"
CLEAN_ARTIFACT_PATH = REPO / "1-Data" / "events_clean.csv"
TABLE_PATH = REPO / "1-Data" / "completed-session" / "cart_session_features.csv"
REBUILT_PATH = TABLE_PATH.with_name("cart_session_features.rebuilt.csv")
JSON_PATH = REPO / "5-Results" / "completed-session__descriptive-stats.json"

# Verbatim from stage1 Sections 3 and 5.
SESSION_GAP_MS = 30 * 60 * 1000
BOT_VELOCITY_THRESHOLD = 500
BOT_CV_THRESHOLD = 0.05
BOT_MIN_EVENTS = 20

TABLE_COLUMNS = [
    "session_key", "visitorid",
    "event_count", "view_count", "addtocart_count", "transaction_count",
    "view_to_cart_ratio", "unique_items_count",
    "session_age_seconds", "inter_event_interval_mean",
    "recency_last_cart_seconds", "has_transaction",
]
TIMING_FEATURES = ["session_age_seconds", "inter_event_interval_mean", "recency_last_cart_seconds"]

# Every descriptive number the submitted paper quotes (paper/arxiv/
# IJACSA_Final_NonBlind.tex; section in brackets), in the precision it is
# printed with. Compared against the computed values at the end.
PAPER = {
    "raw_events": (2_756_101, "§III-A"),
    "observation_days": (137, "§III-A"),
    "clean_events": (2_755_610, "§III-A, §VII"),
    "bot_visitors": (35, "§III-A, §VII"),
    "bot_events": (491, "§III-A, §VII"),
    "bot_events_view": (431, "§VII"),
    "bot_events_addtocart": (60, "§VII"),
    "bot_events_transaction": (0, "§VII ('no transactions')"),
    "sessions_all": (1_761_640, "Table III"),
    "cart_sessions": (43_917, "Abstract, Table I"),
    "cart_visitors": (37_715, "Table I, §III-A"),
    "abandoned_sessions": (31_985, "Table I"),
    "purchased_sessions": (11_932, "Table I"),
    "abandoned_pct": (72.83, "Table I"),
    "purchased_pct": (27.17, "Table I, Abstract, Fig. 4"),
    "median_session_age_purchased": (821.03, "§IV-B, Fig. 2"),
    "median_session_age_abandoned": (172.44, "§IV-B, Fig. 2"),
    "median_interval_purchased": (183.29, "Fig. 2"),
    "median_interval_abandoned": (72.77, "Fig. 2"),
    "median_cart_recency_purchased": (330.82, "§IV-B, Fig. 2"),
    "median_cart_recency_abandoned": (0.0, "§IV-B, Fig. 2"),
    "zero_cart_recency_pct_abandoned": (60.6, "§IV-B"),
    "zero_cart_recency_pct_purchased": (2.1, "§IV-B"),
    "txn_before_first_cart": (172, "§IV-C"),
    "txn_between_first_and_last_cart": (1_573, "§IV-C"),
    "label_ambiguous_sessions": (1_745, "§IV-C"),
    "label_ambiguous_pct": (4.0, "Abstract, §IV-C"),
    "eq2_rows_holding": (43_917, "§IV-A ('holds for all 43,917 rows')"),
    "eq3_rows_holding": (43_917, "§IV-A ('for every row')"),
}

# Paper values that this script shows to be wrong. They are reported next to
# the computed value in the JSON and do not fail the run; any mismatch NOT
# listed here does. Published values are never edited to match.
KNOWN_DISCREPANCIES = {
    "label_ambiguous_sessions": (
        "The paper adds the two groups (172 + 1,573 = 1,745; '§IV-C: A further "
        "1,573 sessions ...'), but 23 sessions belong to both: they have a "
        "transaction before their first addtocart AND one between their first "
        "and last addtocart. The distinct count is 1,722; the 'further' group "
        "has 1,550 sessions not already among the 172."
    ),
    "label_ambiguous_pct": "Follows from label_ambiguous_sessions: 1,722 / 43,917 = 3.92%, printed 3.9%, not 4.0%.",
}


# ============================================================================
# Verbatim copies from stage1_data_cleaning_feature_engineering.py
# ============================================================================

def load_raw(path):
    # stage1 Section 1
    return pd.read_csv(
        path,
        header=0,
        names=["timestamp", "visitorid", "event", "itemid", "transactionid"],
        dtype={
            "timestamp": "int64",
            "visitorid": "int64",
            "event": "category",
            "itemid": "Int64",
            "transactionid": "Int64",
        },
    )


def sessionize(df_raw):
    # stage1 Section 3
    df = df_raw.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)
    df["prev_ts"] = df.groupby("visitorid")["timestamp"].shift(1)
    df["gap_ms"] = df["timestamp"] - df["prev_ts"]
    df["new_session"] = (df["gap_ms"].isna() | (df["gap_ms"] > SESSION_GAP_MS)).astype(int)
    df["session_id"] = df.groupby("visitorid")["new_session"].cumsum()
    df["session_key"] = df["visitorid"].astype(str) + "_" + df["session_id"].astype(str)
    return df


def compute_cv(group):
    # stage1 Section 5
    intervals = group["timestamp"].sort_values().diff().dropna()
    if len(intervals) < BOT_MIN_EVENTS:
        return np.nan
    mean_iv = intervals.mean()
    if mean_iv == 0:
        return 0.0
    return intervals.std() / mean_iv


def find_bots(df):
    # stage1 Section 5
    visitor_stats = df.groupby("visitorid").agg(
        total_events=("event", "count"),
        session_span_ms=("timestamp", lambda x: x.max() - x.min()),
    ).reset_index()
    visitor_stats["span_hours"] = (visitor_stats["session_span_ms"] / 3_600_000).clip(lower=1 / 60)
    visitor_stats["events_per_hour"] = visitor_stats["total_events"] / visitor_stats["span_hours"]

    visitor_cv = (
        df.groupby("visitorid")[["timestamp"]].apply(compute_cv).rename("interval_cv").reset_index()
    )
    visitor_stats = visitor_stats.merge(visitor_cv, on="visitorid")

    is_bot_velocity = visitor_stats["events_per_hour"] > BOT_VELOCITY_THRESHOLD
    is_bot_cv = visitor_stats["interval_cv"].notna() & (visitor_stats["interval_cv"] < BOT_CV_THRESHOLD)
    return set(visitor_stats.loc[is_bot_velocity | is_bot_cv, "visitorid"])


def build_session_features(group):
    # stage1 Section 6, unchanged
    g = group.sort_values("timestamp")

    ts = g["timestamp"].values
    events = g["event"].values
    items = g["itemid"].dropna().values

    view_mask = events == "view"
    cart_mask = events == "addtocart"
    txn_mask = events == "transaction"

    view_cnt = view_mask.sum()
    cart_cnt = cart_mask.sum()

    v2c = view_cnt / cart_cnt if cart_cnt > 0 else float(view_cnt)

    intervals = np.diff(ts) / 1000.0
    mean_interval = intervals.mean() if len(intervals) > 0 else 0.0

    age_sec = (ts[-1] - ts[0]) / 1000.0 if len(ts) > 1 else 0.0

    cart_ts = ts[cart_mask]
    recency_cart = (ts[-1] - cart_ts[-1]) / 1000.0 if len(cart_ts) > 0 else -1.0

    return pd.Series({
        "event_count": len(g),
        "view_count": int(view_cnt),
        "addtocart_count": int(cart_cnt),
        "transaction_count": int(txn_mask.sum()),
        "view_to_cart_ratio": round(v2c, 4),
        "unique_items_count": int(pd.Series(items).nunique()),
        "session_age_seconds": round(age_sec, 2),
        "inter_event_interval_mean": round(mean_interval, 3),
        "recency_last_cart_seconds": round(recency_cart, 2),
        "has_transaction": int(txn_mask.any()),
    })


# ============================================================================
# Table serialisation — matches the committed file's format exactly
# ============================================================================

def format_number(v):
    """The committed CSV writes whole-number floats without '.0' (5, 0) and
    everything else as Python's shortest repr (494.17, 247.084)."""
    v = float(v)
    return str(int(v)) if v.is_integer() else repr(v)


def table_to_csv_bytes(table):
    lines = [",".join(TABLE_COLUMNS)]
    for row in table[TABLE_COLUMNS].itertuples(index=False):
        lines.append(",".join([row[0], str(int(row[1]))] + [format_number(v) for v in row[2:]]))
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


# ============================================================================
# Helpers
# ============================================================================

def round_half_up(x, ndigits):
    """Round as the paper prints: half up, on the decimal value. Python's
    round() works on the binary float, so a median of exactly 330.815
    (stored as 330.81499...) would come out as 330.81 instead of 330.82."""
    q = Decimal(1).scaleb(-ndigits)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def label_ambiguity(cart_events):
    """Sessions whose session-level label cannot describe at least one of their
    cart actions (paper §IV-C). Same rules as 3-Decision-Time: a transaction
    strictly before the first addtocart (build_prefix_dataset.py), or strictly
    between the first and last addtocart (train_prefix_models.py)."""
    before, between = set(), set()
    for key, g in cart_events.groupby("session_key", sort=False):
        ts = g["timestamp"].values
        ev = g["event"].values
        cart_ts, txn_ts = ts[ev == "addtocart"], ts[ev == "transaction"]
        if len(txn_ts) == 0:
            continue
        if txn_ts.min() < cart_ts.min():
            before.add(key)
        if len(cart_ts) >= 2 and np.any((txn_ts > cart_ts.min()) & (txn_ts < cart_ts.max())):
            between.add(key)
    return before, between


def row_for_row(df_clean):
    """Compare the cleaned events with the retained artifact on the four
    columns the paper names, as sorted multisets (the artifact is sorted by
    timestamp with arbitrary tie order)."""
    if not CLEAN_ARTIFACT_PATH.exists():
        return {"status": "skipped", "reason": f"{CLEAN_ARTIFACT_PATH.name} not present"}
    cols = ["visitorid", "timestamp", "event", "itemid"]
    artifact = pd.read_csv(CLEAN_ARTIFACT_PATH, usecols=cols + ["session_key"],
                           dtype={"itemid": "Int64", "event": "string"})
    ours = df_clean[cols + ["session_key"]].astype({"event": "string"})

    def canon(d):
        return d.sort_values(cols + ["session_key"]).reset_index(drop=True)

    a, b = canon(artifact), canon(ours)
    same_rows = len(a) == len(b)
    same_four = same_rows and a[cols].equals(b[cols])
    same_keys = same_rows and a["session_key"].equals(b["session_key"])
    return {
        "status": "match" if same_four else "MISMATCH",
        "artifact": str(CLEAN_ARTIFACT_PATH.relative_to(REPO)),
        "artifact_sha256": sha256(CLEAN_ARTIFACT_PATH),
        "rows_artifact": int(len(a)),
        "rows_rebuilt": int(len(b)),
        "columns_compared": cols,
        "session_key_also_matches": bool(same_keys),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 78)
    print("build_cart_session_table.py — raw events -> cart-session table + paper stats")
    print("=" * 78)

    if not RAW_PATH.exists():
        sys.exit(f"Missing {RAW_PATH} — download the Retail Rocket dataset from Kaggle.")

    stats = {}

    # --- Load + inspect (stage1 Sections 1-2) -------------------------------
    df_raw = load_raw(RAW_PATH)
    event_time = pd.to_datetime(df_raw["timestamp"], unit="ms")
    stats["raw_events"] = len(df_raw)
    stats["observation_days"] = (event_time.max() - event_time.min()).days
    stats["observation_start"] = str(event_time.min().date())
    stats["observation_end"] = str(event_time.max().date())
    print(f"Raw events: {stats['raw_events']:,} over {stats['observation_days']} days")

    # --- Sessionize, then remove bots (stage1 Sections 3, 5) ----------------
    df = sessionize(df_raw)
    print("Flagging bots (CV per visitor takes ~1-2 min)...")
    bots = find_bots(df)
    is_bot_event = df["visitorid"].isin(bots)
    bot_mix = df.loc[is_bot_event, "event"].value_counts()
    df_clean = df[~is_bot_event].copy()

    stats["bot_visitors"] = len(bots)
    stats["bot_events"] = int(is_bot_event.sum())
    stats["bot_events_view"] = int(bot_mix.get("view", 0))
    stats["bot_events_addtocart"] = int(bot_mix.get("addtocart", 0))
    stats["bot_events_transaction"] = int(bot_mix.get("transaction", 0))
    stats["clean_events"] = len(df_clean)
    stats["sessions_all"] = int(df_clean["session_key"].nunique())
    print(f"Bots: {stats['bot_visitors']} visitors, {stats['bot_events']} events "
          f"({stats['bot_events_view']} view / {stats['bot_events_addtocart']} addtocart / "
          f"{stats['bot_events_transaction']} transaction) -> {stats['clean_events']:,} clean events, "
          f"{stats['sessions_all']:,} sessions")

    # --- Cart sessions + features (stage1 Section 6, checkpoint cell 23) ----
    cart_keys = df_clean.loc[df_clean["event"] == "addtocart", "session_key"].unique()
    cart_events = df_clean[df_clean["session_key"].isin(cart_keys)]
    print(f"Building features for {len(cart_keys):,} cart sessions...")
    features = (
        cart_events.groupby("session_key")[["timestamp", "event", "itemid"]]
        .apply(build_session_features)
        .reset_index()
    )
    visitor_per_session = cart_events.groupby("session_key")["visitorid"].first().reset_index()
    table = features.merge(visitor_per_session, on="session_key")[TABLE_COLUMNS]
    # The checkpoint filtered on addtocart_count > 0; assert it selects the same set.
    assert (table["addtocart_count"] > 0).all()

    y = table["has_transaction"]
    stats["cart_sessions"] = len(table)
    stats["cart_visitors"] = int(table["visitorid"].nunique())
    stats["purchased_sessions"] = int((y == 1).sum())
    stats["abandoned_sessions"] = int((y == 0).sum())
    stats["purchased_pct"] = round_half_up(stats["purchased_sessions"] / len(table) * 100, 2)
    stats["abandoned_pct"] = round_half_up(stats["abandoned_sessions"] / len(table) * 100, 2)

    # --- Equations 2 and 3 (§IV-A) ------------------------------------------
    eq2 = table["transaction_count"] == (
        table["event_count"] - table["view_count"] - table["addtocart_count"])
    eq3 = table["has_transaction"] == (table["transaction_count"] > 0).astype(int)
    stats["eq2_rows_holding"] = int(eq2.sum())
    stats["eq3_rows_holding"] = int(eq3.sum())

    # --- Fig. 2 medians and zero-recency shares (§IV-B) ---------------------
    for feat, name in zip(TIMING_FEATURES, ["session_age", "interval", "cart_recency"]):
        med = table.groupby("has_transaction")[feat].median()
        for label, cls in (("purchased", 1), ("abandoned", 0)):
            stats[f"median_{name}_{label}"] = round_half_up(med[cls], 2)
            stats[f"median_{name}_{label}_unrounded"] = float(med[cls])
    zero = table.groupby("has_transaction")["recency_last_cart_seconds"].apply(lambda s: (s == 0).mean())
    stats["zero_cart_recency_pct_purchased"] = round_half_up(zero[1] * 100, 1)
    stats["zero_cart_recency_pct_abandoned"] = round_half_up(zero[0] * 100, 1)

    # --- Label ambiguity (§IV-C) --------------------------------------------
    before, between = label_ambiguity(cart_events)
    stats["txn_before_first_cart"] = len(before)
    stats["txn_between_first_and_last_cart"] = len(between)
    stats["txn_before_and_between_overlap"] = len(before & between)
    stats["txn_between_only"] = len(between - before)
    stats["label_ambiguous_sessions"] = len(before | between)
    stats["label_ambiguous_pct"] = round_half_up(len(before | between) / len(table) * 100, 1)

    # --- Row-for-row check against the retained cleaned artifact (§VII) -----
    print("Row-for-row check against the retained cleaned event log...")
    row_check = row_for_row(df_clean)

    # --- Table: compare byte for byte, never overwrite ----------------------
    rebuilt = table_to_csv_bytes(table.sort_values("session_key"))
    rebuilt_sha = hashlib.sha256(rebuilt).hexdigest()
    if not TABLE_PATH.exists():
        TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        TABLE_PATH.write_bytes(rebuilt)
        table_status = "written (file did not exist)"
    elif TABLE_PATH.read_bytes() == rebuilt:
        table_status = "byte-identical to committed file"
    else:
        REBUILT_PATH.write_bytes(rebuilt)
        table_status = f"DIFFERS from committed file — rebuild saved to {REBUILT_PATH.name}"

    # --- Compare with the paper ---------------------------------------------
    checks = []
    for key, (paper_value, where) in PAPER.items():
        computed = stats[key]
        check = {"quantity": key, "paper": paper_value, "computed": computed,
                 "where_in_paper": where, "match": computed == paper_value}
        if not check["match"] and key in KNOWN_DISCREPANCIES:
            check["known_discrepancy"] = KNOWN_DISCREPANCIES[key]
        checks.append(check)
    failures = [c for c in checks if not c["match"] and "known_discrepancy" not in c]
    # A documented discrepancy that has disappeared means the explanation is stale.
    failures += [{"quantity": f"{k} (listed as discrepancy but now matches)"}
                 for k in KNOWN_DISCREPANCIES
                 if any(c["quantity"] == k and c["match"] for c in checks)]
    if row_check["status"] == "MISMATCH":
        failures.append({"quantity": "row_for_row"})
    if table_status.startswith("DIFFERS"):
        failures.append({"quantity": "cart_session_table"})

    # --- JSON -----------------------------------------------------------------
    payload = {
        "script": "2-Completed-Session/build_cart_session_table.py",
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "libraries": {"pandas": pd.__version__, "numpy": np.__version__},
        },
        "inputs": {
            "raw_events": {"path": str(RAW_PATH.relative_to(REPO)), "sha256": sha256(RAW_PATH)},
        },
        "cart_session_table": {
            "path": str(TABLE_PATH.relative_to(REPO)),
            "rows": len(table),
            "rebuilt_sha256": rebuilt_sha,
            "status": table_status,
        },
        "row_for_row_check": row_check,
        "stats": stats,
        "paper_checks": checks,
        "all_paper_values_match": all(c["match"] for c in checks),
        "known_discrepancies": sorted(c["quantity"] for c in checks if "known_discrepancy" in c),
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # --- Report ---------------------------------------------------------------
    print()
    print(f"{'quantity':<36} {'paper':>12} {'computed':>12}  where")
    for c in checks:
        flag = ("" if c["match"] else
                "   <-- DIFFERS (documented)" if "known_discrepancy" in c else "   <-- DIFFERS")
        print(f"{c['quantity']:<36} {c['paper']!s:>12} {c['computed']!s:>12}  {c['where_in_paper']}{flag}")
    for c in checks:
        if "known_discrepancy" in c:
            print(f"\n  {c['quantity']}: {c['known_discrepancy']}")
    print()
    print(f"txn before AND between (overlap): {stats['txn_before_and_between_overlap']}")
    print(f"Row-for-row check: {row_check['status']}"
          + (f" (session_key also matches: {row_check['session_key_also_matches']})"
             if row_check["status"] != "skipped" else f" ({row_check['reason']})"))
    print(f"Cart-session table: {table_status}")
    print(f"Saved: {JSON_PATH.relative_to(REPO)}")

    if failures:
        print(f"\n{len(failures)} check(s) failed: {[f['quantity'] for f in failures]}")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
