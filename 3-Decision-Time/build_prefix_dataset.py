"""
build_prefix_dataset.py
========================

Point-in-time-correct feature rebuild for cart-abandonment prediction.

WHY THIS SCRIPT EXISTS
-----------------------
An audit of the old Stage-1 pipeline found two leakage mechanisms:

  (a) `transaction_count` was recoverable as
      event_count - view_count - addtocart_count
      (i.e. it silently encoded whether the session ever converted)

  (b) `session_age_seconds` and `recency_last_cart_seconds` were computed
      using the SESSION'S FINAL EVENT as the "current time" endpoint.
      That final event can be *after* the moment we would actually be
      scoring the visitor in production (e.g. after the purchase itself),
      so those features silently peek into the future.

This script fixes both by never computing anything from a full session.
Instead, for every cart session we take a series of "cutoffs" (a point in
time k events after the first add-to-cart) and compute every feature only
from the events that occurred AT OR BEFORE that cutoff (the "prefix").
This is exactly the information a real-time scoring system would have.

Everything below is deliberately UN-OPTIMISED. This is a student project
script: every step is written so it can be explained line-by-line in a
viva, not so it runs fast. Pandas only, no Spark.
"""

import numpy as np
import pandas as pd

# ============================================================================
# CONSTANTS
# ============================================================================

# Raw input file. Path is relative to the repo root (where this script lives).
# We deliberately read from the ORIGINAL raw file and redo all cleaning here,
# rather than reusing events_clean.csv, so the whole pipeline is reproducible
# end-to-end from a single source file.
DATA_PATH = "0-Offline/Data Cleaning & Feature Engineering/1-events.csv"

# Output file for the final point-in-time dataset.
OUTPUT_PATH = "prefix_dataset.parquet"

# A new session starts when the gap since the visitor's previous event
# exceeds this many milliseconds (30 minutes — standard e-commerce timeout).
SESSION_GAP_MS = 30 * 60 * 1000

# Bot-detection thresholds (identical to the original Stage-1 script, so the
# cleaned population matches the numbers already published).
BOT_VELOCITY_THRESHOLD = 500   # events/hour above this => flag as bot
BOT_CV_THRESHOLD = 0.05        # inter-event-interval CV below this => flag as bot
BOT_MIN_EVENTS = 20            # need >= this many events to trust a CV estimate

# The cutoffs we evaluate, expressed as "k events after the first add-to-cart".
# k = 0 means "the moment of the first add-to-cart itself".
CUTOFFS = [0, 1, 2, 3, 5, 10]

# Values published in earlier runs / audits, used purely as fail-loud sanity
# checks so a silent change in the raw file or cleaning logic is caught
# immediately instead of quietly changing every downstream number.
EXPECTED_RAW_ROWS = 2_756_101
EXPECTED_CLEAN_EVENTS = 2_755_610
EXPECTED_BOT_VISITORS = 35
EXPECTED_BOT_EVENTS = 491
EXPECTED_CLEAN_EVENT_MIX = {
    "view": 2_663_881,
    "addtocart": 69_272,
    "transaction": 22_457,
}


# ============================================================================
# STEP 1 — LOAD + CLEAN (bot removal)
# ============================================================================

print("=" * 78)
print("STEP 1 — Load raw events and remove bot traffic")
print("=" * 78)

df_raw = pd.read_csv(
    DATA_PATH,
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

print(f"Loaded {len(df_raw):,} raw rows")

# INPUT ASSERTION — catches a wrong/changed/truncated raw file immediately.
assert len(df_raw) == EXPECTED_RAW_ROWS, (
    f"Raw row count changed! Expected {EXPECTED_RAW_ROWS:,}, got {len(df_raw):,}. "
    "Stop and check whether 1-events.csv was replaced or truncated."
)

# Keep the raw per-event-type mix so we can later assert the bot filter
# removed zero transaction events (transactions are rare and precious —
# losing even one to a "bot" rule would be a red flag, not routine cleaning).
raw_event_mix = df_raw["event"].value_counts().to_dict()

# Sort by (visitorid, timestamp) — required before any gap/interval computation.
df = df_raw.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)

# --- Bot signal 1: velocity (events per hour) ------------------------------
visitor_stats = df.groupby("visitorid").agg(
    total_events=("event", "count"),
    span_ms=("timestamp", lambda x: x.max() - x.min()),
).reset_index()

# clip(lower=1/60) avoids division by zero for visitors whose events all
# happen within under a minute (or at the exact same millisecond).
visitor_stats["span_hours"] = (visitor_stats["span_ms"] / 3_600_000).clip(lower=1 / 60)
visitor_stats["events_per_hour"] = visitor_stats["total_events"] / visitor_stats["span_hours"]
visitor_stats["is_bot_velocity"] = visitor_stats["events_per_hour"] > BOT_VELOCITY_THRESHOLD

# --- Bot signal 2: regularity (coefficient of variation of intervals) ------
def compute_cv(group):
    """CV = std / mean of inter-event gaps. Bots fire at near-constant
    intervals (CV close to 0); humans do not."""
    intervals = group["timestamp"].sort_values().diff().dropna()
    if len(intervals) < BOT_MIN_EVENTS:
        return np.nan
    mean_iv = intervals.mean()
    if mean_iv == 0:
        return 0.0
    return intervals.std() / mean_iv

visitor_cv = (
    df.groupby("visitorid")
    .apply(compute_cv, include_groups=False)
    .rename("interval_cv")
    .reset_index()
)
visitor_stats = visitor_stats.merge(visitor_cv, on="visitorid")
visitor_stats["is_bot_cv"] = (
    visitor_stats["interval_cv"].notna() & (visitor_stats["interval_cv"] < BOT_CV_THRESHOLD)
)

# A visitor is a bot if EITHER signal fires.
visitor_stats["is_bot"] = visitor_stats["is_bot_velocity"] | visitor_stats["is_bot_cv"]
bot_visitors = set(visitor_stats.loc[visitor_stats["is_bot"], "visitorid"])

n_bot_visitors = len(bot_visitors)
n_bot_events = df["visitorid"].isin(bot_visitors).sum()

print(f"Visitors flagged as bots : {n_bot_visitors:,}")
print(f"Events removed           : {n_bot_events:,}")

df_clean = df[~df["visitorid"].isin(bot_visitors)].copy()
clean_event_mix = df_clean["event"].value_counts().to_dict()

# How many of the removed events were transactions? We expect zero: bots
# view/add-to-cart at machine speed, they don't complete purchases.
transactions_removed_by_bot_filter = (
    raw_event_mix.get("transaction", 0) - clean_event_mix.get("transaction", 0)
)
print(f"Transactions removed by bot filter: {transactions_removed_by_bot_filter}")

# ---------------------------------------------------------------------------
# INPUT ASSERTIONS (Addition 2) — fail loudly if the cleaned population
# drifts from the numbers already published in the audit / earlier runs.
# ---------------------------------------------------------------------------
assert n_bot_visitors == EXPECTED_BOT_VISITORS, (
    f"Bot visitor count changed! Expected {EXPECTED_BOT_VISITORS}, got {n_bot_visitors}."
)
assert n_bot_events == EXPECTED_BOT_EVENTS, (
    f"Bot event count changed! Expected {EXPECTED_BOT_EVENTS}, got {n_bot_events}."
)
assert transactions_removed_by_bot_filter == 0, (
    "Bot filter removed a transaction event — this must never happen. "
    f"Removed {transactions_removed_by_bot_filter} transactions."
)
assert len(df_clean) == EXPECTED_CLEAN_EVENTS, (
    f"Post-cleaning event count changed! Expected {EXPECTED_CLEAN_EVENTS:,}, "
    f"got {len(df_clean):,}."
)
for etype, expected_count in EXPECTED_CLEAN_EVENT_MIX.items():
    actual_count = clean_event_mix.get(etype, 0)
    assert actual_count == expected_count, (
        f"Post-cleaning '{etype}' count changed! Expected {expected_count:,}, "
        f"got {actual_count:,}."
    )
print("All input assertions passed.")
print()


# ============================================================================
# STEP 2 — SESSIONIZE
# ============================================================================

print("=" * 78)
print("STEP 2 — Sessionize (30-minute inactivity gap)")
print("=" * 78)

# df_clean is already sorted by (visitorid, timestamp) because it was
# filtered from df, which was sorted earlier. Re-sort anyway to be explicit
# and safe against any future reordering of the cleaning step above.
df_clean = df_clean.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)

df_clean["prev_ts"] = df_clean.groupby("visitorid")["timestamp"].shift(1)
df_clean["gap_ms"] = df_clean["timestamp"] - df_clean["prev_ts"]

# A new session starts on a visitor's very first event (gap_ms is NaN) or
# whenever the gap to the previous event exceeds the 30-minute timeout.
df_clean["new_session"] = (
    df_clean["gap_ms"].isna() | (df_clean["gap_ms"] > SESSION_GAP_MS)
).astype(int)

# Cumulative sum per visitor gives each visitor's sessions numbers 1, 2, 3, ...
df_clean["session_id"] = df_clean.groupby("visitorid")["new_session"].cumsum()
df_clean["session_key"] = (
    df_clean["visitorid"].astype(str) + "_" + df_clean["session_id"].astype(str)
)

n_sessions = df_clean["session_key"].nunique()
print(f"Sessions reconstructed: {n_sessions:,}")
print()


# ============================================================================
# STEP 3 — CART POPULATION
# ============================================================================

print("=" * 78)
print("STEP 3 — Keep only sessions containing at least one add-to-cart")
print("=" * 78)

# Which sessions contain at least one addtocart event?
has_cart = (
    df_clean.groupby("session_key")["event"]
    .apply(lambda s: (s == "addtocart").any())
)
cart_session_keys = set(has_cart[has_cart].index)
n_cart_sessions = len(cart_session_keys)

print(f"Cart sessions (>=1 addtocart): {n_cart_sessions:,}")

# Restrict all further work to events belonging to cart sessions only.
df_cart = df_clean[df_clean["session_key"].isin(cart_session_keys)].copy()

# Informational-only class balance: "did this session ever see a transaction
# ANYWHERE, before or after the cart action?". This is NOT the modelling
# label (see Addition 1 below) — it is only printed here to describe the
# raw population, exactly like the old Stage-1 report did.
session_has_txn = (
    df_cart.groupby("session_key")["event"].apply(lambda s: (s == "transaction").any())
)
n_txn_sessions = int(session_has_txn.sum())
n_no_txn_sessions = n_cart_sessions - n_txn_sessions
print(
    f"(Informational only, NOT the model label) sessions with a transaction "
    f"anywhere: {n_txn_sessions:,} ({n_txn_sessions / n_cart_sessions * 100:.2f}%), "
    f"without: {n_no_txn_sessions:,} ({n_no_txn_sessions / n_cart_sessions * 100:.2f}%)"
)
print()


# ============================================================================
# STEP 4 — CUTOFFS  (the core of this script)
# ============================================================================
#
# ADDITION 1 — LABEL DEFINITION (replaces the old session-level label)
# ---------------------------------------------------------------------------
# The old label ("session contains a transaction") is wrong once we start
# cutting a session into multiple (session, k) rows: a visitor can buy item A
# and then abandon item B within the SAME session. The label must therefore
# be a property of the (session, cutoff) pair, not of the session as a whole:
#
#     y = 1  if a transaction occurs STRICTLY AFTER the cutoff timestamp
#     y = 0  otherwise
#
# A session that has ALREADY converted at-or-before the cutoff has left the
# "risk set" (there is nothing left to predict — it already happened) and
# that row is dropped entirely rather than truncated or relabelled.
#
# For each cart session:
#   t0 = timestamp of the session's FIRST addtocart event
#   for k in CUTOFFS:
#     cutoff_ts = timestamp of the k-th event after t0 (k=0 => t0 itself)
#     require: session has >= k events strictly after t0
#     require: no transaction event at-or-before cutoff_ts
#     prefix   = every event in the session with timestamp <= cutoff_ts
# ============================================================================

print("=" * 78)
print("STEP 4-5 — Build per-cutoff prefixes and point-in-time features")
print("=" * 78)

# Counters for the reporting Addition 1 asks for. Filled in as we iterate.
n_txn_before_first_cart = 0            # sessions disqualified before k=0 even starts
removed_already_converted_by_k = {k: 0 for k in CUTOFFS}
emitted_rows_by_k = {k: 0 for k in CUTOFFS}

output_rows = []

# One session at a time. Un-optimised by design — a Python-level loop over
# ~44k sessions, each holding only a handful of events, so a viva student can
# point at any single line and explain exactly what it does.
for session_key, g in df_cart.groupby("session_key", sort=False):
    g = g.sort_values("timestamp").reset_index(drop=True)

    visitorid = g["visitorid"].iloc[0]
    timestamps = g["timestamp"].values
    events = g["event"].values
    itemids = g["itemid"].values
    is_txn = events == "transaction"
    is_cart = events == "addtocart"
    is_view = events == "view"

    # t0 = timestamp of the FIRST addtocart event in this session.
    first_cart_idx = np.flatnonzero(is_cart)[0]
    t0 = timestamps[first_cart_idx]

    # All transaction timestamps in the FULL session. This uses information
    # beyond the cutoff on purpose — the LABEL is allowed to see the future
    # (that is what makes it a label), it is only the FEATURES (step 5) that
    # must never look past the cutoff.
    txn_timestamps = timestamps[is_txn]

    # If a transaction already happened before the first addtocart, the
    # session has converted before it even enters our risk set: every cutoff
    # (which is always >= t0) will have a transaction at-or-before it, so the
    # session produces zero rows at every k. Report it separately instead of
    # letting it silently disappear into the per-k "already converted" counts.
    if len(txn_timestamps) > 0 and txn_timestamps.min() < t0:
        n_txn_before_first_cart += 1
        continue  # nothing to emit for this session at any k

    # Events strictly after t0, sorted by timestamp — used to find the k-th
    # event after t0 and to check "at least k events after t0".
    after_mask = timestamps > t0
    after_timestamps = timestamps[after_mask]

    for k in CUTOFFS:
        # --- requirement: session must have >= k events after t0 ----------
        if k > 0 and len(after_timestamps) < k:
            continue  # not enough runway yet to reach this cutoff

        # --- locate the cutoff timestamp -----------------------------------
        if k == 0:
            cutoff_ts = t0
        else:
            cutoff_ts = after_timestamps[k - 1]  # k-th event after t0 (1-indexed)

        # --- requirement: session must not already have converted ---------
        already_converted = np.any(txn_timestamps <= cutoff_ts)
        if already_converted:
            removed_already_converted_by_k[k] += 1
            continue

        # --- the prefix: every event at or before the cutoff --------------
        prefix_mask = timestamps <= cutoff_ts
        prefix_ts = timestamps[prefix_mask]
        prefix_events = events[prefix_mask]
        prefix_items = itemids[prefix_mask]

        # --- FORBIDDEN CHECK (Step 6) --------------------------------------
        # A prefix must never contain a transaction event. We just proved
        # `already_converted` is False (no transaction at or before cutoff_ts),
        # so this is a redundant, cheap, fail-loud safety net.
        assert not (prefix_events == "transaction").any(), (
            f"Prefix for session {session_key}, k={k} contains a transaction "
            "event — this must never happen."
        )

        # ====================================================================
        # STEP 5 — FEATURES, computed from the prefix ONLY
        # ====================================================================

        view_count = int((prefix_events == "view").sum())
        addtocart_count = int((prefix_events == "addtocart").sum())
        unique_items_count = int(pd.Series(prefix_items).dropna().nunique())

        # addtocart_count is always >= 1 because the prefix always includes
        # at least the t0 addtocart event itself (cutoff_ts >= t0 for all k).
        view_to_cart_ratio = view_count / addtocart_count

        first_event_ts = prefix_ts[0]          # prefix's earliest event
        seconds_since_first_event = (cutoff_ts - first_event_ts) / 1000.0
        seconds_since_first_cart = (cutoff_ts - t0) / 1000.0

        # Timestamp of the LAST addtocart event within the prefix (there is
        # always at least one: t0 itself).
        prefix_cart_ts = prefix_ts[prefix_events == "addtocart"]
        last_cart_ts = prefix_cart_ts[-1]
        seconds_since_last_cart = (cutoff_ts - last_cart_ts) / 1000.0

        # Second-to-last event in the prefix, i.e. the event immediately
        # before the cutoff event itself. NaN if the prefix has only 1 event
        # (possible at k=0 when the addtocart is the very first event of
        # the session — there is no "previous" event to measure from).
        if len(prefix_ts) >= 2:
            seconds_since_previous_event = (cutoff_ts - prefix_ts[-2]) / 1000.0
        else:
            seconds_since_previous_event = np.nan

        # Mean gap between consecutive prefix events, in seconds. NaN if the
        # prefix has fewer than 2 events (no gaps to average).
        if len(prefix_ts) >= 2:
            gaps_sec = np.diff(prefix_ts) / 1000.0
            inter_event_interval_mean = float(gaps_sec.mean())
        else:
            inter_event_interval_mean = np.nan

        # Number of prefix events that occur strictly after t0. By
        # construction this equals k, but we compute it from the prefix
        # itself (not by writing "= k") to keep every feature demonstrably
        # derived only from the prefix data, per Step 6.
        events_after_cart = int((prefix_ts > t0).sum())

        # Items already added to cart by the cutoff (from prefix addtocart
        # events only), and how many prefix VIEW events touched those items —
        # i.e. "the visitor kept looking at something already in their cart".
        cart_item_ids = set(pd.Series(prefix_items[prefix_events == "addtocart"]).dropna())
        prefix_view_items = pd.Series(prefix_items[prefix_events == "view"]).dropna()
        repeated_views_of_cart_items = int(prefix_view_items.isin(cart_item_ids).sum())

        cutoff_dt = pd.to_datetime(int(cutoff_ts), unit="ms")
        hour_of_day = cutoff_dt.hour
        day_of_week = cutoff_dt.dayofweek  # Monday=0 ... Sunday=6

        # ====================================================================
        # LABEL (Addition 1) — property of (session, cutoff), not of the
        # session as a whole. Uses the FULL session's transaction timestamps
        # (future information), which is correct: the label is what we are
        # trying to predict, it is allowed to see what actually happened next.
        # ====================================================================
        y = int(np.any(txn_timestamps > cutoff_ts))

        output_rows.append({
            "session_key": session_key,
            "visitorid": visitorid,
            "cutoff_k": k,
            "view_count": view_count,
            "addtocart_count": addtocart_count,
            "unique_items_count": unique_items_count,
            "view_to_cart_ratio": view_to_cart_ratio,
            "seconds_since_first_event": seconds_since_first_event,
            "seconds_since_first_cart": seconds_since_first_cart,
            "seconds_since_last_cart": seconds_since_last_cart,
            "seconds_since_previous_event": seconds_since_previous_event,
            "inter_event_interval_mean": inter_event_interval_mean,
            "events_after_cart": events_after_cart,
            "repeated_views_of_cart_items": repeated_views_of_cart_items,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "y": y,
        })

        emitted_rows_by_k[k] += 1

prefix_df = pd.DataFrame(output_rows)

print()
print(f"Cart sessions dropped entirely (transaction before first addtocart): "
      f"{n_txn_before_first_cart:,}")
print()
print("Rows removed by the 'already converted' rule, per cutoff k:")
for k in CUTOFFS:
    print(f"  k={k:>2}: removed {removed_already_converted_by_k[k]:,}")
print()


# ============================================================================
# STEP 7 — ASSERTIONS (fail loudly)
# ============================================================================

print("=" * 78)
print("STEP 7 — Assertions")
print("=" * 78)

# 'event_count' (a whole-session aggregate) must never appear as a feature —
# this was leakage mechanism (a) from the audit.
assert "event_count" not in prefix_df.columns, (
    "'event_count' must never be a feature column — this is the leakage the "
    "audit flagged."
)

# Every row's prefix must obey cutoff_ts as its latest possible timestamp.
# We do not store the raw prefix arrays in the output, so we re-derive the
# guarantee here directly from how seconds_since_first_event / cutoff_ts
# relate: seconds_since_first_event must always be >= 0 (a prefix event can
# never be timestamped before the session's first event), and
# seconds_since_previous_event / seconds_since_last_cart must never be
# negative either (a prefix event can never be timestamped after cutoff_ts).
assert (prefix_df["seconds_since_first_event"] >= 0).all(), (
    "Found a negative seconds_since_first_event — a prefix event occurred "
    "before the session's own first event, which is impossible."
)
assert (prefix_df["seconds_since_last_cart"] >= 0).all(), (
    "Found a negative seconds_since_last_cart — a prefix's addtocart event "
    "timestamp is after the cutoff timestamp."
)
assert (
    prefix_df["seconds_since_previous_event"].dropna() >= 0
).all(), (
    "Found a negative seconds_since_previous_event — a prefix event occurred "
    "after the cutoff timestamp."
)

# No prefix may contain a transaction event — already asserted inline during
# construction (per row), this is the aggregate confirmation.
# (If the inline assert above ever fired, the script would have already
# stopped, so reaching this line means every single row passed.)
print("Confirmed: no prefix ever contained a transaction event (checked per-row above).")

print()
print("Rows and positive rate per cutoff k:")
for k in CUTOFFS:
    sub = prefix_df[prefix_df["cutoff_k"] == k]
    n_rows = len(sub)
    pos_rate = sub["y"].mean() if n_rows > 0 else float("nan")
    print(f"  k={k:>2}: rows={n_rows:>7,}   positive_rate={pos_rate:.4f}")

print()
print(f"Total output rows: {len(prefix_df):,}")
print("All assertions passed.")
print()


# ============================================================================
# OUTPUT
# ============================================================================

FEATURE_COLS = [
    "view_count", "addtocart_count", "unique_items_count", "view_to_cart_ratio",
    "seconds_since_first_event", "seconds_since_first_cart",
    "seconds_since_last_cart", "seconds_since_previous_event",
    "inter_event_interval_mean", "events_after_cart",
    "repeated_views_of_cart_items", "hour_of_day", "day_of_week",
]
FINAL_COLS = ["session_key", "visitorid", "cutoff_k"] + FEATURE_COLS + ["y"]

prefix_df = prefix_df[FINAL_COLS]
prefix_df.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved: {OUTPUT_PATH}  ({len(prefix_df):,} rows, {len(prefix_df.columns)} columns)")
