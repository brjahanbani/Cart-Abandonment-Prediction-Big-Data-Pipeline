"""
clean_events.py
================

Shared bot-removal + sessionization logic, extracted from the point-in-time
pipeline (3-Decision-Time/build_prefix_dataset.py) so it lives in exactly one
place instead of being duplicated by every script that needs a clean,
sessionized event log.

This module is NOT used by anything under 2-Completed-Session/ — that
lineage is frozen and keeps its own inline copy of this same logic, byte for
byte, so its published numbers never depend on a file outside that folder.
This module only backs the decision-time (point-in-time) lineage.

Behaviour is identical to the frozen copy: same 30-minute session gap, same
bot thresholds (velocity > 500 events/hour OR inter-event-interval CV <
0.05, minimum 20 events to trust a CV estimate).
"""

import numpy as np
import pandas as pd

SESSION_GAP_MS = 30 * 60 * 1000

BOT_VELOCITY_THRESHOLD = 500
BOT_CV_THRESHOLD = 0.05
BOT_MIN_EVENTS = 20


def _compute_cv(group):
    """CV = std / mean of inter-event gaps. Bots fire at near-constant
    intervals (CV close to 0); humans do not."""
    intervals = group["timestamp"].sort_values().diff().dropna()
    if len(intervals) < BOT_MIN_EVENTS:
        return np.nan
    mean_iv = intervals.mean()
    if mean_iv == 0:
        return 0.0
    return intervals.std() / mean_iv


def load_raw_events(raw_path):
    """Load the raw Retail Rocket events.csv with explicit dtypes."""
    return pd.read_csv(
        raw_path,
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


def remove_bots(df_raw):
    """
    Sort by (visitorid, timestamp) and drop every event belonging to a
    visitor flagged as a bot by either the velocity or CV rule.

    Returns (df_clean, bot_stats) where bot_stats is a dict with
    n_bot_visitors, n_bot_events, and transactions_removed_by_bot_filter —
    everything a caller needs for fail-loud input assertions.
    """
    df = df_raw.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)

    visitor_stats = df.groupby("visitorid").agg(
        total_events=("event", "count"),
        span_ms=("timestamp", lambda x: x.max() - x.min()),
    ).reset_index()

    visitor_stats["span_hours"] = (visitor_stats["span_ms"] / 3_600_000).clip(lower=1 / 60)
    visitor_stats["events_per_hour"] = visitor_stats["total_events"] / visitor_stats["span_hours"]
    visitor_stats["is_bot_velocity"] = visitor_stats["events_per_hour"] > BOT_VELOCITY_THRESHOLD

    visitor_cv = (
        df.groupby("visitorid")
        .apply(_compute_cv, include_groups=False)
        .rename("interval_cv")
        .reset_index()
    )
    visitor_stats = visitor_stats.merge(visitor_cv, on="visitorid")
    visitor_stats["is_bot_cv"] = (
        visitor_stats["interval_cv"].notna() & (visitor_stats["interval_cv"] < BOT_CV_THRESHOLD)
    )

    visitor_stats["is_bot"] = visitor_stats["is_bot_velocity"] | visitor_stats["is_bot_cv"]
    bot_visitors = set(visitor_stats.loc[visitor_stats["is_bot"], "visitorid"])

    raw_event_mix = df["event"].value_counts().to_dict()
    n_bot_visitors = len(bot_visitors)
    n_bot_events = int(df["visitorid"].isin(bot_visitors).sum())

    df_clean = df[~df["visitorid"].isin(bot_visitors)].copy()
    clean_event_mix = df_clean["event"].value_counts().to_dict()

    transactions_removed_by_bot_filter = (
        raw_event_mix.get("transaction", 0) - clean_event_mix.get("transaction", 0)
    )

    bot_stats = {
        "n_bot_visitors": n_bot_visitors,
        "n_bot_events": n_bot_events,
        "transactions_removed_by_bot_filter": transactions_removed_by_bot_filter,
        "clean_event_mix": clean_event_mix,
    }
    return df_clean, bot_stats


def sessionize(df_clean):
    """
    Sort by (visitorid, timestamp) and assign session_key using a 30-minute
    inactivity gap. Adds prev_ts / gap_ms / new_session / session_id /
    session_key columns.
    """
    df_clean = df_clean.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)

    df_clean["prev_ts"] = df_clean.groupby("visitorid")["timestamp"].shift(1)
    df_clean["gap_ms"] = df_clean["timestamp"] - df_clean["prev_ts"]
    df_clean["new_session"] = (
        df_clean["gap_ms"].isna() | (df_clean["gap_ms"] > SESSION_GAP_MS)
    ).astype(int)
    df_clean["session_id"] = df_clean.groupby("visitorid")["new_session"].cumsum()
    df_clean["session_key"] = (
        df_clean["visitorid"].astype(str) + "_" + df_clean["session_id"].astype(str)
    )
    return df_clean


def clean_and_sessionize(raw_path):
    """Convenience wrapper: load raw -> remove bots -> sessionize.

    Returns (df_clean_sessionized, bot_stats).
    """
    df_raw = load_raw_events(raw_path)
    df_clean, bot_stats = remove_bots(df_raw)
    df_clean = sessionize(df_clean)
    return df_raw, df_clean, bot_stats
