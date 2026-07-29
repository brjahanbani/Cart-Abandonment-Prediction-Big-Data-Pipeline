# Prefix Dataset Rebuild — Decision Log

Running record of the point-in-time feature rebuild (leakage fix). Updated after
every prompt/answer exchange so work can resume cold in a new session.

Context: an audit found two leakage mechanisms in the old `session_features.csv` /
`cart_session_features.csv` pipeline:
(a) `transaction_count` recoverable as `event_count - view_count - addtocart_count`
(b) `session_age` / `cart_recency` used the session's final event as endpoint
    (leaks the future into features computed "at" cart time)

Goal: `build_prefix_dataset.py` — a single, from-raw, point-in-time-correct
feature pipeline producing `prefix_dataset.parquet`.

---

## 2026-07-29 — Step 0: repo inspection (prompt 1)

**Asked:** inspect the repo before writing any code — what preprocessing/training
scripts exist, what they do, whether `events.csv` path still works.

**Found:**
- `0-Offline/Data Cleaning & Feature Engineering/stage1_data_cleaning_feature_engineering.py`
  — original Stage 1 notebook-style script. Raw path is `1-events.csv` in that same
  folder (not `events.csv` at repo root — the literal `DATA_PATH = 'events.csv'` in
  the script does not resolve as-is). Does sessionization (30 min gap), bot removal
  (velocity >500 ev/hr OR interval CV <0.05), then whole-session features including
  `transaction_count` and endpoint-leaky `session_age_seconds` / `recency_last_cart_seconds`.
  Exports `events_clean.csv` and `session_features.csv`.
- `0-Offline/offline_cart_abandonment_prediction.py` — separate 4-model (LR/DT/RF/XGB)
  comparison script, reads `cart_session_features.csv` directly, uses the raw
  (leaky) `session_age_seconds`/`recency_last_cart_seconds` columns as-is. Only
  drops `transaction_count`. **This is a third, more-contaminated results lineage** —
  see prompt 2 answer below.
- `3-Training/train_logistic_regression_baseline.py`, `train_xgboost_baseline.py` —
  load `1-Data/events_clean.csv` + `1-Data/cart_session_features.csv`, recompute
  timing features from non-transaction events only (partial leakage patch, still
  whole-session shaped, not cutoff-based).
- `3-Training/train_lstm_model.py` — hybrid LSTM + 7 dense features.
- `5-Pipeline/*` — Kafka/Spark streaming pipeline (Stage 2), not relevant to this rebuild.
- `1-Data/events_clean.csv`, `1-Data/cart_session_features.csv` — the actual
  artifacts consumed downstream.

**Decision:** new script will read raw from an explicit constant
`'0-Offline/Data Cleaning & Feature Engineering/1-events.csv'` and re-run cleaning
itself (no copy/symlink), so the whole pipeline is reproducible from one file.

---

## 2026-07-29 — Pre-build verification (prompt 2)

User asked for three checks before writing code, plus a skim of
`offline_cart_abandonment_prediction.py`.

1. **Row count.** Raw `1-events.csv` has **2,756,101** data rows (excl. header),
   confirmed via raw line count and `pandas.read_csv` length, agreeing both times.
   → User's papers (2,756,101) are correct. My earlier report of 2,756,102 in the
   Step-0 reply was a miscount and is wrong — use 2,756,101 going forward.

2. **`events_clean.csv` contents.** Confirmed: still contains all three event
   types (view 2,663,881 / addtocart 69,272 / transaction 22,457), timestamps
   still raw int64 milliseconds (~1.43e12 scale), 2,755,610 total rows. Safe to
   use for label assignment and for dropping already-converted sessions, provided
   transactions are excluded again before any *feature* touches the prefix.

3. **Cross-check.** Re-ran the stage-1 cleaning logic (sessionize + bot rule)
   fresh from raw `1-events.csv` and diffed against `events_clean.csv`:
   - Bot visitors flagged: 35, bot events removed: 491 (matches user's earlier
     published numbers exactly)
   - Events after cleaning: 2,755,610 == `events_clean.csv` row count
   - Row-for-row match on `(visitorid, timestamp, event, itemid)`: **all equal**
   → No population drift. `events_clean.csv` is a faithful, reproducible product
   of the raw file.

4. **Third variant check.** `offline_cart_abandonment_prediction.py` is confirmed
   as a **third, non-equivalent results lineage** (distinct from both the Stage-1
   notebook's embedded 5-fold CV and the `3-Training` baselines): it uses the
   stored whole-session `session_age_seconds`/`recency_last_cart_seconds` columns
   untouched (full leakage mechanism (b) present), only strips `transaction_count`,
   and evaluates 4 models on a single 80/20 split rather than CV. It should not be
   cited as a "corrected" baseline — it's the most leakage-contaminated of the three.

**Status:** all pre-build checks passed. Cleared to write `build_prefix_dataset.py`.

---

## 2026-07-29 — Build confirmed, two additions (prompt 3)

User confirmed proceeding, with two additions to the original spec:

**Addition 1 — label redefinition.** The old label ("session contains a
transaction anywhere") is wrong once a session is cut into multiple
`(session, k)` rows — a visitor can buy item A and abandon item B in the same
session. New rule: `y = 1` iff a transaction occurs **strictly after** the
cutoff timestamp; the label is a property of `(session, cutoff_k)`, not of
the session. Required extra reporting: cart sessions with a transaction
*before* their first addtocart, rows removed by the "already converted" rule
per k, and final rows/positive-rate per k (no silent drops).

**Addition 2 — input assertions.** Fail loudly if: raw row count ≠
2,756,101; post-clean event count ≠ 2,755,610; bot removal ≠ 35 visitors /
491 events; transactions removed by bot filter ≠ 0; post-clean event mix ≠
view 2,663,881 / addtocart 69,272 / transaction 22,457. (Verified all five
against the raw file before writing assertions — see prompt 2 cross-check.)

**Built:** `build_prefix_dataset.py` (repo root). Single-file, pandas-only,
heavily commented, deliberately un-optimised (Python loop per cart session)
so every line is explainable in a viva. Reads raw `1-events.csv` from the
constant path, re-runs cleaning + sessionization in-script, restricts to cart
sessions, then per session per `k` in `[0,1,2,3,5,10]`:
- `t0` = first addtocart timestamp; cutoff = k-th event after `t0` (k=0 → t0
  itself)
- drop session-k entirely (not truncate) if a transaction occurred at/before
  cutoff, or if fewer than k events exist after `t0`
- sessions with a transaction *before* `t0` are dropped at every k and
  counted separately
- prefix = events with `timestamp <= cutoff_ts`; all 13 features computed
  only from the prefix (view/cart counts, ratios, four seconds-since-*
  timings, inter-event mean, events_after_cart, repeated_views_of_cart_items,
  hour_of_day, day_of_week)
- label computed from the full session's transaction timestamps (future
  info is correct for a label, forbidden only for features)
- per-row inline assert that no prefix ever contains a transaction event;
  aggregate asserts that `event_count` is not an output column and that
  `seconds_since_*` values are never negative (equivalent to "prefix never
  exceeds cutoff")

**Run 1** hit only an environment gap: `pyarrow`/`fastparquet` missing for
`to_parquet`. Installed `pyarrow` and reran — all logic/assertions had
already passed on run 1 before the write step.

**Actual output (run 2, successful):**
- Bot removal: 35 visitors / 491 events removed, 0 transactions among them —
  matches all 5 input assertions
- Sessions reconstructed: 1,761,640
- Cart sessions (≥1 addtocart): 43,917
- Informational-only "has transaction anywhere" split: 11,932 (27.17%) /
  31,985 (72.83%) — NOT the model label, printed for context only
- Cart sessions dropped entirely (transaction before first addtocart): **172**
- Rows removed by "already converted" rule per k: k=0: 0, k=1: 7,865,
  k=2: 4,229, k=3: 3,160, k=5: 2,182, k=10: 1,337
- Rows / positive rate per k: k=0: 43,745 rows (26.88%), k=1: 19,782
  (19.69%), k=2: 12,276 (20.68%), k=3: 8,513 (21.14%), k=5: 4,874 (22.04%),
  k=10: 1,846 (23.13%)
- **Total output rows: 91,036**, saved to `prefix_dataset.parquet` (17 columns)
- All Step 7 assertions passed (no `event_count` column, no negative
  seconds-since-* values, no transaction ever inside a prefix)

Note the positive rate dips from k=0 to k=1 (26.88% → 19.69%) then rises
again through k=10 (23.13%) — expected under the new label: at k=0 many
"already about to convert" sessions are still in the risk set, then the
fastest converters get removed by the "already converted" rule as k grows,
while the survivors at higher k are a self-selected slower-converting
population with a rising share that still eventually buys.

**Status:** `build_prefix_dataset.py` complete and verified. `prefix_dataset.parquet`
exists at repo root.

## Open items / next steps
- [ ] Decide whether `pyarrow` should be pinned in a requirements file so
      the parquet write doesn't fail fresh again.
- [ ] Point `3-Training` scripts (or a new training script) at
      `prefix_dataset.parquet` instead of the old leaky `cart_session_features.csv`.
- [ ] Re-run the leakage-audit-style sanity checks (AUC, feature correlations)
      on the new prefix dataset to confirm the point-in-time features still
      carry predictive signal without the removed leakage paths.
