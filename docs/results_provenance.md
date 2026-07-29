# Results Provenance

One row per published number. Every row was confirmed by reading the exact
script listed — no number below is guessed from a table caption alone. Paths
are current post-reorg (`2-Completed-Session/`, `3-Decision-Time/`); see
[prefix_rebuild_log.md](prefix_rebuild_log.md) for the reorg itself.

**"Rerun value (2026-07)"** is the number this session's actual re-execution
produced, on the environment recorded in [Environment](#environment) below.
No published number anywhere in this repo or these docs was changed to match
a rerun — where they differ, both values are shown side by side.

| Published figure | Script (exact path) | Population | Split | Leakage status | Rerun value (2026-07) | Reproducible today? |
|---|---|---|---|---|---|---|
| Final Report Table 6 — LR / DT / RF / XGB, 4 models | `2-Completed-Session/offline_cart_abandonment_prediction.py` | 43,917 cart sessions (≥1 addtocart), loaded straight from `cart_session_features.csv` | single 80/20 stratified `train_test_split`, `random_state=42` | **fully leaky** — uses the stored `session_age_seconds` / `recency_last_cart_seconds` columns untouched (computed from each session's *final* event, which for purchase sessions is the transaction itself); only `transaction_count` is dropped as a feature | XGBoost AUC 0.9945 (published 0.9946, diff −0.0001) | Yes |
| Final Report Table 7 — leakage-reduced AUC column | `2-Completed-Session/train_logistic_regression_baseline.py` and `2-Completed-Session/train_xgboost_baseline.py` | same 43,917 cart sessions, merged with `events_clean.csv` | single 80/20 stratified `train_test_split`, `SEED = 42` | **partial patch** — `session_age_seconds`, `inter_event_interval_mean`, `recency_last_cart_seconds` are recomputed from `events[events.event != "transaction"]` (drops the transaction event only), but the recomputed values still span the *entire* session up to its last non-transaction event — not a point-in-time cutoff, so a session's later browsing can still leak into a feature timestamped as if known earlier | XGBoost AUC 0.6863 (published 0.6860, diff +0.0003) | Yes |
| Final Report Table 8 — LSTM iterations 1–4 | **none found** | — | — | — | — | **UNTRACED — permanent.** No sweep code exists anywhere in this repo. Not attempting reconstruction (see gap below). |
| Final Report Table 9 — LSTM classification report | `2-Completed-Session/train_lstm_model.py` | 43,917 cart sessions matched to an event sequence (view/addtocart only, padded/truncated to 50 steps) + 7 session features | single 80/20 stratified `train_test_split`, `SEED = 42` | **partial patch** — same three timing features recomputed from non-transaction events as the LR/XGBoost baselines (see Table 7 row); the LSTM's own event-sequence branch only ever sees `view`/`addtocart` events (never `transaction`), so that branch is point-in-time-safe, but the dense-feature branch inherits the Table 7 partial-patch leakage | AUC 0.6654 (exact match to prior run) | Partially — runs end-to-end and prints one classification report, but cannot confirm which of "iterations 1–4" it corresponds to (only one hyperparameter configuration exists in the repo) |
| Workshop paper Table 3, row 1 — AUC 0.9969, 1,761,640 sessions, 5-fold | `2-Completed-Session/stage1_data_cleaning_feature_engineering.py`, Section 7 ("Predictive Signal Validation") | **all** 1,761,640 reconstructed sessions (not just cart sessions) — includes the 78.3% of sessions with only 1 event. Confirmed exact code path: `session_features = df_clean.groupby('session_key').apply(build_session_features)` (line 462) runs over every session in the bot-cleaned event log, with no cart-session filter applied anywhere before `X = sf_model[FEATURES].fillna(0).values` (line 548) | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on Logistic Regression, `class_weight='balanced'` | **fully leaky** — the 8-feature set fed to the CV explicitly includes `event_count` (a whole-session total) plus `session_age_seconds` and `recency_last_cart_seconds`, both computed from each session's *final* event | AUC-ROC 0.9969 ± 0.0001, sessions 1,761,640 (exact match) | Yes |
| Workshop paper Table 4 — seed-42 benchmark, `original_completed`, ROC-AUC | `2-Completed-Session/reanalysis_controlled_audit.py` (reconstructed — see [Reanalysis reconstruction](#reanalysis-reconstruction) below) | 43,917 cart sessions, `original_completed` feature config (7 features: view_count, addtocart_count, view_to_cart_ratio, unique_items_count, session_age_seconds, inter_event_interval_mean, recency_last_cart_seconds) | single stratified 80/20 split, seed=42; LR/XGB/DT/RF all on the same split | **partial patch** — same leakage profile as Table 7 (timing features span whole session, not point-in-time) | LR 0.8249 (pub 0.825), DT 0.9340 (pub 0.934), RF 0.9837 (pub 0.984), XGB 0.9947 (pub 0.995) — all within ±0.0003 | Yes — reconstructed and reproduces to within tolerance |
| Workshop paper Table 5 — 4 feature configs × {LR, XGB} × 5 seeds, ROC-AUC & PR-AUC | `2-Completed-Session/reanalysis_controlled_audit.py` (reconstructed — see below) | 43,917 cart sessions, 4 feature configs (`target_proxy`, `original_completed`, `timing_only`, `counts_only`) | stratified 80/20, seeds 11/22/33/42/55; 99th-percentile cap fit on train only; training-only standardization for LR | **fully leaky** (`target_proxy` — includes `event_count`, a whole-session total) down to **partial patch** (`original_completed`, `timing_only`, `counts_only` — none touch `event_count` but still use whole-session timing features) | 7 of 8 ROC-AUC cells match to ≤0.0007; 7 of 8 PR-AUC cells match to ≤0.0005; **one cell differs beyond tolerance**: `timing_only` XGB PR-AUC = 0.6563 vs published 0.6530 (diff +0.0033) — reported, not adjusted | Yes — see full per-cell diff table below |
| `prefix_dataset.parquet` — 43,745 rows at k=0, etc. | `3-Decision-Time/build_prefix_dataset.py` | 43,917 cart sessions minus 172 dropped (transaction occurred *before* the session's first addtocart) = 43,745 eligible at k=0; shrinks per cutoff k as sessions convert or run out of events | no train/test split — this script only builds features, one row per `(session_key, cutoff_k)` | **point-in-time** — every feature is computed only from events at or before the cutoff; the label (`y`) is the only place future information is used, by design | k=0 → 43,745 rows (26.88% positive), k=1 → 19,782, k=2 → 12,276, k=3 → 8,513, k=5 → 4,874, k=10 → 1,846 (exact match) | Yes |

## Reanalysis reconstruction

Workshop paper Tables 4 and 5 previously traced to no script in this repo —
they are the paper's primary evidence, from a controlled reanalysis that was
never committed. `2-Completed-Session/reanalysis_controlled_audit.py` was
written directly from the paper's stated protocol (four feature configs,
five seeds, training-only 99th-percentile cap and standardization,
`scale_pos_weight` for XGBoost) and reads the same frozen
`1-Data/completed-session/cart_session_features.csv` as the rest of this
lineage. It asserts Equation 2
(`transaction_count == event_count - view_count - addtocart_count`) and
Equation 3 (`y_s = 1[transaction_count_s > 0]`, i.e.
`has_transaction == (transaction_count > 0)`) before fitting anything. Both
hold for every row.

### Full Table 5 diff (ROC-AUC, mean ± sd across 5 seeds)

| config | model | reproduced | published | diff | flag |
|---|---|---|---|---|---|
| target_proxy | LR | 0.9944 ± 0.0004 | 0.9944 ± 0.0004 | +0.0000 | ok |
| target_proxy | XGB | 0.9997 ± 0.0001 | 0.9997 ± 0.0001 | −0.0000 | ok |
| original_completed | LR | 0.8306 ± 0.0036 | 0.8306 ± 0.0036 | +0.0000 | ok |
| original_completed | XGB | 0.9948 ± 0.0004 | 0.9947 ± 0.0005 | +0.0001 | ok |
| timing_only | LR | 0.8006 ± 0.0032 | 0.8006 ± 0.0032 | +0.0000 | ok |
| timing_only | XGB | 0.8695 ± 0.0031 | 0.8688 ± 0.0035 | +0.0007 | ok |
| counts_only | LR | 0.6061 ± 0.0035 | 0.6061 ± 0.0035 | +0.0000 | ok |
| counts_only | XGB | 0.6061 ± 0.0035 | 0.6065 ± 0.0024 | −0.0004 | ok |

### Full Table 5 diff (PR-AUC)

| config | model | reproduced | published | diff | flag |
|---|---|---|---|---|---|
| target_proxy | LR | 0.9525 | 0.9525 | +0.0000 | ok |
| target_proxy | XGB | 0.9992 | 0.9992 | −0.0000 | ok |
| original_completed | LR | 0.6123 | 0.6123 | +0.0000 | ok |
| original_completed | XGB | 0.9877 | 0.9872 | +0.0005 | ok |
| timing_only | LR | 0.5476 | 0.5476 | +0.0000 | ok |
| **timing_only** | **XGB** | **0.6563** | **0.6530** | **+0.0033** | **DIFFERS** |
| counts_only | LR | 0.3895 | 0.3895 | +0.0000 | ok |
| counts_only | XGB | 0.3943 | 0.3943 | −0.0000 | ok |

Only one cell out of 16 in Table 5 exceeds the ±0.002 tolerance:
`timing_only` XGBoost PR-AUC (reproduced 0.6563 vs. published 0.6530, diff
+0.0033). Every ROC-AUC cell and every other PR-AUC cell matches to within
±0.0007. This is reported as-is — the script was not adjusted to close this
gap, and no published number was changed.

Metric JSON for all 42 runs (4 configs × {LR, XGB} × 5 seeds, plus DT/RF at
seed 42 for `original_completed`) is written to `5-Results/`, named
`completed-session__<model>__stratified-80-20__<config>__s<seed>.json`.

## Why the lineages disagree

Four scripts each own a distinct, non-interchangeable set of numbers:

- **`2-Completed-Session/stage1_data_cleaning_feature_engineering.py`** owns the Table 3 row-1 lineage (AUC 0.9969, all 1,761,640 sessions, 5-fold CV).
- **`2-Completed-Session/offline_cart_abandonment_prediction.py`** owns the Table 6 lineage (4-model comparison, 43,917 cart sessions, single 80/20 split).
- **`2-Completed-Session/train_logistic_regression_baseline.py`** / **`train_xgboost_baseline.py`** own the Table 7 lineage (same 43,917 cart sessions, single 80/20 split, but with three timing features patched).
- **`2-Completed-Session/reanalysis_controlled_audit.py`** owns the Table 4 / Table 5 lineage (same 43,917 cart sessions, but a controlled sweep across 4 feature configs and 5 seeds with training-only capping/scaling).

They disagree because each one answers a different question with a
different population, feature definition, and leakage level, not because
any one of them is "more correct" at reproducing another's number: Table 3
scores a Logistic Regression on the entire session population (mostly
one-event sessions) using whole-session aggregates that include the literal
event count and end-of-session timing; Table 6 restricts to the 43,917 cart
sessions but still uses those same end-of-session timing columns untouched;
Table 7 restricts to the same cart sessions but strips the transaction event
out of the three timing features before recomputing them (a partial fix,
not a point-in-time cutoff); Table 4/5 restricts to the same cart sessions
again but runs a controlled ablation across four feature subsets to isolate
exactly how much of the signal comes from the target-proxy features versus
the timing features versus the count features. Same-looking "AUC" numbers
across these four are not comparable — they were never meant to reproduce
each other.

## The 0.9969 population question

Workshop Table 3 claims AUC 0.9969 on 1,761,640 sessions. The Final Report
gives LR on cart sessions (Table 4) as 0.825. Reading the code confirms
these are two different populations, computed from two different scripts,
with no filtering error in either: `stage1_data_cleaning_feature_engineering.py`
line 462 builds `session_features` from `df_clean.groupby('session_key')` —
the full bot-cleaned event log, all 1,761,640 sessions — and never
restricts to cart sessions before fitting the 5-fold CV at line 566.
`reanalysis_controlled_audit.py` (and the Table 6/7 scripts) load
`cart_session_features.csv` directly, which is already restricted to the
43,917 cart sessions. Both numbers are internally consistent with their own
code; they are not the same experiment and this doc does not attempt to
reconcile which one the paper "should" have used.

## Open provenance gaps

- **Final Report Table 8 (LSTM iterations 1–4) — marked permanently UNTRACED.** No script in the repo sweeps multiple LSTM configurations, seeds, or checkpoints. `train_lstm_model.py` defines exactly one fixed hyperparameter set (`MAX_LEN=50, EPOCHS=30, LR=0.001, SEED=42`, one architecture) and produces one trained model. If "iterations 1–4" refers to four separate manual runs, notebook cells, or hyperparameter attempts made outside this script, that code does not exist in the current repo and cannot be located. Per instruction, no reconstruction of this sweep has been attempted — do not cite Table 8 as reproducible from what's here.

## Environment

Installed package versions used for every rerun in this document (2026-07):

| Package | Version |
|---|---|
| scikit-learn | 1.8.0 |
| xgboost | 3.3.0 |
| numpy | 2.3.3 |
| pandas | 2.3.2 |

Notes for anyone re-running this repo from scratch:

- **`seaborn`** is imported by `stage1_data_cleaning_feature_engineering.py` but was missing from `2-Infra/requirements.txt` until this pass — now pinned there.
- **`stage1_data_cleaning_feature_engineering.py` needs `PYTHONIOENCODING=utf-8` on Windows.** Its default console encoding (cp1252) cannot print the `→` character used in `f'{ts_min.date()}  →  {ts_max.date()}'`; without the env var the script crashes partway through Section 2. This is a pre-existing environment quirk, not something introduced by the repo reorg, and the script's source was not modified to work around it.
- **`1-Data/raw/1-events.csv` is gitignored** (too large for GitHub) and must be downloaded separately from the Retail Rocket dataset on Kaggle before any script in `2-Completed-Session/` or `3-Decision-Time/` can run from scratch.
