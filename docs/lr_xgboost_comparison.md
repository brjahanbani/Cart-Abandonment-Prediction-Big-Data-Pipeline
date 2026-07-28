# Logistic Regression & XGBoost: Offline vs. Leakage-Corrected Baselines

Compares `0-Offline/offline_cart_abandonment_prediction.py` against
`3-Training/train_logistic_regression_baseline.py` and
`3-Training/train_xgboost_baseline.py`. Both pairs train on the same
underlying sessions (`1-Data/cart_session_features.csv`, 43,917 sessions,
27.2% purchased) with the same 80/20 stratified split (`random_state=42`),
so the results are directly comparable — the difference is entirely in how
three of the seven features are computed, plus hyperparameters.

## Results (measured by running both scripts)

| Model | Script | AUC-ROC | F1 (purchased) | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | `0-Offline` | 0.8248 | 0.5857 | 0.59 | 0.58 |
| Logistic Regression | `3-Training` (leakage-corrected) | **0.6544** | 0.44 | 0.44 | 0.44 |
| XGBoost | `0-Offline` | 0.9945 | 0.9358 | 0.91 | 0.96 |
| XGBoost | `3-Training` (leakage-corrected) | **0.6863** | 0.47 | 0.41 | 0.54 |

Both offline models score dramatically higher — LR by ~17 AUC points,
XGBoost by ~31 AUC points. That gap is not a modeling improvement; it comes
from feature leakage in the offline pipeline.

## Root cause: three features leak the outcome

`cart_session_features.csv` (built in Stage 1) computes `session_age_seconds`,
`inter_event_interval_mean`, and `recency_last_cart_seconds` from **all**
events in a session, including the `transaction` event itself. A session
that converts has a `transaction` event appended after the cart action, which
stretches its timespan and shifts these three timing features in a way that
correlates with `has_transaction` — the exact label being predicted. That's
classic target leakage: the feature is only knowable *because* the outcome
already happened.

`0-Offline/offline_cart_abandonment_prediction.py:35` reads these three
columns from `cart_session_features.csv` as-is:
```python
df = pd.read_csv('../1-Data/cart_session_features.csv')
...
X = df[FEATURES].fillna(0).values   # FEATURES includes the 3 leaky columns unchanged
```

Both `3-Training` scripts instead **recompute** those same three columns
after explicitly dropping the `transaction` event first, so only
online-observable behavior (views/cart-adds) informs them:
```python
ev_no_txn = events[events["event"] != "transaction"].copy()   # drop the label-revealing event

timing = ev_no_txn.groupby("session_key")["timestamp"].agg(["min", "max"])
sessions["session_age_seconds"] = ((sessions["_ts_max"] - sessions["_ts_min"]) / 1000).fillna(0)
# ...inter_event_interval_mean and recency_last_cart_seconds rebuilt the same way
```
This mirrors what a real-time scorer sees during a live session — it cannot
know a transaction is coming, so it must not be trained on a version of the
features that assumes one already happened. The other four features
(`view_count`, `addtocart_count`, `view_to_cart_ratio`, `unique_items_count`)
are untouched and identical in both scripts, so they are not a source of the
gap.

## Secondary factor: hyperparameters differ (minor, doesn't explain the gap)

| | `0-Offline` | `3-Training` |
|---|---|---|
| **Logistic Regression** | `max_iter=1000`, default `C` | `max_iter=2000`, default `C` |
| **XGBoost** | `n_estimators=100`, default `max_depth`/`learning_rate` | `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9` |

`3-Training`'s XGBoost is more conservatively tuned (shallower trees, lower
learning rate, subsampling) — the kind of regularization typically added
*because* a first pass overfit on leaky signal. Even so, this tuning
difference is small next to the ~31-point AUC drop; the leakage removal is
the dominant cause.

## Takeaway

- `0-Offline`'s numbers (LR AUC 0.82, XGBoost AUC 0.99) describe a model that
  implicitly "knows" a purchase already happened — not useful as a
  cart-abandonment predictor for a live session in progress.
- `3-Training`'s numbers (LR AUC 0.65, XGBoost AUC 0.69) are the realistic,
  online-computable baselines: what the model can actually know *before* the
  outcome is decided. XGBoost still edges out Logistic Regression once
  leakage is removed (0.6863 vs 0.6544 AUC), consistent with the offline
  ranking, just at correct-scale numbers.
- Use `3-Training`'s figures as the reference baseline when evaluating the
  live LSTM pipeline (`3-Training/train_lstm_model.py`) — comparing the LSTM
  against `0-Offline`'s inflated numbers would be an unfair, leaked
  comparison.
