# Logistic Regression & XGBoost: What's Different Between the Two Versions?

We have two places in the project that train a Logistic Regression and an
XGBoost model on the same cart-session data:

- `0-Offline/offline_cart_abandonment_prediction.py`
- `3-Training/train_logistic_regression_baseline.py` and `3-Training/train_xgboost_baseline.py`

They look similar but are not doing the same thing under the hood. Here's
what actually differs, explained simply.

## 1. Where the input features come from

**`0-Offline`** just loads the ready-made feature file and uses it as-is:
```python
df = pd.read_csv('../1-Data/cart_session_features.csv')
X = df[FEATURES].fillna(0).values
```
It trusts that `session_age_seconds`, `inter_event_interval_mean`, and
`recency_last_cart_seconds` (three of the seven features) are already correct.

**`3-Training`** doesn't trust those three columns — it throws them away and
recalculates them itself:
```python
events   = pd.read_csv("../1-Data/events_clean.csv")
sessions = pd.read_csv("../1-Data/cart_session_features.csv")

# drop the "transaction" event before recomputing timing features
ev_no_txn = events[events["event"] != "transaction"].copy()

timing = ev_no_txn.groupby("session_key")["timestamp"].agg(["min", "max"])
sessions["session_age_seconds"] = ((sessions["_ts_max"] - sessions["_ts_min"]) / 1000).fillna(0)
# ...same idea for inter_event_interval_mean and recency_last_cart_seconds
```

**Why this matters (in plain terms):** the original `cart_session_features.csv`
computed those three timing columns using *every* event in a session —
including the `transaction` event, which only exists if the customer actually
bought something. So a model trained on that raw column is quietly being told
"here's a clue about whether they bought" before it's supposed to know that.
This is called **data leakage** — the model cheats using information it
wouldn't have in real life. `3-Training` fixes this by removing the
`transaction` event first, so the three timing features only reflect what a
live system could actually observe *before* knowing the outcome.

The other four features (`view_count`, `addtocart_count`,
`view_to_cart_ratio`, `unique_items_count`) are identical in both versions —
only the three timing features are recomputed.

## 2. Logistic Regression settings

| Setting | `0-Offline` | `3-Training` |
|---|---|---|
| `max_iter` (how many optimization steps allowed) | 1000 | 2000 |
| Scaling | `StandardScaler` fit once, applied separately | `StandardScaler` inside an sklearn `Pipeline` (same effect, cleaner code) |

`max_iter` is just a safety limit so the solver doesn't stop before it
converges — `3-Training` gives it more room, which mostly matters if the data
is harder to fit (it is, since the leaky shortcut is gone).

## 3. XGBoost settings

| Setting | `0-Offline` | `3-Training` |
|---|---|---|
| `n_estimators` (number of trees) | 100 | 300 |
| `max_depth` (how deep each tree can grow) | default (~6) | 4 (shallower) |
| `learning_rate` (how big a step each tree takes) | default (~0.3) | 0.05 (smaller, slower learning) |
| `subsample` / `colsample_bytree` (randomly use a % of rows/columns per tree) | not set (100%) | 0.9 / 0.9 |

In plain terms: `0-Offline` uses XGBoost's out-of-the-box defaults — deeper
trees that learn fast. `3-Training` uses more, but shallower, trees that
each learn a little at a time, and each tree only sees 90% of the rows/columns
at random. This combination (more trees + smaller learning rate + shallower
depth + row/column subsampling) is a standard way to make the model less
likely to memorize noise and more likely to generalize — useful once you've
removed an easy shortcut like leakage, because the model now has to work
harder to find real patterns instead of overfitting to them.

## Summary

- **Same algorithms** (Logistic Regression, XGBoost), **same raw dataset**,
  but two different feature-engineering approaches and two different sets of
  hyperparameters.
- `0-Offline` = quick, straightforward version using the feature file as-is
  and default model settings.
- `3-Training` = more careful version that recomputes three features to avoid
  data leakage, and tunes the models to compensate for the data being
  genuinely harder to predict once that leakage is gone.
