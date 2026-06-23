import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

SEED = 42

FEAT_COLS = [
    "view_count",
    "addtocart_count",
    "view_to_cart_ratio",
    "unique_items_count",
    "session_age_seconds",
    "inter_event_interval_mean",
    "recency_last_cart_seconds",
]

print("=" * 70)
print("Leakage-Corrected XGBoost Baseline")
print("=" * 70)

events = pd.read_csv("events_clean.csv")
sessions = pd.read_csv("cart_session_features.csv")

ev_no_txn = events[events["event"] != "transaction"].copy()

timing = ev_no_txn.groupby("session_key")["timestamp"].agg(["min", "max"])
sessions = sessions.merge(
    timing.rename(columns={"min": "_ts_min", "max": "_ts_max"}),
    on="session_key",
    how="left",
)

sessions["session_age_seconds"] = (
    (sessions["_ts_max"] - sessions["_ts_min"]) / 1000
).fillna(0)

def mean_iv(grp):
    ts = grp["timestamp"].sort_values().values
    return float(np.mean(np.diff(ts) / 1000)) if len(ts) > 1 else 0.0

iv_map = ev_no_txn.groupby("session_key").apply(mean_iv).to_dict()
sessions["inter_event_interval_mean"] = (
    sessions["session_key"].map(iv_map).fillna(0)
)

last_nontxn = ev_no_txn.groupby("session_key")["timestamp"].max()
last_cart = (
    events[events["event"] == "addtocart"]
    .groupby("session_key")["timestamp"]
    .max()
)

recency = ((last_nontxn - last_cart) / 1000).clip(lower=0)
sessions["recency_last_cart_seconds"] = (
    sessions["session_key"].map(recency).fillna(0)
)

sessions = sessions.drop(columns=["_ts_min", "_ts_max"])

X = sessions[FEAT_COLS].fillna(0)
y = sessions["has_transaction"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=SEED,
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=SEED,
    n_jobs=-1,
)

model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

auc = roc_auc_score(y_test, probs)

print(f"\nAUC-ROC: {auc:.4f}\n")
print(classification_report(y_test, preds, target_names=["Abandoned (0)", "Purchased (1)"]))

importances = pd.DataFrame({
    "feature": FEAT_COLS,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nFeature importance:")
print(importances.to_string(index=False))

print("\nDone.")
