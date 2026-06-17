# train_model.py  —  Hybrid LSTM + Feature model
# ─────────────────────────────────────────────────────────────────────────────
# Two-branch architecture:
#   Branch A : LSTM(32) on event sequences  [event_type, log_interval]
#   Branch B : Dense(16) on 7 session features
#   Merged   : Dense(32) → Dense(1, Sigmoid)
#
# This combines the temporal pattern signal (LSTM) with the aggregate
# statistical signal (Dense), which prior validation showed dominates.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

MAX_LEN   = 50
N_FEATS   = 2       # per timestep: [event_type, norm_interval]
EPOCHS    = 30
BATCH     = 64
LR        = 0.001
SEED      = 42
THRESHOLD = 0.5
EVENT_MAP = {'view': 0, 'addtocart': 1}

# 7 session features (same as offline XGBoost validation)
FEAT_COLS = [
    'view_count', 'addtocart_count', 'view_to_cart_ratio',
    'unique_items_count', 'session_age_seconds',
    'inter_event_interval_mean', 'recency_last_cart_seconds'
]

torch.manual_seed(SEED)
device = torch.device('cpu')

print("=" * 60)
print("  Hybrid LSTM Training  —  Cart Abandonment Prediction")
print("  Branch A : LSTM on event sequences")
print("  Branch B : Dense on 7 session features")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[1/5] Loading data...")
events   = pd.read_csv('events_clean.csv')
sessions = pd.read_csv('cart_session_features.csv')
print(f"  Events   : {len(events):,}")
print(f"  Sessions : {len(sessions):,}")

# ── Fix temporal leakage in 3 time-based features ────────────────────────────
# The precomputed features in cart_session_features.csv used the FINAL event
# timestamp per session. For purchase sessions this final event IS the
# transaction — meaning session_age, interval_mean and recency encode
# transaction timing, causing leakage. We recompute using only non-transaction
# events, which is exactly what the live streaming system sees at prediction time.

ev_no_txn = events[events['event'] != 'transaction'].copy()

# session_age_seconds: duration between first and last NON-transaction event
timing = ev_no_txn.groupby('session_key')['timestamp'].agg(['min', 'max'])
sessions = sessions.merge(
    timing.rename(columns={'min':'_ts_min','max':'_ts_max'}),
    on='session_key', how='left'
)
sessions['session_age_seconds'] = (
    (sessions['_ts_max'] - sessions['_ts_min']) / 1000
).fillna(0)

# inter_event_interval_mean: mean gap between NON-transaction events only
def mean_iv(grp):
    ts = grp['timestamp'].sort_values().values
    return float(np.mean(np.diff(ts) / 1000)) if len(ts) > 1 else 0.0

iv_map = ev_no_txn.groupby('session_key').apply(mean_iv).to_dict()
sessions['inter_event_interval_mean'] = sessions['session_key'].map(iv_map).fillna(0)

# recency_last_cart_seconds: time from last addtocart to last NON-txn event
last_nontxn = ev_no_txn.groupby('session_key')['timestamp'].max()
last_cart   = (events[events['event'] == 'addtocart']
               .groupby('session_key')['timestamp'].max())
recency     = ((last_nontxn - last_cart) / 1000).clip(lower=0)
sessions['recency_last_cart_seconds'] = (
    sessions['session_key'].map(recency).fillna(0)
)
sessions = sessions.drop(columns=['_ts_min','_ts_max'])
print("  Leakage correction applied to 3 time-based features.")

# ── Build event sequences ─────────────────────────────────────────────────────
print("\n[2/5] Building event sequences with timing...")

ev = events[events['event'].isin(['view', 'addtocart'])].copy()
ev = ev.sort_values(['session_key', 'timestamp'])

def build_seq(group):
    ts, etypes = group['timestamp'].values, group['event'].values
    pairs = []
    for i, (t, e) in enumerate(zip(ts, etypes)):
        iv = 0.0 if i == 0 else (ts[i] - ts[i-1]) / 1000.0
        pairs.append([EVENT_MAP[e], float(np.log1p(iv) / 10.0)])
    return pairs

seq_map = ev.groupby('session_key').apply(build_seq).to_dict()
sessions['sequence'] = sessions['session_key'].map(seq_map)
sessions = sessions.dropna(subset=['sequence'])

# Fix recency: replace -1 (no cart) with 0
sessions['recency_last_cart_seconds'] = \
    sessions['recency_last_cart_seconds'].replace(-1, 0)

print(f"  Matched sessions : {len(sessions):,}")

# ── Prepare tensors ───────────────────────────────────────────────────────────
print("\n[3/5] Preparing tensors...")

def pad_seq(seq):
    arr = np.array(seq[:MAX_LEN], dtype='float32')
    if len(arr) < MAX_LEN:
        pad = np.zeros((MAX_LEN - len(arr), N_FEATS), dtype='float32')
        arr = np.vstack([arr, pad])
    return arr

# Sequence tensor: (n, 50, 2)
X_seq = np.stack([pad_seq(s) for s in sessions['sequence']])

# Feature tensor: (n, 7) — standardised
feat_raw = sessions[FEAT_COLS].fillna(0).values.astype('float32')
scaler   = StandardScaler()
X_feat   = scaler.fit_transform(feat_raw).astype('float32')

y = sessions['has_transaction'].values.astype('float32')

print(f"  Sequence tensor : {X_seq.shape}")
print(f"  Feature tensor  : {X_feat.shape}")
print(f"  Purchase rate   : {y.mean()*100:.1f}%")

# ── Split ─────────────────────────────────────────────────────────────────────
(Xs_tr, Xs_te,
 Xf_tr, Xf_te,
 y_tr,  y_te) = train_test_split(
    X_seq, X_feat, y, test_size=0.2, stratify=y, random_state=SEED
)
print(f"\n  Train : {len(y_tr):,}    Test : {len(y_te):,}")

n_neg = (y_tr == 0).sum()
n_pos = (y_tr == 1).sum()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
print(f"  pos_weight : {pos_weight.item():.2f}")

train_ds = TensorDataset(
    torch.tensor(Xs_tr), torch.tensor(Xf_tr), torch.tensor(y_tr)
)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

# ── Model ─────────────────────────────────────────────────────────────────────
print("\n[4/5] Building Hybrid LSTM model...")

class HybridLSTM(nn.Module):
    """
    Branch A: LSTM reads the event sequence step by step
    Branch B: Dense processes the 7 pre-computed session features
    Merge   : concatenate → Dense → output probability
    """
    def __init__(self, n_feats=N_FEATS, n_session=len(FEAT_COLS)):
        super().__init__()
        # Branch A — temporal pattern
        self.lstm    = nn.LSTM(n_feats, 32, batch_first=True)
        # Branch B — aggregate statistics
        self.feat_fc = nn.Linear(n_session, 16)
        # Merge
        self.merge   = nn.Linear(32 + 16, 32)
        self.out     = nn.Linear(32, 1)
        self.relu    = nn.ReLU()
        self.sig     = nn.Sigmoid()
        self.drop    = nn.Dropout(0.3)

    def forward(self, seq, feat):
        _, (h, _) = self.lstm(seq)              # h: (1, batch, 32)
        h = h.squeeze(0)                         # (batch, 32)
        f = self.relu(self.feat_fc(feat))        # (batch, 16)
        x = torch.cat([h, f], dim=1)            # (batch, 48)
        x = self.drop(self.relu(self.merge(x))) # (batch, 32)
        return self.sig(self.out(x)).squeeze(1)  # (batch,)

model     = HybridLSTM().to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n[5/5] Training...")

best_loss, no_improve, patience = float('inf'), 0, 5

for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for Xsb, Xfb, yb in train_dl:
        Xsb, Xfb, yb = Xsb.to(device), Xfb.to(device), yb.to(device)
        optimizer.zero_grad()
        # Raw logits path for BCEWithLogitsLoss
        _, (h, _) = model.lstm(Xsb)
        h = h.squeeze(0)
        f = model.relu(model.feat_fc(Xfb))
        x = torch.cat([h, f], dim=1)
        x = model.drop(model.relu(model.merge(x)))
        logits = model.out(x).squeeze(1)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item()

    avg = total / len(train_dl)
    if avg < best_loss - 1e-4:
        best_loss = avg
        no_improve = 0
        torch.save(model.state_dict(), 'lstm_model_best.pth')
    else:
        no_improve += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>2}/{EPOCHS}  loss: {avg:.4f}  best: {best_loss:.4f}")

    if no_improve >= patience:
        print(f"\n  Early stop at epoch {epoch}")
        break

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EVALUATION ON HELD-OUT TEST SET")
print("=" * 60)

model.load_state_dict(torch.load('lstm_model_best.pth', weights_only=True))
model.eval()

with torch.no_grad():
    Xs_t = torch.tensor(Xs_te, dtype=torch.float32)
    Xf_t = torch.tensor(Xf_te, dtype=torch.float32)
    probs = model(Xs_t, Xf_t).numpy()

preds = (probs >= THRESHOLD).astype(int)
auc   = roc_auc_score(y_te, probs)
print(f"\n  AUC-ROC : {auc:.4f}")
print()
print(classification_report(y_te, preds,
      target_names=['Abandoned (0)', 'Purchased (1)']))

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), 'lstm_model.pth')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
config = {
    'MAX_LEN': MAX_LEN, 'N_FEATS': N_FEATS,
    'EVENT_MAP': EVENT_MAP, 'FEAT_COLS': FEAT_COLS,
    'framework': 'pytorch_hybrid'
}
with open('model_config.pkl', 'wb') as f:
    pickle.dump(config, f)

print("  Saved : lstm_model.pth")
print("  Saved : scaler.pkl")
print("  Saved : model_config.pkl")
print("\n  Done.")
print("=" * 60)
