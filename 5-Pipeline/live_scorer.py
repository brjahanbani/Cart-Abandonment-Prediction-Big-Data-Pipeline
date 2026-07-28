# live_scorer.py  —  Hybrid LSTM version
# ─────────────────────────────────────────────────────────────────────────────
# Polls MongoDB, builds (sequence, features) inputs, runs Hybrid LSTM.
# ─────────────────────────────────────────────────────────────────────────────

import time
import pickle
import numpy as np
from datetime import datetime, timezone
from pymongo import MongoClient
import torch
import torch.nn as nn

MONGO_URI = 'mongodb://localhost:27017/'
MONGO_DB = 'retail_rocket'
MONGO_COL = 'sessions'
THRESHOLD = 0.5
POLL = 5

print("=" * 60)
print("  Live Scorer  —  Hybrid LSTM  (sequence + features)")
print("=" * 60)

with open('model_config.pkl', 'rb') as f:
    config = pickle.load(f)
with open('scaler.pkl',       'rb') as f:
    scaler = pickle.load(f)

MAX_LEN = config['MAX_LEN']
N_FEATS = config['N_FEATS']
FEAT_COLS = config['FEAT_COLS']


class HybridLSTM(nn.Module):
    def __init__(self, n_feats=N_FEATS, n_session=len(FEAT_COLS)):
        super().__init__()
        self.lstm = nn.LSTM(n_feats, 32, batch_first=True)
        self.feat_fc = nn.Linear(n_session, 16)
        self.merge = nn.Linear(48, 32)
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(0.3)

    def forward(self, seq, feat):
        _, (h, _) = self.lstm(seq)
        h = h.squeeze(0)
        f = self.relu(self.feat_fc(feat))
        x = torch.cat([h, f], dim=1)
        x = self.drop(self.relu(self.merge(x)))
        return self.sig(self.out(x)).squeeze(1)


model = HybridLSTM()
model.load_state_dict(torch.load('lstm_model.pth', map_location='cpu',
                                 weights_only=True))
model.eval()
print(f"  Model loaded.  Polling every {POLL}s\n")

client = MongoClient(MONGO_URI)
col    = client[MONGO_DB][MONGO_COL]

# ── Initialise LSTM status in pipeline_stats ─────────────────────────────────────
_stats_col = client[MONGO_DB]['pipeline_stats']
_logs_col  = client[MONGO_DB]['pipeline_logs']
try:
    _stats_col.update_one(
        {'_id': 'pipeline_stats'},
        {'$set': {'lstm.status': 'scoring'}},
        upsert=True,
    )
    _logs_col.insert_one({
        'source':  'lstm',
        'message': f'Hybrid LSTM scorer started. Threshold={THRESHOLD}. Polling every {POLL}s',
        'cls':     'info',
        'ts':      datetime.now(timezone.utc).isoformat(),
    })
except Exception:
    pass

print("-" * 68)
print(f"  {'Session Key':<22} {'Events':>6} {'P(buy)':>8} {'Prediction':>15}")
print("-" * 68)

scored = purchased = 0

while True:
    docs = list(col.find({'purchase_probability': None, 'addtocart_count': {'$gt': 0}}))
    docs = docs[:500]   # score 500 at a time
    if docs:
        keys, seqs, feats = [], [], []

        for d in docs:
            # ── Sequence input ───────────────────────────────────────────────
            etypes = d.get('event_sequence',   [0])
            intervals = d.get('interval_sequence', [0.0] * len(etypes))
            pairs = list(zip(etypes, intervals))
            if len(pairs) == 0:
                arr = np.zeros((MAX_LEN, N_FEATS), dtype='float32')
            else:
                arr = np.array(pairs[:MAX_LEN], dtype='float32')
                if arr.ndim == 1:          # safety: reshape if 1D
                    arr = arr.reshape(-1, N_FEATS)
                if len(arr) < MAX_LEN:
                    pad = np.zeros((MAX_LEN - len(arr), N_FEATS), 'float32')
                    arr = np.vstack([arr, pad])
            seqs.append(arr)

            # ── Feature input ────────────────────────────────────────────────
            feat_vec = [float(d.get(c, 0) or 0) for c in FEAT_COLS]
            feats.append(feat_vec)
            keys.append(d['session_key'])

        Xs = torch.tensor(np.stack(seqs),  dtype=torch.float32)
        Xf = torch.tensor(
            scaler.transform(np.array(feats, dtype='float32')),
            dtype=torch.float32
        )

        with torch.no_grad():
            probs = model(Xs, Xf).numpy()

        now = datetime.now(timezone.utc).isoformat()

        for key, prob in zip(keys, probs):
            label = 'PURCHASE ✓' if prob >= THRESHOLD else 'abandon  ✗'
            seq_len = len(col.find_one({'session_key': key})
                          .get('event_sequence', []))
            print(f"  {key:<22} {seq_len:>6}   {prob:>6.1%}   {label}")

            col.update_one(
                {'session_key': key},
                {'$set': {'purchase_probability': round(float(prob), 4),
                          'predicted_label':      int(prob >= THRESHOLD),
                          'scored_at':            now}}
            )
            scored += 1
            purchased += int(prob >= THRESHOLD)

        pct = purchased / scored * 100 if scored else 0
        print(f"\n  [{scored} scored | {pct:.1f}% predicted purchase]\n")
        print("-" * 68)

        # ── Write scoring-batch stats + log to pipeline_stats ─────────────────────
        try:
            _stats_col.update_one(
                {'_id': 'pipeline_stats'},
                {'$set': {'lstm.status': 'scoring'}},
            )
            batch_purchased = sum(1 for key, prob in zip(keys, probs) if prob >= THRESHOLD)
            _logs_col.insert_one({
                'source':  'lstm',
                'message': (
                    f'Scored {len(keys)} sessions  |  '
                    f'{batch_purchased} PURCHASE  |  '
                    f'{len(keys)-batch_purchased} abandon  |  '
                    f'total scored: {scored}'
                ),
                'cls':     'success' if batch_purchased > 0 else '',
                'ts':      datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            pass

    time.sleep(POLL)
