# live_scorer.py
# ─────────────────────────────────────────────────────────────────────────────
# Polls MongoDB every few seconds for new unscored sessions.
# For each new session:
#   1. Reads the event_sequence field
#   2. Pads it to MAX_LEN
#   3. Feeds it to the LSTM
#   4. Prints the purchase probability
#   5. Updates the MongoDB document with the prediction
#
# Prerequisites:
#   train_model.py must have run first  → lstm_model.h5 + model_config.pkl
#   spark_consumer.py must be running   → MongoDB must have sessions
#
# Usage:
#   python live_scorer.py
# ─────────────────────────────────────────────────────────────────────────────

import time
import pickle
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Config ────────────────────────────────────────────────────────────────────
MONGO_URI      = 'mongodb://localhost:27017/'
MONGO_DB       = 'retail_rocket'
MONGO_COL      = 'sessions'
MODEL_PATH     = 'lstm_model.h5'
CONFIG_PATH    = 'model_config.pkl'
POLL_INTERVAL  = 5        # seconds between MongoDB polls
THRESHOLD      = 0.5      # above this = predicted purchase

# ── Load model and config ──────────────────────────────────────────────────────
print("=" * 60)
print("  Live Scorer — Cart Abandonment LSTM Prediction")
print("=" * 60)
print(f"\n  Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)

with open(CONFIG_PATH, 'rb') as f:
    config = pickle.load(f)
MAX_LEN   = config['MAX_LEN']    # 50
EVENT_MAP = config['EVENT_MAP']  # {'view':0, 'addtocart':1}

print(f"  Model loaded. Max sequence length: {MAX_LEN}")
print(f"\n  Connecting to MongoDB...")
client = MongoClient(MONGO_URI)
col    = client[MONGO_DB][MONGO_COL]
print(f"  Connected to {MONGO_URI}{MONGO_DB}.{MONGO_COL}")

print(f"\n  Polling every {POLL_INTERVAL}s for new sessions...")
print(f"  Purchase threshold: {THRESHOLD}")
print()
print("-" * 60)
print(f"  {'Session Key':<25} {'Seq Len':>7} {'P(purchase)':>12} {'Label':>12}")
print("-" * 60)

# ── Prediction loop ────────────────────────────────────────────────────────────
scored_total   = 0
purchased_pred = 0

while True:
    # Find all sessions not yet scored (purchase_probability is None)
    unscored = list(col.find({'purchase_probability': None}))

    if unscored:
        # Batch all unscored sessions together for efficiency
        session_ids = []
        sequences   = []

        for doc in unscored:
            seq = doc.get('event_sequence', [])
            if len(seq) == 0:
                seq = [0]   # at minimum one event
            session_ids.append(doc['session_key'])
            sequences.append(seq)

        # Pad sequences to MAX_LEN
        X = pad_sequences(
            sequences,
            maxlen=MAX_LEN,
            padding='post',
            truncating='post',
            value=0
        ).reshape(len(sequences), MAX_LEN, 1).astype('float32')

        # LSTM inference (batch)
        probabilities = model.predict(X, verbose=0).flatten()

        # Write predictions back to MongoDB and print results
        now = datetime.utcnow().isoformat()
        for session_key, prob in zip(session_ids, probabilities):
            label = 'PURCHASE ✓' if prob >= THRESHOLD else 'abandon  ✗'

            # Update MongoDB document
            col.update_one(
                {'session_key': session_key},
                {'$set': {
                    'purchase_probability': round(float(prob), 4),
                    'predicted_label':      int(prob >= THRESHOLD),
                    'scored_at':            now
                }}
            )

            # Print live prediction
            seq_len = len(sequences[session_ids.index(session_key)])
            print(f"  {session_key:<25} {seq_len:>7}   "
                  f"{prob:>10.1%}   {label}")

            scored_total   += 1
            purchased_pred += int(prob >= THRESHOLD)

        pct_purchase = purchased_pred / scored_total * 100 if scored_total else 0
        print(f"\n  [{scored_total} scored | {pct_purchase:.1f}% predicted purchase]")
        print("-" * 60)

    # Wait before next poll
    time.sleep(POLL_INTERVAL)
