# train_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Trains an LSTM on cart session event sequences.
# Input:  events_clean.csv   (full cleaned event log)
#         cart_session_features.csv  (43,917 cart sessions with labels)
# Output: lstm_model.h5      (saved Keras model for live_scorer.py)
#         scaler.pkl          (not needed for LSTM, kept for reference)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import pickle

# ── Config ────────────────────────────────────────────────────────────────────
MAX_LEN   = 50    # pad/truncate all sequences to this length
EPOCHS    = 20
BATCH     = 64
SEED      = 42

# Event encoding: we exclude transaction from input (it IS the label)
EVENT_MAP = {'view': 0, 'addtocart': 1}

print("=" * 60)
print("  LSTM Training — Cart Abandonment Prediction")
print("=" * 60)

# ── Step 1: Load data ─────────────────────────────────────────────────────────
print("\n[1/5] Loading data...")

events = pd.read_csv('events_clean.csv')
sessions = pd.read_csv('cart_session_features.csv')

print(f"  Events loaded      : {len(events):,} rows")
print(f"  Cart sessions      : {len(sessions):,} rows")

# ── Step 2: Build event sequences per session ─────────────────────────────────
print("\n[2/5] Building event sequences per session...")

# Keep only events that are NOT transactions (transaction = the label)
events_filtered = events[events['event'].isin(['view', 'addtocart'])].copy()

# Sort by session and timestamp to preserve temporal order
events_filtered = events_filtered.sort_values(['session_key', 'timestamp'])

# Build sequence: for each session_key, collect ordered list of event integers
seq_map = (
    events_filtered
    .groupby('session_key')['event']
    .apply(lambda evts: [EVENT_MAP[e] for e in evts])
    .to_dict()
)

print(f"  Unique sessions with sequences : {len(seq_map):,}")

# Match sequences to labelled cart sessions
sessions['sequence'] = sessions['session_key'].map(seq_map)

# Drop sessions where we couldn't build a sequence (rare edge case)
sessions = sessions.dropna(subset=['sequence'])
print(f"  Sessions after sequence join   : {len(sessions):,}")

# ── Step 3: Prepare LSTM input ────────────────────────────────────────────────
print("\n[3/5] Padding sequences and preparing tensors...")

sequences = sessions['sequence'].tolist()
labels    = sessions['has_transaction'].values

# Pad all sequences to MAX_LEN (shorter → zero-pad, longer → truncate)
X = pad_sequences(
    sequences,
    maxlen=MAX_LEN,
    padding='post',   # zeros added at the END
    truncating='post',
    value=0
)

# LSTM expects shape: (samples, timesteps, features)
X = X.reshape(X.shape[0], MAX_LEN, 1).astype('float32')
y = labels.astype('float32')

print(f"  X shape  : {X.shape}")
print(f"  y shape  : {y.shape}")
print(f"  Purchase rate : {y.mean()*100:.1f}%")

# ── Step 4: Train / test split ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print(f"\n  Train : {len(X_train):,}    Test : {len(X_test):,}")

# Class weights to handle 2.7:1 imbalance without resampling
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
weight_0 = (n_neg + n_pos) / (2 * n_neg)
weight_1 = (n_neg + n_pos) / (2 * n_pos)
class_weights = {0: round(weight_0, 3), 1: round(weight_1, 3)}
print(f"  Class weights : {class_weights}")

# ── Step 5: Build and train LSTM ──────────────────────────────────────────────
print("\n[4/5] Building LSTM model...")

model = Sequential([
    LSTM(32, input_shape=(MAX_LEN, 1)),   # reads sequence, returns final state
    Dense(16, activation='relu'),          # non-linear combination of LSTM output
    Dense(1,  activation='sigmoid')        # purchase probability output
], name='cart_abandonment_lstm')

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\n[5/5] Training...")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,           # stop if val_loss doesn't improve for 3 epochs
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_split=0.15,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EVALUATION ON HELD-OUT TEST SET")
print("=" * 60)

y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob)
print(f"\n  AUC-ROC  : {auc:.4f}")
print()
print(classification_report(
    y_test, y_pred,
    target_names=['Abandoned (0)', 'Purchased (1)']
))

# ── Save model ────────────────────────────────────────────────────────────────
model.save('lstm_model.h5')
print("  Saved : lstm_model.h5")

# Save MAX_LEN and EVENT_MAP so scorer uses the same settings
config = {'MAX_LEN': MAX_LEN, 'EVENT_MAP': EVENT_MAP}
with open('model_config.pkl', 'wb') as f:
    pickle.dump(config, f)
print("  Saved : model_config.pkl")

print("\n  Training complete. Run live_scorer.py next.")
print("=" * 60)
