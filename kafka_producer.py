# kafka_producer.py
# ─────────────────────────────────────────────────────────────────────────────
# Replays events_clean.csv into Kafka topic 'session-events'.
# Messages are keyed by visitorid so all events from one visitor
# go to the same Kafka partition — essential for session reconstruction.
#
# Usage:
#   python kafka_producer.py              # default 500x speed
#   python kafka_producer.py --speed 100  # slower, easier to watch
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import json
import time
import argparse
from kafka import KafkaProducer

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--speed',  type=float, default=500,
                    help='Replay speed multiplier (default: 500x real time)')
parser.add_argument('--topic',  type=str,   default='session-events')
parser.add_argument('--broker', type=str,   default='localhost:9092')
parser.add_argument('--file',   type=str,   default='events_clean.csv')
args = parser.parse_args()

print("=" * 60)
print("  Kafka Producer — Retail Rocket Event Replay")
print("=" * 60)
print(f"  File    : {args.file}")
print(f"  Topic   : {args.topic}")
print(f"  Speed   : {args.speed}x real time")
print(f"  Broker  : {args.broker}")
print()

# ── Load and sort events ───────────────────────────────────────────────────────
print("[1/2] Loading events...")
df = pd.read_csv(args.file).sort_values('timestamp').reset_index(drop=True)
print(f"  {len(df):,} events loaded.")
print(f"  Date range: {pd.to_datetime(df['timestamp'].min(), unit='ms').date()}"
      f" → {pd.to_datetime(df['timestamp'].max(), unit='ms').date()}")
print()

# ── Connect to Kafka ───────────────────────────────────────────────────────────
print("[2/2] Connecting to Kafka...")
producer = KafkaProducer(
    bootstrap_servers=args.broker,
    key_serializer   = lambda k: str(k).encode('utf-8'),
    value_serializer = lambda v: json.dumps(v).encode('utf-8'),
    acks='all'        # wait for broker confirmation before continuing
)
print(f"  Connected to {args.broker}")
print()
print("  Sending events... (Ctrl+C to stop)")
print("-" * 60)

# ── Replay loop ────────────────────────────────────────────────────────────────
prev_ts   = None
sent      = 0
start_run = time.time()

for _, row in df.iterrows():
    # Simulate real-time delay between events (scaled by speed factor)
    if prev_ts is not None:
        real_gap_ms = row['timestamp'] - prev_ts
        sleep_sec   = (real_gap_ms / 1000) / args.speed
        if sleep_sec > 0:
            time.sleep(min(sleep_sec, 0.1))  # cap at 100ms to keep demo fast

    # Build message payload
    message = {
        'timestamp':     int(row['timestamp']),
        'visitorid':     int(row['visitorid']),
        'event':         str(row['event']),
        'itemid':        int(row['itemid'])   if pd.notna(row['itemid'])        else None,
        'transactionid': int(row['transactionid']) if pd.notna(row['transactionid']) else None,
        'session_key':   str(row['session_key'])
    }

    # Key = visitorid → same visitor always goes to same Kafka partition
    producer.send(
        topic=args.topic,
        key=str(row['visitorid']),
        value=message
    )

    prev_ts = row['timestamp']
    sent   += 1

    # Print progress every 10,000 events
    if sent % 10_000 == 0:
        elapsed = time.time() - start_run
        rate    = sent / elapsed
        pct     = sent / len(df) * 100
        print(f"  {sent:>10,} events sent  |  {pct:5.1f}%  |  {rate:,.0f} ev/sec")

producer.flush()
print("-" * 60)
print(f"\n  Done. {sent:,} events sent to topic '{args.topic}'.")
