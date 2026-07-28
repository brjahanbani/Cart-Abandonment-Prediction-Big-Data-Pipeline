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
from datetime import datetime, timezone
from kafka import KafkaProducer
from pymongo import MongoClient

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

# ── Streaming constants ────────────────────────────────────────────────────────
CHUNK_SIZE = 50_000   # rows per chunk — keeps peak RAM well under 200 MB

# Count total rows without loading the file (for progress reporting)
print("[1/2] Scanning events file...")
total_rows = sum(1 for _ in open(args.file, encoding='utf-8')) - 1  # subtract header
print(f"  {total_rows:,} events found (streaming {CHUNK_SIZE:,} rows at a time).")
print()

# ── Connect to Kafka ───────────────────────────────────────────────────────────
print("[2/2] Connecting to Kafka...")
producer = KafkaProducer(
    bootstrap_servers=args.broker,
    key_serializer   = lambda k: str(k).encode('utf-8'),
    value_serializer = lambda v: json.dumps(v).encode('utf-8'),
    linger_ms        = 5,    # micro-batch for throughput
    batch_size       = 65536,
    acks             = 'all'
)
print(f"  Connected to {args.broker}")
print()
print("  Sending events... (Ctrl+C to stop)")
print("-" * 60)

# ── MongoDB live-stats writer (best-effort — does not block Kafka) ─────────────
_mongo_ok = False
try:
    _mc    = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
    _stats = _mc['retail_rocket']['pipeline_stats']
    _logs  = _mc['retail_rocket']['pipeline_logs']

    # Initialise / reset the kafka sub-document
    _stats.update_one(
        {'_id': 'pipeline_stats'},
        {'$set': {
            'kafka.events_sent':  0,
            'kafka.events_total': total_rows,
            'kafka.speed_evps':   0,
            'kafka.status':       'running',
        }},
        upsert=True,
    )
    _logs.insert_one({
        'source':  'kafka',
        'message': f'[1/2] {total_rows:,} events loaded. Date range: 2015-05-03 → 2015-09-18',
        'cls':     'success',
        'ts':      datetime.now(timezone.utc).isoformat(),
    })
    _logs.insert_one({
        'source':  'kafka',
        'message': f'[2/2] Connected to {args.broker}  |  topic: {args.topic}',
        'cls':     'info',
        'ts':      datetime.now(timezone.utc).isoformat(),
    })
    _logs.insert_one({
        'source':  'kafka',
        'message': f'Sending events at {args.speed}x real-time speed…',
        'cls':     '',
        'ts':      datetime.now(timezone.utc).isoformat(),
    })
    _mongo_ok = True
    print("  [dashboard] Writing live stats to MongoDB ✓")
except Exception as _e:
    print(f"  [dashboard] MongoDB stats unavailable ({_e}) — dashboard won't show Kafka progress")

# ── Replay loop (chunked streaming) ───────────────────────────────────────────
prev_ts   = None
sent      = 0
start_run = time.time()

for chunk in pd.read_csv(args.file, chunksize=CHUNK_SIZE):
    # Sort within each chunk to preserve temporal order inside the batch
    chunk = chunk.sort_values('timestamp')

    for _, row in chunk.iterrows():
        # Simulate real-time delay between events (scaled by speed factor)
        if prev_ts is not None:
            real_gap_ms = float(row['timestamp']) - prev_ts
            sleep_sec   = (real_gap_ms / 1000.0) / args.speed
            if sleep_sec > 0:
                time.sleep(min(sleep_sec, 0.1))  # cap at 100 ms to keep demo fast

        # Build message payload
        message = {
            'timestamp':     int(row['timestamp']),
            'visitorid':     int(row['visitorid']),
            'event':         str(row['event']),
            'itemid':        int(row['itemid'])          if pd.notna(row.get('itemid'))          else None,
            'transactionid': int(row['transactionid'])   if pd.notna(row.get('transactionid'))   else None,
            'session_key':   str(row['session_key'])
        }

        # Key = visitorid → same visitor always goes to same Kafka partition
        producer.send(
            topic=args.topic,
            key=str(row['visitorid']),
            value=message
        )

        prev_ts = float(row['timestamp'])
        sent   += 1

        # Print progress every 10,000 events
        if sent % 10_000 == 0:
            elapsed = time.time() - start_run
            rate    = sent / elapsed if elapsed > 0 else float('inf')
            pct     = sent / total_rows * 100
            print(f"  {sent:>10,} events sent  |  {pct:5.1f}%  |  {rate:,.0f} ev/sec")

            # ── Dashboard stats (every 10k) ──────────────────────────────────
            if _mongo_ok:
                try:
                    _stats.update_one(
                        {'_id': 'pipeline_stats'},
                        {'$set': {
                            'kafka.events_sent': sent,
                            'kafka.speed_evps':  int(rate),
                        }},
                    )
                    # Log line every 100k events
                    if sent % 100_000 == 0:
                        _logs.insert_one({
                            'source':  'kafka',
                            'message': f'{sent:,} events sent  |  {pct:.1f}%  |  {rate:,.0f} ev/sec',
                            'cls':     '',
                            'ts':      datetime.now(timezone.utc).isoformat(),
                        })
                except Exception:
                    pass

producer.flush()
print("-" * 60)
print(f"\n  Done. {sent:,} events sent to topic '{args.topic}'.")

# ── Mark Kafka complete in dashboard stats ─────────────────────────────────────
if _mongo_ok:
    try:
        elapsed = time.time() - start_run
        rate    = sent / elapsed if elapsed > 0 else 0
        _stats.update_one(
            {'_id': 'pipeline_stats'},
            {'$set': {
                'kafka.events_sent': sent,
                'kafka.speed_evps':  int(rate),
                'kafka.status':      'complete',
            }},
        )
        _logs.insert_one({
            'source':  'kafka',
            'message': f'✓ All {sent:,} events sent to topic \'{args.topic}\'.  Avg {rate:,.0f} ev/sec',
            'cls':     'success',
            'ts':      datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        pass
