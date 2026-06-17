# spark_consumer.py  —  Batch polling version (no checkpoint required)
# ─────────────────────────────────────────────────────────────────────────────
# Reads from Kafka using Spark BATCH mode every 10 seconds.
# No checkpoint = no hadoop.dll = no Windows DLL errors.
# Demonstrates: Kafka ingestion → Spark processing → MongoDB storage
#
# Usage:  python spark_consumer.py
# ─────────────────────────────────────────────────────────────────────────────

import os, time
os.environ['HADOOP_HOME']        = 'C:\\hadoop'
os.environ['PYSPARK_PYTHON']     = 'python'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python'
os.environ['PYSPARK_SUBMIT_ARGS'] = (
    '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 pyspark-shell'
)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType
)
from pymongo import MongoClient
import numpy as np

KAFKA_BROKER   = 'localhost:9092'
KAFKA_TOPIC    = 'session-events'
MONGO_URI      = 'mongodb://localhost:27017/'
MONGO_DB       = 'retail_rocket'
MONGO_COL      = 'sessions'
POLL_INTERVAL  = 10    # seconds between Spark batch reads

EVENT_SCHEMA = StructType([
    StructField('timestamp',     LongType(),   True),
    StructField('visitorid',     LongType(),   True),
    StructField('event',         StringType(), True),
    StructField('itemid',        LongType(),   True),
    StructField('transactionid', LongType(),   True),
    StructField('session_key',   StringType(), True),
])

# ── Spark session ──────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName('CartAbandonmentPipeline') \
    .config('spark.sql.shuffle.partitions', '4') \
    .config('spark.driver.memory', '2g') \
    .getOrCreate()

spark.sparkContext.setLogLevel('ERROR')   # suppress WARN noise

print("=" * 60)
print("  Spark Consumer  —  Cart Abandonment Pipeline")
print("  Mode: Kafka batch polling (no checkpoint required)")
print("=" * 60)
print(f"  Kafka  : {KAFKA_BROKER} / {KAFKA_TOPIC}")
print(f"  MongoDB: {MONGO_URI}{MONGO_DB}.{MONGO_COL}")
print(f"  Polling every {POLL_INTERVAL}s\n")
print("  Listening... (Ctrl+C to stop)\n")
print("-" * 60)

batch_num = 0

while True:
    try:
        # ── Read ALL available messages from Kafka (batch mode) ────────────
        raw = spark.read \
            .format('kafka') \
            .option('kafka.bootstrap.servers', KAFKA_BROKER) \
            .option('subscribe', KAFKA_TOPIC) \
            .option('startingOffsets', 'earliest') \
            .option('endingOffsets',   'latest') \
            .option('failOnDataLoss',  'false') \
            .load()

        count = raw.count()
        if count == 0:
            print(f"  Batch {batch_num} — no messages yet, waiting...")
            time.sleep(POLL_INTERVAL)
            continue

        batch_num += 1

        # ── Parse JSON ─────────────────────────────────────────────────────
        events = raw.select(
            F.from_json(F.col('value').cast('string'), EVENT_SCHEMA).alias('d')
        ).select('d.*').filter(F.col('session_key').isNotNull())

        n_events = events.count()
        print(f"\n  Batch {batch_num}  |  {n_events:,} events from Kafka")

        # ── Aggregate features per session ─────────────────────────────────
        agg = events.groupBy('session_key').agg(
            F.first('visitorid').alias('visitor_id'),
            F.min('timestamp').alias('session_start_ts'),
            F.count('*').alias('event_count'),
            F.sum(F.when(F.col('event') == 'view',        1).otherwise(0))
             .alias('view_count'),
            F.sum(F.when(F.col('event') == 'addtocart',   1).otherwise(0))
             .alias('addtocart_count'),
            F.sum(F.when(F.col('event') == 'transaction', 1).otherwise(0))
             .alias('transaction_count'),
            F.countDistinct('itemid').alias('unique_items_count'),
            ((F.max('timestamp') - F.min('timestamp')) / 1000)
             .alias('session_age_raw'),
            F.max(F.when(F.col('event') == 'transaction', 1).otherwise(0))
             .alias('has_transaction'),
            F.collect_list(
                F.when(F.col('event').isin(['view', 'addtocart']),
                       F.struct(
                           F.col('timestamp').alias('ts'),
                           F.when(F.col('event') == 'view', 0)
                            .otherwise(1).alias('etype')
                       ))
            ).alias('event_structs'),
        ).withColumn(
            'view_to_cart_ratio',
            F.when(F.col('addtocart_count') > 0,
                   F.col('view_count') / F.col('addtocart_count'))
             .otherwise(F.col('view_count').cast('float'))
        )

        n_sessions = agg.count()
        print(f"  → {n_sessions} sessions aggregated")

        # ── Write to MongoDB ───────────────────────────────────────────────
        client  = MongoClient(MONGO_URI)
        col     = client[MONGO_DB][MONGO_COL]
        written = 0

        for row in agg.collect():
            # Build event sequence and intervals (for LSTM scorer)
            structs   = sorted(
                [s for s in row['event_structs'] if s is not None],
                key=lambda s: s['ts']
            )
            etypes    = [s['etype'] for s in structs]
            ts_list   = [s['ts']   for s in structs]
            intervals = [0.0]
            for i in range(1, len(ts_list)):
                sec = (ts_list[i] - ts_list[i-1]) / 1000.0
                intervals.append(round(float(np.log1p(sec) / 10.0), 5))

            doc = {
                'session_key':              row['session_key'],
                'visitor_id':               row['visitor_id'],
                'session_start_ts':         row['session_start_ts'],
                'view_count':               row['view_count'],
                'addtocart_count':          row['addtocart_count'],
                'transaction_count':        row['transaction_count'],
                'view_to_cart_ratio':       round(float(row['view_to_cart_ratio'] or 0), 4),
                'unique_items_count':       row['unique_items_count'],
                'session_age_seconds':      round(float(row['session_age_raw'] or 0), 2),
                'event_sequence':           etypes,
                'interval_sequence':        intervals,
                'sequence_length':          len(etypes),
                'has_transaction':          row['has_transaction'],
                'purchase_probability':     None,
                'scored_at':               None,
            }

            col.update_one(
                {'session_key': row['session_key']},
                {'$set': doc},
                upsert=True
            )
            written += 1

        client.close()
        print(f"  → {written} documents upserted to MongoDB ✓")

    except KeyboardInterrupt:
        print("\n  Stopped by user.")
        break
    except Exception as e:
        print(f"  Error: {e}")

    time.sleep(POLL_INTERVAL)

spark.stop()
