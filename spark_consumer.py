# spark_consumer.py
# ─────────────────────────────────────────────────────────────────────────────
# Reads the 'session-events' Kafka topic with Spark Structured Streaming.
# For each micro-batch:
#   - Parses JSON events
#   - Groups by session_key
#   - Computes 7 session features + event sequence
#   - Upserts results to MongoDB collection 'sessions'
#
# Prerequisites:
#   docker-compose up -d          (Kafka + MongoDB must be running)
#   pip install pyspark kafka-python pymongo
#
# Usage:
#   spark-submit \
#     --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
#     spark_consumer.py
#
# OR simply:
#   python spark_consumer.py
# ─────────────────────────────────────────────────────────────────────────────

import os
os.environ['PYSPARK_SUBMIT_ARGS'] = (
    '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 pyspark-shell'
)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, IntegerType
)
from pymongo import MongoClient
import json

# ── Config ────────────────────────────────────────────────────────────────────
KAFKA_BROKER   = 'localhost:9092'
KAFKA_TOPIC    = 'session-events'
MONGO_URI      = 'mongodb://localhost:27017/'
MONGO_DB       = 'retail_rocket'
MONGO_COL      = 'sessions'
BATCH_INTERVAL = '10 seconds'    # process a micro-batch every 10 seconds

# ── Kafka message schema ───────────────────────────────────────────────────────
EVENT_SCHEMA = StructType([
    StructField('timestamp',     LongType(),    True),
    StructField('visitorid',     LongType(),    True),
    StructField('event',         StringType(),  True),
    StructField('itemid',        LongType(),    True),
    StructField('transactionid', LongType(),    True),
    StructField('session_key',   StringType(),  True),
])

# ── Spark session ─────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName('CartAbandonmentPipeline') \
    .config('spark.sql.shuffle.partitions', '8') \
    .config('spark.executor.memory', '2g') \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')  # reduce log noise

print("=" * 60)
print("  Spark Consumer — Cart Abandonment Pipeline")
print("=" * 60)
print(f"  Kafka  : {KAFKA_BROKER} / {KAFKA_TOPIC}")
print(f"  Mongo  : {MONGO_URI}{MONGO_DB}.{MONGO_COL}")
print(f"  Batch  : every {BATCH_INTERVAL}")
print()

# ── Read from Kafka ────────────────────────────────────────────────────────────
raw_stream = spark.readStream \
    .format('kafka') \
    .option('kafka.bootstrap.servers', KAFKA_BROKER) \
    .option('subscribe', KAFKA_TOPIC) \
    .option('startingOffsets', 'latest') \
    .option('maxOffsetsPerTrigger', 50_000) \
    .load()

# Parse JSON value column into structured columns
events_stream = raw_stream.select(
    F.from_json(
        F.col('value').cast('string'),
        EVENT_SCHEMA
    ).alias('data')
).select('data.*')

# ── Process each micro-batch ───────────────────────────────────────────────────
def process_batch(batch_df, batch_id):
    """
    Called by Spark for every micro-batch.
    Computes session features and upserts to MongoDB.
    """
    if batch_df.isEmpty():
        return

    count = batch_df.count()
    print(f"\n  Batch {batch_id} | {count:,} events received")

    # ── Aggregate features per session ────────────────────────────────────────
    sessions = batch_df.groupBy('session_key').agg(

        # Identifiers
        F.first('visitorid').alias('visitor_id'),

        # Session timestamps
        F.min('timestamp').alias('session_start_ts'),
        F.max('timestamp').alias('session_end_ts'),

        # Event type counts
        F.count('*').alias('event_count'),
        F.sum(F.when(F.col('event') == 'view',        1).otherwise(0)).alias('view_count'),
        F.sum(F.when(F.col('event') == 'addtocart',   1).otherwise(0)).alias('addtocart_count'),
        F.sum(F.when(F.col('event') == 'transaction', 1).otherwise(0)).alias('transaction_count'),

        # Unique items browsed
        F.countDistinct('itemid').alias('unique_items_count'),

        # Session age (ms → seconds)
        ((F.max('timestamp') - F.min('timestamp')) / 1000).alias('session_age_seconds'),

        # Has purchase label
        F.max(F.when(F.col('event') == 'transaction', 1).otherwise(0)).alias('has_transaction'),

        # Event sequence for LSTM: ordered list of non-transaction event types
        # view=0, addtocart=1 (transaction excluded — it's the label)
        F.collect_list(
            F.when(F.col('event').isin(['view', 'addtocart']),
                   F.when(F.col('event') == 'view', 0).otherwise(1))
        ).alias('event_sequence_raw'),

    )

    # Compute derived features that need two columns
    sessions = sessions \
        .withColumn(
            'view_to_cart_ratio',
            F.when(F.col('addtocart_count') > 0,
                   F.col('view_count') / F.col('addtocart_count'))
            .otherwise(F.col('view_count').cast('float'))
        ) \
        .withColumn(
            'sequence_length',
            F.size('event_sequence_raw')
        )

    n_sessions = sessions.count()
    print(f"  → {n_sessions} sessions aggregated")

    # ── Write to MongoDB ──────────────────────────────────────────────────────
    client = MongoClient(MONGO_URI)
    col    = client[MONGO_DB][MONGO_COL]

    written = 0
    for row in sessions.collect():

        # Build the document
        doc = {
            'session_key':              row['session_key'],
            'visitor_id':               row['visitor_id'],
            'session_start_ts':         row['session_start_ts'],
            'view_count':               row['view_count'],
            'addtocart_count':          row['addtocart_count'],
            'transaction_count':        row['transaction_count'],
            'view_to_cart_ratio':       round(float(row['view_to_cart_ratio'] or 0), 4),
            'unique_items_count':       row['unique_items_count'],
            'session_age_seconds':      round(float(row['session_age_seconds'] or 0), 2),
            'event_sequence':           list(row['event_sequence_raw']),
            'sequence_length':          row['sequence_length'],
            'has_transaction':          row['has_transaction'],
            'purchase_probability':     None,    # filled by live_scorer.py
            'scored_at':                None,
        }

        # Upsert: update if session_key exists, insert if new
        col.update_one(
            {'session_key': row['session_key']},
            {'$set': doc},
            upsert=True
        )
        written += 1

    client.close()
    print(f"  → {written} documents upserted to MongoDB ✓")

# ── Start streaming ────────────────────────────────────────────────────────────
print("  Listening for events... (Ctrl+C to stop)\n")

query = events_stream.writeStream \
    .foreachBatch(process_batch) \
    .trigger(processingTime=BATCH_INTERVAL) \
    .option('checkpointLocation', '/tmp/spark_checkpoint') \
    .start()

query.awaitTermination()
