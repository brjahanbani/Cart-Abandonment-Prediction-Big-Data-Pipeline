"""
dashboard_server.py  —  Real-time Pipeline Dashboard Bridge
────────────────────────────────────────────────────────────
Reads live data from MongoDB (written by kafka_producer, spark_consumer,
and live_scorer) and exposes REST endpoints consumed by the HTML dashboard.

Run:   uvicorn dashboard_server:app --host 0.0.0.0 --port 8765
Open:  http://localhost:8765
────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import traceback
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pymongo import MongoClient

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Cart Abandonment — Pipeline Dashboard API", version="1.0.0")

# Allow the HTML to call the API even when opened directly from disk
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

MONGO_URI    = "mongodb://localhost:27017/"
MONGO_DB     = "retail_rocket"
TOTAL_EVENTS = 2_755_610
MAX_SESSIONS = 1_523_895


def _db():
    """Open a short-lived MongoDB connection (reconnects on each request)."""
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)[MONGO_DB]


# ── Serve the dashboard HTML ───────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """Serve pipeline_dashboard.html so everything is same-origin."""
    html = (Path(__file__).parent / "pipeline_dashboard.html").read_text(encoding="utf-8")
    return html


# ── /api/stats ────────────────────────────────────────────────────────────────
@app.get("/api/stats")
def api_stats():
    """
    Returns real-time pipeline statistics sourced directly from MongoDB:
      - kafka  : events sent + throughput (written by kafka_producer.py)
      - spark  : batch count + session count (derived from sessions collection)
      - lstm   : scored / purchased / avg confidence (aggregated from sessions)
    """
    try:
        db = _db()

        # ── pipeline_stats document (written by each script) ──────────────────
        doc   = db["pipeline_stats"].find_one({"_id": "pipeline_stats"}) or {}
        kafka = doc.get("kafka", {})
        spark = doc.get("spark", {})
        lstm  = doc.get("lstm",  {})

        # ── Spark: real session count from MongoDB ────────────────────────────
        total_sessions = db["sessions"].count_documents({})

        # ── LSTM: aggregate over sessions collection for accuracy ─────────────
        agg = list(db["sessions"].aggregate([
            {"$match": {"purchase_probability": {"$ne": None}}},
            {"$group": {
                "_id":        None,
                "count":      {"$sum": 1},
                "purchased":  {"$sum": "$predicted_label"},
                "total_prob": {"$sum": "$purchase_probability"},
            }},
        ]))
        if agg:
            a         = agg[0]
            scored    = a["count"]
            purchased = a["purchased"]
            avg_conf  = round(a["total_prob"] / scored * 100, 1) if scored else 0.0
        else:
            scored = purchased = 0
            avg_conf = 0.0

        return {
            "ok": True,
            "kafka": {
                "events_sent":  kafka.get("events_sent",  0),
                "events_total": TOTAL_EVENTS,
                "speed_evps":   kafka.get("speed_evps",   0),
                "status":       kafka.get("status",       "idle"),
            },
            "spark": {
                "batches":           spark.get("batches",           0),
                "sessions":          total_sessions,
                "last_batch_events": spark.get("last_batch_events", 0),
                "status":            spark.get("status",            "idle"),
            },
            "lstm": {
                "scored":    scored,
                "purchased": purchased,
                "avg_conf":  avg_conf,
                "status":    lstm.get("status", "idle"),
            },
        }

    except Exception:
        traceback.print_exc()
        return {"ok": False, "error": "MongoDB unavailable — is the server running?"}


# ── /api/logs/{source} ────────────────────────────────────────────────────────
@app.get("/api/logs/{source}")
def api_logs(source: str, limit: int = 60):
    """
    Returns the most recent log lines for a given pipeline stage.
    source: 'kafka' | 'spark'
    """
    try:
        db   = _db()
        docs = list(
            db["pipeline_logs"]
            .find({"source": source}, {"_id": 0})
            .sort("_id", -1)
            .limit(limit)
        )
        docs.reverse()          # oldest → newest for the frontend
        return {"ok": True, "logs": docs}
    except Exception:
        return {"ok": False, "logs": []}


# ── /api/predictions ──────────────────────────────────────────────────────────
@app.get("/api/predictions")
def api_predictions(limit: int = 100):
    """
    Returns the most recently scored sessions with their LSTM prediction.
    """
    try:
        db   = _db()
        docs = list(
            db["sessions"]
            .find(
                {"purchase_probability": {"$ne": None}},
                {
                    "_id":                  0,
                    "session_key":          1,
                    "sequence_length":      1,
                    "purchase_probability": 1,
                    "predicted_label":      1,
                },
            )
            .sort("_id", -1)
            .limit(limit)
        )
        return {"ok": True, "predictions": docs}
    except Exception:
        return {"ok": False, "predictions": []}


# ── /api/health ───────────────────────────────────────────────────────────────
@app.get("/api/health")
def api_health():
    """Quick connectivity check used by the dashboard on load."""
    try:
        _db().command("ping")
        return {"ok": True, "mongo": "connected"}
    except Exception as e:
        return {"ok": False, "mongo": str(e)}
