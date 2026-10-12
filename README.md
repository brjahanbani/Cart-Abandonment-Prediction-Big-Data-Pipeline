# Cart Abandonment Prediction — Leakage Audit + Big Data Pipeline

## Paper

"When Offline Performance Misleads: A Leakage Audit of Completed-Session
Features for Cart-Abandonment Prediction" — Babak Rad Jahanbani, Ahmet Sayar.

- **IJACSA:** manuscript MS-17-10-0048, **accepted for publication** (October 2026); DOI pending.
- **arXiv:** submitted 2026-09-21, cs.AI.

Manuscript sources (IJACSA and arXiv) and the build are in
[paper/](paper/README.md). Every published number is mapped to the script that
produces it in [docs/results_provenance.md](docs/results_provenance.md).

## Folder layout
| Folder | Contents |
|---|---|
| `0-Shared/` | `clean_events.py` — bot removal + 30-minute sessionization, used by the decision-time lineage |
| `1-Data/` | `raw/` (Retail Rocket CSVs — gitignored, download from Kaggle), `completed-session/cart_session_features.csv`, `decision-time/prefix_dataset.parquet`, `events_clean.csv` (gitignored) |
| `2-Completed-Session/` | **Frozen.** The scripts behind every number in the submitted paper, leakage included by design (see its README). `reanalysis_controlled_audit.py` produces the primary evidence (Tables IV–V) |
| `2-Infra/` | `docker-compose.yml`, `requirements.txt` |
| `3-Decision-Time/` | Point-in-time rebuild: `build_prefix_dataset.py`, `train_prefix_models.py` (not in the submitted paper) |
| `4-Model-Artifacts/` | `lstm_model.pth`, `model_config.pkl`, `scaler.pkl` — produced by `2-Completed-Session/train_lstm_model.py` |
| `5-Pipeline/` | Live demo: `kafka_producer.py`, `spark_consumer.py`, `live_scorer.py`, `dashboard_server.py`, `pipeline_dashboard.html` (engineering context, not evidence) |
| `5-Results/` | Metric JSONs (`<lineage>__<model>__<split>__<k>__<seed>.json`) and decision-time curves |
| `docs/` | `results_provenance.md`, `prefix_rebuild_log.md`, `lr_xgboost_comparison.md` |
| `paper/` | IJACSA source, arXiv submission, `build.py` |

Scripts reference `1-Data/` and `4-Model-Artifacts/` via relative paths
(e.g. `../1-Data/events_clean.csv`), so **run each script from inside its own
folder** (`cd` into it first) — the `../` paths resolve from there.

### Live demo — run order
| File | Purpose | Run order |
|---|---|---|
| `2-Infra/docker-compose.yml` | Launches Kafka + MongoDB | Step 1 |
| `2-Completed-Session/train_lstm_model.py` | Trains LSTM, saves to `4-Model-Artifacts/` | Step 3 |
| `5-Pipeline/kafka_producer.py` | Replays events_clean.csv to Kafka | Step 4 |
| `5-Pipeline/spark_consumer.py` | Reads Kafka → features → MongoDB | Step 4 |
| `5-Pipeline/live_scorer.py` | Reads MongoDB → LSTM → predictions | Step 4 |

---

### Step 1 — Start infrastructure (one command)
```bash
cd 2-Infra
docker-compose up -d
```
Wait ~30 seconds for Kafka and MongoDB to be ready.
Verify: `docker ps` should show 3 containers running.

---

### Step 2 — Install Python dependencies
```bash
pip install -r 2-Infra/requirements.txt
```

---

### Step 3 — Train the LSTM (run once, takes ~2-3 minutes)
```bash
cd 2-Completed-Session
python train_lstm_model.py
```
Outputs directly into `4-Model-Artifacts/`: `lstm_model.pth`, `scaler.pkl`, `model_config.pkl`.
Retraining overwrites them in place (they are tracked in git) — `5-Pipeline/live_scorer.py` picks up the new weights automatically.

---

### Step 4 — Open 3 terminals for the live demo

```bash
cd 5-Pipeline
uvicorn dashboard_server:app --host 0.0.0.0 --port 8765
```
🌐 http://localhost:8765

**Terminal 1 — Kafka Producer**
```bash
cd 5-Pipeline
python kafka_producer.py --speed 99999
```
You will see events streaming in real time.

**Terminal 2 — Spark Consumer**
```bash
cd 5-Pipeline
python spark_consumer.py
```
You will see Spark processing micro-batches and writing to MongoDB.

**Terminal 3 — Live Scorer**
```bash
cd 5-Pipeline
python live_scorer.py
```
You will see real-time predictions appearing as sessions complete.

---

### Optional: MongoDB Compass GUI
Connect to `mongodb://localhost:27017` and open the `retail_rocket.sessions`
collection to watch documents appear and predictions fill in live.

---

### Stop everything
```bash
docker-compose down
```
