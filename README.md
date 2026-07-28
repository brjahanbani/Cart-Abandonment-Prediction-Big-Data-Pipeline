# Cart Abandonment Prediction — Big Data Pipeline
## Kafka + Spark + MongoDB + LSTM

### Folder layout
| Folder | Contents |
|---|---|
| `0-Offline/` | `offline_cart_abandonment_prediction.py` — offline data-mining comparison (Logistic Regression, Decision Tree, Random Forest, XGBoost) on `1-Data/cart_session_features.csv` |
| `1-Data/` | `events_clean.csv`, `cart_session_features.csv` — from Stage 1 preprocessing |
| `2-Infra/` | `docker-compose.yml`, `requirements.txt` |
| `3-Training/` | `train_model.py` (LSTM), plus baseline scripts (`train_logistic_online_baseline.py`, `train_xgboost_corrected_baseline.py`) |
| `4-Model-Artifacts/` | `lstm_model.pth`, `model_config.pkl`, `scaler.pkl` — produced by `3-Training/train_model.py` |
| `5-Pipeline/` | `kafka_producer.py`, `spark_consumer.py`, `live_scorer.py`, `dashboard_server.py`, `pipeline_dashboard.html` |

Scripts read their input files (data/model) by bare filename from the current
directory, so **run each script from inside its own folder** (`cd` into it
first). `events_clean.csv` and the model artifacts are hardlinked into
`5-Pipeline/` so `kafka_producer.py` and `live_scorer.py` can find them
without any path changes.

### Project files & run order
| File | Purpose | Run order |
|---|---|---|
| `0-Offline/offline_cart_abandonment_prediction.py` | Offline model comparison (data mining stage) | Step 0 (optional) |
| `2-Infra/docker-compose.yml` | Launches Kafka + MongoDB | Step 1 |
| `3-Training/train_model.py` | Trains LSTM, saves lstm_model.h5 | Step 2 |
| `5-Pipeline/kafka_producer.py` | Replays events_clean.csv to Kafka | Step 3 |
| `5-Pipeline/spark_consumer.py` | Reads Kafka → features → MongoDB | Step 4 |
| `5-Pipeline/live_scorer.py` | Reads MongoDB → LSTM → predictions | Step 5 |

---

### Step 0 — Offline data-mining comparison (optional)
```bash
cd 0-Offline
python offline_cart_abandonment_prediction.py
```
Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost on
`1-Data/cart_session_features.csv` and saves comparison charts
(`dm_results_comparison.png`, `dm_roc_curves.png`, `dm_feature_importance.png`)
into `0-Offline/`. Independent of the live pipeline below.

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
cd 3-Training
python train_model.py
```
Outputs (into `3-Training/`): `lstm_model.pth`, `scaler.pkl`, `model_config.pkl`.
Move/copy them into `4-Model-Artifacts/` (and re-link into `5-Pipeline/`) if you retrain.

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
