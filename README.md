# Cart Abandonment Prediction — Big Data Pipeline
## Kafka + Spark + MongoDB + LSTM

### Project files
| File | Purpose | Run order |
|---|---|---|
| `docker-compose.yml` | Launches Kafka + MongoDB | Step 1 |
| `train_model.py` | Trains LSTM, saves lstm_model.h5 | Step 2 |
| `kafka_producer.py` | Replays events_clean.csv to Kafka | Step 3 |
| `spark_consumer.py` | Reads Kafka → features → MongoDB | Step 4 |
| `live_scorer.py` | Reads MongoDB → LSTM → predictions | Step 5 |

### Data files required (same folder as scripts)
- `events_clean.csv` — from Stage 1 preprocessing
- `cart_session_features.csv` — from Stage 1 preprocessing

---

### Step 1 — Start infrastructure (one command)
```bash
docker-compose up -d
```
Wait ~30 seconds for Kafka and MongoDB to be ready.
Verify: `docker ps` should show 3 containers running.

---

### Step 2 — Install Python dependencies
```bash
pip install -r requirements.txt
```

---

### Step 3 — Train the LSTM (run once, takes ~2-3 minutes)
```bash
python train_model.py
```
Outputs: `lstm_model.h5` and `model_config.pkl`

---

### Step 4 — Open 3 terminals for the live demo

**Terminal 1 — Kafka Producer**
```bash
python kafka_producer.py --speed 99999
```
You will see events streaming in real time.

**Terminal 2 — Spark Consumer**
```bash
python spark_consumer.py
```
You will see Spark processing micro-batches and writing to MongoDB.

**Terminal 3 — Live Scorer**
```bash
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
