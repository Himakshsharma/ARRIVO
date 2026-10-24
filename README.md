# Arrivo — Shipment ETA Prediction & Delay Risk Engine

> Final-year B.Tech CSE (AI/ML) major project · Python · scikit-learn / XGBoost / LightGBM · SHAP · FastAPI · Streamlit · Docker

Late deliveries cost logistics companies money and customer trust. **Arrivo** looks at a truck shipment *at the moment it is dispatched* and predicts:

| Output | Type | Example |
|---|---|---|
| **ETA** | Regression | 85.6 hrs → arrives Thu 24 Jul, 08:05 |
| **Delay risk** | Calibrated classification | 97.6% chance of arriving late → **High** |
| **Top reasons** | SHAP explanations | "Rain at origin (32 mm) and destination (31 mm) (+10.5 hrs)" |

Users: logistics operations managers who need to spot at-risk shipments early and warn customers before they complain.

---

## Results (held-out test set: Jul–Dec 2025, 2,942 trips)

| Task | Model | Key metrics | vs. baseline |
|---|---|---|---|
| ETA (hours) | Gradient Boosting | **MAE 3.70 h**, RMSE 5.43 h, R² 0.933 | Planner's own ETA: MAE 5.08 h → **27% lower error** |
| Delay risk | Logistic Regression (class-weighted, calibrated) | **ROC-AUC 0.854**, PR-AUC 0.794, F1 0.724, Recall 0.80 | Random guess AUC 0.50 |

Risk bands are honest (test set): shipments rated **Low** were late 14% of the time, **Medium** 53%, **High** 84%.

<details><summary>Full comparison tables</summary>

**ETA regression** (selected on validation MAE)

| Model | Val MAE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Planner's ETA (no ML) | 5.06 | 5.08 | 7.40 | 0.875 |
| Linear Regression | 3.99 | 4.26 | 5.93 | 0.920 |
| Random Forest | 3.79 | 3.92 | 5.85 | 0.922 |
| **Gradient Boosting** | **3.41** | **3.70** | **5.43** | **0.933** |

**Delay classification** (selected on validation ROC-AUC; threshold tuned for F1 on validation)

| Model | Val AUC | Test AUC | Test F1 | Test Precision | Test Recall |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.833** | **0.854** | **0.724** | 0.659 | 0.803 |
| Random Forest | 0.822 | 0.837 | 0.701 | 0.665 | 0.741 |
| Gradient Boosting | 0.826 | 0.843 | 0.696 | 0.576 | 0.880 |

</details>

> **Note on these numbers.** They were produced in a build environment without XGBoost/LightGBM/SHAP/Optuna/MLflow, so scikit-learn's Gradient Boosting stood in for the boosted models and a built-in random search stood in for Optuna. On your machine `pip install -r requirements-dev.txt` adds XGBoost and LightGBM to the comparison automatically. Re-run `python -m src.pipeline` and update this table from `reports/metrics/*.csv`.

What drives delay risk (mean |SHAP|, grouped): route's track record › rain › supplier reliability › dispatch hour › monsoon › festivals.

---

## Architecture

```mermaid
flowchart LR
  subgraph Data
    A[Kaggle truck trips<br/>or synthetic twin] --> B[load.py<br/>canonical schema]
    B --> C[clean.py<br/>dedupe · outliers · targets]
    C --> D[enrich.py]
    W[Open-Meteo<br/>weather] -.cached.-> D
    O[OSRM<br/>road route] -.cached.-> D
    H[holidays + Indian<br/>festival calendar] --> D
  end
  D --> F[build_features.py<br/>time split · route history<br/>train-only, out-of-fold]
  F --> M1[train_eta.py<br/>LR → RF → XGB → LGBM]
  F --> M2[train_delay.py<br/>+ class weights · calibration]
  M1 & M2 --> T[(MLflow runs)]
  M1 & M2 --> J[(models/*.joblib<br/>model_card.json)]
  J --> P[predict.py + shap_utils.py<br/>ETA · risk · top-3 reasons]
  P --> API[FastAPI<br/>/predict · /predict/batch · /alert]
  API --> UI[Streamlit dashboard<br/>form · map · ops table · insights]
  API --> LLM[Customer alert<br/>Claude or template]
```

The **same feature code** runs in training and in the API, so there is no train/serve skew.

---

## Dataset

**Recommended: Kaggle "Delivery Truck Trips Data"** (Indian truck bookings with origin/destination coordinates, planned ETA, actual arrival, distance, vehicle type, supplier, material). It is the only candidate with a *real* planned-vs-actual delay label. See [docs/phase_summaries.md](docs/phase_summaries.md#phase-1) for the comparison with the Amazon Delivery and DataCo datasets.

**How data is loaded**
- Put the Kaggle file (e.g. `Delivery truck trip data.xlsx`) in `data/raw/` → it is detected automatically (any name containing "truck").
- If it is absent, Arrivo generates a **synthetic twin** with the same columns: ~15,000 trips across 27 Indian hubs (2023–2025), built from realistic effects (vehicle speeds, monsoon and regional rainfall normals, festivals, hills, route congestion, supplier reliability, breakdowns) plus deliberate data-quality problems for the cleaning step. The repository's committed results come from this synthetic data — say so clearly in your report, and re-run on the Kaggle file for your final numbers.

Data dictionary: [docs/data_dictionary.md](docs/data_dictionary.md) · Feature list: [docs/feature_list.md](docs/feature_list.md)

---

## Quick start

```bash
# 1. environment (Python 3.11 recommended)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 2. (optional) put the Kaggle file into data/raw/

# 3. data → features → models → SHAP → reports   (~5–10 min)
python -m src.pipeline --trials 30

# 4. API  → http://localhost:8000/docs
uvicorn api.main:app --reload --port 8000

# 5. dashboard (second terminal) → http://localhost:8501
streamlit run dashboard/app.py

# 6. tests and experiment tracking
pytest -q
mlflow ui --backend-store-uri ./mlruns
```

Or everything in Docker: `docker compose up --build` (API on 8000, dashboard on 8501).

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `ARRIVO_OFFLINE` | unset | `1` = never call Open-Meteo/OSRM (use cached data or climatology) |
| `ARRIVO_LIVE_ENRICHMENT` | `1` for Kaggle-trained models, `0` for synthetic | look up live weather/route at prediction time |
| `ARRIVO_API_URL` | `http://localhost:8000` | where the dashboard finds the API (falls back to in-process if unreachable) |
| `ANTHROPIC_API_KEY` | unset | enables AI-written customer alerts (template used otherwise) |
| `ARRIVO_LLM_MODEL` | `claude-haiku-5-5` | model for the alerts |

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + which models are loaded |
| GET | `/meta` | supported cities, vehicle and cargo types, risk bands |
| GET | `/model/metrics` | model card + comparison tables |
| POST | `/predict` | one shipment → ETA, arrival time, delay probability, risk level, top-3 reasons |
| POST | `/predict/batch` | CSV upload → all shipments ranked by risk; bad rows reported, not fatal |
| POST | `/alert` | bonus: plain-English customer message |

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "shipment_id": "ARV-1", "origin_city": "Mumbai", "destination_city": "Kolkata",
  "dispatch_time": "2025-07-20T18:30:00", "vehicle_type": "32 FT Multi-Axle",
  "material_type": "Cement & Steel", "supplier_id": "SUP007"}'
```

Unknown cities work too if you pass `origin_lat/origin_lon` (or `destination_lat/destination_lon`). Weather can be overridden (`dest_rain_mm`, …) for what-if analysis.

---

## Deployment

Step-by-step guides for **Render** (blueprint `render.yaml`, two free services) and **Hugging Face Spaces** (single container, `deploy/huggingface/`) are in [docs/deployment.md](docs/deployment.md). The Docker image trains the models at build time if none are present, so pickled models always match the installed library versions.

---

## Project structure

```
arrivo/
├── data/            raw/ · interim/ (clean data, API caches) · processed/ (features, sample batch)
├── notebooks/       01_eda · 02_features · 03_modeling · 04_explainability (executed, with outputs)
├── src/
│   ├── config.py    all paths, constants, feature lists, split dates
│   ├── data/        load.py · clean.py · enrich.py · geo.py (hubs, climate normals, festivals)
│   ├── features/    build_features.py (time/route features, route history, time split)
│   ├── models/      common.py · train_eta.py · train_delay.py · evaluate.py · predict.py
│   ├── explain/     shap_utils.py (SHAP back-ends, reason groups, plots)
│   ├── llm/         alerts.py (bonus: customer alerts)
│   ├── eda.py       11 EDA plots + auto-written insights
│   └── pipeline.py  end-to-end run
├── api/             main.py (FastAPI) · schemas.py (Pydantic) · service.py (logic)
├── dashboard/       app.py (Streamlit) · client.py (API client with embedded fallback)
├── models/          eta_model.joblib · delay_model.joblib · route_stats.joblib · model_card.json
├── reports/         figures/ (EDA, SHAP, test plots) · metrics/ (comparison tables, runs) · eda_insights.md
├── tests/           32 tests: data, features, leakage, explanations, predictor, service, API
├── docs/            data dictionary · feature list · phase summaries · deployment · viva prep
├── deploy/huggingface/   Space README + Dockerfile
├── Dockerfile · docker-compose.yml · render.yaml · Makefile
└── requirements.txt · requirements-dev.txt
```

## Screenshots

Add after running the dashboard: `docs/screenshots/predict.png`, `operations.png`, `insights.png`.

## Limitations & future work

- Weather "on the dispatch date" uses observed weather in training; in production you would use the forecast available at dispatch time (slightly less accurate).
- No live GPS: predictions are made once at dispatch. A natural extension is re-predicting mid-route from GPS pings.
- Logistic regression is linear in rainfall, so extreme rain (>200 mm) saturates risk at ~100%; tree models handle this more gracefully.
- Retrain periodically (e.g. monthly) and monitor drift — the delay rate already shifts between years.

## Team

[Your names] · Guide: [Guide name] · [College], [Year]
