# Methodology

How the project was built, phase by phase: what was done, the results, the files involved and the reasoning behind each decision.

All numbers below are from the committed run on the synthetic twin dataset (14,892 clean trips). Re-running on the Kaggle file regenerates every table and figure.

---

## Phase 1 — Setup & data selection <a id="phase-1"></a>

**Done:** repository structure, `requirements.txt` / `requirements-dev.txt`, config module, README; dataset comparison; loaders for the Kaggle file and a synthetic twin; data dictionary.

**Dataset comparison**

| | Delivery Truck Trips (Kaggle) | Amazon Delivery Dataset (Kaggle) | DataCo Smart Supply Chain (Kaggle) |
|---|---|---|---|
| Size | ~6.9k truck bookings (small) | ~43k last-mile orders | ~180k orders |
| Domain fit | ✅ Indian inter-city trucking | ⚠️ last-mile food/retail (minutes) | ⚠️ global e-commerce (days) |
| Coordinates | ✅ origin + destination lat/lon | ✅ store + drop lat/lon | ⚠️ store only |
| Timestamps | ✅ dispatch, planned ETA, actual ETA | ✅ order/pickup time | ⚠️ dates, shipping days |
| Delay label | ✅ real (actual vs planned ETA) | ❌ no planned time → arbitrary threshold | ✅ `Late_delivery_risk` (days granularity) |
| Weather/route enrichment | ✅ easy (coordinates + dates) | ✅ | ❌ hard |
| Leakage risk | GPS ping columns (dropped) | weather/traffic recorded at delivery | delivery-status columns |

**Recommendation:** Delivery Truck Trips — it is the only one with a genuine *planned vs actual* label and inter-city routes, which is exactly our problem. Its weaknesses (small, short time window, messy) are handled by careful cleaning, and by a synthetic twin with the same schema covering 3 years (2023–2025) so the time-based split, seasonality and festivals can be demonstrated end to end.

**Files:** `src/config.py`, `src/data/load.py`, `src/data/geo.py`, `docs/data_dictionary.md`, `requirements*.txt`, `README.md`

**Rationale**
- We chose the dataset by *label quality* first: a model is only as good as its target, and only the truck dataset has a real planned-vs-actual delay.
- One canonical schema means the rest of the pipeline doesn't care which dataset is used — swapping in new data is a one-file change.
- The synthetic twin is generated from documented, realistic causal effects, so we know the "true" drivers and can check SHAP finds them (it does: rain, route history, supplier, festivals, rush hour).
- Personal data (driver names/phones) is never loaded — privacy by design.

---

## Phase 2 — EDA & cleaning

**Done:** 11 EDA plots (`reports/figures/eda_*.png`), auto-written insights (`reports/eda_insights.md`), notebook `01_eda.ipynb`, cleaning with a logged report.

**Key results:** 150 duplicates and 108 impossible trips removed, 150 distances and 450 vehicle types imputed → 14,892 trips; 35.8% late. Monsoon months 49% late vs 28% in dry months; heavy rain at destination 87% late vs 29% on dry days; festival windows 49% vs 34%; rush-hour dispatch 42% vs 25% at night; route delay rates range 0–88%.

**Rationale**
- Outliers were removed only when *physically impossible* (speed outside 5–90 km/h, negative duration) — genuine long delays are kept, because they are exactly what we want to predict.
- Delay = actual > planned (grace configurable) matches the dataset's own on-time flag, so the label is not our invention.
- Transit time is right-skewed, so MAE (robust, in hours) is our headline metric; RMSE is reported to show we also control big misses.
- EDA directly motivated the features: rain, monsoon, festivals, dispatch hour and route history all showed large effects.

---

## Phase 3 — Feature engineering & enrichment

**Done:** 32 features in 7 families (`docs/feature_list.md`), Open-Meteo weather with on-disk cache + climatology fallback, OSRM route with cache + fallback, `holidays` + custom Indian festival calendar, route/supplier history computed from training data only (out-of-fold), notebook `02_features.ipynb`.

**Key result:** out-of-fold encoding matters — the route delay rate correlates 0.47 with the label in-sample but only 0.36 out-of-fold (0.33 on validation). The in-sample number would have been leakage.

**Rationale**
- Every feature is available at dispatch time; a unit test fails if a post-delivery column is ever added.
- API results are cached and every external call has a fallback, so training is reproducible and the live API never crashes because a third-party service is down.
- Route history uses smoothing so a route seen 3 times can't get a 100% delay rate from 3/3 late trips.
- The same `base_features` + `RouteHistory` code runs at training and prediction time → no train/serve skew.

---

## Phase 4 — Modelling

**Done:** time-based split (train 2023–24: 9,126 · validation H1-2025: 2,824 · test H2-2025: 2,942); baseline → ensemble models for both tasks; tuning (Optuna TPE when installed, random search otherwise); every run logged (MLflow + `reports/metrics/runs.jsonl`); class weighting, F1-optimal threshold and probability calibration; models saved with joblib; notebook `03_modeling.ipynb`.

**Key results (test):** ETA MAE 3.70 h (planner 5.08 h, −27%), R² 0.933. Delay ROC-AUC 0.854, F1 0.724, recall 0.80. Calibration: Low band 14% late, Medium 53%, High 84%.

**Rationale**
- Random splits would let the model train on December and test on the previous June — it would "see the future". The time split mimics deployment.
- Model selection and threshold tuning use validation only; the test set is touched once, at the end.
- Imbalance (~36% late) is handled three ways: class weights, tuning the threshold for F1 instead of using 0.5, and re-calibrating probabilities so "80% risk" really means ~80%.
- Logistic regression won on delay risk — we kept the simpler model because it won on validation; complexity must earn its place.
- Tuning didn't beat the default Gradient Boosting on validation, so we kept defaults — reporting that honestly is better than over-fitting the validation set.

---

## Phase 5 — Explainability

**Done:** `src/explain/shap_utils.py` (exact linear SHAP, native TreeSHAP for XGBoost/LightGBM, `shap.TreeExplainer`, Saabas fallback), global importance + summary (beeswarm) plots, local waterfall plots, reason groups, top-3 human-readable reasons; notebook `04_explainability.ipynb`.

**Key result:** additivity check passes (base + ΣSHAP = prediction, max error 3e-13). Grouped delay drivers: route track record › rain › supplier › dispatch hour › monsoon › festival — the same factors that built the synthetic data.

**Rationale**
- SHAP values add up exactly to the prediction, so "+2.1 hrs from rain" is a literal decomposition, not a vague importance score.
- Correlated features (road km, straight-line km, drive time) split credit; we sum them into groups for manager-facing reasons — legal because SHAP is additive.
- Reasons are ranked by impact on *risk* but show the effect in *hours* from the ETA model — the two numbers managers care about.
- At-risk shipments show what pushes risk up; low-risk shipments show what keeps them safe.

---

## Phase 6 — API

**Done:** FastAPI app (`/health`, `/meta`, `/model/metrics`, `/predict`, `/predict/batch`, `/alert`), Pydantic validation (India bounding box, origin ≠ destination, unknown cities need coordinates, sane ranges), per-row error reporting in batches, 5 MB upload limit, CORS, 32 pytest tests.

**Rationale**
- Validation happens before the model: bad input returns a clear 422, never a 500.
- Business logic lives in `api/service.py`, separate from HTTP code, so it is unit-tested directly and reused by the dashboard.
- Batch scoring validates every row, scores valid rows in one vectorised call, and reports bad rows instead of failing the whole file.
- Models are loaded once per process (cached), not per request.

---

## Phase 7 — Dashboard

**Done:** Streamlit app with 4 pages — *Check a shipment* (form, ETA cards, risk gauge, reasons, route arc map, what-if weather), *Operations view* (CSV upload or sample batch, KPI tiles, risk-sorted table with progress bars, download, map), *Model insights* (metrics, comparison tables, calibration, SHAP/EDA figures), *About*. Calls the API; falls back to in-process mode when the API is down.

**Rationale**
- The dashboard is a thin client: all logic stays behind the API, so a mobile app or ERP could reuse it.
- The ops view is designed for action: sorted by risk, filterable, downloadable.
- What-if sliders (rain) let managers see how conditions change the prediction — a live demo of explainability.

---

## Phase 8 — Customer alerts

`src/llm/alerts.py` turns a prediction into a 2–4 sentence customer message. With `ARRIVO_LLM_API_KEY` set it calls an OpenAI-compatible LLM endpoint; otherwise it uses a deterministic template. Exposed at `POST /alert` and in the dashboard.

**Rationale**
- The LLM only rephrases facts we pass in (ETA, risk, reasons); the prompt forbids invented causes or promises — a hallucination guardrail.
- Ops wording ("supplier's past reliability") is translated into customer-friendly phrases ("loading time at the dispatch point").
- Any LLM failure falls back to the template, so the feature is always available.

---

## Phase 9 — Deployment & documentation

**Done:** one Docker image (API or dashboard by start command) that trains at build time if models are missing; `docker-compose.yml`; Render blueprint; Hugging Face Spaces single-container setup; README with architecture diagram and results; deployment guide.
