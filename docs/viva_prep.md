# Viva & interview preparation

## 30-second pitch

> "Every logistics company promises an ETA, and roughly one in three truck shipments in our data misses it. Arrivo predicts, at the moment a truck is dispatched, when it will really arrive, how likely it is to be late, and *why*. We combine trip history with live weather, road routes and an Indian festival calendar, train gradient-boosted and calibrated models on a strict time-based split, and explain every prediction with SHAP — 'heavy rain at destination, +6 hours'. Our ETA error is 27% lower than the planner's own estimate, and when we call a shipment high-risk, it is late 84% of the time. Ops managers get a ranked list of at-risk shipments and a one-click customer alert, through a FastAPI service and a Streamlit dashboard deployed with Docker."

---

## Project report outline

1. **Introduction** — problem, motivation (cost of late deliveries, customer trust), objectives (ETA, delay risk, reasons), scope and users.
2. **Literature / existing approaches** — rule-based planners (distance ÷ speed), ETA in ride-hailing and parcel logistics, gradient boosting for tabular data, SHAP for explainability, calibration.
3. **Dataset** — candidate comparison, chosen dataset, synthetic twin and its generating assumptions, data dictionary, ethics/privacy.
4. **Methodology**
   4.1 Cleaning and target definition · 4.2 EDA findings · 4.3 Enrichment (weather, routes, calendar) · 4.4 Feature engineering and leakage prevention · 4.5 Time-based validation · 4.6 Models and tuning · 4.7 Imbalance, thresholds and calibration · 4.8 Explainability
5. **System design** — architecture diagram, API design, dashboard, LLM alert layer, Docker and deployment.
6. **Results** — comparison tables, test metrics, calibration table, SHAP global/local figures, example predictions.
7. **Testing** — unit/integration tests, leakage tests, additivity test.
8. **Discussion** — what worked, why logistic regression was competitive, limitations (forecast vs observed weather, no live GPS, synthetic data).
9. **Conclusion & future work** — mid-route re-prediction from GPS, drift monitoring, model registry, multi-modal freight.
10. **References** · **Appendix** (feature list, API examples, screenshots).

---

## 10-slide presentation outline

1. **Title** — Arrivo: Shipment ETA & Delay Risk Engine; team; guide.
2. **The problem** — late deliveries; one-line stat from EDA (36% late); who suffers (ops, customers).
3. **What we built** — 3 outputs (ETA, risk, reasons) + screenshot of the prediction page.
4. **Data** — dataset choice table; cleaning numbers; delay definition.
5. **What EDA told us** — 3 plots: monsoon by month, rain vs delay, festival effect.
6. **Features** — 7 feature families; leakage rule; route history out-of-fold (0.47 → 0.36).
7. **Models & validation** — time split diagram; comparison table; planner MAE 5.08 → 3.70 h.
8. **Trust: calibration & explainability** — risk-band table (14/53/84%), SHAP summary + one waterfall.
9. **Product demo** — architecture diagram; live demo: single prediction → ops batch → customer alert.
10. **Impact, limitations, future work** — and thank you / Q&A.

---

## 20 likely viva / interview questions

**1. Why a time-based split instead of a random split?**
In production we always predict the future from the past. A random split mixes months, so the model trains on December trips and is tested on June trips — it effectively sees the future (seasonality, drift, route stats), and scores look better than reality. Our split: train 2023–24, validate H1-2025, test H2-2025.

**2. What is data leakage, and how did you prevent it?**
Leakage is when training uses information that won't exist at prediction time. We (a) never use post-delivery columns (actual ETA, delay flags, GPS pings) — enforced by a test; (b) compute route/supplier history from training data only; (c) use out-of-fold history for training rows so a row never sees its own label; (d) tune and calibrate only on validation.

**3. How did you define "delayed"?**
`actual_hours > planned_hours` (grace period configurable, default 0), which matches the dataset's own on-time flag. The ETA target is the actual transit time in hours.

**4. Why MAE as the main ETA metric?**
It's in hours and easy to explain ("off by 3.7 hours on average"), and it's robust to the long right tail of breakdowns. RMSE is also reported because it penalises big misses, and R² shows explained variance.

**5. Why ROC-AUC, and why not accuracy, for delay risk?**
With 36% late, a model saying "on time" for everything gets 64% accuracy and is useless. ROC-AUC measures ranking quality across all thresholds — what an ops manager needs to prioritise. We also report precision, recall and F1 at the chosen threshold.

**6. How did you handle class imbalance?**
Class weights (`class_weight='balanced'`, `scale_pos_weight`), an F1-optimal threshold chosen on validation (0.34 instead of 0.5), and probability calibration. We avoided SMOTE because synthetic minority rows can distort time-ordered data.

**7. What is probability calibration and why did you need it?**
Class weighting inflates probabilities. Platt (sigmoid) scaling fitted on validation maps scores back to real frequencies. Proof: High-band shipments (mean predicted 82%) were actually late 84% of the time.

**8. Why did logistic regression beat Random Forest and Gradient Boosting for delay?**
On this data the delay signal is mostly additive (rain, route record, supplier, hour), the time split rewards simpler models that generalise, and the class-weighted linear model is well regularised. We pick the model that wins on validation — complexity must earn its place. For ETA, boosting won clearly because effects are non-linear.

**9. Explain SHAP in simple terms.**
Start from the average prediction. Each feature pushes it up or down; the SHAP value is the size of that push, computed fairly using game theory (Shapley values — average marginal contribution over all orders of adding features). They add up exactly to the prediction, which our test checks.

**10. Correlated features split SHAP credit — how did you deal with it?**
Road km, straight-line km and drive time are highly correlated, so individually their SHAP values can look odd. Since SHAP values are additive, we sum them into "reason groups" (route length, rain, route history…) for manager-facing reasons, and keep per-feature plots for analysis.

**11. Why gradient boosting for tabular data rather than deep learning?**
On medium-sized tabular data boosted trees are usually more accurate, train in seconds, need little preprocessing, handle non-linearities and interactions, and have exact, fast SHAP explanations. Deep nets need more data and tuning and are harder to explain.

**12. How does Optuna work?**
It uses a Tree-structured Parzen Estimator: it models which hyper-parameter regions gave good vs bad scores and samples more from the good regions — smarter than grid or random search for the same budget. We tune against validation MAE/AUC and keep the tuned model only if it actually beats the default.

**13. What does the route-history feature do and why smoothing?**
It's the training-period delay rate and overrun ratio for each origin→destination pair. Smoothing `(late + 20·global)/(n + 20)` stops a route seen twice from getting a 100% rate. Unseen routes get the global average.

**14. How is weather used, and isn't that leakage?**
We use the weather on the dispatch date at origin and destination. That information is available at dispatch from a forecast, so it is not leakage — but training uses observed weather, which is slightly better than a forecast. In production you'd train on archived forecasts; we list this as a limitation.

**15. What happens if Open-Meteo or OSRM is down?**
Every call is cached on disk and has a fallback: climatology weather from regional monthly normals, and haversine × detour factor for distance. `ARRIVO_OFFLINE=1` forces offline mode. The API never fails because a third party is down.

**16. How do training and the API stay consistent?**
Both use the same functions (`base_features`, `RouteHistory.transform`, `feature_matrix`) and the same feature list from `config.py`; models are saved as full scikit-learn pipelines including preprocessing. The Docker image trains at build time so library versions match the pickles.

**17. How would you monitor this model in production?**
Log every prediction and the actual outcome; track MAE, AUC and calibration weekly; monitor input drift (rainfall, route mix, delay rate — it already shifts from 38% in 2024 to 31% in 2025); retrain monthly or when drift crosses a threshold; register versions in MLflow.

**18. How did you validate the API?**
Pydantic validation (ranges, India bounding box, origin ≠ destination, unknown cities need coordinates), 32 pytest tests covering data cleaning, features, leakage, SHAP additivity, predictor contract, batch error handling and HTTP endpoints.

**19. Why did you use a synthetic dataset — doesn't that make results meaningless?**
The real Kaggle truck dataset is small and covers a short period, which makes a 3-way time split and seasonal/festival effects hard to demonstrate. The synthetic twin has the identical schema and is generated from documented, realistic effects, so the full pipeline can be shown end to end, and we can verify that SHAP recovers the true drivers. The pipeline runs unchanged on the Kaggle file — drop it in `data/raw/` and re-run.

**20. What would you improve with more time?**
Re-predict mid-journey from GPS pings; train on archived weather *forecasts*; quantile regression for ETA intervals ("arrives between 20:00 and 23:00"); per-region models; drift monitoring with Evidently; an MLflow model registry with staged rollouts; A/B test the customer alerts.

---

## Bonus quick-fire answers

- **R² of 0.93 — is that suspicious?** Distance explains most transit time; the real test is beating the planner (MAE 5.08 → 3.70 h), which we do.
- **Precision vs recall trade-off?** For ops, missing a late shipment (FN) is costlier than an extra check (FP), so our threshold favours recall (0.80).
- **Why FastAPI?** Async, automatic Swagger docs, Pydantic validation built in, very fast.
- **Why Streamlit?** A data-app UI in pure Python; it calls the API so the UI can be replaced without touching the model.
- **Random seed?** `RANDOM_SEED = 42` in `config.py`, used by the data generator, splits, models and tuning.
