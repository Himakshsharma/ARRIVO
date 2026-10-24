# EDA insights (auto-generated from the data)

1. **Data quality** – 150 duplicate bookings and 108 impossible trips (negative or >10-day durations, speeds outside 5–90 km/h) were removed; 150 missing distances were estimated from coordinates × median detour factor 1.203; 450 missing vehicle types became 'Unknown'. 14,892 clean trips remain.
2. **Targets** – 35.8% of trips are late (moderate imbalance, ~1 late : 2 on-time). Transit time is right-skewed (skewness 1.06): most trips take 15–45 hrs, a long tail comes from long-haul routes and breakdowns, so MAE is a fairer headline metric than RMSE.
3. **Monsoon** – average delay rate is 49% in Jun–Sep vs 28% in the dry months: the single strongest seasonal signal. Weather features are essential.
4. **Rain at destination** – delay rate rises from 29% on dry days to 87% under heavy rain.
5. **Festivals** – trips dispatched in a festival window are late 49% of the time vs 34% normally, so a custom Indian festival calendar adds signal the plain holiday list misses.
6. **Dispatch time** – rush-hour dispatches are late 42% of the time vs 25% for night runs (empty highways, no city entry restrictions).
7. **Routes differ a lot** – among routes with ≥30 trips, delay rates range from 0% to 88%. That justifies a route-history feature (computed from training data only).
8. **Vehicle / cargo** – delay rates vary by cargo (e.g. heavy 'Cement & Steel' 44%) and vehicle size, so both are kept as categorical features.
9. **Drift** – yearly delay rate: {'2023': 38.4, '2024': 38.6, '2025': 31.4}. Operations change over time, which is exactly why we use a time-based split: a random split would hide this and overstate accuracy.
10. **The planner is systematically optimistic** in bad conditions (EDA plot 3): planned time is roughly distance ÷ nominal speed and ignores weather, festivals and route quirks – the gap our models learn.
