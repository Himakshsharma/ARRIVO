# Feature list (32 features, all known at dispatch time)

Defined once in `src/config.py` (`NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`) and used by both training and the API.
`check_no_leakage()` fails the pipeline if any post-delivery column is added.

## Route & schedule
| Feature | How it is computed | Why it helps |
|---|---|---|
| `road_distance_km` | dataset distance, or OSRM, or haversine × detour | longer trips take longer and accumulate more risk |
| `haversine_km` | great-circle distance between coordinates | distance even when road km is missing |
| `route_detour_ratio` | road km ÷ straight-line km | winding / mountain roads are slower |
| `osrm_duration_hrs` | OSRM free-flow driving time (fallback: km ÷ 55) | road-type information (expressway vs state road) |
| `planned_hours` | planner's ETA − dispatch | the planner's view; the model learns its systematic errors |
| `planned_speed_kmph` | road km ÷ planned hours | a tight (fast) plan is easier to miss |

## Time of dispatch
| Feature | Definition | Why |
|---|---|---|
| `dispatch_hour` | 0–23 | traffic and city entry restrictions vary by hour |
| `dispatch_weekday` | 0 = Monday | weekly patterns |
| `dispatch_month` | 1–12 | seasonality |
| `is_weekend` | Sat/Sun | lighter traffic, fewer working warehouses |
| `is_peak_hour` | 08–10 or 17–20 h | slow exit from the origin city |
| `is_night_dispatch` | 22–04 h | empty highways |

## Calendar & season
| Feature | Definition | Why |
|---|---|---|
| `is_national_holiday` | `holidays.India` (fallback: 26 Jan, 15 Aug, 2 Oct) | closures |
| `is_festival_window` | within ±2–5 days of Diwali, Dussehra, Holi, Ganesh Chaturthi, Eid, Pongal, Christmas | freight congestion, driver shortages |
| `days_to_festival` | days to the nearest festival (capped at 30) | build-up before festivals |
| `is_monsoon` | June–September | rain, waterlogging, landslides |

## Geography
| Feature | Definition | Why |
|---|---|---|
| `origin_is_hilly`, `dest_is_hilly` | hill-region flag | slower ghats / mountain roads |
| `origin_region`, `dest_region` (categorical) | nearest hub's region | regional road quality and climate |

## Weather on the dispatch date (Open-Meteo, cached)
| Feature | Definition | Why |
|---|---|---|
| `origin_rain_mm`, `dest_rain_mm` | daily precipitation | rain slows trucks and causes waterlogging |
| `origin_temp_c`, `dest_temp_c` | daily mean temperature | heat waves, fog in winter |
| `max_wind_kmph` | max of origin/destination daily max wind | storms / cyclones |
| `is_heavy_rain` | either side ≥ 35 mm | simple, explainable flag |

## History (computed from TRAINING data only)
| Feature | Definition | Why |
|---|---|---|
| `route_hist_delay_rate` | smoothed share of late trips on this origin→destination route | some routes are chronically late |
| `route_hist_overrun_ratio` | smoothed mean of actual ÷ planned hours on the route | how much the plan is usually missed by |
| `route_hist_trips` | number of training trips on the route | confidence / familiarity |
| `supplier_hist_delay_rate` | smoothed late share for the supplier | reliable vs unreliable transporters |

Smoothing: `(late trips + 20 × global rate) ÷ (trips + 20)` so rarely-seen routes are pulled towards the average.
Training rows get **out-of-fold** values (5 folds) so a row never sees its own label; unseen routes/suppliers get the global rate.

## Vehicle & cargo (categorical, one-hot encoded)
| Feature | Example values |
|---|---|
| `vehicle_type` | 32 FT Multi-Axle, 24 FT Open Body, 14 FT Tata 407, Unknown |
| `material_type` | FMCG, Auto Parts, Electronics, Pharma, Textiles, Cement & Steel, Perishables |

## Explicitly NOT used (leakage)
`actual_hours`, `trip_end_time`, `actual_eta`, `delay_hours`, `is_delayed`, `ontime`, `delay`, GPS ping columns — all only known after (or during) the trip.
